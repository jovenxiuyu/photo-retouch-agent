"""SiliconFlow provider tests: vision brain plus image editing.

Every HTTP call below is served by ``httpx.MockTransport``. No request leaves the
machine and no billable API is touched.
"""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_agent_brain_factory,
    get_job_store,
    get_provider_factory,
    get_storage_service,
)
from app.brains.factory import AgentBrainFactory
from app.core.config import Settings
from app.main import create_app
from app.providers.factory import ImageProviderFactory
from app.providers.mock_image_provider import MockImageProvider
from app.providers.siliconflow_image_provider import (
    SiliconFlowImageProvider,
    SiliconFlowProviderError,
)
from app.schemas.retouch import RetouchPlan
from app.services.job_store import JobStore
from app.services.storage import StorageService
from tests.conftest import make_image_bytes, make_landscape_bytes, make_portrait_bytes

VISION_MODEL = "Qwen/Qwen3.5-35B-A3B"
IMAGE_MODEL = "Qwen/Qwen-Image-Edit-2509"
API_KEY = "sk-siliconflow-test-secret"
BASE_URL = "https://api.siliconflow.cn/v1"


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "data_dir": tmp_path / "data",
        "image_provider": "mock",
        "brain_provider": "local",
        "dashscope_api_key": None,
        "ark_api_key": None,
        "zhipu_api_key": None,
        "deepseek_api_key": None,
        "siliconflow_api_key": None,
        "rate_limit_per_second": 0,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _plan(plan_id: str = "natural", prompt: str = "自然提亮肤色并保留本人特征。") -> RetouchPlan:
    return RetouchPlan(
        plan_id=plan_id,
        domain_type="portrait",
        title=plan_id,
        description=plan_id,
        intensity="natural",
        edit_prompt=prompt,
        negative_prompt="避免改变五官",
        expected_changes=["提亮"],
    )


def _install_mock_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Force every provider/brain client built in-process onto a mock transport."""
    real_client = httpx.AsyncClient

    def build(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr("app.brains.openai_compatible.httpx.AsyncClient", build)


def test_siliconflow_is_registered_for_vision_and_editing(tmp_path: Path) -> None:
    without_key = _settings(tmp_path)
    action_caps = ImageProviderFactory(without_key).capabilities()
    entry = next(item for item in action_caps["actionProviders"] if item["id"] == "siliconflow")
    assert entry["model"] == IMAGE_MODEL
    assert entry["configured"] is False
    assert entry["requiresApiKey"] is True
    assert action_caps["siliconflowConfigured"] is False
    assert action_caps["siliconflowImageModel"] == IMAGE_MODEL

    brain_caps = AgentBrainFactory(without_key).capabilities()
    brain_entry = next(
        item for item in brain_caps["brainProviders"] if item["id"] == "siliconflow"
    )
    assert brain_entry["model"] == VISION_MODEL
    assert brain_entry["visionMode"] == "direct"
    assert brain_entry["configured"] is False
    assert brain_caps["siliconflowVisionConfigured"] is False

    with_key = _settings(tmp_path, siliconflow_api_key=API_KEY)
    assert ImageProviderFactory(with_key).capabilities()["siliconflowConfigured"] is True
    assert AgentBrainFactory(with_key).capabilities()["siliconflowVisionConfigured"] is True


def test_missing_key_is_rejected_before_any_request(tmp_path: Path) -> None:
    config = _settings(tmp_path)

    with pytest.raises(HTTPException) as action_error:
        ImageProviderFactory(config).create("siliconflow")
    assert action_error.value.status_code == 400
    assert "SILICONFLOW_API_KEY" in action_error.value.detail
    assert API_KEY not in action_error.value.detail

    with pytest.raises(HTTPException) as brain_error:
        AgentBrainFactory(config).create("siliconflow")
    assert brain_error.value.status_code == 400
    assert "SILICONFLOW_API_KEY" in brain_error.value.detail
    assert "X-Agent-API-Key" in brain_error.value.detail


def test_siliconflow_brain_sends_the_real_source_photo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "portrait.png"
    source.write_bytes(make_portrait_bytes())
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "editPrompt": "提亮面部并保留本人特征",
                                    "negativePrompt": "不要磨皮过度",
                                    "expectedChanges": ["面部提亮"],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    _install_mock_transport(monkeypatch, handler)
    brain = AgentBrainFactory(_settings(tmp_path, siliconflow_api_key=API_KEY)).create(
        "siliconflow"
    )

    plan = asyncio.run(brain.optimize(source, _plan(), "黑眼圈淡一点"))

    assert captured["url"] == f"{BASE_URL}/chat/completions"
    payload = captured["payload"]
    assert payload["model"] == VISION_MODEL
    assert payload["messages"][0]["role"] == "system"
    image_part = payload["messages"][1]["content"][0]
    assert image_part["type"] == "image_url"
    data_url = image_part["image_url"]["url"]
    assert data_url.startswith("data:image/png;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == source.read_bytes()
    assert plan.edit_prompt == "提亮面部并保留本人特征"
    assert brain.provider_name == "siliconflow"
    assert brain.vision_mode == "direct"


def test_image_edit_sends_image_and_prompt_without_unsupported_size(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "outputs" / "out_test.png"
    source.write_bytes(make_image_bytes())
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers["authorization"]
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                headers={"x-request-id": "req-123"},
                json={
                    "images": [{"url": "https://images.example.test/result.png"}],
                    "timings": {"inference": 1.5},
                },
            )
        return httpx.Response(
            200,
            content=make_image_bytes(),
            headers={"content-type": "image/png"},
        )

    provider = SiliconFlowImageProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=IMAGE_MODEL,
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(provider.edit_image(source, output, _plan(), "背景干净一点"))

    assert captured["url"] == f"{BASE_URL}/images/generations"
    assert captured["authorization"] == f"Bearer {API_KEY}"
    payload = captured["payload"]
    assert payload["model"] == IMAGE_MODEL
    assert payload["image"].startswith("data:image/png;base64,")
    assert base64.b64decode(payload["image"].split(",", 1)[1]) == source.read_bytes()
    assert "背景干净一点" in payload["prompt"]
    assert "image_size" not in payload
    assert "size" not in payload
    assert "image2" not in payload


def test_result_is_saved_locally_with_provenance_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    outputs = tmp_path / "outputs"
    output = outputs / "out_abc.png"
    source.write_bytes(make_image_bytes())
    result_bytes = make_image_bytes(size=(16, 16))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                headers={"x-request-id": "req-abc"},
                json={"images": [{"url": "https://images.example.test/result.png"}]},
            )
        return httpx.Response(
            200, content=result_bytes, headers={"content-type": "image/png"}
        )

    provider = SiliconFlowImageProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=IMAGE_MODEL,
        transport=httpx.MockTransport(handler),
    )

    asyncio.run(provider.edit_image(source, output, _plan("bold", "大胆造型。"), ""))

    assert output.read_bytes() == result_bytes
    metadata_path = outputs / "_meta" / f"{output.name}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["provider"] == "siliconflow"
    assert metadata["model"] == IMAGE_MODEL
    assert metadata["planId"] == "bold"
    assert metadata["requestId"] == "req-abc"
    assert metadata["imageCount"] == 1
    assert metadata["bytes"] == len(result_bytes)
    assert API_KEY not in metadata_path.read_text(encoding="utf-8")
    # Metadata must never be mistaken for the generated picture itself.
    assert [path.name for path in outputs.glob(f"{output.stem}.*")] == [output.name]


def test_three_directions_produce_independent_prompts(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    outputs = tmp_path / "outputs"
    source.write_bytes(make_image_bytes())
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            prompts.append(json.loads(request.content)["prompt"])
            return httpx.Response(
                200, json={"images": [{"url": "https://images.example.test/r.png"}]}
            )
        return httpx.Response(
            200, content=make_image_bytes(), headers={"content-type": "image/png"}
        )

    provider = SiliconFlowImageProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=IMAGE_MODEL,
        transport=httpx.MockTransport(handler),
    )
    directions = [
        ("natural", "自然优化：轻微提亮与控油。"),
        ("feature", "特征强化：放大本人辨识度。"),
        ("bold", "大胆突破：尝试明显不同的造型方向。"),
    ]

    for plan_id, prompt in directions:
        asyncio.run(
            provider.edit_image(
                source, outputs / f"out_{plan_id}.png", _plan(plan_id, prompt), ""
            )
        )

    assert len(prompts) == 3
    assert len(set(prompts)) == 3
    for (_, prompt), sent in zip(directions, prompts):
        assert prompt in sent
    # Every direction keeps the same identity-preservation constraints.
    assert all("不换脸" in sent for sent in prompts)


def test_failure_does_not_fall_back_to_mock_or_text_to_image(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "outputs" / "out_fail.png"
    source.write_bytes(make_image_bytes())
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    provider = SiliconFlowImageProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=IMAGE_MODEL,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(SiliconFlowProviderError) as error:
        asyncio.run(provider.edit_image(source, output, _plan(), ""))

    assert "rate limited" in str(error.value)
    assert not output.exists()
    assert len(requests) == 1
    assert requests[0]["image"].startswith("data:image/png;base64,")


def test_provider_errors_never_leak_the_api_key(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "outputs" / "out.png"
    source.write_bytes(make_image_bytes())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": f"bad key {API_KEY}"}})

    provider = SiliconFlowImageProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=IMAGE_MODEL,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(SiliconFlowProviderError) as error:
        asyncio.run(provider.edit_image(source, output, _plan(), ""))

    assert API_KEY not in str(error.value)
    assert "[redacted]" in str(error.value)


def test_factory_returns_a_siliconflow_provider_and_keeps_mock_available(
    tmp_path: Path,
) -> None:
    config = _settings(tmp_path, siliconflow_api_key=API_KEY)

    provider = ImageProviderFactory(config).create("siliconflow")

    assert isinstance(provider, SiliconFlowImageProvider)
    assert provider.model_name == IMAGE_MODEL
    assert provider.endpoint == f"{BASE_URL}/images/generations"
    assert isinstance(ImageProviderFactory(config).create("mock"), MockImageProvider)


def _portrait_plan(client: TestClient) -> tuple[str, dict]:
    upload = client.post(
        "/api/photos/upload",
        files={"file": ("portrait.png", make_portrait_bytes(), "image/png")},
    )
    image_id = upload.json()["imageId"]
    analysis = client.post("/api/photos/analyze", json={"imageId": image_id})
    plans = client.post("/api/retouch/plans", json={"analysis": analysis.json()})
    return image_id, plans.json()[0]


def test_api_rejects_siliconflow_actions_without_a_key(client: TestClient) -> None:
    image_id, plan = _portrait_plan(client)

    response = client.post(
        "/api/retouch/jobs",
        headers={"X-Agent-Provider": "local", "X-Action-Provider": "siliconflow"},
        json={"sourceImageId": image_id, "plan": plan, "userInstruction": ""},
    )

    assert response.status_code == 400
    assert "SILICONFLOW_API_KEY" in response.json()["detail"]
    assert "secret" not in response.text.lower()


def test_portrait_and_landscape_routing_still_hold_with_siliconflow_configured(
    tmp_path: Path,
) -> None:
    config = _settings(tmp_path, siliconflow_api_key=API_KEY)
    app = create_app(_settings=config)
    app.dependency_overrides[get_storage_service] = lambda: StorageService(config)
    app.dependency_overrides[get_job_store] = lambda: JobStore(config)
    app.dependency_overrides[get_provider_factory] = lambda: ImageProviderFactory(config)
    app.dependency_overrides[get_agent_brain_factory] = lambda: AgentBrainFactory(config)

    with TestClient(
        app, headers={"X-Agent-Provider": "local", "X-Action-Provider": "mock"}
    ) as client:
        portrait_id = client.post(
            "/api/photos/upload",
            files={"file": ("portrait.png", make_portrait_bytes(), "image/png")},
        ).json()["imageId"]
        landscape_id = client.post(
            "/api/photos/upload",
            files={"file": ("landscape.png", make_landscape_bytes(), "image/png")},
        ).json()["imageId"]

        portrait = client.post(
            "/api/photos/analyze", json={"imageId": portrait_id, "domainType": "auto"}
        ).json()
        landscape = client.post(
            "/api/photos/analyze", json={"imageId": landscape_id, "domainType": "auto"}
        ).json()

        assert portrait["domainType"] == "portrait"
        assert landscape["domainType"] == "general"

        portrait_plans = client.post(
            "/api/retouch/plans", json={"analysis": portrait}
        ).json()
        assert [plan["planId"] for plan in portrait_plans] == ["natural", "avatar", "mood"]

    app.dependency_overrides.clear()
