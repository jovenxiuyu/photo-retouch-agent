"""The studio workstation: quote first, spend only on an explicit confirmation.

Every test here stays offline. ``/studio/quote`` and ``/studio/status`` are pure
catalogue math, and the two ``/studio/render`` cases are refused *before* any
request is built -- one for the missing ``confirmPaid``, one for the missing API
key -- so a green suite can never bill anything.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_storage_service, get_studio_service
from app.core.config import Settings
from app.main import create_app
from app.services.storage import StorageService
from app.services.studio_service import StudioService


@pytest.fixture()
def studio_client(temp_settings: Settings) -> Iterator[TestClient]:
    """A client whose studio service is pinned to the offline test settings.

    The studio service is overridden rather than left on the container: the
    container reads the real ``.env``, and a render test must never be able to
    reach a live key.
    """
    app = create_app(_settings=temp_settings)
    storage = StorageService(temp_settings)
    app.dependency_overrides[get_storage_service] = lambda: storage
    app.dependency_overrides[get_studio_service] = lambda: StudioService(
        temp_settings, storage=storage
    )
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_status_lists_the_catalogue_without_a_key(studio_client: TestClient) -> None:
    response = studio_client.get("/api/studio/status")

    assert response.status_code == 200
    body = response.json()
    assert body["configured"] is False
    assert body["provider"] == "siliconflow"
    assert body["defaultEditModel"] == "Qwen/Qwen-Image-Edit-2509"
    assert body["defaultTextModel"] == "Qwen/Qwen-Image"
    assert body["maxRenders"] == 4
    assert {model["mode"] for model in body["models"]} == {"edit", "text2image"}
    edit = next(m for m in body["models"] if m["id"] == "Qwen/Qwen-Image-Edit-2509")
    assert edit["requiresSource"] is True


def test_quote_counts_one_paid_request_per_render(studio_client: TestClient) -> None:
    response = studio_client.post(
        "/api/studio/quote",
        json={"prompt": "一只在窗台上晒太阳的橘猫", "mode": "text2image", "count": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "Qwen/Qwen-Image"
    assert body["mode"] == "text2image"
    assert body["count"] == 2
    assert body["paidRequests"] == 2
    assert body["imageSize"] == "1328x1328"
    assert body["requiresConfirmation"] is True
    assert "2 次真实付费请求" in body["note"]


def test_quote_clamps_a_missing_count_to_one(studio_client: TestClient) -> None:
    response = studio_client.post(
        "/api/studio/quote",
        json={"prompt": "夜景霓虹街道路口", "mode": "text2image", "count": 0},
    )

    assert response.status_code == 200
    assert response.json()["paidRequests"] == 1


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"prompt": "   ", "mode": "text2image"}, "提示词不能为空"),
        ({"prompt": "换一件大衣", "mode": "edit"}, "请先上传一张输入图"),
        ({"prompt": "出四张", "mode": "text2image", "count": 5}, "一次最多生成 4 张"),
        (
            {
                "prompt": "出图",
                "mode": "text2image",
                "modelId": "Tongyi-MAI/Z-Image",
                "imageSize": "1328x1328",
            },
            "不支持这个尺寸",
        ),
        (
            {"prompt": "出图", "mode": "text2image", "modelId": "no/such-model"},
            "还没有接入模型",
        ),
        (
            {"prompt": "出图", "mode": "text2image", "modelId": "Qwen/Qwen-Image-Edit-2509"},
            "不属于",
        ),
    ],
)
def test_quote_refuses_invalid_requests(
    studio_client: TestClient, payload: dict, detail: str
) -> None:
    response = studio_client.post("/api/studio/quote", json=payload)

    assert response.status_code == 400
    assert detail in response.json()["detail"]


def test_render_without_confirmation_is_refused(studio_client: TestClient) -> None:
    response = studio_client.post(
        "/api/studio/render",
        json={"prompt": "一只在窗台上晒太阳的橘猫", "mode": "text2image", "confirmPaid": False},
    )

    assert response.status_code == 400
    assert "未确认费用" in response.json()["detail"]


def test_render_without_a_key_reports_the_missing_setting(studio_client: TestClient) -> None:
    response = studio_client.post(
        "/api/studio/render",
        json={"prompt": "一只在窗台上晒太阳的橘猫", "mode": "text2image", "confirmPaid": True},
    )

    assert response.status_code == 400
    assert "SILICONFLOW_API_KEY" in response.json()["detail"]


def test_history_is_empty_before_the_first_render(studio_client: TestClient) -> None:
    response = studio_client.get("/api/studio/history")

    assert response.status_code == 200
    assert response.json() == []
