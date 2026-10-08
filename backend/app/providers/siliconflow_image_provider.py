from __future__ import annotations
import base64
import json
import mimetypes
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import httpx
from PIL import Image, UnidentifiedImageError

from app.schemas.retouch import RetouchPlan

# SiliconFlow's Qwen-Image-Edit family rejects `image_size`. The constant below is
# deliberately absent from the request body and only documents that omission.
UNSUPPORTED_PARAMETERS = ("image_size",)

METADATA_DIRNAME = "_meta"


class SiliconFlowProviderError(RuntimeError):
    """Safe SiliconFlow error that never contains API credentials."""


class SiliconFlowImageProvider:
    """SiliconFlow image-edit provider.

    Sends a JSON POST to ``{base_url}/images/generations`` with the flat body
    ``{model, prompt, image}``. ``image`` is the source photo encoded as a data URL,
    so the edit always starts from the real input image instead of a text prompt.
    Exactly one output image is requested per call; ``image2``/``image3`` are
    reserved for a future multi-reference feature and are unused by the UI.
    """

    provider_name = "siliconflow"
    output_extension = ".png"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout_seconds: float = 180,
        download_limit_bytes: int = 30 * 1024 * 1024,
        reference_image_paths: Sequence[Path] = (),
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.download_limit_bytes = download_limit_bytes
        self.reference_image_paths = tuple(reference_image_paths)
        self._transport = transport

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/images/generations"

    async def edit_image(
        self,
        source_path: Path,
        output_path: Path,
        plan: RetouchPlan,
        user_instruction: str,
    ) -> None:
        payload = self._build_payload(source_path, plan, user_instruction)

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            transport=self._transport,
        ) as client:
            response = await client.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            result = self._parse_response(response)
            image_url = self._extract_image_url(result)
            image_response = await client.get(image_url)
            self._save_download(image_response, output_path)

        self._write_metadata(
            output_path,
            plan=plan,
            response=response,
            payload=result,
            content_type=image_response.headers.get("content-type", ""),
        )

    def _build_payload(
        self,
        source_path: Path,
        plan: RetouchPlan,
        user_instruction: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": self._build_prompt(plan, user_instruction),
            "image": self._encode_image(source_path),
        }
        # Optional multi-reference inputs: omitted entirely unless explicitly set,
        # because the single-image flow is the only one the UI drives today.
        for index, reference in enumerate(self.reference_image_paths[:2], start=2):
            payload[f"image{index}"] = self._encode_image(reference)
        for unsupported in UNSUPPORTED_PARAMETERS:
            payload.pop(unsupported, None)
        return payload

    @staticmethod
    def _encode_image(path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type or not mime_type.startswith("image/"):
            raise SiliconFlowProviderError("Unsupported input image type.")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _build_prompt(plan: RetouchPlan, user_instruction: str) -> str:
        instructions = [
            plan.edit_prompt,
            "在原始照片的基础上编辑，必须保持同一个人：人物身份、面部几何、五官比例、"
            "年龄表现与姿态都不改变；不换脸、不替换成陌生人、不改变性别。",
            "不得新增或删除人物，不得添加文字或水印；避免塑料皮肤与明显 AI 生成痕迹。",
        ]
        if user_instruction.strip():
            instructions.append(f"用户补充要求：{user_instruction.strip()}")
        return "\n".join(instructions)

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise SiliconFlowProviderError(
                f"SiliconFlow API returned invalid JSON (HTTP {response.status_code})."
            ) from exc

        if response.is_error or payload.get("error"):
            error = payload.get("error") or {}
            if isinstance(error, dict):
                message = error.get("message") or error.get("code")
            else:
                message = str(error)
            message = message or payload.get("message") or "Request failed."
            message = str(message).replace(self._api_key, "[redacted]")
            raise SiliconFlowProviderError(
                f"SiliconFlow API call failed (HTTP {response.status_code}): {message}"
            )
        if not isinstance(payload, dict):
            raise SiliconFlowProviderError("SiliconFlow API returned an unexpected payload.")
        return payload

    @staticmethod
    def _extract_image_url(payload: dict[str, Any]) -> str:
        candidates: list[Any] = []
        for key in ("images", "data"):
            entries = payload.get(key)
            if isinstance(entries, list) and entries:
                candidates.append(entries[0])
        for entry in candidates:
            if not isinstance(entry, dict):
                continue
            url = entry.get("url") or entry.get("imageUrl")
            if url:
                return str(url)
        raise SiliconFlowProviderError(
            "SiliconFlow API response did not contain an output image."
        )

    def _save_download(self, response: httpx.Response, output_path: Path) -> None:
        if response.is_error:
            raise SiliconFlowProviderError(
                f"Failed to download SiliconFlow output image (HTTP {response.status_code})."
            )
        content_type = response.headers.get("content-type", "")
        if content_type and not content_type.startswith("image/"):
            raise SiliconFlowProviderError("SiliconFlow output URL did not return an image.")
        if len(response.content) > self.download_limit_bytes:
            raise SiliconFlowProviderError("SiliconFlow output image exceeded the download limit.")
        try:
            with Image.open(BytesIO(response.content)) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise SiliconFlowProviderError(
                "SiliconFlow output image could not be decoded."
            ) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)

    def _write_metadata(
        self,
        output_path: Path,
        plan: RetouchPlan,
        response: httpx.Response,
        payload: dict[str, Any],
        content_type: str,
    ) -> None:
        """Persist non-secret provenance next to the generated image.

        The file lives in a ``_meta`` subdirectory so image-id lookups, which glob
        ``outputs/<image_id>.*``, never mistake metadata for a picture.
        """
        metadata_dir = output_path.parent / METADATA_DIRNAME
        metadata_dir.mkdir(parents=True, exist_ok=True)
        request_id = (
            response.headers.get("x-request-id")
            or response.headers.get("x-silicon-request-id")
            or payload.get("requestId")
            or payload.get("id")
        )
        image_count = len(payload.get("images") or payload.get("data") or [1])
        metadata = {
            "provider": self.provider_name,
            "model": self.model_name,
            "planId": plan.plan_id,
            "intensity": plan.intensity,
            "outputFile": output_path.name,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "requestId": str(request_id) if request_id else None,
            "imageCount": image_count,
            "bytes": output_path.stat().st_size,
            "contentType": content_type or None,
            "seed": payload.get("seed"),
            "timings": payload.get("timings"),
            "usage": payload.get("usage"),
            "referenceImages": len(self.reference_image_paths),
        }
        (metadata_dir / f"{output_path.name}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
