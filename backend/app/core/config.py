from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    app_name: str = "Photo Retouch Agent API"
    app_version: str = "0.3.0"
    max_upload_bytes: int = 10 * 1024 * 1024
    allowed_image_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")
    data_dir: Path = PROJECT_ROOT / "data"
    frontend_dir: Path = PROJECT_ROOT / "frontend"
    image_provider: str = field(
        default_factory=lambda: os.getenv("PHOTO_AGENT_IMAGE_PROVIDER", "mock").strip().lower()
    )
    brain_provider: str = field(
        default_factory=lambda: os.getenv("PHOTO_AGENT_BRAIN_PROVIDER", "local").strip().lower()
    )
    dashscope_api_key: str | None = field(
        default_factory=lambda: _optional_env("DASHSCOPE_API_KEY"),
        repr=False,
    )
    dashscope_workspace_id: str | None = field(
        default_factory=lambda: _optional_env("DASHSCOPE_WORKSPACE_ID")
    )
    dashscope_endpoint: str | None = field(
        default_factory=lambda: _optional_env("DASHSCOPE_ENDPOINT")
    )
    dashscope_region: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_REGION", "beijing").strip().lower()
    )
    dashscope_image_model: str = field(
        default_factory=lambda: os.getenv(
            "DASHSCOPE_IMAGE_MODEL", "qwen-image-2.0-pro"
        ).strip()
    )
    dashscope_wan_model: str = field(
        default_factory=lambda: os.getenv(
            "DASHSCOPE_WAN_MODEL", "wan2.7-image-pro"
        ).strip()
    )
    dashscope_vision_endpoint: str | None = field(
        default_factory=lambda: _optional_env("DASHSCOPE_VISION_ENDPOINT")
    )
    dashscope_vision_model: str = field(
        default_factory=lambda: os.getenv(
            "DASHSCOPE_VISION_MODEL", "qwen3-vl-plus"
        ).strip()
    )
    provider_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("PHOTO_AGENT_PROVIDER_TIMEOUT", "180"))
    )
    provider_download_limit_bytes: int = 30 * 1024 * 1024
    api_key: str | None = field(
        default_factory=lambda: _optional_env("PHOTO_AGENT_API_KEY"),
        repr=False,
    )
    rate_limit_per_second: int = field(
        default_factory=lambda: int(os.getenv("PHOTO_AGENT_RATE_LIMIT", "10"))
    )
    max_concurrent_jobs: int = field(
        default_factory=lambda: int(os.getenv("PHOTO_AGENT_MAX_CONCURRENT", "3"))
    )
    database_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("PHOTO_AGENT_DATABASE_PATH", str(PROJECT_ROOT / "data" / "photo_retouch.db"))
        )
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("PHOTO_AGENT_LOG_LEVEL", "INFO").strip().upper()
    )
    deepseek_api_key: str | None = field(
        default_factory=lambda: _optional_env("DEEPSEEK_API_KEY"),
        repr=False,
    )
    deepseek_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "DEEPSEEK_ENDPOINT", "https://api.deepseek.com/chat/completions"
        ).strip()
    )
    deepseek_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
    )
    zhipu_api_key: str | None = field(
        default_factory=lambda: _optional_env("ZHIPU_API_KEY")
        or _optional_env("ZHIPUAI_API_KEY"),
        repr=False,
    )
    zhipu_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "ZHIPU_ENDPOINT",
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        ).strip()
    )
    zhipu_model: str = field(
        default_factory=lambda: os.getenv("ZHIPU_MODEL", "glm-5v-turbo").strip()
    )
    ark_api_key: str | None = field(
        default_factory=lambda: _optional_env("ARK_API_KEY"),
        repr=False,
    )
    ark_chat_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "ARK_CHAT_ENDPOINT",
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        ).strip()
    )
    ark_vision_model: str = field(
        default_factory=lambda: os.getenv(
            "ARK_VISION_MODEL", "doubao-seed-2-0-lite-260215"
        ).strip()
    )
    ark_image_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "ARK_IMAGE_ENDPOINT",
            "https://ark.cn-beijing.volces.com/api/v3/images/generations",
        ).strip()
    )
    ark_image_model: str = field(
        default_factory=lambda: os.getenv(
            "ARK_IMAGE_MODEL", "doubao-seedream-5-0-260128"
        ).strip()
    )
    siliconflow_api_key: str | None = field(
        default_factory=lambda: _optional_env("SILICONFLOW_API_KEY"),
        repr=False,
    )
    siliconflow_base_url: str = field(
        default_factory=lambda: (
            _optional_env("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1"
        ).rstrip("/")
    )
    siliconflow_vision_model: str = field(
        default_factory=lambda: os.getenv(
            "SILICONFLOW_VISION_MODEL", "Qwen/Qwen3.5-35B-A3B"
        ).strip()
    )
    siliconflow_image_model: str = field(
        default_factory=lambda: os.getenv(
            "SILICONFLOW_IMAGE_MODEL", "Qwen/Qwen-Image-Edit-2509"
        ).strip()
    )
    # --- strict face identity lock ---------------------------------------
    # The styling providers repaint the whole person, so the generated face is
    # *not* trustworthy even when the prompt begs for identity preservation.
    # ``strict`` restores the original facial pixels after generation; ``off``
    # keeps the raw provider output (unlocked) for debugging only.
    face_lock_mode: str = field(
        default_factory=lambda: os.getenv(
            "PHOTO_AGENT_FACE_LOCK_MODE", "strict"
        ).strip().lower()
    )
    face_landmark_model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("PHOTO_AGENT_FACE_LANDMARK_MODEL")
            or str(PROJECT_ROOT / "data" / "models" / "face_landmarker.task")
        )
    )
    face_lock_residual_ratio: float = field(
        default_factory=lambda: float(
            os.getenv("PHOTO_AGENT_FACE_LOCK_RESIDUAL_RATIO", "0.045")
        )
    )
    face_lock_strict_residual_ratio: float = field(
        default_factory=lambda: float(
            os.getenv("PHOTO_AGENT_FACE_LOCK_STRICT_RESIDUAL_RATIO", "0.030")
        )
    )
    face_lock_manual_mask_path: Path | None = field(
        default_factory=lambda: (
            Path(value)
            if (value := _optional_env("PHOTO_AGENT_FACE_LOCK_MASK"))
            else None
        )
    )

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def face_lock_enabled(self) -> bool:
        return self.face_lock_mode not in ("", "off", "none", "disabled")


settings = Settings()
