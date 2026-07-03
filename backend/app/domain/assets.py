from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PhotoAsset:
    image_id: str
    filename: str
    content_type: str
    width: int
    height: int
    size_bytes: int
    object_key: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("image_id is required")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.size_bytes < 0:
            raise ValueError("size_bytes cannot be negative")

