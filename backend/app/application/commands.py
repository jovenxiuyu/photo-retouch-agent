from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalyzePhotoCommand:
    image_id: str
    domain_type: str

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("image_id is required")
        if not self.domain_type:
            raise ValueError("domain_type is required")

