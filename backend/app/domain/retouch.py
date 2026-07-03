from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetouchPlan:
    plan_id: str
    domain_type: str
    title: str
    description: str
    intensity: str
    edit_prompt: str
    negative_prompt: str
    expected_changes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("plan_id is required")
        if not self.domain_type:
            raise ValueError("domain_type is required")

