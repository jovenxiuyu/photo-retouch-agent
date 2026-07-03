from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum

from app.domain.retouch import RetouchPlan


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


_ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.RUNNING, JobStatus.CANCELED}),
    JobStatus.RUNNING: frozenset(
        {
            JobStatus.RETRYING,
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELED,
        }
    ),
    JobStatus.RETRYING: frozenset(
        {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELED}
    ),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELED: frozenset(),
}


class InvalidJobTransition(ValueError):
    pass


@dataclass(frozen=True)
class RetouchJob:
    job_id: str
    source_image_id: str
    plan: RetouchPlan
    model_provider: str
    model_name: str
    brain_provider: str
    brain_model: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    base_image_id: str | None = None
    user_instruction: str = ""
    output_image_ids: tuple[str, ...] = ()
    output_urls: tuple[str, ...] = ()
    error_message: str | None = None
    attempt: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id is required")
        if not self.source_image_id:
            raise ValueError("source_image_id is required")
        if self.attempt < 0:
            raise ValueError("attempt cannot be negative")

    @property
    def plan_id(self) -> str:
        return self.plan.plan_id

    def transition(
        self,
        target: JobStatus,
        *,
        now: datetime,
        error_message: str | None = None,
        output_image_ids: tuple[str, ...] | None = None,
        output_urls: tuple[str, ...] | None = None,
    ) -> "RetouchJob":
        if target not in _ALLOWED_TRANSITIONS[self.status]:
            raise InvalidJobTransition(
                f"cannot transition job {self.job_id} "
                f"from {self.status.value} to {target.value}"
            )

        attempt = self.attempt
        if target is JobStatus.RUNNING:
            attempt += 1

        return replace(
            self,
            status=target,
            updated_at=now,
            attempt=attempt,
            error_message=error_message,
            output_image_ids=(
                output_image_ids
                if output_image_ids is not None
                else self.output_image_ids
            ),
            output_urls=output_urls if output_urls is not None else self.output_urls,
        )

