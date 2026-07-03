from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.domain.jobs import InvalidJobTransition, JobStatus
from app.schemas.job import RetouchJob
from app.schemas.mappers import job_to_domain, job_to_schema
from app.schemas.retouch import RetouchPlan


def _job() -> RetouchJob:
    now = datetime.now(timezone.utc)
    plan = RetouchPlan(
        plan_id="plan_1",
        domain_type="portrait",
        title="Natural",
        description="Natural portrait retouch",
        intensity="medium",
        edit_prompt="Retouch naturally",
        negative_prompt="plastic skin",
        expected_changes=["skin"],
    )
    return RetouchJob(
        job_id="job_1",
        source_image_id="img_1",
        plan_id=plan.plan_id,
        plan=plan,
        model_provider="mock",
        model_name="mock-model",
        brain_provider="local",
        brain_model="rule-based",
        status="queued",
        created_at=now,
        updated_at=now,
    )


def test_job_domain_model_enforces_state_machine() -> None:
    queued = job_to_domain(_job())
    running = queued.transition(
        JobStatus.RUNNING,
        now=datetime.now(timezone.utc),
    )
    succeeded = running.transition(
        JobStatus.SUCCEEDED,
        now=datetime.now(timezone.utc),
        output_image_ids=("out_1",),
        output_urls=("/data/outputs/out_1.png",),
    )

    assert running.attempt == 1
    assert succeeded.status is JobStatus.SUCCEEDED
    assert succeeded.output_image_ids == ("out_1",)

    with pytest.raises(InvalidJobTransition):
        succeeded.transition(
            JobStatus.RUNNING,
            now=datetime.now(timezone.utc),
        )


def test_job_schema_round_trip_preserves_public_contract() -> None:
    schema = _job()

    restored = job_to_schema(job_to_domain(schema))

    assert restored.model_dump(by_alias=True) == schema.model_dump(by_alias=True)

