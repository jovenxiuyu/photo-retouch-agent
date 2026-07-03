from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.application.commands import AnalyzePhotoCommand
from app.application.use_cases import (
    AnalyzePhoto,
    AssetNotFoundError,
    CreateRetouchPlans,
)
from app.domain.analysis import PhotoAnalysis, SubjectSummary
from app.domain.retouch import RetouchPlan


def _analysis() -> PhotoAnalysis:
    return PhotoAnalysis(
        image_id="img_1",
        domain_type="portrait",
        scene_type="portrait",
        subjects=SubjectSummary(
            count=1,
            position="center",
            face_visibility="clear",
        ),
    )


class FakeAssets:
    def resolve_path(self, image_id: str) -> Path | None:
        return Path("/tmp/image.png") if image_id == "img_1" else None


class FakeAnalyzers:
    def analyze(
        self,
        image_id: str,
        image_path: Path,
        domain_type: str,
    ) -> PhotoAnalysis:
        assert image_path == Path("/tmp/image.png")
        assert domain_type == "portrait"
        return _analysis()


class FakeBrain:
    async def enrich(
        self,
        image_id: str,
        image_path: Path,
        baseline: PhotoAnalysis,
    ) -> PhotoAnalysis:
        return PhotoAnalysis(
            **{
                **baseline.__dict__,
                "recommended_styles": ("natural",),
                "brain_provider": "fake",
            }
        )


class FakePlanners:
    def create_plans(self, analysis: PhotoAnalysis) -> tuple[RetouchPlan, ...]:
        return (
            RetouchPlan(
                plan_id="plan_1",
                domain_type=analysis.domain_type,
                title="Natural",
                description="Natural retouch",
                intensity="medium",
                edit_prompt="Retouch naturally",
                negative_prompt="plastic skin",
            ),
        )


def test_analyze_photo_orchestrates_ports() -> None:
    use_case = AnalyzePhoto(FakeAssets(), FakeAnalyzers())

    result = asyncio.run(
        use_case.execute(
            AnalyzePhotoCommand(image_id="img_1", domain_type="portrait"),
            FakeBrain(),
        )
    )

    assert result.brain_provider == "fake"
    assert result.recommended_styles == ("natural",)


def test_analyze_photo_reports_missing_asset() -> None:
    use_case = AnalyzePhoto(FakeAssets(), FakeAnalyzers())

    with pytest.raises(AssetNotFoundError):
        asyncio.run(
            use_case.execute(
                AnalyzePhotoCommand(image_id="missing", domain_type="portrait"),
                FakeBrain(),
            )
        )


def test_create_plans_uses_plan_catalog() -> None:
    plans = CreateRetouchPlans(FakePlanners()).execute(_analysis())

    assert plans[0].plan_id == "plan_1"
