from __future__ import annotations

from app.application.commands import AnalyzePhotoCommand
from app.application.ports import (
    AnalysisCatalog,
    AssetReader,
    PlanCatalog,
    PromptAnalyzer,
)
from app.domain.analysis import PhotoAnalysis
from app.domain.retouch import RetouchPlan


class AssetNotFoundError(LookupError):
    pass


class AnalyzePhoto:
    def __init__(
        self,
        assets: AssetReader,
        analyzers: AnalysisCatalog,
    ) -> None:
        self._assets = assets
        self._analyzers = analyzers

    async def execute(
        self,
        command: AnalyzePhotoCommand,
        brain: PromptAnalyzer,
    ) -> PhotoAnalysis:
        image_path = self._assets.resolve_path(command.image_id)
        if image_path is None:
            raise AssetNotFoundError(command.image_id)

        baseline = self._analyzers.analyze(
            command.image_id,
            image_path,
            command.domain_type,
        )
        return await brain.enrich(command.image_id, image_path, baseline)


class CreateRetouchPlans:
    def __init__(self, planners: PlanCatalog) -> None:
        self._planners = planners

    def execute(self, analysis: PhotoAnalysis) -> tuple[RetouchPlan, ...]:
        return tuple(self._planners.create_plans(analysis))

