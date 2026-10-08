from __future__ import annotations

from app.application.commands import AnalyzePhotoCommand
from app.application.ports import (
    AnalysisCatalog,
    AssetReader,
    PlanCatalog,
    PromptAnalyzer,
    StylingCatalog,
)
from app.domain.analysis import PhotoAnalysis
from app.domain.retouch import RetouchPlan
from app.domain.styling import StylingProposal


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


class CreateStylingProposals:
    """Design three complete looks for the person in an analysed portrait.

    This is pure computation over the analysis (no network, no image model),
    so it never incurs a paid request.
    """

    def __init__(self, stylists: StylingCatalog) -> None:
        self._stylists = stylists

    def execute(self, analysis: PhotoAnalysis) -> tuple[StylingProposal, ...]:
        return tuple(self._stylists.create_styling_proposals(analysis))
