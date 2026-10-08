from __future__ import annotations

from pathlib import Path

from app.application.ports import (
    AnalysisCatalog,
    AssetReader,
    PlanCatalog,
    PromptAnalyzer,
)
from app.brains.base import PromptBrain
from app.domain.analysis import PhotoAnalysis
from app.domain.retouch import RetouchPlan
from app.domains.registry import DomainRegistry
from app.schemas.mappers import (
    analysis_to_domain,
    analysis_to_schema,
    plan_to_domain,
)
from app.services.storage import StorageService


class StorageAssetReader(AssetReader):
    def __init__(self, storage: StorageService) -> None:
        self._storage = storage

    def resolve_path(self, image_id: str) -> Path | None:
        return self._storage.resolve_image_path(image_id)


class RegistryAnalysisCatalog(AnalysisCatalog):
    def __init__(self, registry: DomainRegistry) -> None:
        self._registry = registry

    def analyze(
        self,
        image_id: str,
        image_path: Path,
        domain_type: str,
    ) -> PhotoAnalysis:
        resolved = self._registry.resolve_domain_type(domain_type, image_path)
        analyzer = self._registry.get_analyzer(resolved)
        return analysis_to_domain(analyzer.analyze(image_id, image_path))


class BrainPromptAnalyzer(PromptAnalyzer):
    def __init__(self, brain: PromptBrain) -> None:
        self._brain = brain

    async def enrich(
        self,
        image_id: str,
        image_path: Path,
        baseline: PhotoAnalysis,
    ) -> PhotoAnalysis:
        enriched = await self._brain.analyze(
            image_id,
            image_path,
            analysis_to_schema(baseline),
        )
        return analysis_to_domain(enriched)


class RegistryPlanCatalog(PlanCatalog):
    def __init__(self, registry: DomainRegistry) -> None:
        self._registry = registry

    def create_plans(self, analysis: PhotoAnalysis) -> tuple[RetouchPlan, ...]:
        planner = self._registry.get_planner(analysis.domain_type)
        plans = planner.create_plans(analysis_to_schema(analysis))
        return tuple(plan_to_domain(plan) for plan in plans)

