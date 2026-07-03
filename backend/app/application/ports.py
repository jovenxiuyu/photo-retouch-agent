from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from app.domain.analysis import PhotoAnalysis
from app.domain.retouch import RetouchPlan


class AssetReader(Protocol):
    def resolve_path(self, image_id: str) -> Path | None:
        """Resolve an asset to a local readable path for processing."""


class AnalysisCatalog(Protocol):
    def analyze(
        self,
        image_id: str,
        image_path: Path,
        domain_type: str,
    ) -> PhotoAnalysis:
        """Run the deterministic/domain-specific baseline analyzer."""


class PromptAnalyzer(Protocol):
    async def enrich(
        self,
        image_id: str,
        image_path: Path,
        baseline: PhotoAnalysis,
    ) -> PhotoAnalysis:
        """Enrich a baseline analysis with the selected AI brain."""


class PlanCatalog(Protocol):
    def create_plans(self, analysis: PhotoAnalysis) -> Sequence[RetouchPlan]:
        """Create retouch plan candidates for a domain analysis."""

