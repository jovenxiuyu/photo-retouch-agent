"""AI personal styling: design complete looks before any image is generated.

Both endpoints here are free:

* ``GET /styling/providers`` reports the styling vocabulary source plus the
  cost guardrails so the UI can show expected requests/cost before generating.
* ``POST /styling/proposals`` turns an existing portrait analysis into three
  complete, mutually distinct looks. It is pure computation over the analysis
  and never calls a paid model.

Generating a look still goes through ``POST /api/retouch/jobs`` with the chosen
proposal posted as the plan and the *original* image id as ``sourceImageId``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import (
    get_agent_brain_factory,
    get_create_styling_proposals_use_case,
    get_domain_registry,
    get_provider_factory,
)
from app.application.use_cases import CreateStylingProposals
from app.brains.factory import AgentBrainFactory
from app.domains.registry import DomainRegistry
from app.providers.factory import ImageProviderFactory
from app.schemas.mappers import analysis_to_domain, styling_proposal_to_schema
from app.schemas.styling import StylingProposal, StylingProposalsRequest

router = APIRouter(prefix="/styling", tags=["styling"])

# Generating one look is exactly one image-edit request with the current
# SiliconFlow Qwen-Image-Edit flow. Kept as one constant so the UI copy and the
# guardrail can never drift apart.
_REQUESTS_PER_GENERATION = 1


@router.get("/providers")
def get_styling_providers(
    image_factory: Annotated[ImageProviderFactory, Depends(get_provider_factory)],
    brain_factory: Annotated[AgentBrainFactory, Depends(get_agent_brain_factory)],
    domains: Annotated[DomainRegistry, Depends(get_domain_registry)],
) -> dict:
    vocabulary = domains.style_vocabulary
    return {
        **image_factory.capabilities(),
        **brain_factory.capabilities(),
        "stylingVocabulary": {
            "available": vocabulary.available,
            "sourceLabel": vocabulary.source_label,
            "loadedFiles": list(vocabulary.loaded_files),
        },
        "estimatedRequestsPerGeneration": _REQUESTS_PER_GENERATION,
        "billingNote": (
            "Proposals are free (no model calls). Generating one look costs "
            f"{_REQUESTS_PER_GENERATION} image-edit request and only runs when "
            "you press a generate button."
        ),
    }


@router.post(
    "/proposals",
    response_model=list[StylingProposal],
    response_model_by_alias=True,
)
def create_styling_proposals(
    request: StylingProposalsRequest,
    use_case: Annotated[
        CreateStylingProposals,
        Depends(get_create_styling_proposals_use_case),
    ],
) -> list[StylingProposal]:
    analysis = analysis_to_domain(request.analysis)
    proposals = use_case.execute(analysis)
    return [styling_proposal_to_schema(proposal) for proposal in proposals]
