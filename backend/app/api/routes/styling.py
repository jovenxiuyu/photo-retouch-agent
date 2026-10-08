"""AI personal styling: design complete looks before any image is generated.

There are two clearly separated design modes:

* ``template`` -- the free, deterministic ``_Axis`` composition. Zero model
  calls, zero cost.
* ``ai`` -- a real vision model *authors* the three looks from the person's
  observable features plus the stylekit catalogue. This is exactly one paid
  model request, made only when the caller explicitly asks for it, and it is
  never triggered as a fallback: if the model fails, the request fails with
  502 instead of quietly returning templates as if they were AI designs.

Generating a look still goes through ``POST /api/retouch/jobs`` with the chosen
proposal posted as the plan and the *original* image id as ``sourceImageId``.
Designing is free of image generation; nothing here ever calls Image Edit.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.api.dependencies import (
    get_agent_brain_factory,
    get_create_styling_proposals_use_case,
    get_domain_registry,
    get_provider_factory,
    get_storage_service,
)
from app.application.use_cases import CreateStylingProposals
from app.brains.factory import AgentBrainFactory
from app.brains.openai_compatible import PromptBrainError
from app.brains.styling_planner import AIStylingPlanner
from app.core.errors import bad_request
from app.domains.portrait.ai_stylist import (
    build_ai_proposals,
    person_identity_anchors,
    preset_catalogue,
    valid_preset_ids,
)
from app.domains.registry import DomainRegistry
from app.infrastructure.storage import StorageService
from app.providers.factory import ImageProviderFactory
from app.schemas.mappers import analysis_to_domain, styling_proposal_to_schema
from app.schemas.styling import StylingProposal, StylingProposalsRequest

router = APIRouter(prefix="/styling", tags=["styling"])

_TEMPLATE_MODE = "template"
_AI_MODE = "ai"
_DESIGN_MODES = (_TEMPLATE_MODE, _AI_MODE)

# Generating one look is exactly one image-edit request with the current
# SiliconFlow Qwen-Image-Edit flow. Kept as one constant so the UI copy and the
# guardrail can never drift apart.
_REQUESTS_PER_GENERATION = 1

# Authoring three looks with the styling planner is deliberately batched into a
# single model call, so the AI design pass costs one request, not three.
_AI_DESIGN_REQUESTS = 1


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
        "stylingDesignModes": {
            _TEMPLATE_MODE: {
                "id": _TEMPLATE_MODE,
                "label": "免费模板预览",
                "isAi": False,
                "paidRequests": 0,
                "costNote": "免费生成，0 次模型调用，不产生任何费用。",
            },
            _AI_MODE: {
                "id": _AI_MODE,
                "label": "AI 个性化设计",
                "isAi": True,
                "paidRequests": _AI_DESIGN_REQUESTS,
                "costNote": (
                    f"由所选视觉模型一次生成三套方案，预计 {_AI_DESIGN_REQUESTS} 次"
                    "模型调用，只在你点击按钮时发起。"
                ),
            },
        },
        "stylingAiDesignRequests": _AI_DESIGN_REQUESTS,
        "estimatedRequestsPerGeneration": _REQUESTS_PER_GENERATION,
        "billingNote": (
            "Template proposals are free (no model calls); AI styling design uses "
            f"{_AI_DESIGN_REQUESTS} model request and only runs when you press the "
            "design button. Generating one look costs "
            f"{_REQUESTS_PER_GENERATION} image-edit request and only runs when "
            "you press a generate button."
        ),
    }


@router.post(
    "/proposals",
    response_model=list[StylingProposal],
    response_model_by_alias=True,
)
async def create_styling_proposals(
    request: StylingProposalsRequest,
    use_case: Annotated[
        CreateStylingProposals,
        Depends(get_create_styling_proposals_use_case),
    ],
    brain_factory: Annotated[AgentBrainFactory, Depends(get_agent_brain_factory)],
    storage: Annotated[StorageService, Depends(get_storage_service)],
    domains: Annotated[DomainRegistry, Depends(get_domain_registry)],
    agent_provider: Annotated[str | None, Header(alias="X-Agent-Provider")] = None,
    agent_api_key: Annotated[str | None, Header(alias="X-Agent-API-Key")] = None,
    agent_workspace_id: Annotated[
        str | None, Header(alias="X-Agent-Workspace-Id")
    ] = None,
) -> list[StylingProposal]:
    analysis = analysis_to_domain(request.analysis)
    mode = (request.mode or _TEMPLATE_MODE).strip().lower()
    if mode not in _DESIGN_MODES:
        raise bad_request(f"Unsupported styling design mode: {mode}")

    # Styling is portrait-only; this also raises a 400 for other domains in both
    # modes, before any model call is made.
    domains.get_stylist(analysis.domain_type)

    if mode == _TEMPLATE_MODE:
        proposals = use_case.execute(analysis)
    else:
        proposals = await _design_with_ai(
            analysis=analysis,
            request=request,
            brain_factory=brain_factory,
            storage=storage,
            domains=domains,
            agent_provider=agent_provider,
            agent_api_key=agent_api_key,
            agent_workspace_id=agent_workspace_id,
        )
    return [styling_proposal_to_schema(proposal) for proposal in proposals]


async def _design_with_ai(
    *,
    analysis,
    request: StylingProposalsRequest,
    brain_factory: AgentBrainFactory,
    storage: StorageService,
    domains: DomainRegistry,
    agent_provider: str | None,
    agent_api_key: str | None,
    agent_workspace_id: str | None,
):
    brain = brain_factory.create(agent_provider, agent_api_key, agent_workspace_id)
    provider_config = getattr(brain, "provider_config", None)
    if not callable(provider_config):
        raise bad_request(
            "The selected agent brain cannot author AI styling designs. "
            "Choose a vision model provider (e.g. SiliconFlow)."
        )

    vocabulary = domains.style_vocabulary
    facts = analysis.face_facts or {}
    image_path = storage.resolve_image_path(analysis.image_id)
    catalogue = preset_catalogue(vocabulary)

    planner = AIStylingPlanner(**provider_config())
    try:
        payload = await planner.plan(
            image_path=image_path,
            face_facts=facts,
            preset_catalogue=catalogue,
            valid_preset_ids=valid_preset_ids(vocabulary, catalogue),
            identity_anchors=person_identity_anchors(facts),
            user_instruction=request.user_instruction,
            analysis_summary=_analysis_summary(analysis),
        )
        return build_ai_proposals(
            payload,
            facts,
            analysis.domain_type,
            vocabulary,
            getattr(brain, "provider_name", "") or "",
        )
    except (PromptBrainError, RuntimeError) as exc:
        # Never fall back to the free template while claiming an AI design: the
        # caller must be able to tell the design failed.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI styling design failed: {exc}",
        ) from exc


def _analysis_summary(analysis) -> dict:
    return {
        "sceneType": analysis.scene_type,
        "domainType": analysis.domain_type,
        "recommendedStyles": list(analysis.recommended_styles),
        "portraitSuggestions": list(analysis.portrait_suggestions),
        "riskFlags": list(analysis.risk_flags),
        "brainProvider": analysis.brain_provider,
        "visionMode": analysis.vision_mode,
    }
