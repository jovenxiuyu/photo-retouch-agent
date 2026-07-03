from __future__ import annotations
from typing import Annotated

from fastapi import APIRouter, Depends, Header

from app.api.dependencies import (
    get_analyze_photo_use_case,
    get_agent_brain_factory,
    get_provider_factory,
)
from app.api.application_adapters import BrainPromptAnalyzer
from app.api.model_selection import require_model_selection
from app.application.commands import AnalyzePhotoCommand
from app.application.use_cases import AnalyzePhoto, AssetNotFoundError
from app.brains.factory import AgentBrainFactory
from app.core.errors import not_found
from app.providers.factory import ImageProviderFactory
from app.schemas.analysis import AnalyzeRequest, PhotoAnalysis
from app.schemas.mappers import analysis_to_schema

router = APIRouter(prefix="/photos", tags=["analysis"])


@router.post("/analyze", response_model=PhotoAnalysis, response_model_by_alias=True)
async def analyze_photo(
    request: AnalyzeRequest,
    use_case: Annotated[AnalyzePhoto, Depends(get_analyze_photo_use_case)],
    brain_factory: Annotated[AgentBrainFactory, Depends(get_agent_brain_factory)],
    action_factory: Annotated[ImageProviderFactory, Depends(get_provider_factory)],
    brain_name: Annotated[str | None, Header(alias="X-Agent-Provider")] = None,
    brain_api_key: Annotated[str | None, Header(alias="X-Agent-API-Key")] = None,
    brain_workspace_id: Annotated[
        str | None, Header(alias="X-Agent-Workspace-Id")
    ] = None,
    action_name: Annotated[str | None, Header(alias="X-Action-Provider")] = None,
    action_api_key: Annotated[str | None, Header(alias="X-Action-API-Key")] = None,
    action_workspace_id: Annotated[
        str | None, Header(alias="X-Action-Workspace-Id")
    ] = None,
    legacy_action_name: Annotated[str | None, Header(alias="X-AI-Provider")] = None,
    legacy_action_api_key: Annotated[
        str | None, Header(alias="X-AI-API-Key")
    ] = None,
    legacy_action_workspace_id: Annotated[
        str | None, Header(alias="X-AI-Workspace-Id")
    ] = None,
) -> PhotoAnalysis:
    selected_brain = require_model_selection(
        brain_name,
        None,
        "Agent brain",
    )
    selected_action = require_model_selection(
        action_name,
        legacy_action_name,
        "Agent action",
    )
    brain = brain_factory.create(
        selected_brain,
        brain_api_key,
        brain_workspace_id,
    )
    action_factory.create(
        selected_action,
        action_api_key or legacy_action_api_key,
        action_workspace_id or legacy_action_workspace_id,
    )
    try:
        analysis = await use_case.execute(
            AnalyzePhotoCommand(
                image_id=request.image_id,
                domain_type=request.domain_type,
            ),
            BrainPromptAnalyzer(brain),
        )
    except AssetNotFoundError as exc:
        raise not_found("Image not found.") from exc
    return analysis_to_schema(analysis)
