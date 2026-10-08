from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.analysis import PhotoAnalysis


class StylingReference(BaseModel):
    """The stylekit preset a proposal was composed from (traceability)."""

    model_config = ConfigDict(populate_by_name=True)

    preset_id: str = Field(alias="presetId")
    bucket: str
    name: str
    source_file: str = Field(alias="sourceFile")
    vocabulary_ref: str = Field(alias="vocabularyRef")
    prompt: str = ""


class StylingProposal(BaseModel):
    """One complete look designed for the person in the uploaded photo.

    The retouch fields (``planId``/``editPrompt``/``negativePrompt``/... ) are
    intentionally identical to ``RetouchPlan`` so a chosen proposal can be sent
    straight to ``POST /api/retouch/jobs`` without any translation.
    """

    model_config = ConfigDict(populate_by_name=True)

    plan_id: str = Field(alias="planId")
    domain_type: str = Field(alias="domainType")
    title: str
    name: str
    design_axis: str = Field(alias="designAxis")
    design_direction: str = Field(alias="designDirection")
    description: str
    design_reason: str = Field(alias="designReason")
    intensity: str
    hair_plan: str = Field(alias="hairPlan")
    makeup_plan: str = Field(alias="makeupPlan")
    outfit_plan: str = Field(alias="outfitPlan")
    accessory_plan: str = Field(alias="accessoryPlan")
    palette: str
    identity_anchors: list[str] = Field(alias="identityAnchors")
    why_it_fits: str = Field(alias="whyItFits")
    expected_changes: list[str] = Field(alias="expectedChanges")
    edit_prompt: str = Field(alias="editPrompt")
    negative_prompt: str = Field(alias="negativePrompt")
    reference_presets: list[StylingReference] = Field(
        default_factory=list, alias="referencePresets"
    )
    estimated_paid_requests: int = Field(default=1, alias="estimatedPaidRequests")


class StylingProposalsRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    analysis: PhotoAnalysis
