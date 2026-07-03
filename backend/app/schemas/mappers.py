from __future__ import annotations

from app.domain.analysis import (
    PhotoAnalysis as DomainPhotoAnalysis,
    SubjectSummary as DomainSubjectSummary,
)
from app.domain.assets import PhotoAsset as DomainPhotoAsset
from app.domain.jobs import JobStatus, RetouchJob as DomainRetouchJob
from app.domain.retouch import RetouchPlan as DomainRetouchPlan
from app.schemas.analysis import (
    PhotoAnalysis as PhotoAnalysisSchema,
    SubjectSummary as SubjectSummarySchema,
)
from app.schemas.job import RetouchJob as RetouchJobSchema
from app.schemas.photo import PhotoAsset as PhotoAssetSchema
from app.schemas.retouch import RetouchPlan as RetouchPlanSchema


def plan_to_domain(plan: RetouchPlanSchema) -> DomainRetouchPlan:
    return DomainRetouchPlan(
        plan_id=plan.plan_id,
        domain_type=plan.domain_type,
        title=plan.title,
        description=plan.description,
        intensity=plan.intensity,
        edit_prompt=plan.edit_prompt,
        negative_prompt=plan.negative_prompt,
        expected_changes=tuple(plan.expected_changes),
    )


def plan_to_schema(plan: DomainRetouchPlan) -> RetouchPlanSchema:
    return RetouchPlanSchema(
        plan_id=plan.plan_id,
        domain_type=plan.domain_type,
        title=plan.title,
        description=plan.description,
        intensity=plan.intensity,
        edit_prompt=plan.edit_prompt,
        negative_prompt=plan.negative_prompt,
        expected_changes=list(plan.expected_changes),
    )


def job_to_domain(job: RetouchJobSchema) -> DomainRetouchJob:
    if job.plan is None:
        raise ValueError("retouch job plan is required")
    return DomainRetouchJob(
        job_id=job.job_id,
        source_image_id=job.source_image_id,
        base_image_id=job.base_image_id,
        plan=plan_to_domain(job.plan),
        user_instruction=job.user_instruction,
        model_provider=job.model_provider,
        model_name=job.model_name,
        brain_provider=job.brain_provider,
        brain_model=job.brain_model,
        status=JobStatus(job.status),
        output_image_ids=tuple(job.output_image_ids),
        output_urls=tuple(job.output_urls),
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def job_to_schema(job: DomainRetouchJob) -> RetouchJobSchema:
    return RetouchJobSchema(
        job_id=job.job_id,
        source_image_id=job.source_image_id,
        base_image_id=job.base_image_id,
        plan_id=job.plan_id,
        plan=plan_to_schema(job.plan),
        user_instruction=job.user_instruction,
        model_provider=job.model_provider,
        model_name=job.model_name,
        brain_provider=job.brain_provider,
        brain_model=job.brain_model,
        status=job.status.value,
        output_image_ids=list(job.output_image_ids),
        output_urls=list(job.output_urls),
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def analysis_to_domain(analysis: PhotoAnalysisSchema) -> DomainPhotoAnalysis:
    return DomainPhotoAnalysis(
        image_id=analysis.image_id,
        domain_type=analysis.domain_type,
        scene_type=analysis.scene_type,
        subjects=DomainSubjectSummary(
            count=analysis.subjects.count,
            position=analysis.subjects.position,
            face_visibility=analysis.subjects.face_visibility,
        ),
        lighting_issues=tuple(analysis.lighting_issues),
        background_issues=tuple(analysis.background_issues),
        portrait_suggestions=tuple(analysis.portrait_suggestions),
        composition_suggestions=tuple(analysis.composition_suggestions),
        recommended_styles=tuple(analysis.recommended_styles),
        risk_flags=tuple(analysis.risk_flags),
        brain_provider=analysis.brain_provider,
        brain_model=analysis.brain_model,
        vision_mode=analysis.vision_mode,
    )


def analysis_to_schema(analysis: DomainPhotoAnalysis) -> PhotoAnalysisSchema:
    return PhotoAnalysisSchema(
        image_id=analysis.image_id,
        domain_type=analysis.domain_type,
        scene_type=analysis.scene_type,
        subjects=SubjectSummarySchema(
            count=analysis.subjects.count,
            position=analysis.subjects.position,
            face_visibility=analysis.subjects.face_visibility,
        ),
        lighting_issues=list(analysis.lighting_issues),
        background_issues=list(analysis.background_issues),
        portrait_suggestions=list(analysis.portrait_suggestions),
        composition_suggestions=list(analysis.composition_suggestions),
        recommended_styles=list(analysis.recommended_styles),
        risk_flags=list(analysis.risk_flags),
        brain_provider=analysis.brain_provider,
        brain_model=analysis.brain_model,
        vision_mode=analysis.vision_mode,
    )


def asset_to_domain(asset: PhotoAssetSchema) -> DomainPhotoAsset:
    return DomainPhotoAsset(
        image_id=asset.image_id,
        filename=asset.filename,
        content_type=asset.content_type,
        width=asset.width,
        height=asset.height,
        size_bytes=asset.size_bytes,
        object_key=asset.url,
        created_at=asset.created_at,
    )


def asset_to_schema(asset: DomainPhotoAsset) -> PhotoAssetSchema:
    return PhotoAssetSchema(
        image_id=asset.image_id,
        filename=asset.filename,
        content_type=asset.content_type,
        width=asset.width,
        height=asset.height,
        size_bytes=asset.size_bytes,
        url=asset.object_key,
        created_at=asset.created_at,
    )

