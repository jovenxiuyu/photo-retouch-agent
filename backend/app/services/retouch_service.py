from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4

from app.brains.base import PromptBrain
from app.brains.local import LocalPromptBrain
from app.core.config import Settings, settings as default_settings
from app.core.errors import bad_request, not_found
from app.domains.portrait import face_lock as face_lock_domain
from app.providers.base import ImageEditProvider
from app.providers.mock_image_provider import MockImageProvider
from app.schemas.job import CreateRetouchJobRequest, RefineRetouchJobRequest, RetouchJob
from app.services.job_store import JobStore
from app.services.storage import StorageService


class RetouchService:
    def __init__(
        self,
        storage: StorageService,
        jobs: JobStore,
        provider: ImageEditProvider | None = None,
        brain: PromptBrain | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.storage = storage
        self.jobs = jobs
        self.provider = provider or MockImageProvider()
        self.brain = brain or LocalPromptBrain()
        self.settings = settings or default_settings

    # ------------------------------------------------------------- face lock
    def _run_face_lock(
        self,
        source_path,
        generated_path,
        generated_image_id: str,
    ) -> face_lock_domain.FaceLockReport | None:
        """Restore the original face into the generated image (strict mode only).

        Returns ``None`` when the lock is switched off, so the caller keeps the
        previous behaviour untouched. A returned report is *never* silently
        treated as success: the caller checks :attr:`FaceLockReport.ok` and fails
        the job when the strict lock could not be verified.
        """
        config = face_lock_domain.config_from_settings(self.settings)
        if not getattr(self.settings, "face_lock_enabled", False):
            return None
        if config.mode != face_lock_domain.FACE_LOCK_STRICT:
            return None

        locked_image_id = f"lock_{uuid4().hex[:12]}"
        locked_filename = f"{locked_image_id}.png"
        locked_path = self.storage.config.outputs_dir / locked_filename

        report = face_lock_domain.apply_face_lock(
            original_path=source_path,
            generated_path=generated_path,
            output_path=locked_path,
            model_path=getattr(self.settings, "face_landmark_model_path", None),
            config=config,
            manual_mask_path=getattr(self.settings, "face_lock_manual_mask_path", None),
        )
        report.diagnostics["preLockImageId"] = generated_image_id
        report.diagnostics["preLockPath"] = f"/data/outputs/{generated_path.name}"
        report.diagnostics["lockedImageId"] = locked_image_id if report.ok else None
        report.diagnostics["lockedPath"] = (
            f"/data/outputs/{locked_filename}" if report.ok else None
        )
        report.diagnostics["manualMaskPath"] = (
            str(self.settings.face_lock_manual_mask_path)
            if getattr(self.settings, "face_lock_manual_mask_path", None)
            else None
        )
        return report

    def _finish_job(
        self,
        job: RetouchJob,
        source_path,
        output_image_id: str,
        output_filename: str,
        output_path,
    ) -> RetouchJob:
        """Record the provider output, applying the strict face lock when enabled."""
        report = self._run_face_lock(source_path, output_path, output_image_id)
        if report is not None:
            job.face_lock = report.to_dict()
        if report is None or report.ok:
            job.status = "succeeded"
            if report is None:
                job.output_image_ids = [output_image_id]
                job.output_urls = [f"/data/outputs/{output_filename}"]
            else:
                locked_filename = f"{report.diagnostics['lockedImageId']}.png"
                job.output_image_ids = [report.diagnostics["lockedImageId"]]
                job.output_urls = [f"/data/outputs/{locked_filename}"]
                job.error_message = None
        else:
            # A styling render whose strict face lock failed is not a success.
            job.status = "failed"
            job.output_image_ids = []
            job.output_urls = []
            job.error_message = (
                f"保脸失败（{report.status.value}）：{report.message} "
                "生成图未通过严格人脸锁定校验，请人工检查保护区域或调整掩膜后重试。"
            )
        job.updated_at = datetime.now(timezone.utc)
        return job

    async def create_job_queued(self, request: CreateRetouchJobRequest) -> RetouchJob:
        """Create a job in 'queued' status without executing the AI workflow."""
        base_image_id = request.base_image_id or request.source_image_id
        source_path = self.storage.resolve_image_path(base_image_id)
        if source_path is None:
            raise not_found("Base image not found.")

        now = datetime.now(timezone.utc)
        job = RetouchJob(
            job_id=f"job_{uuid4().hex[:12]}",
            source_image_id=request.source_image_id,
            base_image_id=request.base_image_id,
            plan_id=request.plan.plan_id,
            plan=request.plan,
            user_instruction=request.user_instruction,
            model_provider=self.provider.provider_name,
            model_name=self.provider.model_name,
            brain_provider=self.brain.provider_name,
            brain_model=self.brain.model_name,
            status="queued",
            output_image_ids=[],
            output_urls=[],
            error_message=None,
            created_at=now,
            updated_at=now,
        )
        self.jobs.save(job)
        return job

    async def execute_job(self, job_id: str) -> RetouchJob:
        """Execute the AI workflow for a queued job."""
        job = self.jobs.get(job_id)
        base_image_id = job.base_image_id or job.source_image_id
        source_path = self.storage.resolve_image_path(base_image_id)

        job.status = "running"
        job.updated_at = datetime.now(timezone.utc)
        self.jobs.save(job)

        try:
            effective_plan = await self.brain.optimize(
                source_path,
                job.plan,
                job.user_instruction,
            )
            job.plan = effective_plan
            output_image_id = f"out_{uuid4().hex[:12]}"
            output_filename = f"{output_image_id}{self.provider.output_extension}"
            output_path = self.storage.config.outputs_dir / output_filename
            await self.provider.edit_image(
                source_path=source_path,
                output_path=output_path,
                plan=effective_plan,
                user_instruction=job.user_instruction,
            )
            self._finish_job(
                job, source_path, output_image_id, output_filename, output_path
            )
        except Exception as exc:  # pragma: no cover - defensive status capture
            job.status = "failed"
            job.error_message = str(exc)
            job.updated_at = datetime.now(timezone.utc)

        return self.jobs.save(job)

    async def create_job(self, request: CreateRetouchJobRequest) -> RetouchJob:
        base_image_id = request.base_image_id or request.source_image_id
        source_path = self.storage.resolve_image_path(base_image_id)
        if source_path is None:
            raise not_found("Base image not found.")

        now = datetime.now(timezone.utc)
        job = RetouchJob(
            job_id=f"job_{uuid4().hex[:12]}",
            source_image_id=request.source_image_id,
            base_image_id=request.base_image_id,
            plan_id=request.plan.plan_id,
            plan=request.plan,
            user_instruction=request.user_instruction,
            model_provider=self.provider.provider_name,
            model_name=self.provider.model_name,
            brain_provider=self.brain.provider_name,
            brain_model=self.brain.model_name,
            status="running",
            output_image_ids=[],
            output_urls=[],
            error_message=None,
            created_at=now,
            updated_at=now,
        )
        self.jobs.save(job)

        try:
            effective_plan = await self.brain.optimize(
                source_path,
                request.plan,
                request.user_instruction,
            )
            job.plan = effective_plan
            output_image_id = f"out_{uuid4().hex[:12]}"
            output_filename = f"{output_image_id}{self.provider.output_extension}"
            output_path = self.storage.config.outputs_dir / output_filename
            await self.provider.edit_image(
                source_path=source_path,
                output_path=output_path,
                plan=effective_plan,
                user_instruction=request.user_instruction,
            )
            self._finish_job(
                job, source_path, output_image_id, output_filename, output_path
            )
        except Exception as exc:  # pragma: no cover - defensive status capture
            job.status = "failed"
            job.error_message = str(exc)
            job.updated_at = datetime.now(timezone.utc)

        return self.jobs.save(job)

    def with_providers(
        self,
        provider: ImageEditProvider,
        brain: PromptBrain,
    ) -> "RetouchService":
        return RetouchService(self.storage, self.jobs, provider, brain, self.settings)

    def get_job(self, job_id: str) -> RetouchJob:
        return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 20, offset: int = 0) -> list[RetouchJob]:
        return self.jobs.list_jobs(limit=limit, offset=offset)

    async def refine_job(self, job_id: str, request: RefineRetouchJobRequest) -> RetouchJob:
        parent = self.jobs.get(job_id)
        if parent.status != "succeeded" or not parent.output_image_ids:
            raise bad_request("Only succeeded jobs with output can be refined.")

        plan = request.plan or parent.plan
        if plan is None:
            raise bad_request("A retouch plan is required for refinement.")

        return await self.create_job(
            CreateRetouchJobRequest(
                source_image_id=parent.source_image_id,
                base_image_id=parent.output_image_ids[-1],
                plan=plan,
                user_instruction=request.user_instruction,
            )
        )
