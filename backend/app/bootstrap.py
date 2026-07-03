from __future__ import annotations

from dataclasses import dataclass

from app.brains.factory import AgentBrainFactory
from app.core.config import Settings
from app.domains.registry import DomainRegistry
from app.infrastructure.job_store import JobStore
from app.infrastructure.storage import StorageService
from app.providers.factory import ImageProviderFactory
from app.services.task_queue import TaskQueue


@dataclass
class AppContainer:
    settings: Settings
    storage: StorageService
    jobs: JobStore
    image_providers: ImageProviderFactory
    brains: AgentBrainFactory
    domains: DomainRegistry
    task_queue: TaskQueue


def build_container(settings: Settings) -> AppContainer:
    return AppContainer(
        settings=settings,
        storage=StorageService(settings),
        jobs=JobStore(settings),
        image_providers=ImageProviderFactory(settings),
        brains=AgentBrainFactory(settings),
        domains=DomainRegistry(),
        task_queue=TaskQueue(max_concurrent=settings.max_concurrent_jobs),
    )

