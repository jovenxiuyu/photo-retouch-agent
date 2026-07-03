from __future__ import annotations

from dataclasses import dataclass

from app.application.ports import JobExecutor
from app.services.task_queue import TaskQueue


@dataclass
class InProcessJobDispatcher:
    """Dispatch jobs through the local queue behind the worker port."""

    queue: TaskQueue

    async def dispatch(self, job_id: str, executor: JobExecutor) -> None:
        async def execute() -> None:
            await executor(job_id)

        await self.queue.enqueue(job_id, execute)
