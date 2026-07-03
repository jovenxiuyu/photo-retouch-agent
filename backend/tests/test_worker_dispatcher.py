from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine

from app.worker.in_process import InProcessJobDispatcher


class ImmediateQueue:
    def __init__(self) -> None:
        self.job_ids: list[str] = []

    async def enqueue(
        self,
        job_id: str,
        coro_factory: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        self.job_ids.append(job_id)
        await coro_factory()


def test_in_process_dispatcher_forwards_job_to_executor() -> None:
    queue = ImmediateQueue()
    dispatcher = InProcessJobDispatcher(queue)  # type: ignore[arg-type]
    executed: list[str] = []

    async def execute(job_id: str) -> object:
        executed.append(job_id)
        return {"jobId": job_id}

    asyncio.run(dispatcher.dispatch("job_123", execute))

    assert queue.job_ids == ["job_123"]
    assert executed == ["job_123"]
