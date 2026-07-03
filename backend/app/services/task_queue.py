from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger("photo_retouch.task_queue")


class TaskQueue:
    """In-process background task runner with concurrency control."""

    def __init__(self, max_concurrent: int = 3) -> None:
        self._max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    async def enqueue(
        self,
        job_id: str,
        coro_factory: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        semaphore = self._get_semaphore()

        async def _run() -> None:
            async with semaphore:
                logger.info("task_started job_id=%s", job_id)
                try:
                    await coro_factory()
                except Exception:
                    logger.exception("task_failed job_id=%s", job_id)

        asyncio.create_task(_run())
        logger.info("task_enqueued job_id=%s", job_id)
