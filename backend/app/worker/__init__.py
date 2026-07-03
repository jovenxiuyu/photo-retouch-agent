"""Background job execution adapters."""

from app.worker.in_process import InProcessJobDispatcher

__all__ = ["InProcessJobDispatcher"]
