"""Compatibility import for the local job repository.

New code should import from ``app.infrastructure.job_store``.
"""

from app.infrastructure.job_store import JobStore

__all__ = ["JobStore"]
