"""Compatibility imports for the storage adapter.

New code should import from ``app.infrastructure.storage``.
"""

from app.infrastructure.storage import (
    CONTENT_TYPE_EXTENSIONS,
    StorageService,
    ensure_data_dirs,
)

__all__ = ["CONTENT_TYPE_EXTENSIONS", "StorageService", "ensure_data_dirs"]
