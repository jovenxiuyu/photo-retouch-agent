from __future__ import annotations

from dataclasses import dataclass

from app.plugins.registry import PluginDefinition, PluginRegistry
from app.providers.base import ImageEditProvider


@dataclass(frozen=True)
class ImageProviderPlugin(PluginDefinition[ImageEditProvider]):
    """Registration contract for an image-edit provider."""


class ImageProviderRegistry(PluginRegistry[ImageEditProvider]):
    """Registry specialized for image-edit providers."""


__all__ = ["ImageProviderPlugin", "ImageProviderRegistry"]
