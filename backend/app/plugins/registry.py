from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

TPlugin = TypeVar("TPlugin")


@dataclass(frozen=True)
class PluginDefinition(Generic[TPlugin]):
    """A named builder and its safe, public capability metadata."""

    plugin_id: str
    build: Callable[[str | None, str | None], TPlugin]
    capability: Callable[[], dict[str, Any]]
    aliases: tuple[str, ...] = ()


class PluginRegistry(Generic[TPlugin]):
    """Instance-local registry that keeps plugin order deterministic."""

    def __init__(self) -> None:
        self._plugins: list[PluginDefinition[TPlugin]] = []
        self._by_name: dict[str, PluginDefinition[TPlugin]] = {}

    def register(self, plugin: PluginDefinition[TPlugin]) -> None:
        names = (plugin.plugin_id, *plugin.aliases)
        normalized_names = tuple(self._normalize(name) for name in names)
        duplicates = [name for name in normalized_names if name in self._by_name]
        if duplicates:
            raise ValueError(f"Plugin name already registered: {duplicates[0]}")

        self._plugins.append(plugin)
        for name in normalized_names:
            self._by_name[name] = plugin

    def get(self, plugin_id: str) -> PluginDefinition[TPlugin]:
        return self._by_name[self._normalize(plugin_id)]

    def capabilities(self) -> list[dict[str, Any]]:
        return [plugin.capability() for plugin in self._plugins]

    @property
    def plugin_ids(self) -> tuple[str, ...]:
        return tuple(plugin.plugin_id for plugin in self._plugins)

    @staticmethod
    def _normalize(name: str) -> str:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("Plugin name cannot be empty.")
        return normalized
