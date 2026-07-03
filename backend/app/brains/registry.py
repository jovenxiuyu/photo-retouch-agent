from __future__ import annotations

from dataclasses import dataclass

from app.brains.base import PromptBrain
from app.plugins.registry import PluginDefinition, PluginRegistry


@dataclass(frozen=True)
class AgentBrainPlugin(PluginDefinition[PromptBrain]):
    """Registration contract for a prompt-planning brain."""


class AgentBrainRegistry(PluginRegistry[PromptBrain]):
    """Registry specialized for prompt-planning brains."""


__all__ = ["AgentBrainPlugin", "AgentBrainRegistry"]
