from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.brains.factory import AgentBrainFactory
from app.brains.local import LocalPromptBrain
from app.brains.registry import AgentBrainPlugin
from app.core.config import Settings
from app.plugins.registry import PluginDefinition, PluginRegistry
from app.providers.factory import ImageProviderFactory
from app.providers.mock_image_provider import MockImageProvider
from app.providers.registry import ImageProviderPlugin


def test_registry_normalizes_names_and_rejects_alias_collisions() -> None:
    registry: PluginRegistry[str] = PluginRegistry()
    registry.register(
        PluginDefinition(
            plugin_id="First",
            aliases=("legacy",),
            build=lambda _key, _workspace: "first",
            capability=lambda: {"id": "first"},
        )
    )

    assert registry.get(" FIRST ").plugin_id == "First"
    assert registry.get("LEGACY").plugin_id == "First"
    assert registry.plugin_ids == ("First",)

    with pytest.raises(ValueError, match="legacy"):
        registry.register(
            PluginDefinition(
                plugin_id="second",
                aliases=("LEGACY",),
                build=lambda _key, _workspace: "second",
                capability=lambda: {"id": "second"},
            )
        )


def test_image_factory_accepts_a_plugin_without_core_changes(tmp_path) -> None:
    factory = ImageProviderFactory(
        Settings(data_dir=tmp_path / "data", image_provider="mock")
    )
    factory.register(
        ImageProviderPlugin(
            plugin_id="custom-image",
            build=lambda _key, _workspace: MockImageProvider(),
            capability=lambda: {
                "id": "custom-image",
                "label": "Custom Image",
                "model": "custom-v1",
                "configured": True,
                "requiresApiKey": False,
                "workspaceSupported": False,
            },
        )
    )

    provider = factory.create("CUSTOM-IMAGE")

    assert isinstance(provider, MockImageProvider)
    assert factory.capabilities()["actionProviders"][-1]["id"] == "custom-image"


def test_brain_factory_plugins_are_isolated_per_application(tmp_path) -> None:
    config = Settings(data_dir=tmp_path / "data", brain_provider="local")
    customized = AgentBrainFactory(config)
    customized.register(
        AgentBrainPlugin(
            plugin_id="custom-brain",
            build=lambda _key, _workspace: LocalPromptBrain(),
            capability=lambda: {
                "id": "custom-brain",
                "label": "Custom Brain",
                "model": "custom-v1",
                "configured": True,
                "visionMode": "derived",
                "requiresApiKey": False,
                "workspaceSupported": False,
            },
        )
    )

    assert isinstance(customized.create("custom-brain"), LocalPromptBrain)
    assert customized.capabilities()["brainProviders"][-1]["id"] == "custom-brain"

    isolated = AgentBrainFactory(config)
    with pytest.raises(HTTPException, match="Unsupported"):
        isolated.create("custom-brain")
