# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .plugin_config import load_workflow_plugin_specs
from .errors import WorkflowPluginCapabilityError
from .interfaces import CAPABILITY_METHOD_NAMES
from .plugin_loader import instantiate_workflow_plugins


class WorkflowPluginRuntime:
    def __init__(self, plugins: Sequence[Any]) -> None:
        self._plugins = list(plugins)

    @classmethod
    def from_config(cls, config_path: str | Path) -> "WorkflowPluginRuntime":
        target = Path(config_path).expanduser().resolve()
        specs = load_workflow_plugin_specs(target)
        plugins = instantiate_workflow_plugins(specs, base_path=target.parent)
        return cls(plugins)

    @property
    def plugins(self) -> list[Any]:
        return list(self._plugins)

    def iter_capability(self, capability: str) -> Iterable[Any]:
        for plugin in self._plugins:
            capabilities = getattr(plugin, "capabilities", ())
            if capability in capabilities:
                yield plugin

    def first_capability(self, capability: str) -> Any:
        for plugin in self.iter_capability(capability):
            return plugin
        raise WorkflowPluginCapabilityError(f"No workflow plugin provides {capability}.")

    def invoke_first(
        self,
        capability: str,
        *,
        coercer: Callable[[Any], Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        plugin = self.first_capability(capability)
        method_name = CAPABILITY_METHOD_NAMES[capability]
        result = getattr(plugin, method_name)(**kwargs)
        if coercer is None:
            return result
        return coercer(result)

    def invoke_all(
        self,
        capability: str,
        *,
        coercer: Callable[[Any], Any] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        results: list[Any] = []
        for plugin in self.iter_capability(capability):
            method_name = CAPABILITY_METHOD_NAMES[capability]
            result = getattr(plugin, method_name)(**kwargs)
            if coercer is not None:
                result = coercer(result)
            results.append(result)
        return results


__all__ = ["WorkflowPluginRuntime"]
