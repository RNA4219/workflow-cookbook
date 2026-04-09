from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from tools.perf.structured_logger import InferenceLogPlugin


class InferencePluginLoadError(ValueError):
    pass


@dataclass(frozen=True)
class InferencePluginSpec:
    factory: str
    options: Mapping[str, Any] | None = None
    enabled: bool = True


def load_plugin_factory(import_path: str) -> Callable[..., Any]:
    module_name, separator, attr_name = import_path.partition(":")
    if separator == "" or not module_name or not attr_name:
        raise InferencePluginLoadError(
            "Plugin factory path must use 'module:attribute' format"
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise InferencePluginLoadError(f"Plugin module could not be imported: {module_name}") from exc
    try:
        factory = getattr(module, attr_name)
    except AttributeError as exc:
        raise InferencePluginLoadError(
            f"Plugin factory attribute could not be resolved: {import_path}"
        ) from exc
    if not callable(factory):
        raise InferencePluginLoadError(f"Plugin factory is not callable: {import_path}")
    return factory


def instantiate_inference_plugin(
    spec: InferencePluginSpec,
    *,
    factory_loader: Callable[[str], Callable[..., Any]] = load_plugin_factory,
) -> InferenceLogPlugin:
    factory = factory_loader(spec.factory)
    plugin = factory(**dict(spec.options or {}))
    if not hasattr(plugin, "handle_inference") or not callable(plugin.handle_inference):
        raise InferencePluginLoadError(
            f"Instantiated plugin does not implement handle_inference(): {spec.factory}"
        )
    return plugin


def instantiate_inference_plugins(
    specs: Sequence[InferencePluginSpec],
    *,
    factory_loader: Callable[[str], Callable[..., Any]] = load_plugin_factory,
) -> list[InferenceLogPlugin]:
    plugins: list[InferenceLogPlugin] = []
    for spec in specs:
        if not spec.enabled:
            continue
        plugins.append(instantiate_inference_plugin(spec, factory_loader=factory_loader))
    return plugins


__all__ = [
    "InferencePluginLoadError",
    "InferencePluginSpec",
    "instantiate_inference_plugin",
    "instantiate_inference_plugins",
    "load_plugin_factory",
]
