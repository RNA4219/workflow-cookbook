from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .errors import WorkflowPluginCapabilityError, WorkflowPluginLoadError
from .interfaces import CAPABILITY_METHOD_NAMES


@dataclass(frozen=True)
class WorkflowPluginSpec:
    factory: str
    options: Mapping[str, Any] | None = None
    enabled: bool = True
    python_paths: Sequence[str] | None = None


def _extend_sys_path(paths: Sequence[str] | None, *, base_path: Path | None = None) -> None:
    if not paths:
        return
    for raw_path in paths:
        candidate = Path(raw_path)
        if base_path is not None and not candidate.is_absolute():
            candidate = (base_path / candidate).resolve()
        else:
            candidate = candidate.resolve()
        rendered = str(candidate)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)


def load_plugin_factory(
    import_path: str,
    *,
    python_paths: Sequence[str] | None = None,
    base_path: Path | None = None,
) -> Callable[..., Any]:
    module_name, separator, attr_name = import_path.partition(":")
    if separator == "" or not module_name or not attr_name:
        raise WorkflowPluginLoadError(
            "Plugin factory path must use 'module:attribute' format"
        )

    _extend_sys_path(python_paths, base_path=base_path)

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise WorkflowPluginLoadError(f"Plugin module could not be imported: {module_name}") from exc

    try:
        factory = getattr(module, attr_name)
    except AttributeError as exc:
        raise WorkflowPluginLoadError(
            f"Plugin factory attribute could not be resolved: {import_path}"
        ) from exc
    if not callable(factory):
        raise WorkflowPluginLoadError(f"Plugin factory is not callable: {import_path}")
    return factory


def instantiate_workflow_plugin(
    spec: WorkflowPluginSpec,
    *,
    factory_loader: Callable[..., Callable[..., Any]] = load_plugin_factory,
    base_path: Path | None = None,
) -> Any:
    factory = factory_loader(
        spec.factory,
        python_paths=spec.python_paths,
        base_path=base_path,
    )
    plugin = factory(**dict(spec.options or {}))
    capabilities = getattr(plugin, "capabilities", None)
    if not isinstance(capabilities, Sequence) or isinstance(capabilities, (str, bytes)):
        raise WorkflowPluginLoadError(
            f"Instantiated plugin does not expose a capabilities sequence: {spec.factory}"
        )
    for capability in capabilities:
        method_name = CAPABILITY_METHOD_NAMES.get(str(capability))
        if method_name is None:
            raise WorkflowPluginCapabilityError(
                f"Plugin declares unsupported capability '{capability}': {spec.factory}"
            )
        if not callable(getattr(plugin, method_name, None)):
            raise WorkflowPluginCapabilityError(
                f"Plugin capability '{capability}' requires callable '{method_name}': {spec.factory}"
            )
    return plugin


def instantiate_workflow_plugins(
    specs: Sequence[WorkflowPluginSpec],
    *,
    factory_loader: Callable[..., Callable[..., Any]] = load_plugin_factory,
    base_path: Path | None = None,
) -> list[Any]:
    plugins: list[Any] = []
    for spec in specs:
        if not spec.enabled:
            continue
        plugins.append(
            instantiate_workflow_plugin(
                spec,
                factory_loader=factory_loader,
                base_path=base_path,
            )
        )
    return plugins


__all__ = [
    "WorkflowPluginLoadError",
    "WorkflowPluginSpec",
    "instantiate_workflow_plugin",
    "instantiate_workflow_plugins",
    "load_plugin_factory",
]
