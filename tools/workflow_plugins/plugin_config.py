from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.workflow_plugins.plugin_loader import WorkflowPluginSpec

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]


class WorkflowPluginConfigError(ValueError):
    pass


def load_workflow_plugin_specs_from_mapping(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[WorkflowPluginSpec]:
    raw_specs: Sequence[Any]
    if isinstance(payload, Mapping):
        raw_specs = payload.get("workflow_plugins", [])
    else:
        raw_specs = payload
    if not isinstance(raw_specs, Sequence) or isinstance(raw_specs, (str, bytes)):
        raise WorkflowPluginConfigError("Workflow plugin config must contain a sequence")

    specs: list[WorkflowPluginSpec] = []
    for index, item in enumerate(raw_specs):
        if not isinstance(item, Mapping):
            raise WorkflowPluginConfigError(f"Plugin spec at index {index} must be a mapping")
        factory = item.get("factory")
        if not isinstance(factory, str) or not factory.strip():
            raise WorkflowPluginConfigError(f"Plugin spec at index {index} requires factory")
        options = item.get("options")
        if options is not None and not isinstance(options, Mapping):
            raise WorkflowPluginConfigError(
                f"Plugin spec at index {index} options must be a mapping"
            )
        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            raise WorkflowPluginConfigError(
                f"Plugin spec at index {index} enabled must be a boolean"
            )
        python_paths = item.get("python_paths")
        if python_paths is not None and (
            not isinstance(python_paths, Sequence) or isinstance(python_paths, (str, bytes))
        ):
            raise WorkflowPluginConfigError(
                f"Plugin spec at index {index} python_paths must be a sequence"
            )
        specs.append(
            WorkflowPluginSpec(
                factory=factory.strip(),
                options=dict(options) if options is not None else None,
                enabled=enabled,
                python_paths=[str(path) for path in python_paths] if python_paths is not None else None,
            )
        )
    return specs


def load_workflow_plugin_specs_from_path(path: str | Path) -> list[WorkflowPluginSpec]:
    target = Path(path).expanduser()
    suffix = target.suffix.lower()
    raw = target.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(raw)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise WorkflowPluginConfigError("YAML plugin config requires PyYAML to be installed")
        payload = yaml.safe_load(raw)  # type: ignore[union-attr]
    else:
        raise WorkflowPluginConfigError(f"Unsupported workflow plugin config format: {suffix}")
    if not isinstance(payload, (Mapping, Sequence)) or isinstance(payload, (str, bytes)):
        raise WorkflowPluginConfigError("Workflow plugin config root must be a mapping or sequence")
    return load_workflow_plugin_specs_from_mapping(payload)


def load_workflow_plugin_specs(
    source: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | Path,
) -> list[WorkflowPluginSpec]:
    if isinstance(source, (str, Path)):
        return load_workflow_plugin_specs_from_path(source)
    return load_workflow_plugin_specs_from_mapping(source)


__all__ = [
    "WorkflowPluginConfigError",
    "load_workflow_plugin_specs",
    "load_workflow_plugin_specs_from_mapping",
    "load_workflow_plugin_specs_from_path",
]
