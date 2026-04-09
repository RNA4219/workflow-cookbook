from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.protocols.plugin_loader import InferencePluginLoadError, InferencePluginSpec

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]


class InferencePluginConfigError(ValueError):
    pass


def load_inference_plugin_specs_from_mapping(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[InferencePluginSpec]:
    raw_specs: Sequence[Any]
    if isinstance(payload, Mapping):
        raw_specs = payload.get("inference_plugins", [])
    else:
        raw_specs = payload
    if not isinstance(raw_specs, Sequence) or isinstance(raw_specs, (str, bytes)):
        raise InferencePluginConfigError("Plugin config must contain a sequence of plugin specs")

    specs: list[InferencePluginSpec] = []
    for index, item in enumerate(raw_specs):
        if not isinstance(item, Mapping):
            raise InferencePluginConfigError(f"Plugin spec at index {index} must be a mapping")
        factory = item.get("factory")
        if not isinstance(factory, str) or not factory.strip():
            raise InferencePluginConfigError(f"Plugin spec at index {index} requires factory")
        options = item.get("options")
        if options is not None and not isinstance(options, Mapping):
            raise InferencePluginConfigError(
                f"Plugin spec at index {index} options must be a mapping"
            )
        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            raise InferencePluginConfigError(
                f"Plugin spec at index {index} enabled must be a boolean"
            )
        specs.append(
            InferencePluginSpec(
                factory=factory.strip(),
                options=dict(options) if options is not None else None,
                enabled=enabled,
            )
        )
    return specs


def load_inference_plugin_specs_from_path(path: str | Path) -> list[InferencePluginSpec]:
    target = Path(path).expanduser()
    suffix = target.suffix.lower()
    raw = target.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(raw)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise InferencePluginConfigError(
                "YAML plugin config requires PyYAML to be installed"
            )
        payload = yaml.safe_load(raw)  # type: ignore[union-attr]
    else:
        raise InferencePluginConfigError(
            f"Unsupported plugin config format: {target.suffix}"
        )
    if not isinstance(payload, (Mapping, Sequence)) or isinstance(payload, (str, bytes)):
        raise InferencePluginConfigError("Plugin config root must be a mapping or sequence")
    return load_inference_plugin_specs_from_mapping(payload)


def load_inference_plugin_specs(
    source: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | Path,
) -> list[InferencePluginSpec]:
    if isinstance(source, (str, Path)):
        return load_inference_plugin_specs_from_path(source)
    return load_inference_plugin_specs_from_mapping(source)


__all__ = [
    "InferencePluginConfigError",
    "InferencePluginLoadError",
    "InferencePluginSpec",
    "load_inference_plugin_specs",
    "load_inference_plugin_specs_from_mapping",
    "load_inference_plugin_specs_from_path",
]
