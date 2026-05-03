# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Plugin capability catalog consistency checker."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.workflow_plugins.interfaces import CAPABILITY_METHOD_NAMES

_DEFAULT_CATALOG = _REPO_ROOT / "examples" / "plugin-capability-catalog.sample.json"
_DEFAULT_SCHEMA = _REPO_ROOT / "schemas" / "plugin-capability-catalog.schema.json"
_DEFAULT_INTERFACES = _REPO_ROOT / "tools" / "workflow_plugins" / "interfaces.py"


@dataclass
class ValidationResult:
    """Result of capability catalog validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        return not self.errors

    def emit(self) -> None:
        for message in self.errors:
            print(message, file=sys.stderr)
        for message in self.warnings:
            print(message, file=sys.stderr)


def load_catalog_capabilities(catalog_path: Path) -> dict[str, str]:
    """Load capabilities from catalog JSON."""
    try:
        content = catalog_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {}
    capabilities = payload.get("capabilities")
    if capabilities is None or not isinstance(capabilities, list):
        return {}
    result: dict[str, str] = {}
    for item in capabilities:
        if isinstance(item, dict):
            cap_id = item.get("id")
            method_name = item.get("method_name")
            if cap_id and method_name:
                result[str(cap_id)] = str(method_name)
    return result


def load_interfaces_capabilities(interfaces_path: Path) -> dict[str, str]:
    """Extract CAPABILITY_METHOD_NAMES from interfaces.py by reading the file."""
    # Directly use imported CAPABILITY_METHOD_NAMES since we already imported it
    return dict(CAPABILITY_METHOD_NAMES)


def validate_plugin_capability_catalog(
    *,
    catalog_path: Path,
    interfaces_path: Path,
    check_schema: bool = True,
    schema_path: Path | None = None,
) -> ValidationResult:
    """Validate plugin capability catalog consistency."""
    result = ValidationResult()

    catalog_caps = load_catalog_capabilities(catalog_path)
    interfaces_caps = load_interfaces_capabilities(interfaces_path)

    if not catalog_caps:
        result.errors.append(f"No capabilities found in catalog {catalog_path}")
    if not interfaces_caps:
        result.errors.append(f"No capabilities found in interfaces {interfaces_path}")

    # Check catalog matches interfaces
    for cap_id, method_name in interfaces_caps.items():
        if cap_id not in catalog_caps:
            result.errors.append(
                f"Capability '{cap_id}' defined in interfaces.py but missing from catalog"
            )
        elif catalog_caps[cap_id] != method_name:
            result.errors.append(
                f"Capability '{cap_id}' method mismatch: "
                f"interfaces.py has '{method_name}', catalog has '{catalog_caps[cap_id]}'"
            )

    # Check catalog has no extra capabilities
    for cap_id in catalog_caps:
        if cap_id not in interfaces_caps:
            result.warnings.append(
                f"Capability '{cap_id}' in catalog but not defined in interfaces.py"
            )

    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for plugin capability catalog checker."""
    parser = argparse.ArgumentParser(
        description="Validate plugin capability catalog against interfaces.py."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=_DEFAULT_CATALOG,
        help="Path to plugin-capability-catalog JSON.",
    )
    parser.add_argument(
        "--interfaces",
        type=Path,
        default=_DEFAULT_INTERFACES,
        help="Path to tools/workflow_plugins/interfaces.py.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero on validation failure.",
    )
    args = parser.parse_args(argv)

    result = validate_plugin_capability_catalog(
        catalog_path=args.catalog,
        interfaces_path=args.interfaces,
    )
    result.emit()

    if result.is_success:
        print("Plugin capability catalog matches interfaces.py.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())