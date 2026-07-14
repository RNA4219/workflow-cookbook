# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFEST = _REPO_ROOT / "governance" / "repo-responsibilities.yaml"


def validate_manifest(payload: dict[str, Any]) -> list[str]:
    """Return deterministic ownership errors for the canonical responsibility manifest."""
    errors: list[str] = []
    if payload.get("schema_version") != "repo-responsibilities/v1":
        errors.append("schema_version must be repo-responsibilities/v1")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        return [*errors, "capabilities must be a non-empty list"]

    names: list[str] = []
    for index, item in enumerate(capabilities):
        if not isinstance(item, dict):
            errors.append(f"capabilities[{index}] must be an object")
            continue
        for field in ("capability", "owner_repo", "canonical_path", "responsibility"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"capabilities[{index}].{field} must be a non-empty string")
        capability = item.get("capability")
        if isinstance(capability, str) and capability.strip():
            names.append(capability)

    for name, count in sorted(Counter(names).items()):
        if count != 1:
            errors.append(f"capability {name!r} has {count} canonical owners; expected exactly one")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the canonical repository responsibility manifest.")
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true", help="Compatibility flag for CI invocation.")
    args = parser.parse_args()
    payload = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    errors = validate_manifest(payload if isinstance(payload, dict) else {})
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {args.manifest} defines one owner for each capability")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
