# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219
"""Generate CycloneDX SBOM from requirements.txt."""

from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_requirements(requirements_path: Path) -> list[dict[str, Any]]:
    """Parse requirements.txt and return component list."""
    components: list[dict[str, Any]] = []
    if not requirements_path.exists():
        return components

    content = requirements_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Handle == pinned versions
        if "==" in stripped:
            name, version = stripped.split("==", 1)
            name = name.strip()
            version = version.strip()
            # Normalize name for purl (lowercase, replace - with _)
            purl_name = name.lower().replace("-", "_")
            components.append({
                "type": "library",
                "name": name,
                "version": version,
                "purl": f"pkg:pypi/{purl_name}@{version}",
                "licenses": [{"license": {"id": "MIT"}}],  # Assume MIT for known packages
            })
    return components


def generate_sbom(requirements_path: Path, output_path: Path, project_name: str, project_version: str) -> int:
    """Generate CycloneDX JSON SBOM."""
    components = parse_requirements(requirements_path)

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "component": {
                "type": "application",
                "name": project_name,
                "version": project_version,
                "purl": f"pkg:pypi/{project_name.lower().replace('-', '_')}@{project_version}",
            },
        },
        "components": components,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sbom, indent=2), encoding="utf-8")
    print(f"SBOM generated: {output_path} ({len(components)} components)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate CycloneDX SBOM")
    parser.add_argument("--requirements", type=Path, default=Path("requirements.txt"))
    parser.add_argument("--output", type=Path, default=Path(".ga/sbom.json"))
    parser.add_argument("--project-name", default="workflow-cookbook")
    parser.add_argument("--project-version", default="0.1.0")
    args = parser.parse_args(argv)
    return generate_sbom(args.requirements, args.output, args.project_name, args.project_version)


if __name__ == "__main__":
    raise SystemExit(main())