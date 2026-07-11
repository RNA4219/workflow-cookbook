#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Visualize transitive dependencies for workflow-cookbook.

Generates a dependency graph from requirements.txt and pyproject.toml,
outputting to docs/security/Dependency_Tree.md.
"""

from __future__ import annotations

import argparse
import subprocess  # nosec B404
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_FILE = _REPO_ROOT / "docs" / "security" / "Dependency_Tree.md"


def _get_direct_dependencies() -> dict[str, list[str]]:
    """Parse direct dependencies from requirements.txt and pyproject.toml."""
    direct: dict[str, list[str]] = {"runtime": [], "dev": []}

    # requirements.txt (runtime)
    req_file = _REPO_ROOT / "requirements.txt"
    if req_file.exists():
        for line in req_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                # Extract package name (before == or >=)
                pkg = line.split("==")[0].split(">=")[0].split("<")[0].strip()
                if pkg:
                    direct["runtime"].append(pkg)

    # pyproject.toml dev dependencies (from pyproject.toml)
    pyproject = _REPO_ROOT / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        # Simple parse for dev dependencies
        in_dev = False
        for line in content.splitlines():
            if "dev = [" in line:
                in_dev = True
            elif in_dev and "]" in line:
                in_dev = False
            elif in_dev and '"' in line:
                # Extract package name
                pkg = line.split('"')[1].split("==")[0].strip()
                if pkg and pkg not in ["pytest", "pytest-cov", "bandit", "pip-audit"]:
                    pass  # already known
                if pkg:
                    direct["dev"].append(pkg)

    return direct


def _run_pipdeptree() -> str:
    """Run pipdeptree and return output."""
    try:
        result = subprocess.run(  # nosec B603,B607
            ["pipdeptree", "--json-tree"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return ""
    except FileNotFoundError:
        print("pipdeptree not installed. Run: pip install pipdeptree", file=sys.stderr)
        return ""


def _filter_deps(tree_json: str, direct_deps: list[str]) -> list[dict[str, Any]]:
    """Filter dependency tree to only include workflow-cookbook deps."""
    import json

    try:
        tree = json.loads(tree_json)
    except json.JSONDecodeError:
        return []

    # Find packages that match our direct dependencies
    filtered: list[dict[str, Any]] = []
    for pkg in tree:
        pkg_name = pkg.get("package_name", "").lower().replace("-", "_")
        for direct in direct_deps:
            if direct.lower().replace("-", "_") == pkg_name:
                filtered.append(pkg)
                break

    return filtered


def _format_dep_tree(deps: list[dict[str, Any]], indent: int = 0) -> str:
    """Format dependency tree as markdown."""
    lines = []
    for dep in deps:
        name = dep.get("package_name", "unknown")
        version = dep.get("installed_version", "unknown")
        req_version = dep.get("required_version", "any")

        prefix = "  " * indent
        if indent == 0:
            lines.append(f"{prefix}- **{name}** `{version}` (required: `{req_version}`)")
        else:
            lines.append(f"{prefix}- {name} `{version}`")

        sub_deps = dep.get("dependencies", [])
        if sub_deps:
            lines.append(_format_dep_tree(sub_deps, indent + 1))

    return "\n".join(lines)


def _generate_markdown(direct: dict[str, list[str]], tree_json: str) -> str:
    """Generate full markdown document."""
    import datetime

    now = datetime.datetime.now(datetime.UTC)

    runtime_deps = _filter_deps(tree_json, direct["runtime"])
    dev_deps = _filter_deps(tree_json, direct["dev"])

    content = f"""---
intent_id: INT-SEC-014
owner: security
status: active
last_reviewed_at: {now.strftime("%Y-%m-%d")}
next_review_due: {(now + datetime.timedelta(days=30)).strftime("%Y-%m-%d")}
---

# Dependency Tree

本ドキュメントは workflow-cookbook の依存関係（transitive dependencies）を可視化する。

生成コマンド: `python tools/security/visualize_dependencies.py`

## Runtime Dependencies

直接依存（requirements.txt）:

| Package | Version |
| --- | --- |
"""

    for pkg in direct["runtime"]:
        content += f"| {pkg} | pinned |\n"

    content += "\n### Transitive Dependencies\n\n"

    if runtime_deps:
        content += _format_dep_tree(runtime_deps) + "\n"
    else:
        content += "(pipdeptree で transitive deps を取得)\n"

    content += """
## Dev Dependencies

直接依存（pyproject.toml dev）:

| Package | Version |
| --- | --- |
"""

    for pkg in ["pytest", "pytest-cov", "bandit", "pip-audit"]:
        content += f"| {pkg} | pinned |\n"

    content += "\n### Transitive Dependencies\n\n"

    if dev_deps:
        content += _format_dep_tree(dev_deps) + "\n"
    else:
        content += "（dev deps の transitive は CI 再現性に影響しない）\n"

    content += """
## 可視化方法

### pipdeptree

```bash
pip install pipdeptree
pipdeptree --json-tree > .ga/dependency-tree.json
```

### requirements.txt ベース

本プロジェクトは `requirements.txt` を lockfile 相当として扱う。
transitive dependencies は pipdeptree で可視化し、
docs/security/Dependency_Tree.md に記録。

## 参照

- [Dependency Governance](./Dependency_Governance.md)
- [Enterprise Readiness Assessment](../reports/enterprise-readiness-assessment-20260417.md)
"""

    return content


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize transitive dependencies."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_OUTPUT_FILE,
        help="Output markdown file.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Output JSON file (optional).",
    )
    args = parser.parse_args(argv)

    direct = _get_direct_dependencies()
    tree_json = _run_pipdeptree()

    if not tree_json:
        print("Warning: pipdeptree output empty, generating partial doc.", file=sys.stderr)

    markdown = _generate_markdown(direct, tree_json)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote dependency tree to {args.output}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(tree_json, encoding="utf-8")
        print(f"Wrote JSON to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())