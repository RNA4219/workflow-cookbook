# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.workflow_plugins.errors import WorkflowPluginCapabilityError
from tools.workflow_plugins.interfaces import coerce_acceptance_index_result
from tools.workflow_plugins.runtime import WorkflowPluginRuntime


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate docs/acceptance/INDEX.md via workflow plugin.")
    parser.add_argument("--plugin-config", type=Path, required=True, help="Workflow plugin config path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "docs" / "acceptance" / "INDEX.md",
        help="Acceptance index output path.",
    )
    args = parser.parse_args(argv)

    runtime = WorkflowPluginRuntime.from_config(args.plugin_config)
    try:
        response = runtime.invoke_first(
            "acceptance.index",
            coercer=coerce_acceptance_index_result,
            repo_root=_REPO_ROOT,
        )
    except WorkflowPluginCapabilityError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not response.markdown.strip():
        print("Plugin returned an empty acceptance index.", file=sys.stderr)
        return 1
    args.output.write_text(response.markdown.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote acceptance index to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
