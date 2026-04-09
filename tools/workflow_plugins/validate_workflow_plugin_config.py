# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.workflow_plugins.errors import WorkflowPluginError
from tools.workflow_plugins.plugin_config import WorkflowPluginConfigError, load_workflow_plugin_specs
from tools.workflow_plugins.runtime import WorkflowPluginRuntime


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate workflow plugin config and optional instantiation.")
    parser.add_argument("--plugin-config", type=Path, required=True, help="Workflow plugin config path.")
    parser.add_argument(
        "--instantiate",
        action="store_true",
        help="Import and instantiate plugins in addition to validating config shape.",
    )
    parser.add_argument("--emit-json", action="store_true", help="Emit validation summary as JSON.")
    args = parser.parse_args(argv)

    try:
        specs = load_workflow_plugin_specs(args.plugin_config)
        payload: dict[str, object] = {
            "plugin_config": str(Path(args.plugin_config).expanduser().resolve()),
            "valid": True,
            "specs": [
                {
                    "factory": spec.factory,
                    "enabled": spec.enabled,
                    "python_paths": list(spec.python_paths or []),
                    "options_keys": sorted((spec.options or {}).keys()),
                }
                for spec in specs
            ],
        }
        if args.instantiate:
            runtime = WorkflowPluginRuntime.from_config(args.plugin_config)
            payload["plugins"] = [
                {
                    "type": type(plugin).__name__,
                    "capabilities": list(getattr(plugin, "capabilities", ())),
                }
                for plugin in runtime.plugins
            ]
    except (WorkflowPluginConfigError, WorkflowPluginError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.emit_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Validated {len(specs)} workflow plugin spec(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
