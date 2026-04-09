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

from tools.workflow_plugins.errors import WorkflowPluginCapabilityError
from tools.workflow_plugins.interfaces import (
    as_jsonable,
    coerce_docs_ack_result,
    coerce_docs_resolve_result,
    coerce_docs_stale_result,
)
from tools.workflow_plugins.runtime import WorkflowPluginRuntime


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Resolve / ack / stale-check docs through workflow plugins.")
    parser.add_argument("--plugin-config", type=Path, required=True, help="Workflow plugin config path.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser("resolve", help="Resolve docs for a task.")
    resolve_parser.add_argument("--task-id", required=True)
    resolve_parser.add_argument("--intent-id")

    ack_parser = subparsers.add_parser("ack", help="Acknowledge docs for a task.")
    ack_parser.add_argument("--task-id", required=True)
    ack_parser.add_argument("--doc-id", action="append", required=True)
    ack_parser.add_argument("--reader", default="workflow-cookbook")

    stale_parser = subparsers.add_parser("stale", help="Run stale check for a task.")
    stale_parser.add_argument("--task-id", required=True)
    stale_parser.add_argument("--check", action="store_true", help="Exit non-zero when stale docs exist.")

    args = parser.parse_args(argv)
    runtime = WorkflowPluginRuntime.from_config(args.plugin_config)

    try:
        if args.command == "resolve":
            payload = as_jsonable(
                runtime.invoke_first(
                    "docs.resolve",
                    coercer=coerce_docs_resolve_result,
                    repo_root=_REPO_ROOT,
                    task_id=args.task_id,
                    intent_id=args.intent_id,
                )
            )
        elif args.command == "ack":
            payload = as_jsonable(
                runtime.invoke_first(
                    "docs.ack",
                    coercer=coerce_docs_ack_result,
                    repo_root=_REPO_ROOT,
                    task_id=args.task_id,
                    doc_ids=args.doc_id,
                    reader=args.reader,
                )
            )
        else:
            payload = as_jsonable(
                runtime.invoke_first(
                    "docs.stale_check",
                    coercer=coerce_docs_stale_result,
                    repo_root=_REPO_ROOT,
                    task_id=args.task_id,
                )
            )
            if args.check and payload.get("stale"):
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 1
    except WorkflowPluginCapabilityError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
