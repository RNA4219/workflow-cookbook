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

from tools.workflow_plugins.interfaces import (
    as_jsonable,
    coerce_task_acceptance_sync_report,
)
from tools.workflow_plugins.runtime import WorkflowPluginRuntime


def _collect_report_errors(report) -> list[str]:
    errors = list(report.errors)
    for task in report.tasks:
        if not isinstance(task, dict):
            continue
        status = str(task.get("status", "")).strip().lower()
        acceptance_ids = task.get("acceptance_ids") or []
        if status == "done" and not acceptance_ids:
            errors.append(
                f"Done task '{task.get('task_id', 'unknown')}' does not have an acceptance record."
            )
    return errors


def _merge_reports(reports) -> dict[str, object]:
    merged_report: dict[str, object] = {
        "tasks": [],
        "acceptances": [],
        "errors": [],
        "warnings": [],
    }
    for report in reports:
        merged_report["tasks"].extend(report.tasks)
        merged_report["acceptances"].extend(report.acceptances)
        merged_report["errors"].extend(report.errors)
        merged_report["warnings"].extend(report.warnings)
    return merged_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Task Seed / Acceptance linkage using workflow plugins."
    )
    parser.add_argument(
        "--plugin-config",
        type=Path,
        required=True,
        help="Workflow plugin config path.",
    )
    parser.add_argument(
        "--emit-json",
        action="store_true",
        help="Emit the raw sync report as JSON.",
    )
    args = parser.parse_args(argv)

    runtime = WorkflowPluginRuntime.from_config(args.plugin_config)
    reports = runtime.invoke_all(
        "task_state.sync",
        coercer=coerce_task_acceptance_sync_report,
        repo_root=_REPO_ROOT,
    )

    if not reports:
        print("No workflow plugin provides task_state.sync.", file=sys.stderr)
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    merged_report = _merge_reports(reports)
    for report in reports:
        errors.extend(_collect_report_errors(report))
        warnings.extend(report.warnings)

    if args.emit_json:
        print(json.dumps(as_jsonable(merged_report), ensure_ascii=False, indent=2))

    for warning in warnings:
        print(warning, file=sys.stderr)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("Task Seed / Acceptance linkage is synchronized across configured workflow plugins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
