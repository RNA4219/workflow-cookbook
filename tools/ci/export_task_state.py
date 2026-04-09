#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Export TaskState: Task Seeds, Acceptances, Evidence relationships.

Generates JSON/YAML export of the task state for external consumption.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASKS_DIR = _REPO_ROOT / "docs" / "tasks"
_ACCEPTANCE_DIR = _REPO_ROOT / "docs" / "acceptance"
_OUTPUT_DIR = _REPO_ROOT / ".workflow-cache"


@dataclass
class TaskState:
    task_id: str
    intent_id: str
    status: str
    acceptance_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    file_path: str = ""


@dataclass
class AcceptanceState:
    acceptance_id: str
    task_id: str
    status: str
    reviewed_at: str
    file_path: str = ""


@dataclass
class EvidenceState:
    evidence_id: str
    task_id: str
    timestamp: str
    file_path: str = ""


@dataclass
class TaskStateExport:
    exported_at: str
    tasks: list[dict[str, Any]] = field(default_factory=list)
    acceptances: list[dict[str, Any]] = field(default_factory=list)
    evidences: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _parse_front_matter(content: str) -> dict[str, Any]:
    """Parse YAML front matter."""
    import re
    if not content.startswith("---"):
        return {}

    end_match = re.search(r"\n---\s*$|\n---\s*\n", content[3:])
    if not end_match:
        return {}

    front_matter = content[3 : end_match.start() + 3]
    result: dict[str, Any] = {}

    for line in front_matter.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2:
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
        result[key] = value

    return result


def scan_tasks(tasks_dir: Path) -> list[TaskState]:
    """Scan task files."""
    tasks = []
    if not tasks_dir.exists():
        return tasks

    for task_file in tasks_dir.glob("*.md"):
        content = task_file.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)

        task_id = fm.get("task_id", "")
        if not task_id:
            continue

        tasks.append(TaskState(
            task_id=task_id,
            intent_id=fm.get("intent_id", ""),
            status=fm.get("status", "unknown"),
            file_path=str(task_file.relative_to(_REPO_ROOT)),
        ))

    return tasks


def scan_acceptances(acceptance_dir: Path) -> list[AcceptanceState]:
    """Scan acceptance files."""
    acceptances = []
    if not acceptance_dir.exists():
        return acceptances

    for acc_file in acceptance_dir.glob("AC-*.md"):
        content = acc_file.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)

        acceptance_id = fm.get("acceptance_id", "")
        if not acceptance_id:
            continue

        acceptances.append(AcceptanceState(
            acceptance_id=acceptance_id,
            task_id=fm.get("task_id", ""),
            status=fm.get("status", "unknown"),
            reviewed_at=fm.get("reviewed_at", ""),
            file_path=str(acc_file.relative_to(_REPO_ROOT)),
        ))

    return acceptances


def scan_evidences(evidence_dir: Path) -> list[EvidenceState]:
    """Scan evidence files."""
    evidences = []
    if not evidence_dir.exists():
        return evidences

    for ev_file in evidence_dir.glob("**/*.json"):
        try:
            data = json.loads(ev_file.read_text(encoding="utf-8"))
            evidence_id = data.get("evidence_id", ev_file.stem)
            evidences.append(EvidenceState(
                evidence_id=evidence_id,
                task_id=data.get("taskSeedId", data.get("task_id", "")),
                timestamp=data.get("startTime", ""),
                file_path=str(ev_file.relative_to(_REPO_ROOT)),
            ))
        except (json.JSONDecodeError, KeyError):
            continue

    return evidences


def build_export(
    tasks: list[TaskState],
    acceptances: list[AcceptanceState],
    evidences: list[EvidenceState],
) -> TaskStateExport:
    """Build export with relationships."""
    from datetime import datetime, timezone

    export = TaskStateExport(
        exported_at=datetime.now(timezone.utc).isoformat(),
    )

    # Build lookup maps
    acc_by_task: dict[str, list[str]] = {}
    for acc in acceptances:
        if acc.task_id:
            acc_by_task.setdefault(acc.task_id, []).append(acc.acceptance_id)

    ev_by_task: dict[str, list[str]] = {}
    for ev in evidences:
        if ev.task_id:
            ev_by_task.setdefault(ev.task_id, []).append(ev.evidence_id)

    # Build tasks with relationships
    for task in tasks:
        task_dict = asdict(task)
        task_dict["acceptance_ids"] = acc_by_task.get(task.task_id, [])
        task_dict["evidence_ids"] = ev_by_task.get(task.task_id, [])
        export.tasks.append(task_dict)

    # Add acceptances and evidences
    for acc in acceptances:
        export.acceptances.append(asdict(acc))

    for ev in evidences:
        export.evidences.append(asdict(ev))

    # Summary
    export.summary = {
        "total_tasks": len(tasks),
        "total_acceptances": len(acceptances),
        "total_evidences": len(evidences),
        "tasks_with_acceptance": len([t for t in tasks if acc_by_task.get(t.task_id)]),
        "tasks_with_evidence": len([t for t in tasks if ev_by_task.get(t.task_id)]),
        "status_counts": {
            "tasks": {
                "done": len([t for t in tasks if t.status == "done"]),
                "active": len([t for t in tasks if t.status == "active"]),
                "pending": len([t for t in tasks if t.status == "pending"]),
            },
            "acceptances": {
                "approved": len([a for a in acceptances if a.status == "approved"]),
                "rejected": len([a for a in acceptances if a.status == "rejected"]),
                "draft": len([a for a in acceptances if a.status == "draft"]),
            },
        },
    }

    return export


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export TaskState for external consumption."
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=_TASKS_DIR,
        help="Tasks directory.",
    )
    parser.add_argument(
        "--acceptance-dir",
        type=Path,
        default=_ACCEPTANCE_DIR,
        help="Acceptance directory.",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=_OUTPUT_DIR,
        help="Evidence directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (default: stdout).",
    )

    args = parser.parse_args(argv)

    tasks = scan_tasks(args.tasks_dir)
    acceptances = scan_acceptances(args.acceptance_dir)
    evidences = scan_evidences(args.evidence_dir)

    export = build_export(tasks, acceptances, evidences)

    output_json = json.dumps(asdict(export), ensure_ascii=False, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json, encoding="utf-8")
        print(f"Exported task state to {args.output}")
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())