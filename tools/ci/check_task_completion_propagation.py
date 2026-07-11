#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2026 RNA4219

"""Check Task Seed completion propagation to completion-record.

Detects Task Seeds with status: done that are not reflected in
completion-record.md and generates nudges for propagation.

SKILL-DRAFT-002: Task Seed Completion Auto-Propagation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASKS_DIR = _REPO_ROOT / "docs" / "tasks"
_COMPLETION_RECORD = _REPO_ROOT / "docs" / "completion-record.md"
_NUDGE_DIR = _REPO_ROOT / ".workflow-cache" / "nudges"


def _parse_front_matter(content: str) -> dict[str, str]:
    """Parse YAML front matter from markdown content."""
    if not content.startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    front_matter = parts[1].strip()
    result: dict[str, str] = {}
    for line in front_matter.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def _get_done_task_ids(tasks_dir: Path, repo_root: Path = _REPO_ROOT) -> list[dict[str, Any]]:
    """Get all Task Seeds with status: done."""
    done_tasks: list[dict[str, Any]] = []
    if not tasks_dir.exists():
        return done_tasks

    for task_file in tasks_dir.glob("task-*.md"):
        content = task_file.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)
        status = fm.get("status", "").strip().lower()

        if status == "done":
            task_id = fm.get("task_id", task_file.stem)
            title = ""
            for line in content.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            # Handle relative_to for paths outside repo_root
            try:
                task_path = str(task_file.relative_to(repo_root))
            except ValueError:
                task_path = str(task_file)

            done_tasks.append({
                "task_id": task_id,
                "file": task_file.name,
                "title": title,
                "path": task_path,
                "last_reviewed_at": fm.get("last_reviewed_at", ""),
            })

    return done_tasks


def _get_recorded_task_ids(completion_path: Path) -> set[str]:
    """Get task IDs already recorded in completion-record."""
    recorded: set[str] = set()

    if not completion_path.exists():
        return recorded

    content = completion_path.read_text(encoding="utf-8")

    # Find task links in completion entries
    patterns = [
        r"docs/tasks/task-[^)]+.md",
        r"docs/tasks/[^\)]+",
        r"task-[^\)]+.md",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, content):
            # Extract task_id from link
            link = match.group()
            # task_id from file name: task-example-name.md -> example-name
            task_match = re.search(r"task-([^\.)]+)", link)
            if task_match:
                recorded.add(task_match.group(1))

    # Also check for explicit task_id references in tables
    task_id_pattern = re.compile(r"task_id:\s*[\d-]+")
    for match in task_id_pattern.finditer(content):
        recorded.add(match.group().split(":")[1].strip())

    return recorded


def check_task_completion_propagation(
    tasks_dir: Path = _TASKS_DIR,
    completion_path: Path = _COMPLETION_RECORD,
    repo_root: Path = _REPO_ROOT,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Check that done Task Seeds are recorded in completion-record.

    Returns nudges for unpropagated tasks and warnings.
    """
    nudges: list[dict[str, Any]] = []
    warnings: list[str] = []

    done_tasks = _get_done_task_ids(tasks_dir, repo_root)
    recorded_ids = _get_recorded_task_ids(completion_path)

    now = datetime.now()

    for task in done_tasks:
        task_id = task["task_id"]

        # Check if task_id or file name is in completion-record
        is_recorded = (
            task_id in recorded_ids
            or task["file"].replace(".md", "") in recorded_ids
            or any(task["file"].replace("task-", "").replace(".md", "") in rid for rid in recorded_ids)
        )

        if not is_recorded:
            nudge = {
                "nudge_id": f"NUDGE-TASK-{task_id}-{now.strftime('%Y%m%d')}",
                "reason": f"Task Seed '{task_id}' ({task['file']}) status: done but not in completion-record",
                "target_kind": "task_seed",
                "target_ref": task["path"],
                "suggested_action": "Add entry to docs/completion-record.md with link to task file",
                "created_at": now.isoformat(),
                "priority": "medium",
                "category": "completion_propagation",
                "blocking": False,
                "task_title": task["title"],
                "task_file": task["file"],
            }
            nudges.append(nudge)

    return nudges, warnings


def write_nudges(nudges: list[dict[str, Any]], output_dir: Path) -> int:
    """Write nudge records to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for nudge in nudges:
        nudge_path = output_dir / f"{nudge['nudge_id']}.json"
        nudge_path.write_text(
            json.dumps(nudge, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Task Seed completion propagation"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run check mode (exit 1 on unpropagated tasks)",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=_TASKS_DIR,
        help="Path to docs/tasks directory",
    )
    parser.add_argument(
        "--completion-path",
        type=Path,
        default=_COMPLETION_RECORD,
        help="Path to completion-record.md",
    )
    parser.add_argument(
        "--write-nudges",
        action="store_true",
        help="Write nudge records to .workflow-cache/nudges/",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    nudges, warnings = check_task_completion_propagation(
        tasks_dir=args.tasks_dir,
        completion_path=args.completion_path,
    )

    if args.write_nudges and nudges:
        written = write_nudges(nudges, _NUDGE_DIR)
        print(f"Wrote {written} nudge records to {_NUDGE_DIR}")

    if args.json:
        output = {
            "nudges": nudges,
            "warnings": warnings,
            "summary": {
                "total_done_tasks": len(_get_done_task_ids(args.tasks_dir)),
                "unpropagated": len(nudges),
                "warnings": len(warnings),
            },
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        if warnings:
            for w in warnings:
                print(f"WARNING: {w}")
        if nudges:
            for n in nudges:
                print(f"NUDGE: {n['nudge_id']}")
                print(f"  Reason: {n['reason']}")
                print(f"  Action: {n['suggested_action']}")
        else:
            print("All done Task Seeds are propagated to completion-record")

    if args.check and nudges:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())