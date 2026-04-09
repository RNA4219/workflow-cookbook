"""Tests for tools.ci.export_task_state."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.ci.export_task_state import (
    _parse_front_matter,
    build_export,
    TaskState,
    AcceptanceState,
    EvidenceState,
)


class TestParseFrontMatter:
    def test_parses_yaml_front_matter(self) -> None:
        content = dedent("""
            ---
            task_id: TEST-001
            status: done
            ---
            # Content
            """).strip()
        result = _parse_front_matter(content)
        assert result.get("task_id") == "TEST-001"
        assert result.get("status") == "done"

    def test_returns_empty_for_no_front_matter(self) -> None:
        result = _parse_front_matter("# No front matter")
        assert result == {}

    def test_handles_quoted_values(self) -> None:
        content = dedent("""
            ---
            task_id: "TEST-001"
            status: 'done'
            ---
            """).strip()
        result = _parse_front_matter(content)
        assert result.get("task_id") == "TEST-001"
        assert result.get("status") == "done"


class TestBuildExport:
    def test_builds_export_with_relationships(self) -> None:
        tasks = [TaskState(
            task_id="TASK-001",
            intent_id="INT-001",
            status="done",
            file_path="tasks/task.md",
        )]
        acceptances = [AcceptanceState(
            acceptance_id="AC-001",
            task_id="TASK-001",
            status="approved",
            reviewed_at="2026-04-10",
            file_path="acceptance/AC-001.md",
        )]
        evidences = [EvidenceState(
            evidence_id="EV-001",
            task_id="TASK-001",
            timestamp="2026-04-10T00:00:00Z",
            file_path="evidence/ev.json",
        )]

        export = build_export(tasks, acceptances, evidences)

        assert export.summary["total_tasks"] == 1
        assert export.summary["total_acceptances"] == 1
        assert export.summary["total_evidences"] == 1
        assert len(export.tasks) == 1
        assert "AC-001" in export.tasks[0]["acceptance_ids"]
        assert "EV-001" in export.tasks[0]["evidence_ids"]

    def test_builds_export_with_status_counts(self) -> None:
        tasks = [
            TaskState(task_id="TASK-001", intent_id="INT-001", status="done", file_path=""),
            TaskState(task_id="TASK-002", intent_id="INT-001", status="active", file_path=""),
            TaskState(task_id="TASK-003", intent_id="INT-001", status="pending", file_path=""),
        ]
        acceptances = [
            AcceptanceState(acceptance_id="AC-001", task_id="", status="approved", reviewed_at="", file_path=""),
            AcceptanceState(acceptance_id="AC-002", task_id="", status="rejected", reviewed_at="", file_path=""),
        ]
        evidences = []

        export = build_export(tasks, acceptances, evidences)

        assert export.summary["status_counts"]["tasks"]["done"] == 1
        assert export.summary["status_counts"]["tasks"]["active"] == 1
        assert export.summary["status_counts"]["tasks"]["pending"] == 1
        assert export.summary["status_counts"]["acceptances"]["approved"] == 1
        assert export.summary["status_counts"]["acceptances"]["rejected"] == 1

    def test_builds_export_with_no_data(self) -> None:
        export = build_export([], [], [])

        assert export.summary["total_tasks"] == 0
        assert export.summary["total_acceptances"] == 0
        assert export.summary["total_evidences"] == 0
        assert len(export.tasks) == 0
    def test_builds_export_with_relationships(self, tmp_path: Path) -> None:
        from tools.ci.export_task_state import TaskState, AcceptanceState, EvidenceState

        tasks = [TaskState(
            task_id="TASK-001",
            intent_id="INT-001",
            status="done",
            file_path="tasks/task.md",
        )]
        acceptances = [AcceptanceState(
            acceptance_id="AC-001",
            task_id="TASK-001",
            status="approved",
            reviewed_at="2026-04-10",
            file_path="acceptance/AC-001.md",
        )]
        evidences = [EvidenceState(
            evidence_id="EV-001",
            task_id="TASK-001",
            timestamp="2026-04-10T00:00:00Z",
            file_path="evidence/ev.json",
        )]

        export = build_export(tasks, acceptances, evidences)

        assert export.summary["total_tasks"] == 1
        assert export.summary["total_acceptances"] == 1
        assert export.summary["total_evidences"] == 1
        assert len(export.tasks) == 1
        assert "AC-001" in export.tasks[0]["acceptance_ids"]
        assert "EV-001" in export.tasks[0]["evidence_ids"]