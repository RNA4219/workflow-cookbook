"""Tests for tools.ci.check_task_acceptance_bidirectional."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.ci.check_task_acceptance_bidirectional import (
    AcceptanceRecord,
    TaskRecord,
    SyncReport,
    _parse_front_matter,
    _scan_tasks,
    _scan_acceptances,
    validate_bidirectional_sync,
)


class TestParseFrontMatter:
    def test_parses_valid_front_matter(self) -> None:
        content = dedent("""
            ---
            task_id: TEST-001
            status: done
            owner: test
            ---
            # Content
            """).strip()
        result = _parse_front_matter(content)
        assert result["task_id"] == "TEST-001"
        assert result["status"] == "done"
        assert result["owner"] == "test"

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

    def test_returns_empty_for_no_front_matter(self) -> None:
        content = "# No front matter\n\nContent"
        result = _parse_front_matter(content)
        assert result == {}


class TestScanTasks:
    def test_scans_task_files(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "tasks"
        task_dir.mkdir()
        task_file = task_dir / "task-test.md"
        task_file.write_text(dedent("""
            ---
            task_id: TASK-001
            status: done
            ---
            # Task
            """).strip())

        tasks = _scan_tasks(task_dir)
        assert len(tasks) == 1
        assert tasks[0].task_id == "TASK-001"
        assert tasks[0].status == "done"

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        tasks = _scan_tasks(tmp_path / "nonexistent")
        assert tasks == []


class TestScanAcceptances:
    def test_scans_acceptance_files(self, tmp_path: Path) -> None:
        acc_dir = tmp_path / "acceptance"
        acc_dir.mkdir()
        acc_file = acc_dir / "AC-20260410-01.md"
        acc_file.write_text(dedent("""
            ---
            acceptance_id: AC-20260410-01
            task_id: TASK-001
            status: approved
            ---
            # Acceptance
            """).strip())

        acceptances = _scan_acceptances(acc_dir)
        assert len(acceptances) == 1
        assert acceptances[0].acceptance_id == "AC-20260410-01"
        assert acceptances[0].task_id == "TASK-001"
        assert acceptances[0].status == "approved"


class TestValidateBidirectionalSync:
    def test_valid_sync(self) -> None:
        tasks = [TaskRecord(task_id="TASK-001", status="done", file_path=Path("t.md"))]
        acceptances = [
            AcceptanceRecord(
                acceptance_id="AC-001",
                task_id="TASK-001",
                status="approved",
                file_path=Path("a.md"),
            )
        ]
        report = validate_bidirectional_sync(tasks, acceptances)
        assert report.errors == []
        assert report.warnings == []

    def test_error_done_task_without_acceptance(self) -> None:
        tasks = [TaskRecord(task_id="TASK-001", status="done", file_path=Path("t.md"))]
        acceptances: list[AcceptanceRecord] = []
        report = validate_bidirectional_sync(tasks, acceptances)
        assert len(report.errors) == 1
        assert "no acceptance record" in report.errors[0]

    def test_warning_done_task_with_unapproved_acceptance(self) -> None:
        tasks = [TaskRecord(task_id="TASK-001", status="done", file_path=Path("t.md"))]
        acceptances = [
            AcceptanceRecord(
                acceptance_id="AC-001",
                task_id="TASK-001",
                status="draft",
                file_path=Path("a.md"),
            )
        ]
        report = validate_bidirectional_sync(tasks, acceptances)
        assert report.errors == []
        assert len(report.warnings) == 1
        assert "none approved" in report.warnings[0]

    def test_error_approved_acceptance_without_task_id(self) -> None:
        tasks: list[TaskRecord] = []
        acceptances = [
            AcceptanceRecord(
                acceptance_id="AC-001",
                task_id="",
                status="approved",
                file_path=Path("a.md"),
            )
        ]
        report = validate_bidirectional_sync(tasks, acceptances)
        assert len(report.errors) == 1
        assert "no task_id reference" in report.errors[0]

    def test_error_approved_acceptance_references_nonexistent_task(self) -> None:
        tasks: list[TaskRecord] = []
        acceptances = [
            AcceptanceRecord(
                acceptance_id="AC-001",
                task_id="NONEXISTENT",
                status="approved",
                file_path=Path("a.md"),
            )
        ]
        report = validate_bidirectional_sync(tasks, acceptances)
        assert len(report.errors) == 1
        assert "non-existent task" in report.errors[0]

    def test_warning_approved_acceptance_references_non_done_task(self) -> None:
        tasks = [TaskRecord(task_id="TASK-001", status="active", file_path=Path("t.md"))]
        acceptances = [
            AcceptanceRecord(
                acceptance_id="AC-001",
                task_id="TASK-001",
                status="approved",
                file_path=Path("a.md"),
            )
        ]
        report = validate_bidirectional_sync(tasks, acceptances)
        assert report.errors == []
        assert len(report.warnings) == 1
        assert "expected 'done'" in report.warnings[0]

    def test_ignores_non_done_tasks(self) -> None:
        tasks = [TaskRecord(task_id="TASK-001", status="active", file_path=Path("t.md"))]
        acceptances: list[AcceptanceRecord] = []
        report = validate_bidirectional_sync(tasks, acceptances)
        assert report.errors == []
        assert report.warnings == []

    def test_ignores_non_approved_acceptances(self) -> None:
        tasks: list[TaskRecord] = []
        acceptances = [
            AcceptanceRecord(
                acceptance_id="AC-001",
                task_id="",
                status="draft",
                file_path=Path("a.md"),
            )
        ]
        report = validate_bidirectional_sync(tasks, acceptances)
        assert report.errors == []
        assert report.warnings == []