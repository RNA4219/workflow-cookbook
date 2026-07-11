#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2026 RNA4219

"""Tests for check_task_completion_propagation.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.ci.check_task_completion_propagation import (
    _get_done_task_ids,
    _get_recorded_task_ids,
    _parse_front_matter,
    check_task_completion_propagation,
)


@pytest.fixture
def temp_tasks_dir(tmp_path: Path) -> Path:
    """Create temporary tasks directory."""
    tasks_dir = tmp_path / "docs" / "tasks"
    tasks_dir.mkdir(parents=True)
    return tasks_dir


@pytest.fixture
def temp_completion_record(tmp_path: Path) -> Path:
    """Create temporary completion-record."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    completion = docs_dir / "completion-record.md"
    completion.write_text("# Completion Record\n", encoding="utf-8")
    return completion


@pytest.fixture
def temp_repo_root(tmp_path: Path) -> Path:
    """Create temporary repo root."""
    return tmp_path


def write_task_seed(path: Path, task_id: str, status: str, title: str = "Test Task") -> None:
    """Write a Task Seed file."""
    content = f"""---
task_id: {task_id}
status: {status}
last_reviewed_at: 2026-05-01
---

# Task Seed: {title}

## 背景

Test background.

## 完了条件

Test conditions.
"""
    path.write_text(content, encoding="utf-8")


class TestParseFrontMatter:
    """Test _parse_front_matter function."""

    def test_valid_front_matter(self, temp_tasks_dir: Path) -> None:
        task_path = temp_tasks_dir / "task-test-001.md"
        write_task_seed(task_path, "001", "done")
        content = task_path.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)
        assert fm.get("task_id") == "001"
        assert fm.get("status") == "done"

    def test_no_front_matter(self) -> None:
        content = "# No front matter\n\nContent"
        fm = _parse_front_matter(content)
        assert fm == {}

    def test_empty_front_matter(self) -> None:
        content = "---\n---\n# Content"
        fm = _parse_front_matter(content)
        assert fm == {}


class TestGetDoneTaskIds:
    """Test _get_done_task_ids function."""

    def test_single_done_task(
        self, temp_tasks_dir: Path, temp_repo_root: Path
    ) -> None:
        task_path = temp_tasks_dir / "task-done-001.md"
        write_task_seed(task_path, "20260501-01", "done", "Done Task")
        done_tasks = _get_done_task_ids(temp_tasks_dir, temp_repo_root)
        assert len(done_tasks) == 1
        assert done_tasks[0]["task_id"] == "20260501-01"

    def test_multiple_tasks(
        self, temp_tasks_dir: Path, temp_repo_root: Path
    ) -> None:
        write_task_seed(temp_tasks_dir / "task-done-001.md", "001", "done")
        write_task_seed(temp_tasks_dir / "task-pending-002.md", "002", "pending")
        write_task_seed(temp_tasks_dir / "task-done-003.md", "003", "done")
        done_tasks = _get_done_task_ids(temp_tasks_dir, temp_repo_root)
        assert len(done_tasks) == 2

    def test_empty_dir(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        done_tasks = _get_done_task_ids(empty_dir, tmp_path)
        assert len(done_tasks) == 0


class TestGetRecordedTaskIds:
    """Test _get_recorded_task_ids function."""

    def test_tasks_link_in_record(self, temp_completion_record: Path) -> None:
        content = """# Completion Record

## 2026-05-01 Test Entry

| 項目 | 状態 | 正本 |
|---|---|---|
| Test | 完了 | docs/tasks/task-test-001.md |
"""
        temp_completion_record.write_text(content, encoding="utf-8")
        recorded = _get_recorded_task_ids(temp_completion_record)
        assert "test-001" in recorded

    def test_empty_record(self, temp_completion_record: Path) -> None:
        recorded = _get_recorded_task_ids(temp_completion_record)
        assert len(recorded) == 0


class TestCheckTaskCompletionPropagation:
    """Test check_task_completion_propagation function."""

    def test_propagated_task(
        self,
        temp_tasks_dir: Path,
        temp_completion_record: Path,
        temp_repo_root: Path,
    ) -> None:
        write_task_seed(temp_tasks_dir / "task-done-001.md", "001", "done")
        content = """## 2026-05-01 Test

| Test | 完了 | docs/tasks/task-done-001.md |
"""
        temp_completion_record.write_text(content, encoding="utf-8")
        nudges, warnings = check_task_completion_propagation(
            tasks_dir=temp_tasks_dir,
            completion_path=temp_completion_record,
            repo_root=temp_repo_root,
        )
        assert len(nudges) == 0

    def test_unpropagated_task(
        self,
        temp_tasks_dir: Path,
        temp_completion_record: Path,
        temp_repo_root: Path,
    ) -> None:
        write_task_seed(temp_tasks_dir / "task-done-001.md", "001", "done", "Unpropagated Task")
        nudges, warnings = check_task_completion_propagation(
            tasks_dir=temp_tasks_dir,
            completion_path=temp_completion_record,
            repo_root=temp_repo_root,
        )
        assert len(nudges) == 1
        assert nudges[0]["target_kind"] == "task_seed"
        assert "completion-record" in nudges[0]["suggested_action"]

    def test_pending_task_not_checked(
        self,
        temp_tasks_dir: Path,
        temp_completion_record: Path,
        temp_repo_root: Path,
    ) -> None:
        write_task_seed(temp_tasks_dir / "task-pending-001.md", "001", "pending")
        nudges, warnings = check_task_completion_propagation(
            tasks_dir=temp_tasks_dir,
            completion_path=temp_completion_record,
            repo_root=temp_repo_root,
        )
        assert len(nudges) == 0


class TestNudgeSchemaCompliance:
    """Test generated nudges comply with PeriodicNudge schema."""

    def test_required_fields(
        self,
        temp_tasks_dir: Path,
        temp_completion_record: Path,
        temp_repo_root: Path,
    ) -> None:
        write_task_seed(temp_tasks_dir / "task-done-001.md", "001", "done")
        nudges, _ = check_task_completion_propagation(
            tasks_dir=temp_tasks_dir,
            completion_path=temp_completion_record,
            repo_root=temp_repo_root,
        )
        assert len(nudges) > 0

        nudge = nudges[0]
        required_fields = [
            "nudge_id",
            "reason",
            "target_kind",
            "target_ref",
            "suggested_action",
            "created_at",
        ]
        for field in required_fields:
            assert field in nudge

    def test_blocking_is_false(
        self,
        temp_tasks_dir: Path,
        temp_completion_record: Path,
        temp_repo_root: Path,
    ) -> None:
        write_task_seed(temp_tasks_dir / "task-done-001.md", "001", "done")
        nudges, _ = check_task_completion_propagation(
            tasks_dir=temp_tasks_dir,
            completion_path=temp_completion_record,
            repo_root=temp_repo_root,
        )
        assert nudges[0]["blocking"] is False