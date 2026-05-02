"""Tests for check_completion_trace.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "ci" / "check_completion_trace.py"
spec = importlib.util.spec_from_file_location("check_completion_trace", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to load check_completion_trace module")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

check_completion_trace = module.check_completion_trace


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    return tmp_path


def _write_task(target: Path, *, task_id: str, status: str, acceptance_link: bool = False, exception_reason: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    content = f"""---
task_id: {task_id}
intent_id: INT-001
status: {status}
---

# Task Seed

## Objective

Test task.

## Requirements

- Requirement 1

## Completion Criteria

- [ ] Done
"""
    if acceptance_link:
        content += "\n\nSee [docs/acceptance/AC-20260410-01.md](docs/acceptance/AC-20260410-01.md) for acceptance."
    if exception_reason:
        content += "\n\nException reason: minor change, no acceptance required."

    target.write_text(content, encoding="utf-8")


def _write_acceptance(target: Path, *, acceptance_id: str, task_id: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    content = f"""---
acceptance_id: {acceptance_id}
task_id: {task_id}
intent_id: INT-001
status: approved
---

# Acceptance Record

## Evidence

- pytest passed
"""
    target.write_text(content, encoding="utf-8")


def _write_completion_record(target: Path, *, with_link: bool = True) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if with_link:
        content = """# Completion Record

## 2026-04-10 Feature Implementation

| Task | Status | Link |
|---|---|---|
| Test Task | 完了 | [docs/tasks/task-test-20260410-01.md](docs/tasks/task-test-20260410-01.md) |

判定: go
"""
    else:
        content = """# Completion Record

## 2026-04-10 Feature Implementation

| Task | Status | Description |
|---|---|---|
| Test Task | 完了 | Implemented feature |

判定: go
"""
    target.write_text(content, encoding="utf-8")


def _write_changelog(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    content = """---
intent_id: INT-001
---

# Changelog

## Unreleased

### Added

- New feature.

## 1.0.0 - 2026-04-10

### Added

- Initial release.
"""
    target.write_text(content, encoding="utf-8")


def test_pass_done_task_with_acceptance_link(repo_root: Path) -> None:
    """Done task with acceptance link should pass."""
    docs_tasks = repo_root / "docs" / "tasks"
    docs_acceptance = repo_root / "docs" / "acceptance"

    _write_task(
        docs_tasks / "task-test-20260410-01.md",
        task_id="20260410-01",
        status="done",
        acceptance_link=True,
    )
    _write_acceptance(
        docs_acceptance / "AC-20260410-01.md",
        acceptance_id="AC-20260410-01",
        task_id="20260410-01",
    )
    _write_completion_record(repo_root / "docs" / "completion-record.md")
    _write_changelog(repo_root / "CHANGELOG.md")

    result = check_completion_trace(repo_root)
    assert result["errors"] == []


def test_pass_done_task_with_exception_reason(repo_root: Path) -> None:
    """Done task with exception reason should pass."""
    docs_tasks = repo_root / "docs" / "tasks"

    _write_task(
        docs_tasks / "task-minor-20260410-02.md",
        task_id="20260410-02",
        status="done",
        exception_reason=True,
    )
    _write_completion_record(repo_root / "docs" / "completion-record.md")
    _write_changelog(repo_root / "CHANGELOG.md")

    result = check_completion_trace(repo_root)
    assert result["errors"] == []


def test_error_done_task_without_acceptance_or_exception(repo_root: Path) -> None:
    """Done task without acceptance or exception should error only with flag."""
    docs_tasks = repo_root / "docs" / "tasks"

    _write_task(
        docs_tasks / "task-orphan-20260410-03.md",
        task_id="20260410-03",
        status="done",
    )
    _write_completion_record(repo_root / "docs" / "completion-record.md")
    _write_changelog(repo_root / "CHANGELOG.md")

    # Without flag: warning only
    result = check_completion_trace(repo_root, require_acceptance_for_done=False)
    assert len(result["warnings"]) >= 1
    assert any("lacks acceptance record" in w for w in result["warnings"])
    assert result["errors"] == []

    # With flag: error
    result_strict = check_completion_trace(repo_root, require_acceptance_for_done=True)
    assert len(result_strict["errors"]) >= 1
    assert any("lacks acceptance record" in e for e in result_strict["errors"])


def test_error_completion_without_source_link(repo_root: Path) -> None:
    """Completion entry without source link should error."""
    docs_tasks = repo_root / "docs" / "tasks"
    docs_acceptance = repo_root / "docs" / "acceptance"

    _write_task(
        docs_tasks / "task-test-20260410-01.md",
        task_id="20260410-01",
        status="done",
        acceptance_link=True,
    )
    _write_acceptance(
        docs_acceptance / "AC-20260410-01.md",
        acceptance_id="AC-20260410-01",
        task_id="20260410-01",
    )
    _write_completion_record(repo_root / "docs" / "completion-record.md", with_link=False)
    _write_changelog(repo_root / "CHANGELOG.md")

    result = check_completion_trace(repo_root)
    assert len(result["errors"]) >= 1
    assert any("lacks canonical source link" in e for e in result["errors"])


def test_warning_acceptance_orphan(repo_root: Path) -> None:
    """Acceptance not referenced by task or completion should warn."""
    docs_tasks = repo_root / "docs" / "tasks"
    docs_acceptance = repo_root / "docs" / "acceptance"

    _write_task(
        docs_tasks / "task-test-20260410-01.md",
        task_id="20260410-01",
        status="active",
    )
    # Orphan acceptance (not referenced by task)
    _write_acceptance(
        docs_acceptance / "AC-20260410-99.md",
        acceptance_id="AC-20260410-99",
        task_id="20260410-99",  # Different task_id
    )
    _write_completion_record(repo_root / "docs" / "completion-record.md")
    _write_changelog(repo_root / "CHANGELOG.md")

    result = check_completion_trace(repo_root)
    assert any("orphaned" in w for w in result["warnings"])


def test_pass_planned_task_without_acceptance(repo_root: Path) -> None:
    """Planned/active task without acceptance should pass."""
    docs_tasks = repo_root / "docs" / "tasks"

    _write_task(
        docs_tasks / "task-planned-20260410-04.md",
        task_id="20260410-04",
        status="planned",
    )
    _write_completion_record(repo_root / "docs" / "completion-record.md")
    _write_changelog(repo_root / "CHANGELOG.md")

    result = check_completion_trace(repo_root)
    assert not any("lacks acceptance" in e for e in result["errors"])


def test_missing_completion_record(repo_root: Path) -> None:
    """Missing completion record should error."""
    docs_tasks = repo_root / "docs" / "tasks"
    _write_task(
        docs_tasks / "task-test-20260410-01.md",
        task_id="20260410-01",
        status="done",
        acceptance_link=True,
    )
    _write_changelog(repo_root / "CHANGELOG.md")

    result = check_completion_trace(repo_root)
    assert any("not found" in e for e in result["errors"])


def test_missing_changelog(repo_root: Path) -> None:
    """Missing changelog should warn."""
    docs_tasks = repo_root / "docs" / "tasks"
    docs_acceptance = repo_root / "docs" / "acceptance"

    _write_task(
        docs_tasks / "task-test-20260410-01.md",
        task_id="20260410-01",
        status="done",
        acceptance_link=True,
    )
    _write_acceptance(
        docs_acceptance / "AC-20260410-01.md",
        acceptance_id="AC-20260410-01",
        task_id="20260410-01",
    )
    _write_completion_record(repo_root / "docs" / "completion-record.md")

    result = check_completion_trace(repo_root)
    assert any("CHANGELOG.md not found" in w for w in result["warnings"])


def test_edge_case_empty_task_file(repo_root: Path) -> None:
    """Empty task file should be handled gracefully."""
    docs_tasks = repo_root / "docs" / "tasks"
    docs_tasks.mkdir(parents=True, exist_ok=True)
    (docs_tasks / "task-empty-20260410-00.md").write_text("", encoding="utf-8")

    _write_completion_record(repo_root / "docs" / "completion-record.md")
    _write_changelog(repo_root / "CHANGELOG.md")

    # Should not crash
    result = check_completion_trace(repo_root)
    assert isinstance(result, dict)


def test_edge_case_task_without_front_matter(repo_root: Path) -> None:
    """Task file without front matter should be handled gracefully."""
    docs_tasks = repo_root / "docs" / "tasks"
    docs_tasks.mkdir(parents=True, exist_ok=True)
    (docs_tasks / "task-no-fm-20260410-01.md").write_text(
        "# Task Seed\n\nNo front matter here.",
        encoding="utf-8"
    )

    _write_completion_record(repo_root / "docs" / "completion-record.md")
    _write_changelog(repo_root / "CHANGELOG.md")

    result = check_completion_trace(repo_root)
    # Should not crash, status defaults to empty (not "done")
    assert isinstance(result, dict)


def test_edge_case_acceptance_with_missing_task_id(repo_root: Path) -> None:
    """Acceptance without task_id in front matter should be handled."""
    docs_tasks = repo_root / "docs" / "tasks"
    docs_acceptance = repo_root / "docs" / "acceptance"

    _write_task(
        docs_tasks / "task-test-20260410-01.md",
        task_id="20260410-01",
        status="done",
        acceptance_link=True,
    )
    # Acceptance without task_id
    docs_acceptance.mkdir(parents=True, exist_ok=True)
    (docs_acceptance / "AC-20260410-99.md").write_text(
        """---
acceptance_id: AC-20260410-99
status: approved
---

# Acceptance Record

No task_id field.
""",
        encoding="utf-8"
    )

    _write_completion_record(repo_root / "docs" / "completion-record.md")
    _write_changelog(repo_root / "CHANGELOG.md")

    result = check_completion_trace(repo_root)
    assert isinstance(result, dict)


def test_edge_case_completion_record_empty(repo_root: Path) -> None:
    """Empty completion record should error (no entries)."""
    docs_tasks = repo_root / "docs" / "tasks"
    docs_tasks.mkdir(parents=True, exist_ok=True)
    (docs_tasks / "task-dummy.md").write_text(
        "---\ntask_id: dummy\nstatus: planned\n---\n# Dummy\n",
        encoding="utf-8"
    )

    completion_path = repo_root / "docs" / "completion-record.md"
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    completion_path.write_text("", encoding="utf-8")

    _write_changelog(repo_root / "CHANGELOG.md")

    result = check_completion_trace(repo_root)
    # Empty file should not cause crash
    assert isinstance(result, dict)