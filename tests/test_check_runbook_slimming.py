"""Tests for check_runbook_slimming.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "ci" / "check_runbook_slimming.py"
spec = importlib.util.spec_from_file_location("check_runbook_slimming", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to load check_runbook_slimming module")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

check_runbook_slimming = module.check_runbook_slimming
MAX_COMPLETED_TABLE_ROWS = module.MAX_COMPLETED_TABLE_ROWS


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    return tmp_path


def _write_runbook(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _write_completion_record(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_pass_no_completed_tables(repo_root: Path) -> None:
    """RUNBOOK without completed tables should pass."""
    runbook_content = """
---
intent_id: INT-001
---

# Runbook

## Execute

- Prepare and run commands
- Verify output

## Rollback

Steps to rollback.

| Step | Action |
|---|---|
| 1 | Check version |
| 2 | Restore backup |
"""
    _write_runbook(repo_root / "RUNBOOK.md", runbook_content)

    warnings = check_runbook_slimming(repo_root)
    assert warnings == {}


def test_pass_short_completed_reference(repo_root: Path) -> None:
    """RUNBOOK with short completed reference should pass."""
    runbook_content = """
---
intent_id: INT-001
---

# Runbook

## Execute

Current operations.

## Completed

| Task | Status | Link |
|---|---|---|
| Feature A | 完了 | [docs/tasks/task-feature-a.md](docs/tasks/task-feature-a.md) |

See [docs/completion-record.md](docs/completion-record.md) for details.
"""
    _write_runbook(repo_root / "RUNBOOK.md", runbook_content)

    warnings = check_runbook_slimming(repo_root)
    assert warnings == {}


def test_warning_large_completed_table(repo_root: Path) -> None:
    """Large completed table should trigger warning."""
    rows = []
    for i in range(MAX_COMPLETED_TABLE_ROWS + 5):
        rows.append(f"| Task {i} | 完了 | Description {i} |")

    runbook_content = f"""
---
intent_id: INT-001
---

# Runbook

## Completed

| Task | Status | Description |
|---|---|---|
{chr(10).join(rows)}
"""
    _write_runbook(repo_root / "RUNBOOK.md", runbook_content)

    warnings = check_runbook_slimming(repo_root)
    assert len(warnings) == 1
    runbook_path = repo_root / "RUNBOOK.md"
    assert runbook_path in warnings
    assert any("Large completed table" in w for w in warnings[runbook_path])


def test_warning_missing_canonical_link(repo_root: Path) -> None:
    """Completed table without canonical links should trigger warning."""
    runbook_content = """
---
intent_id: INT-001
---

# Runbook

## Completed

| Task | Status | Description |
|---|---|---|
| Feature A | 完了 | Implemented feature A |
| Feature B | 完了 | Implemented feature B |
| Feature C | 完了 | Implemented feature C |
"""
    _write_runbook(repo_root / "RUNBOOK.md", runbook_content)

    warnings = check_runbook_slimming(repo_root)
    assert len(warnings) == 1
    runbook_path = repo_root / "RUNBOOK.md"
    assert runbook_path in warnings
    assert any("lacks canonical links" in w for w in warnings[runbook_path])


def test_warning_duplicate_completion_detail(repo_root: Path) -> None:
    """Duplicate details in RUNBOOK and completion-record should trigger warning."""
    runbook_content = """
---
intent_id: INT-001
---

# Runbook

## Completed

| Task | Status | Link |
|---|---|---|
| Feature A | 完了 | [docs/tasks/task-a.md](docs/tasks/task-a.md) |
| Feature B | 完了 | [docs/tasks/task-b.md](docs/tasks/task-b.md) |
| Feature C | 完了 | [docs/tasks/task-c.md](docs/tasks/task-c.md) |
"""
    completion_content = """
# Completion Record

## 2026-04-10 Feature Implementation

| Task | Status | Link |
|---|---|---|
| Feature A | 完了 | [docs/tasks/task-a.md](docs/tasks/task-a.md) |
| Feature B | 完了 | [docs/tasks/task-b.md](docs/tasks/task-b.md) |
| Feature C | 完了 | [docs/tasks/task-c.md](docs/tasks/task-c.md) |

判定: go
"""
    _write_runbook(repo_root / "RUNBOOK.md", runbook_content)
    _write_completion_record(repo_root / "docs" / "completion-record.md", completion_content)

    warnings = check_runbook_slimming(repo_root)
    assert len(warnings) == 1
    runbook_path = repo_root / "RUNBOOK.md"
    assert runbook_path in warnings
    assert any("Duplicate" in w for w in warnings[runbook_path])


def test_pass_current_ops_exception(repo_root: Path) -> None:
    """Current operational tables should pass (exception)."""
    runbook_content = """
---
intent_id: INT-001
---

# Runbook

## Operational Readiness Backlog

| Task | Status | Link |
|---|---|---|
| Branch Protection | active | [docs/tasks/task-branch.md](docs/tasks/task-branch.md) |
| Release Drill | planned | [docs/tasks/task-drill.md](docs/tasks/task-drill.md) |
| Supply Chain | planned | [docs/tasks/task-supply.md](docs/tasks/task-supply.md) |
| Dependency Review | 完了 | [docs/tasks/task-deps.md](docs/tasks/task-deps.md) |

## Rollback

| Step | Action |
|---|---|---|
| 1 | Check version |
| 2 | Restore backup |
"""
    _write_runbook(repo_root / "RUNBOOK.md", runbook_content)

    warnings = check_runbook_slimming(repo_root)
    assert warnings == {}


def test_pass_with_completion_record_link(repo_root: Path) -> None:
    """RUNBOOK linking to completion-record should pass."""
    runbook_content = """
---
intent_id: INT-001
---

# Runbook

## Execute

Current operations.

Completed tasks are tracked in [docs/completion-record.md](docs/completion-record.md).
"""
    _write_runbook(repo_root / "RUNBOOK.md", runbook_content)

    warnings = check_runbook_slimming(repo_root)
    assert warnings == {}


def test_missing_runbook(repo_root: Path) -> None:
    """Missing RUNBOOK should return error."""
    warnings = check_runbook_slimming(repo_root)
    assert len(warnings) == 1
    runbook_path = repo_root / "RUNBOOK.md"
    assert runbook_path in warnings
    assert any("not found" in w for w in warnings[runbook_path])


def test_edge_case_empty_file(repo_root: Path) -> None:
    """Empty RUNBOOK should pass (no content to check)."""
    runbook_path = repo_root / "RUNBOOK.md"
    runbook_path.parent.mkdir(parents=True, exist_ok=True)
    runbook_path.write_text("", encoding="utf-8")

    warnings = check_runbook_slimming(repo_root)
    assert warnings == {}


def test_edge_case_malformed_table(repo_root: Path) -> None:
    """Malformed table (incomplete rows) should be handled gracefully."""
    runbook_content = """
---
intent_id: INT-001
---

# Runbook

## Completed

| Task | Status
|---|
| Feature A | 完了
| Feature B | 完了 |
"""
    _write_runbook(repo_root / "RUNBOOK.md", runbook_content)

    # Should not crash, may or may not warn depending on detection
    warnings = check_runbook_slimming(repo_root)
    assert isinstance(warnings, dict)


def test_edge_case_table_in_code_block(repo_root: Path) -> None:
    """Tables inside code blocks should not be detected as real tables."""
    runbook_content = """
---
intent_id: INT-001
---

# Runbook

## Example

```markdown
| Task | Status |
|---|---|
| Example | 完了 |
| Sample | 完了 |
```

This is just an example, not real completed items.
"""
    _write_runbook(repo_root / "RUNBOOK.md", runbook_content)

    warnings = check_runbook_slimming(repo_root)
    # Code block tables should not trigger warnings
    assert warnings == {}