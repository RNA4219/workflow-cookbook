from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence, Tuple

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "ci" / "check_front_matter.py"
spec = importlib.util.spec_from_file_location("check_front_matter", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to load check_front_matter module")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

REQUIRED_FIELDS = tuple(module.REQUIRED_FIELDS)
INCIDENT_REQUIRED_FIELDS = tuple(module.INCIDENT_REQUIRED_FIELDS)
validate_markdown_front_matter = module.validate_markdown_front_matter
validate_incident_front_matter = module.validate_incident_front_matter
validate_task_seed_front_matter = module.validate_task_seed_front_matter
TASK_REQUIRED_FIELDS = tuple(module.TASK_REQUIRED_FIELDS)

FieldPairs = Sequence[Tuple[str, str]]


def _write_markdown(target: Path, pairs: FieldPairs, body: str = "Body") -> None:
    front_matter = ["---", *[f"{key}: {value}" for key, value in pairs], "---", body, ""]
    target.write_text("\n".join(front_matter), encoding="utf-8")


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    return tmp_path


def test_validate_markdown_front_matter_pass(repo_root: Path) -> None:
    _write_markdown(
        repo_root / "README.md",
        (
            ("intent_id", "INT-123"),
            ("owner", "sample-owner"),
            ("status", "active"),
            ("last_reviewed_at", "2024-01-01"),
            ("next_review_due", "2024-02-01"),
        ),
    )

    assert validate_markdown_front_matter(repo_root) == {}


def test_validate_markdown_front_matter_missing_fields(repo_root: Path) -> None:
    _write_markdown(
        repo_root / "README.md",
        (
            ("intent_id", "INT-999"),
            ("owner", "reviewer"),
            ("status", "active"),
            ("last_reviewed_at", "2024-03-10"),
            ("next_review_due", "2024-04-10"),
        ),
    )

    _write_markdown(
        repo_root / "CHANGELOG.md",
        (
            ("intent_id", "INT-777"),
            ("status", "draft"),
            ("last_reviewed_at", "2024-01-01"),
        ),
    )

    _write_markdown(
        repo_root / "GUIDE.md",
        (
            ("intent_id", "INT-888"),
            ("owner", "maintainer"),
            ("status", "deprecated"),
            ("last_reviewed_at", "2024-02-01"),
            ("next_review_due", "2024-03-01"),
        ),
    )

    missing = validate_markdown_front_matter(repo_root)

    assert missing == {
        repo_root / "CHANGELOG.md": [field for field in REQUIRED_FIELDS if field in {"owner", "next_review_due"}]
    }


def test_validate_incident_front_matter_pass(repo_root: Path) -> None:
    docs_dir = repo_root / "docs"
    docs_dir.mkdir()
    _write_markdown(
        docs_dir / "IN-20250115-001.md",
        (
            ("incident_id", "IN-20250115-001"),
            ("occurred_at", "2025-01-15T04:12:00+09:00"),
            ("owner", "oncall@example"),
            ("status", "resolved"),
            ("linked_pr", "https://example.com/pr/123"),
            ("runbook", "../RUNBOOK.md"),
        ),
    )

    assert validate_incident_front_matter(repo_root) == {}


def test_validate_incident_front_matter_missing_fields(repo_root: Path) -> None:
    docs_dir = repo_root / "docs"
    docs_dir.mkdir()
    _write_markdown(
        docs_dir / "IN-20250115-002.md",
        (
            ("incident_id", "IN-20250115-002"),
            ("occurred_at", "2025-01-15T05:30:00+09:00"),
            ("owner", "oncall@example"),
            ("status", "resolved"),
            ("linked_pr", "https://example.com/pr/456"),
        ),
    )

    missing = validate_incident_front_matter(repo_root)

    assert missing == {
        docs_dir / "IN-20250115-002.md": [field for field in INCIDENT_REQUIRED_FIELDS if field == "runbook"]
    }


def test_validate_task_seed_front_matter_pass(repo_root: Path) -> None:
    tasks_dir = repo_root / "docs" / "tasks"
    tasks_dir.mkdir(parents=True)
    _write_markdown(
        tasks_dir / "task-sample.md",
        (
            ("task_id", "20260410-01"),
            ("intent_id", "INT-001"),
            ("owner", "docs-core"),
            ("status", "active"),
        ),
    )

    assert validate_task_seed_front_matter(repo_root) == {}


def test_validate_task_seed_front_matter_missing_fields(repo_root: Path) -> None:
    tasks_dir = repo_root / "docs" / "tasks"
    tasks_dir.mkdir(parents=True)
    _write_markdown(
        tasks_dir / "task-sample.md",
        (
            ("intent_id", "INT-001"),
            ("owner", "docs-core"),
        ),
    )

    missing = validate_task_seed_front_matter(repo_root)

    assert missing == {
        tasks_dir / "task-sample.md": [field for field in TASK_REQUIRED_FIELDS if field in {"task_id", "status"}]
    }
