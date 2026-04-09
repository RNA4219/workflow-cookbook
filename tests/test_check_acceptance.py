from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "ci" / "check_acceptance.py"
spec = importlib.util.spec_from_file_location("check_acceptance", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to load check_acceptance module")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

REQUIRED_FIELDS = tuple(module.REQUIRED_FIELDS)
REQUIRED_HEADINGS = tuple(module.REQUIRED_HEADINGS)
validate_acceptance_docs = module.validate_acceptance_docs


def _write_acceptance(target: Path, *, include_all: bool = True) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        "acceptance_id: AC-20260410-01",
        "task_id: 20260410-01",
        "intent_id: INT-001",
        "owner: reviewer",
        "status: approved",
        "reviewed_at: 2026-04-10",
        "reviewed_by: reviewer",
        "---",
        "# Acceptance Record",
        "",
        "## Scope",
        "",
        "- in",
        "",
        "## Acceptance Criteria",
        "",
        "- [x] ok",
        "",
        "## Evidence",
        "",
        "- pytest",
        "",
        "## Verification Result",
        "",
        "- approved",
        "",
    ]
    if not include_all:
        lines.remove("reviewed_by: reviewer")
        lines.remove("## Evidence")
    target.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    return tmp_path


def test_validate_acceptance_docs_pass(repo_root: Path) -> None:
    docs_dir = repo_root / "docs" / "acceptance"
    docs_dir.mkdir(parents=True)
    (docs_dir / "README.md").write_text("# Acceptance\n", encoding="utf-8")
    (docs_dir / "ACCEPTANCE_TEMPLATE.md").write_text("# Template\n", encoding="utf-8")
    (docs_dir / "INDEX.md").write_text("# Acceptance Index\n", encoding="utf-8")
    _write_acceptance(docs_dir / "AC-20260410-01.md")

    assert validate_acceptance_docs(repo_root) == {}


def test_validate_acceptance_docs_missing_fields_and_headings(repo_root: Path) -> None:
    docs_dir = repo_root / "docs" / "acceptance"
    docs_dir.mkdir(parents=True)
    _write_acceptance(docs_dir / "AC-20260410-02.md", include_all=False)

    missing = validate_acceptance_docs(repo_root)

    assert missing == {
        docs_dir / "AC-20260410-02.md": [
            "front matter:reviewed_by",
            "heading:## Evidence",
        ]
    }


def test_validate_acceptance_docs_missing_directory(repo_root: Path) -> None:
    missing = validate_acceptance_docs(repo_root)

    assert missing == {
        repo_root / "docs" / "acceptance": ["docs/acceptance directory is missing"]
    }
