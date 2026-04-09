"""Tests for tools.ci.generate_acceptance_index_standalone."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.ci.generate_acceptance_index_standalone import (
    _parse_front_matter,
    scan_acceptances,
    AcceptanceInfo,
)


class TestParseFrontMatter:
    def test_parses_yaml_front_matter(self) -> None:
        content = dedent("""
            ---
            acceptance_id: AC-001
            task_id: TASK-001
            status: approved
            reviewed_at: 2026-04-10
            ---
            # Content
            """).strip()
        result = _parse_front_matter(content)
        assert result.get("acceptance_id") == "AC-001"
        assert result.get("task_id") == "TASK-001"
        assert result.get("status") == "approved"

    def test_returns_empty_for_no_front_matter(self) -> None:
        content = "# No front matter"
        result = _parse_front_matter(content)
        assert result == {}

    def test_handles_quoted_values(self) -> None:
        content = dedent("""
            ---
            acceptance_id: "AC-001"
            status: 'approved'
            ---
            """).strip()
        result = _parse_front_matter(content)
        assert result.get("acceptance_id") == "AC-001"
        assert result.get("status") == "approved"


class TestScanAcceptances:
    def test_scans_acceptance_files(self, tmp_path: Path) -> None:
        acc_dir = tmp_path / "acceptance"
        acc_dir.mkdir()
        acc_file = acc_dir / "AC-20260410-01.md"
        acc_file.write_text(dedent("""
            ---
            acceptance_id: AC-20260410-01
            task_id: TASK-001
            intent_id: INT-001
            status: approved
            reviewed_at: 2026-04-10
            ---
            # Acceptance
            """).strip())

        acceptances = scan_acceptances(acc_dir)
        assert len(acceptances) == 1
        assert acceptances[0].acceptance_id == "AC-20260410-01"
        assert acceptances[0].task_id == "TASK-001"
        assert acceptances[0].status == "approved"

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        acceptances = scan_acceptances(tmp_path / "nonexistent")
        assert acceptances == []

    def test_skips_files_without_acceptance_id(self, tmp_path: Path) -> None:
        acc_dir = tmp_path / "acceptance"
        acc_dir.mkdir()
        acc_file = acc_dir / "AC-invalid.md"
        acc_file.write_text(dedent("""
            ---
            task_id: TASK-001
            ---
            # No acceptance_id
            """).strip())

        acceptances = scan_acceptances(acc_dir)
        assert acceptances == []

    def test_uses_reviewed_by_as_fallback(self, tmp_path: Path) -> None:
        acc_dir = tmp_path / "acceptance"
        acc_dir.mkdir()
        acc_file = acc_dir / "AC-001.md"
        acc_file.write_text(dedent("""
            ---
            acceptance_id: AC-001
            reviewed_by: 2026-04-10
            ---
            """).strip())

        acceptances = scan_acceptances(acc_dir)
        assert len(acceptances) == 1
        assert acceptances[0].reviewed_at == "2026-04-10"


class TestGenerateIndexMarkdown:
    def test_generates_summary_table(self, tmp_path: Path) -> None:
        # Create actual acceptance files to avoid relative_to error
        acc_dir = tmp_path / "docs" / "acceptance"
        acc_dir.mkdir(parents=True)
        acc_file1 = acc_dir / "AC-001.md"
        acc_file2 = acc_dir / "AC-002.md"
        acc_file1.write_text("---\nacceptance_id: AC-001\n---")
        acc_file2.write_text("---\nacceptance_id: AC-002\n---")

        acceptances = [
            AcceptanceInfo(
                acceptance_id="AC-001",
                task_id="TASK-001",
                intent_id="INT-001",
                status="approved",
                reviewed_at="2026-04-10",
                file_path=acc_file1,
            ),
            AcceptanceInfo(
                acceptance_id="AC-002",
                task_id="TASK-002",
                intent_id="INT-001",
                status="rejected",
                reviewed_at="2026-04-09",
                file_path=acc_file2,
            ),
        ]

        # Import with patched _REPO_ROOT
        import importlib
        import tools.ci.generate_acceptance_index_standalone as gen_module
        original_repo_root = gen_module._REPO_ROOT
        gen_module._REPO_ROOT = tmp_path

        markdown = gen_module.generate_index_markdown(acceptances)

        gen_module._REPO_ROOT = original_repo_root

        assert "# Acceptance Index" in markdown
        assert "## Summary" in markdown
        assert "approved" in markdown
        assert "rejected" in markdown
        assert "## Records" in markdown
        assert "AC-001" in markdown
        assert "AC-002" in markdown

    def test_handles_empty_acceptances(self) -> None:
        import importlib
        import tools.ci.generate_acceptance_index_standalone as gen_module
        markdown = gen_module.generate_index_markdown([])
        assert "# Acceptance Index" in markdown
        assert "## Summary" in markdown
        assert "**Total** | **0**" in markdown

    def test_sorts_by_acceptance_id_descending(self, tmp_path: Path) -> None:
        acc_dir = tmp_path / "docs" / "acceptance"
        acc_dir.mkdir(parents=True)
        acc_file1 = acc_dir / "AC-001.md"
        acc_file2 = acc_dir / "AC-002.md"
        acc_file1.write_text("---\n---")
        acc_file2.write_text("---\n---")

        acceptances = [
            AcceptanceInfo(
                acceptance_id="AC-001",
                task_id="TASK-001",
                intent_id="INT-001",
                status="approved",
                reviewed_at="2026-04-10",
                file_path=acc_file1,
            ),
            AcceptanceInfo(
                acceptance_id="AC-002",
                task_id="TASK-002",
                intent_id="INT-001",
                status="approved",
                reviewed_at="2026-04-10",
                file_path=acc_file2,
            ),
        ]

        import tools.ci.generate_acceptance_index_standalone as gen_module
        original_repo_root = gen_module._REPO_ROOT
        gen_module._REPO_ROOT = tmp_path

        markdown = gen_module.generate_index_markdown(acceptances)

        gen_module._REPO_ROOT = original_repo_root

        # AC-002 should appear first (sorted descending)
        assert markdown.find("AC-002") < markdown.find("AC-001")