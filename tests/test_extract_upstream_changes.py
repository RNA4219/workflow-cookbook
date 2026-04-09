"""Tests for tools.ci.extract_upstream_changes main function."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools.ci.extract_upstream_changes as eu_module


class TestMain:
    def test_main_prints_no_changes(self, tmp_path: Path, capsys) -> None:
        upstream_md = tmp_path / "UPSTREAM.md"
        upstream_md.write_text("No changes")

        weekly_log = tmp_path / "WEEKLY.md"
        weekly_log.write_text("No table")

        result = eu_module.main([
            "--upstream-md", str(upstream_md),
            "--weekly-log", str(weekly_log),
        ])

        assert result == 0
        captured = capsys.readouterr()
        assert "No new upstream changes" in captured.out

    def test_main_prints_new_changes(self, tmp_path: Path, capsys) -> None:
        upstream_md = tmp_path / "UPSTREAM.md"
        upstream_md.write_text(dedent("""
            ## 2026-04-10 - Source A

            Description

            **Impact:** low
            """).strip())

        weekly_log = tmp_path / "WEEKLY.md"
        weekly_log.write_text("| Source | Date | Notes |\n|---|---|---|\n")

        result = eu_module.main([
            "--upstream-md", str(upstream_md),
            "--weekly-log", str(weekly_log),
            "--print-changes",
        ])

        assert result == 0
        captured = capsys.readouterr()
        assert "Source A" in captured.out

    def test_main_updates_log(self, tmp_path: Path) -> None:
        upstream_md = tmp_path / "UPSTREAM.md"
        upstream_md.write_text(dedent("""
            ## 2026-04-10 - Source A

            Description

            **Impact:** low
            """).strip())

        weekly_log = tmp_path / "WEEKLY.md"
        weekly_log.write_text("| Source | Date | Notes |\n|---|---|---|\n")

        result = eu_module.main([
            "--upstream-md", str(upstream_md),
            "--weekly-log", str(weekly_log),
            "--update-log",
        ])

        assert result == 0
        content = weekly_log.read_text()
        assert "2026-04-10" in content