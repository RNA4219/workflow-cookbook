"""Tests for tools.ci.extract_upstream_changes."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.ci.extract_upstream_changes import (
    _parse_upstream_md,
    _parse_weekly_log,
    extract_new_changes,
    generate_weekly_update,
    UpstreamChange,
)


class TestParseUpstreamMd:
    def test_parses_change_entries(self) -> None:
        content = dedent("""
            ## 2026-04-10 - Test Source

            Description of the change.

            **Impact:** low

            ## 2026-04-09 - Another Source

            Another description.
            """).strip()

        changes = _parse_upstream_md(content)
        assert len(changes) == 2
        assert changes[0].date == "2026-04-10"
        assert changes[0].source == "Test Source"
        assert changes[0].impact == "low"

    def test_returns_empty_for_no_changes(self) -> None:
        content = "# No changes"
        changes = _parse_upstream_md(content)
        assert changes == []

    def test_extracts_impact_from_description(self) -> None:
        content = dedent("""
            ## 2026-04-10 - Test Source

            Some description.

            **Impact:** critical

            More text.
            """).strip()

        changes = _parse_upstream_md(content)
        assert len(changes) == 1
        assert changes[0].impact == "critical"


class TestParseWeeklyLog:
    def test_parses_table_entries(self) -> None:
        content = dedent("""
            | Source | Date | Notes |
            |---|---|---|
            | Provider A | 2026-04-10 | Reviewed |
            | Provider B | 2026-04-09 | Pending |
            """).strip()

        reviewed = _parse_weekly_log(content)
        assert reviewed["Provider A"] == "2026-04-10"
        assert reviewed["Provider B"] == "2026-04-09"

    def test_returns_empty_for_no_table(self) -> None:
        content = "No table"
        reviewed = _parse_weekly_log(content)
        assert reviewed == {}


class TestExtractNewChanges:
    def test_extracts_newer_changes(self) -> None:
        changes = [
            UpstreamChange(date="2026-04-10", source="Source A", description="New", impact="low"),
            UpstreamChange(date="2026-04-08", source="Source A", description="Old", impact="low"),
        ]
        reviewed = {"Source A": "2026-04-09"}

        new_changes = extract_new_changes(changes, reviewed)
        assert len(new_changes) == 1
        assert new_changes[0].date == "2026-04-10"

    def test_returns_all_for_unreviewed_source(self) -> None:
        changes = [
            UpstreamChange(date="2026-04-10", source="Source A", description="New", impact="low"),
        ]
        reviewed: dict = {}

        new_changes = extract_new_changes(changes, reviewed)
        assert len(new_changes) == 1


class TestGenerateWeeklyUpdate:
    def test_generates_update_with_changes(self) -> None:
        changes = [
            UpstreamChange(
                date="2026-04-10",
                source="Source A",
                description="Test change",
                impact="medium",
            ),
        ]

        update = generate_weekly_update(changes)
        assert "# Upstream Weekly Log" in update
        assert "2026-04-10" in update
        assert "Source A" in update
        assert "Test change" in update
        assert "**Impact:** medium" in update

    def test_generates_update_without_changes(self) -> None:
        update = generate_weekly_update([])
        assert "No new upstream changes detected" in update