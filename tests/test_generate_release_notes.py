"""Tests for tools.ci.generate_release_notes."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.ci.generate_release_notes import (
    _parse_changelog,
    scan_releases_dir,
    generate_release_notes,
    main,
    ReleaseInfo,
)


class TestParseChangelog:
    def test_parses_version_entries(self) -> None:
        content = dedent("""
            ## v1.0.0 - 2026-04-10
            - First release
            - Added feature X

            ## v0.9.0 - 2026-04-01
            - Beta release
            """).strip()

        releases = _parse_changelog(content)
        assert len(releases) == 2
        assert releases[0].version == "v1.0.0"
        assert releases[0].date == "2026-04-10"
        assert "First release" in releases[0].changes
        assert releases[1].version == "v0.9.0"

    def test_handles_versions_without_dates(self) -> None:
        content = dedent("""
            ## 1.0.0
            - First release
            """).strip()

        releases = _parse_changelog(content)
        assert len(releases) == 1
        assert releases[0].version == "1.0.0"
        assert releases[0].date == ""

    def test_cleans_up_change_numbers(self) -> None:
        content = dedent("""
            ## v1.0.0 - 2026-04-10
            - 0075: Some change with prefix
            - Another change
            """).strip()

        releases = _parse_changelog(content)
        assert "Some change with prefix" in releases[0].changes
        assert "Another change" in releases[0].changes

    def test_returns_empty_for_no_versions(self) -> None:
        content = "# CHANGELOG\n\nNo releases yet"
        releases = _parse_changelog(content)
        assert releases == []


class TestScanReleasesDir:
    def test_scans_release_files(self, tmp_path: Path) -> None:
        releases_dir = tmp_path / "releases"
        releases_dir.mkdir()
        release_file = releases_dir / "v1.0.0.md"
        release_file.write_text(dedent("""
            ---
            date: 2026-04-10
            ---
            # v1.0.0
            - Feature A added
            - Bug fix
            """).strip())

        releases = scan_releases_dir(releases_dir)
        assert len(releases) == 1
        assert releases[0].version == "v1.0.0"
        assert releases[0].date == "2026-04-10"
        assert len(releases[0].changes) == 2

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        releases = scan_releases_dir(tmp_path / "nonexistent")
        assert releases == []

    def test_only_scans_v_prefix_files(self, tmp_path: Path) -> None:
        releases_dir = tmp_path / "releases"
        releases_dir.mkdir()
        (releases_dir / "v1.0.0.md").write_text("# v1.0.0")
        (releases_dir / "release.md").write_text("# Other")

        releases = scan_releases_dir(releases_dir)
        assert len(releases) == 1
        assert releases[0].version == "v1.0.0"


class TestGenerateReleaseNotes:
    def test_generates_release_notes(self) -> None:
        changelog_releases = [
            ReleaseInfo(version="v1.0.0", date="2026-04-10", changes=["Feature A"]),
        ]
        file_releases = []

        notes = generate_release_notes(changelog_releases, file_releases)
        assert "# Release Notes" in notes
        assert "## v1.0.0" in notes
        assert "Released: 2026-04-10" in notes
        assert "- Feature A" in notes

    def test_merges_releases(self) -> None:
        changelog_releases = [
            ReleaseInfo(version="v1.0.0", date="2026-04-10", changes=["Feature A"]),
        ]
        file_releases = [
            ReleaseInfo(
                version="v1.0.0",
                date="",
                changes=["Feature B"],
                file_path=Path("releases/v1.0.0.md"),
            ),
        ]

        notes = generate_release_notes(changelog_releases, file_releases)
        assert "- Feature A" in notes
        assert "- Feature B" in notes

    def test_limits_changes_to_top_10(self) -> None:
        changes = [f"Change {i}" for i in range(15)]
        changelog_releases = [
            ReleaseInfo(version="v1.0.0", date="2026-04-10", changes=changes),
        ]

        notes = generate_release_notes(changelog_releases, [])
        assert "... and 5 more" in notes

    def test_sorts_versions_descending(self) -> None:
        changelog_releases = [
            ReleaseInfo(version="v1.0.0", date="2026-04-10", changes=["A"]),
            ReleaseInfo(version="v2.0.0", date="2026-04-11", changes=["B"]),
        ]

        notes = generate_release_notes(changelog_releases, [])
        # v2.0.0 should appear before v1.0.0
        assert notes.find("v2.0.0") < notes.find("v1.0.0")


class TestMain:
    def test_main_outputs_to_stdout(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("## v1.0.0 - 2026-04-10\n- Test\n")

        result = main([
            "--changelog", str(changelog),
            "--releases-dir", str(tmp_path / "releases"),
        ])

        assert result == 0

    def test_main_outputs_latest(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("## v1.0.0 - 2026-04-10\n- Test\n")

        result = main([
            "--changelog", str(changelog),
            "--releases-dir", str(tmp_path / "releases"),
            "--latest",
        ])

        assert result == 0

    def test_main_writes_to_file(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("## v1.0.0 - 2026-04-10\n- Test\n")
        output = tmp_path / "RELEASE_NOTES.md"

        result = main([
            "--changelog", str(changelog),
            "--releases-dir", str(tmp_path / "releases"),
            "--output", str(output),
        ])

        assert result == 0
        assert output.exists()