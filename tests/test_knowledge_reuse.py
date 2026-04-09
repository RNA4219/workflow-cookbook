"""Tests for tools.ci.knowledge_reuse."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.ci.knowledge_reuse import (
    _parse_front_matter,
    _extract_title,
    search_knowledge,
    KnowledgeEntry,
)


class TestParseFrontMatter:
    def test_parses_yaml_front_matter(self) -> None:
        content = dedent("""
            ---
            date: 2026-04-10
            tags: security, release
            ---
            # Content
            """).strip()
        result = _parse_front_matter(content)
        assert result.get("date") == "2026-04-10"
        assert result.get("tags") == "security, release"

    def test_returns_empty_for_no_front_matter(self) -> None:
        result = _parse_front_matter("# No front matter")
        assert result == {}


class TestExtractTitle:
    def test_extracts_h1_title(self) -> None:
        content = "# Incident Title\n\nContent"
        title = _extract_title(content)
        assert title == "Incident Title"

    def test_returns_untitled_for_no_title(self) -> None:
        title = _extract_title("Content only")
        assert title == "Untitled"


class TestScanReleases:
    def test_scans_release_files(self, tmp_path: Path) -> None:
        # Create releases directory with actual file
        releases_dir = tmp_path / "docs" / "releases"
        releases_dir.mkdir(parents=True)
        (releases_dir / "v1.0.0.md").write_text(dedent("""
            ---
            date: 2026-04-10
            ---
            # Release v1.0.0
            """).strip())

        import tools.ci.knowledge_reuse as kr_module
        original_repo_root = kr_module._REPO_ROOT
        original_releases_dir = kr_module._RELEASES_DIR
        kr_module._REPO_ROOT = tmp_path
        kr_module._RELEASES_DIR = releases_dir

        entries = kr_module.scan_releases()

        kr_module._REPO_ROOT = original_repo_root
        kr_module._RELEASES_DIR = original_releases_dir

        assert len(entries) == 1
        assert entries[0].entry_type == "release"
        assert entries[0].entry_id == "v1.0.0"
        assert entries[0].title == "Release v1.0.0"

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        import tools.ci.knowledge_reuse as kr_module
        original_releases_dir = kr_module._RELEASES_DIR
        kr_module._RELEASES_DIR = tmp_path / "nonexistent"

        entries = kr_module.scan_releases()

        kr_module._RELEASES_DIR = original_releases_dir

        assert entries == []


class TestScanAcceptances:
    def test_scans_acceptance_files(self, tmp_path: Path) -> None:
        acc_dir = tmp_path / "docs" / "acceptance"
        acc_dir.mkdir(parents=True)
        (acc_dir / "AC-001.md").write_text(dedent("""
            ---
            acceptance_id: AC-001
            task_id: TASK-001
            reviewed_at: 2026-04-10
            ---
            """).strip())

        import tools.ci.knowledge_reuse as kr_module
        original_repo_root = kr_module._REPO_ROOT
        original_acc_dir = kr_module._ACCEPTANCE_DIR
        kr_module._REPO_ROOT = tmp_path
        kr_module._ACCEPTANCE_DIR = acc_dir

        entries = kr_module.scan_acceptances()

        kr_module._REPO_ROOT = original_repo_root
        kr_module._ACCEPTANCE_DIR = original_acc_dir

        assert len(entries) == 1
        assert entries[0].entry_type == "acceptance"
        assert entries[0].entry_id == "AC-001"
        assert entries[0].related_entries == ["TASK-001"]

    def test_skips_files_without_acceptance_id(self, tmp_path: Path) -> None:
        acc_dir = tmp_path / "docs" / "acceptance"
        acc_dir.mkdir(parents=True)
        (acc_dir / "AC-invalid.md").write_text(dedent("""
            ---
            task_id: TASK-001
            ---
            """).strip())

        import tools.ci.knowledge_reuse as kr_module
        original_repo_root = kr_module._REPO_ROOT
        original_acc_dir = kr_module._ACCEPTANCE_DIR
        kr_module._REPO_ROOT = tmp_path
        kr_module._ACCEPTANCE_DIR = acc_dir

        entries = kr_module.scan_acceptances()

        kr_module._REPO_ROOT = original_repo_root
        kr_module._ACCEPTANCE_DIR = original_acc_dir

        assert entries == []


class TestScanIncidents:
    def test_scans_incident_files(self, tmp_path: Path) -> None:
        incidents_dir = tmp_path / "docs"
        incidents_dir.mkdir(parents=True)
        (incidents_dir / "IN-001.md").write_text(dedent("""
            ---
            date: 2026-04-10
            tags: security,urgent
            ---
            # Incident 001
            """).strip())

        import tools.ci.knowledge_reuse as kr_module
        original_repo_root = kr_module._REPO_ROOT
        original_incidents_dir = kr_module._INCIDENTS_DIR
        kr_module._REPO_ROOT = tmp_path
        kr_module._INCIDENTS_DIR = incidents_dir

        entries = kr_module.scan_incidents()

        kr_module._REPO_ROOT = original_repo_root
        kr_module._INCIDENTS_DIR = original_incidents_dir

        assert len(entries) == 1
        assert entries[0].entry_type == "incident"
        assert entries[0].entry_id == "IN-001"
        # Tags are split by comma without strip, so expect ["security", "urgent"]
        assert "security" in entries[0].tags[0]


class TestBuildKnowledgeIndex:
    def test_builds_index(self, tmp_path: Path) -> None:
        releases_dir = tmp_path / "docs" / "releases"
        releases_dir.mkdir(parents=True)
        (releases_dir / "v1.0.0.md").write_text(dedent("""
            ---
            date: 2026-04-10
            ---
            # Release
            """).strip())

        acc_dir = tmp_path / "docs" / "acceptance"
        acc_dir.mkdir(parents=True)
        (acc_dir / "AC-001.md").write_text(dedent("""
            ---
            acceptance_id: AC-001
            task_id: TASK-001
            ---
            """).strip())

        incidents_dir = tmp_path / "docs"
        # Already created by releases_dir/acc_dir parents=True
        (incidents_dir / "IN-001.md").write_text("# Incident")

        import tools.ci.knowledge_reuse as kr_module
        original_repo_root = kr_module._REPO_ROOT
        original_releases_dir = kr_module._RELEASES_DIR
        original_acc_dir = kr_module._ACCEPTANCE_DIR
        original_incidents_dir = kr_module._INCIDENTS_DIR
        kr_module._REPO_ROOT = tmp_path
        kr_module._RELEASES_DIR = releases_dir
        kr_module._ACCEPTANCE_DIR = acc_dir
        kr_module._INCIDENTS_DIR = incidents_dir

        index = kr_module.build_knowledge_index()

        kr_module._REPO_ROOT = original_repo_root
        kr_module._RELEASES_DIR = original_releases_dir
        kr_module._ACCEPTANCE_DIR = original_acc_dir
        kr_module._INCIDENTS_DIR = original_incidents_dir

        assert "entries" in index
        assert "by_type" in index
        assert "release" in index["by_type"]
        assert "acceptance" in index["by_type"]
        assert "incident" in index["by_type"]
        assert len(index["entries"]) >= 1


class TestSearchKnowledge:
    def test_searches_by_title(self) -> None:
        index = {
            "entries": [
                {
                    "type": "release",
                    "id": "v1.0.0",
                    "title": "Security Release",
                    "date": "2026-04-10",
                    "related": [],
                    "tags": [],
                    "path": "",
                },
            ],
        }

        results = search_knowledge(index, "security")
        assert len(results) == 1
        assert results[0]["id"] == "v1.0.0"

    def test_searches_by_id(self) -> None:
        index = {
            "entries": [
                {
                    "type": "acceptance",
                    "id": "AC-001",
                    "title": "Acceptance Record",
                    "date": "",
                    "related": [],
                    "tags": [],
                    "path": "",
                },
            ],
        }

        results = search_knowledge(index, "AC-001")
        assert len(results) == 1

    def test_searches_by_tag(self) -> None:
        index = {
            "entries": [
                {
                    "type": "incident",
                    "id": "IN-001",
                    "title": "Incident",
                    "date": "",
                    "related": [],
                    "tags": ["urgent", "security"],
                    "path": "",
                },
            ],
        }

        results = search_knowledge(index, "urgent")
        assert len(results) == 1

    def test_returns_empty_for_no_match(self) -> None:
        index = {"entries": []}
        results = search_knowledge(index, "nothing")
        assert results == []