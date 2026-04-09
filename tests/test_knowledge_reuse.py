"""Tests for tools.ci.knowledge_reuse."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.ci.knowledge_reuse import (
    _parse_front_matter,
    _extract_title,
    scan_releases,
    scan_acceptances,
    scan_incidents,
    build_knowledge_index,
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