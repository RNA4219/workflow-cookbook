"""Tests for tools.ci.generate_docs_index."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.ci.generate_docs_index import (
    _parse_front_matter,
    _extract_title,
    _check_links,
    scan_directory,
    generate_index_markdown,
    DocInfo,
)


class TestParseFrontMatter:
    def test_parses_yaml_front_matter(self) -> None:
        content = dedent("""
            ---
            adr_id: ADR-001
            status: accepted
            last_reviewed: 2026-04-10
            ---
            # Content
            """).strip()
        result = _parse_front_matter(content)
        assert result.get("adr_id") == "ADR-001"
        assert result.get("status") == "accepted"

    def test_returns_empty_for_no_front_matter(self) -> None:
        result = _parse_front_matter("# No front matter")
        assert result == {}

    def test_handles_quoted_values(self) -> None:
        content = dedent("""
            ---
            adr_id: "ADR-001"
            status: 'accepted'
            ---
            """).strip()
        result = _parse_front_matter(content)
        assert result.get("adr_id") == "ADR-001"
        assert result.get("status") == "accepted"


class TestExtractTitle:
    def test_extracts_h1_title(self) -> None:
        content = "# My Document Title\n\nContent"
        title = _extract_title(content)
        assert title == "My Document Title"

    def test_returns_untitled_for_no_title(self) -> None:
        content = "Content without title"
        title = _extract_title(content)
        assert title == "Untitled"

    def test_ignores_h2_and_below(self) -> None:
        content = "## Secondary Title\n# Primary Title"
        title = _extract_title(content)
        assert title == "Primary Title"


class TestCheckLinks:
    def test_extracts_markdown_links(self, tmp_path: Path) -> None:
        content = "See [README](README.md) and [Docs](docs/guide.md)"
        broken = _check_links(content, tmp_path / "test.md", tmp_path)
        # These should be broken since files don't exist
        assert len(broken) == 2

    def test_ignores_http_links(self, tmp_path: Path) -> None:
        content = "See [External](https://example.com/doc)"
        broken = _check_links(content, tmp_path / "test.md", tmp_path)
        assert broken == []

    def test_accepts_existing_files(self, tmp_path: Path) -> None:
        # Create referenced file
        ref_file = tmp_path / "README.md"
        ref_file.write_text("# Reference")

        content = "See [README](README.md)"
        broken = _check_links(content, tmp_path / "test.md", tmp_path)
        assert broken == []

    def test_handles_absolute_paths(self, tmp_path: Path) -> None:
        content = "See [Absolute](/missing.md)"
        broken = _check_links(content, tmp_path / "test.md", tmp_path)
        assert len(broken) == 1


class TestScanDirectory:
    def test_scans_markdown_files(self, tmp_path: Path) -> None:
        adr_dir = tmp_path / "ADR"
        adr_dir.mkdir()
        adr_file = adr_dir / "ADR-001.md"
        adr_file.write_text(dedent("""
            ---
            adr_id: ADR-001
            status: accepted
            last_reviewed: 2026-04-10
            ---
            # ADR-001: Test Decision
            """).strip())

        docs = scan_directory(adr_dir, tmp_path)
        assert len(docs) == 1
        assert docs[0].title == "ADR-001: Test Decision"
        assert docs[0].adr_id == "ADR-001"
        assert docs[0].status == "accepted"

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        docs = scan_directory(tmp_path / "nonexistent", tmp_path)
        assert docs == []

    def test_skips_index_and_readme(self, tmp_path: Path) -> None:
        adr_dir = tmp_path / "ADR"
        adr_dir.mkdir()
        (adr_dir / "INDEX.md").write_text("# Index")
        (adr_dir / "README.md").write_text("# README")
        (adr_dir / "ADR-001.md").write_text("# ADR")

        docs = scan_directory(adr_dir, tmp_path)
        assert len(docs) == 1
        assert docs[0].file_path.name == "ADR-001.md"

    def test_checks_for_broken_links(self, tmp_path: Path) -> None:
        adr_dir = tmp_path / "ADR"
        adr_dir.mkdir()
        adr_file = adr_dir / "ADR-001.md"
        adr_file.write_text("See [Missing](missing.md)")

        docs = scan_directory(adr_dir, tmp_path)
        assert len(docs) == 1
        assert len(docs[0].broken_links) == 1


class TestGenerateIndexMarkdown:
    def test_generates_index_content(self) -> None:
        docs = [
            DocInfo(
                file_path=Path("ADR/ADR-001.md"),
                title="Test ADR",
                adr_id="ADR-001",
                status="accepted",
                last_reviewed="2026-04-10",
                broken_links=[],
            ),
        ]

        markdown = generate_index_markdown(docs, "ADR", "../")
        assert "# ADR Index" in markdown
        assert "Total: 1 documents" in markdown
        assert "ADR-001.md" in markdown
        assert "Test ADR" in markdown

    def test_reports_broken_links(self) -> None:
        docs = [
            DocInfo(
                file_path=Path("ADR/ADR-001.md"),
                title="Test ADR",
                adr_id="ADR-001",
                status="accepted",
                last_reviewed="2026-04-10",
                broken_links=["README -> README.md"],
            ),
        ]

        markdown = generate_index_markdown(docs, "ADR", "../")
        assert "## Broken Links" in markdown
        assert "README -> README.md" in markdown

    def test_handles_empty_docs(self) -> None:
        markdown = generate_index_markdown([], "ADR", "../")
        assert "No documents found" in markdown