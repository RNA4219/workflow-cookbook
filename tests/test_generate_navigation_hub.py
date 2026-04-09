"""Tests for tools.ci.generate_navigation_hub."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.ci.generate_navigation_hub import (
    generate_hub,
    main,
    NAV_SECTIONS,
)


class TestGenerateHub:
    def test_generates_hub_content(self) -> None:
        content = generate_hub()
        assert "# Navigation Hub" in content
        assert "Generated:" in content
        assert "Quick entry points" in content

    def test_includes_all_sections(self) -> None:
        content = generate_hub()
        for section in NAV_SECTIONS:
            assert f"## {section['title']}" in content
            assert section["description"] in content

    def test_includes_quick_commands(self) -> None:
        content = generate_hub()
        assert "## Quick Commands" in content
        assert "pytest" in content
        assert "update.py" in content

    def test_generates_entry_table(self) -> None:
        content = generate_hub()
        assert "| Entry | Path | Description |" in content
        # Check that entries are rendered
        for section in NAV_SECTIONS:
            for name, path, desc in section["entries"]:
                assert f"| [{name}]" in content


class TestNavSections:
    def test_sections_have_required_fields(self) -> None:
        for section in NAV_SECTIONS:
            assert "title" in section
            assert "description" in section
            assert "entries" in section
            assert len(section["entries"]) > 0

    def test_entries_have_three_elements(self) -> None:
        for section in NAV_SECTIONS:
            for entry in section["entries"]:
                assert len(entry) == 3
                assert isinstance(entry[0], str)  # name
                assert isinstance(entry[1], str)  # path
                assert isinstance(entry[2], str)  # description


class TestMain:
    def test_main_writes_output(self, tmp_path: Path) -> None:
        output_path = tmp_path / "NAVIGATION.md"

        result = main(["--output", str(output_path)])

        assert result == 0
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "# Navigation Hub" in content