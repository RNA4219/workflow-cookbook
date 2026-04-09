"""Tests for tools.ci.check_sample_docs_sync."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.ci.check_sample_docs_sync import (
    _extract_doc_refs,
    scan_samples,
    validate_sync,
)


class TestExtractDocRefs:
    def test_extracts_markdown_links(self) -> None:
        content = "See [README](../README.md) and [Docs](docs/guide.md)"
        refs = _extract_doc_refs(content, Path("test.md"))
        assert "../README.md" in refs
        assert "docs/guide.md" in refs

    def test_ignores_http_links(self) -> None:
        content = "See [External](https://example.com/doc)"
        refs = _extract_doc_refs(content, Path("test.md"))
        assert "https://example.com/doc" not in refs

    def test_extracts_paths_from_json(self) -> None:
        content = json.dumps({
            "config": "path/to/config.yaml",
            "doc": "docs/readme.md",
        })
        refs = _extract_doc_refs(content, Path("test.json"))
        assert any("config.yaml" in r for r in refs)
        assert any("readme.md" in r for r in refs)


class TestScanSamples:
    def test_scans_sample_files(self, tmp_path: Path) -> None:
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        sample_file = examples_dir / "sample.json"
        sample_file.write_text(json.dumps({"ref": "docs/test.md"}))

        samples = scan_samples(examples_dir, tmp_path)
        assert len(samples) == 1
        assert samples[0].file_path == sample_file

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        samples = scan_samples(tmp_path / "nonexistent", tmp_path)
        assert samples == []


class TestValidateSync:
    def test_validates_existing_refs(self, tmp_path: Path) -> None:
        # Create sample
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        sample_file = examples_dir / "sample.json"
        sample_file.write_text(json.dumps({"doc": "test.md"}))

        # Create referenced doc
        doc_file = tmp_path / "test.md"
        doc_file.write_text("# Test")

        samples = scan_samples(examples_dir, tmp_path)
        report = validate_sync(samples, tmp_path)

        assert report.errors == []

    def test_reports_missing_refs(self, tmp_path: Path) -> None:
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        sample_file = examples_dir / "sample.json"
        # Use a path with "/" so it's extracted as a doc ref
        sample_file.write_text(json.dumps({"doc": "docs/nonexistent.md"}))

        # Need to pass repo_root as tmp_path for resolution
        samples = scan_samples(examples_dir, tmp_path)
        report = validate_sync(samples, tmp_path)

        # Check that there are missing refs
        assert len(report.errors) >= 1
        assert "missing" in report.errors[0].lower() or "nonexistent" in report.errors[0].lower()