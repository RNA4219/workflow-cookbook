"""Tests for tools.ci.check_release_evidence."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from textwrap import dedent

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "ci" / "check_release_evidence.py"
spec = importlib.util.spec_from_file_location("check_release_evidence", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to load check_release_evidence module")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

validate_release_evidence = module.validate_release_evidence
load_changelog_versions = module.load_changelog_versions
load_release_doc_versions = module.load_release_doc_versions
load_release_doc_title_version = module.load_release_doc_title_version
ValidationResult = module.ValidationResult


def _write_release_doc(path: Path, version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"# workflow-cookbook v{version}",
                "",
                "## Summary",
                "",
                "- sample",
            ]
        ),
        encoding="utf-8",
    )


class TestValidationResult:
    def test_is_success_with_no_errors(self) -> None:
        result = ValidationResult()
        assert result.is_success is True

    def test_is_success_with_errors(self) -> None:
        result = ValidationResult(errors=["error"])
        assert result.is_success is False

    def test_emit_outputs_errors(self, capsys) -> None:
        result = ValidationResult(
            errors=["error1"],
            warnings=["warning1"],
        )
        result.emit()
        captured = capsys.readouterr()
        assert "error1" in captured.err
        assert "warning1" in captured.err


class TestLoadChangelogVersions:
    def test_extracts_versions(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(dedent("""
            # Changelog

            ## 1.0.0 - 2026-04-10
            - First release

            ## 0.9.0 - 2026-04-01
            - Beta
            """).strip())

        versions = load_changelog_versions(changelog)
        assert versions == ["1.0.0", "0.9.0"]

    def test_returns_empty_for_no_versions(self, tmp_path: Path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("No versions")

        versions = load_changelog_versions(changelog)
        assert versions == []


class TestLoadReleaseDocVersions:
    def test_extracts_versions(self, tmp_path: Path) -> None:
        releases_dir = tmp_path / "releases"
        releases_dir.mkdir()
        (releases_dir / "v1.0.0.md").write_text("# content")
        (releases_dir / "v0.9.0.md").write_text("# content")

        versions = load_release_doc_versions(releases_dir)
        assert "1.0.0" in versions
        assert "0.9.0" in versions

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        versions = load_release_doc_versions(tmp_path / "nonexistent")
        assert versions == {}


class TestLoadReleaseDocTitleVersion:
    def test_extracts_title_version(self, tmp_path: Path) -> None:
        release_file = tmp_path / "v1.0.0.md"
        release_file.write_text("# workflow-cookbook v1.0.0\n\nContent")

        version = load_release_doc_title_version(release_file)
        assert version == "1.0.0"

    def test_returns_none_for_no_title(self, tmp_path: Path) -> None:
        release_file = tmp_path / "v1.0.0.md"
        release_file.write_text("No title here")

        version = load_release_doc_title_version(release_file)
        assert version is None


def test_validate_release_evidence_pass(tmp_path: Path, monkeypatch) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    releases_dir = tmp_path / "docs" / "releases"
    changelog.write_text(
        "\n".join(
            [
                "# Changelog",
                "",
                "## [Unreleased]",
                "",
                "## 1.0.0 - 2025-10-16",
            ]
        ),
        encoding="utf-8",
    )
    _write_release_doc(releases_dir / "v1.0.0.md", "1.0.0")
    monkeypatch.setattr(module, "load_git_tag_versions", lambda repo_root: {"1.0.0"})

    result = validate_release_evidence(
        repo_root=tmp_path,
        changelog_path=changelog,
        releases_dir=releases_dir,
    )

    assert result.errors == []


def test_validate_release_evidence_reports_missing_release_doc(tmp_path: Path, monkeypatch) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    releases_dir = tmp_path / "docs" / "releases"
    releases_dir.mkdir(parents=True)
    changelog.write_text(
        "\n".join(
            [
                "# Changelog",
                "",
                "## [Unreleased]",
                "",
                "## 1.0.0 - 2025-10-16",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "load_git_tag_versions", lambda repo_root: {"1.0.0"})

    result = validate_release_evidence(
        repo_root=tmp_path,
        changelog_path=changelog,
        releases_dir=releases_dir,
    )

    assert "CHANGELOG version 1.0.0 is missing docs/releases/v1.0.0.md." in result.errors
    assert "Git tag v1.0.0 is missing docs/releases/v1.0.0.md." in result.errors


def test_validate_release_evidence_reports_title_mismatch(tmp_path: Path, monkeypatch) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    releases_dir = tmp_path / "docs" / "releases"
    changelog.write_text(
        "\n".join(["# Changelog", "", "## [Unreleased]", "", "## 1.0.0 - 2025-10-16"]),
        encoding="utf-8",
    )
    _write_release_doc(releases_dir / "v1.0.0.md", "1.0.1")
    monkeypatch.setattr(module, "load_git_tag_versions", lambda repo_root: {"1.0.0"})

    result = validate_release_evidence(
        repo_root=tmp_path,
        changelog_path=changelog,
        releases_dir=releases_dir,
    )

    assert f"{releases_dir / 'v1.0.0.md'} title does not match filename version v1.0.0." in result.errors


def test_validate_release_evidence_reports_missing_changelog_version(tmp_path: Path, monkeypatch) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    releases_dir = tmp_path / "docs" / "releases"
    changelog.write_text(
        "\n".join(["# Changelog", "", "## [Unreleased]", "", "## 1.0.0 - 2025-10-16"]),
        encoding="utf-8",
    )
    _write_release_doc(releases_dir / "v1.0.0.md", "1.0.0")
    _write_release_doc(releases_dir / "v2.0.0.md", "2.0.0")
    monkeypatch.setattr(module, "load_git_tag_versions", lambda repo_root: {"1.0.0"})

    result = validate_release_evidence(
        repo_root=tmp_path,
        changelog_path=changelog,
        releases_dir=releases_dir,
    )

    assert any("not present in CHANGELOG" in e for e in result.errors)


def test_validate_release_evidence_no_git_tags(tmp_path: Path, monkeypatch) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    releases_dir = tmp_path / "docs" / "releases"
    changelog.write_text(
        "\n".join(["# Changelog", "", "## 1.0.0 - 2025-10-16"]),
        encoding="utf-8",
    )
    _write_release_doc(releases_dir / "v1.0.0.md", "1.0.0")
    monkeypatch.setattr(module, "load_git_tag_versions", lambda repo_root: set())

    result = validate_release_evidence(
        repo_root=tmp_path,
        changelog_path=changelog,
        releases_dir=releases_dir,
    )

    assert any("No git tags" in w for w in result.warnings)