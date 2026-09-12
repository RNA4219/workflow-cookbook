# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Tests for check_version_consistency.py."""

from __future__ import annotations

from pathlib import Path

from tools.ci.check_version_consistency import (
    ValidationResult,
    load_changelog_versions,
    load_git_tag_versions,
    load_pyproject_version,
    load_readme_badge_version,
    load_release_doc_versions,
    validate_version_consistency,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


class TestLoadFunctions:
    """Test version extraction from various sources."""

    def test_load_pyproject_version(self) -> None:
        pyproject = _REPO_ROOT / "pyproject.toml"
        version = load_pyproject_version(pyproject)
        assert version is not None
        assert isinstance(version, str)
        # Format: X.Y.Z
        parts = version.split(".")
        assert len(parts) == 3

    def test_load_readme_badge_version(self) -> None:
        readme = _REPO_ROOT / "README.md"
        version = load_readme_badge_version(readme)
        assert version is not None
        assert isinstance(version, str)

    def test_load_changelog_versions(self) -> None:
        changelog = _REPO_ROOT / "CHANGELOG.md"
        versions = load_changelog_versions(changelog)
        assert len(versions) > 0
        # All versions should be X.Y.Z format
        for v in versions:
            parts = v.split(".")
            assert len(parts) == 3

    def test_load_git_tag_versions(self) -> None:
        versions = load_git_tag_versions(_REPO_ROOT)
        assert len(versions) > 0

    def test_load_release_doc_versions(self) -> None:
        releases_dir = _REPO_ROOT / "docs" / "releases"
        versions = load_release_doc_versions(releases_dir)
        assert len(versions) > 0


class TestValidationResult:
    """Test ValidationResult class."""

    def test_is_success_no_errors(self) -> None:
        result = ValidationResult()
        assert result.is_success

    def test_is_success_with_errors(self) -> None:
        result = ValidationResult(errors=["error1"])
        assert not result.is_success

    def test_warnings_do_not_affect_success(self) -> None:
        result = ValidationResult(warnings=["warning1"])
        assert result.is_success


class TestValidateVersionConsistency:
    """Test full validation."""

    def test_validation_returns_result(self) -> None:
        result = validate_version_consistency(
            repo_root=_REPO_ROOT,
            pyproject_path=_REPO_ROOT / "pyproject.toml",
            readme_path=_REPO_ROOT / "README.md",
            changelog_path=_REPO_ROOT / "CHANGELOG.md",
            releases_dir=_REPO_ROOT / "docs" / "releases",
        )
        assert isinstance(result, ValidationResult)

    def test_validation_passes_after_fix(self) -> None:
        """Test that validation passes after version fixes."""
        result = validate_version_consistency(
            repo_root=_REPO_ROOT,
            pyproject_path=_REPO_ROOT / "pyproject.toml",
            readme_path=_REPO_ROOT / "README.md",
            changelog_path=_REPO_ROOT / "CHANGELOG.md",
            releases_dir=_REPO_ROOT / "docs" / "releases",
        )
        # After fixing pyproject.toml, validation should pass (errors empty)
        # A fully aligned pending release may emit a non-failing tag warning.
        assert result.is_success
        assert len(result.errors) == 0

    def test_validation_reports_no_changelog_warning_for_tagged_release(self) -> None:
        """Test tagged CHANGELOG entries do not produce missing-tag warnings."""
        result = validate_version_consistency(
            repo_root=_REPO_ROOT,
            pyproject_path=_REPO_ROOT / "pyproject.toml",
            readme_path=_REPO_ROOT / "README.md",
            changelog_path=_REPO_ROOT / "CHANGELOG.md",
            releases_dir=_REPO_ROOT / "docs" / "releases",
        )
        assert not any("1.2.0" in w and "no git tag" in w for w in result.warnings)

    @staticmethod
    def _write_sources(
        root: Path,
        *,
        package_version: str,
        readme_version: str,
        changelog_versions: list[str],
        release_versions: list[str],
    ) -> tuple[Path, Path, Path, Path]:
        pyproject = root / "pyproject.toml"
        readme = root / "README.md"
        changelog = root / "CHANGELOG.md"
        releases = root / "docs" / "releases"
        releases.mkdir(parents=True)
        pyproject.write_text(
            f'[project]\nversion = "{package_version}"\n', encoding="utf-8"
        )
        readme.write_text(
            f"![Version](https://img.shields.io/badge/version-{readme_version}-blue.svg)\n",
            encoding="utf-8",
        )
        changelog.write_text(
            "\n".join(f"## {version} - 2026-09-13" for version in changelog_versions),
            encoding="utf-8",
        )
        for version in release_versions:
            (releases / f"v{version}.md").write_text(
                f"# workflow-cookbook v{version}\n", encoding="utf-8"
            )
        return pyproject, readme, changelog, releases

    def test_tagged_release_passes_when_all_sources_align(self, tmp_path: Path) -> None:
        pyproject, readme, changelog, releases = self._write_sources(
            tmp_path,
            package_version="1.2.0",
            readme_version="1.2.0",
            changelog_versions=["1.2.0"],
            release_versions=["1.2.0"],
        )

        result = validate_version_consistency(
            repo_root=tmp_path,
            pyproject_path=pyproject,
            readme_path=readme,
            changelog_path=changelog,
            releases_dir=releases,
            git_tag_versions={"1.2.0"},
        )

        assert result.is_success
        assert result.warnings == []

    def test_pending_release_passes_when_all_sources_align(self, tmp_path: Path) -> None:
        pyproject, readme, changelog, releases = self._write_sources(
            tmp_path,
            package_version="1.3.0",
            readme_version="1.3.0",
            changelog_versions=["1.3.0", "1.2.0"],
            release_versions=["1.3.0", "1.2.0"],
        )

        result = validate_version_consistency(
            repo_root=tmp_path,
            pyproject_path=pyproject,
            readme_path=readme,
            changelog_path=changelog,
            releases_dir=releases,
            git_tag_versions={"1.2.0"},
        )

        assert result.is_success
        assert any("git tag v1.3.0 is pending" in warning for warning in result.warnings)

    def test_pending_release_requires_aligned_readme_changelog_and_note(
        self, tmp_path: Path
    ) -> None:
        pyproject, readme, changelog, releases = self._write_sources(
            tmp_path,
            package_version="1.3.0",
            readme_version="1.2.0",
            changelog_versions=["1.2.0"],
            release_versions=["1.2.0"],
        )

        result = validate_version_consistency(
            repo_root=tmp_path,
            pyproject_path=pyproject,
            readme_path=readme,
            changelog_path=changelog,
            releases_dir=releases,
            git_tag_versions={"1.2.0"},
        )

        assert not result.is_success
        assert any("README badge" in error for error in result.errors)
        assert any("CHANGELOG.md missing" in error for error in result.errors)
        assert any("docs/releases/v1.3.0.md missing" in error for error in result.errors)

    def test_package_version_behind_latest_tag_fails(self, tmp_path: Path) -> None:
        pyproject, readme, changelog, releases = self._write_sources(
            tmp_path,
            package_version="1.1.0",
            readme_version="1.1.0",
            changelog_versions=["1.2.0", "1.1.0"],
            release_versions=["1.2.0", "1.1.0"],
        )

        result = validate_version_consistency(
            repo_root=tmp_path,
            pyproject_path=pyproject,
            readme_path=readme,
            changelog_path=changelog,
            releases_dir=releases,
            git_tag_versions={"1.2.0", "1.1.0"},
        )

        assert not result.is_success
        assert any("git tag latest 1.2.0" in error for error in result.errors)

    def test_unrelated_untagged_release_note_fails(self, tmp_path: Path) -> None:
        pyproject, readme, changelog, releases = self._write_sources(
            tmp_path,
            package_version="1.3.0",
            readme_version="1.3.0",
            changelog_versions=["1.3.0", "1.2.0"],
            release_versions=["1.4.0", "1.3.0", "1.2.0"],
        )

        result = validate_version_consistency(
            repo_root=tmp_path,
            pyproject_path=pyproject,
            readme_path=readme,
            changelog_path=changelog,
            releases_dir=releases,
            git_tag_versions={"1.2.0"},
        )

        assert not result.is_success
        assert any("v1.4.0.md exists but no git tag" in error for error in result.errors)
