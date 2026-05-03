# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Tests for check_version_consistency.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.ci.check_version_consistency import (
    load_changelog_versions,
    load_git_tag_versions,
    load_pyproject_version,
    load_readme_badge_version,
    load_release_doc_versions,
    validate_version_consistency,
    ValidationResult,
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
        # Only warning: CHANGELOG 1.2.0 without tag
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
