# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Version consistency checker across multiple sources."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_DEFAULT_README = _REPO_ROOT / "README.md"
_DEFAULT_CHANGELOG = _REPO_ROOT / "CHANGELOG.md"
_DEFAULT_RELEASES_DIR = _REPO_ROOT / "docs" / "releases"

_PYPROJECT_VERSION_RE = re.compile(r'^version\s*=\s*"(\d+\.\d+\.\d+)"')
_README_BADGE_RE = re.compile(r"badge/version-(\d+\.\d+\.\d+)-")
_CHANGELOG_VERSION_RE = re.compile(r"^## (\d+\.\d+\.\d+) - ")
_TAG_RE = re.compile(r"^v(\d+\.\d+\.\d+)$")
_RELEASE_DOC_RE = re.compile(r"^v(\d+\.\d+\.\d+)\.md$")


@dataclass
class ValidationResult:
    """Result of version consistency validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        return not self.errors

    def emit(self) -> None:
        for message in self.errors:
            print(message, file=sys.stderr)
        for message in self.warnings:
            print(message, file=sys.stderr)


def load_pyproject_version(pyproject_path: Path) -> str | None:
    """Extract version from pyproject.toml."""
    for line in pyproject_path.read_text(encoding="utf-8").splitlines():
        match = _PYPROJECT_VERSION_RE.match(line.strip())
        if match:
            return match.group(1)
    return None


def load_readme_badge_version(readme_path: Path) -> str | None:
    """Extract version from README.md badge."""
    for line in readme_path.read_text(encoding="utf-8").splitlines():
        match = _README_BADGE_RE.search(line)
        if match:
            return match.group(1)
    return None


def load_changelog_versions(changelog_path: Path) -> list[str]:
    """Extract versions from CHANGELOG.md (sorted by appearance order)."""
    versions: list[str] = []
    for line in changelog_path.read_text(encoding="utf-8").splitlines():
        match = _CHANGELOG_VERSION_RE.match(line.strip())
        if match:
            versions.append(match.group(1))
    return versions


def load_git_tag_versions(repo_root: Path) -> set[str]:
    """Load versions from git tags."""
    try:
        completed = subprocess.run(  # nosec B603,B607  # git command with fixed args
            ["git", "tag"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()

    versions: set[str] = set()
    for raw in completed.stdout.splitlines():
        match = _TAG_RE.match(raw.strip())
        if match:
            versions.add(match.group(1))
    return versions


def load_release_doc_versions(releases_dir: Path) -> set[str]:
    """Load versions from docs/releases/v*.md files."""
    versions: set[str] = set()
    for path in releases_dir.glob("v*.md"):
        match = _RELEASE_DOC_RE.match(path.name)
        if match:
            versions.add(match.group(1))
    return versions


def validate_version_consistency(
    *,
    repo_root: Path,
    pyproject_path: Path,
    readme_path: Path,
    changelog_path: Path,
    releases_dir: Path,
) -> ValidationResult:
    """Validate version consistency across all sources."""
    result = ValidationResult()

    pyproject_version = load_pyproject_version(pyproject_path)
    readme_version = load_readme_badge_version(readme_path)
    changelog_versions = load_changelog_versions(changelog_path)
    tag_versions = load_git_tag_versions(repo_root)
    release_doc_versions = load_release_doc_versions(releases_dir)

    # Check pyproject.toml exists
    if not pyproject_version:
        result.errors.append("pyproject.toml: version field not found")

    # Check README badge exists
    if not readme_version:
        result.warnings.append("README.md: version badge not found")

    # Check CHANGELOG has versions
    if not changelog_versions:
        result.errors.append("CHANGELOG.md: no version entries found")

    # Determine expected latest version from git tags
    if tag_versions:
        # Sort by semantic version
        sorted_tags = sorted(
            tag_versions, key=lambda v: tuple(int(x) for x in v.split("."))
        )
        latest_tag = sorted_tags[-1]
    else:
        result.warnings.append("No git tags found; cannot determine expected version")
        latest_tag = None

    # Cross-validate sources
    if pyproject_version and latest_tag:
        if pyproject_version != latest_tag:
            result.errors.append(
                f"pyproject.toml version {pyproject_version} != git tag latest {latest_tag}"
            )

    if readme_version and latest_tag:
        if readme_version != latest_tag:
            result.warnings.append(
                f"README badge version {readme_version} != git tag latest {latest_tag}"
            )

    # Check changelog contains all tagged versions
    changelog_set = set(changelog_versions)
    for version in tag_versions:
        if version not in changelog_set:
            result.errors.append(
                f"CHANGELOG.md missing version {version} (git tag exists)"
            )

    # Check release docs match git tags
    for version in tag_versions:
        if version not in release_doc_versions:
            result.errors.append(
                f"docs/releases/v{version}.md missing (git tag exists)"
            )

    # Check release docs not backed by git tags
    for version in release_doc_versions:
        if version not in tag_versions:
            result.errors.append(
                f"docs/releases/v{version}.md exists but no git tag v{version}"
            )

    # Check CHANGELOG entries not backed by git tags (excluding Unreleased)
    for version in changelog_set:
        if version not in tag_versions:
            result.warnings.append(
                f"CHANGELOG.md has version {version} but no git tag v{version}"
            )

    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for version consistency checker."""
    parser = argparse.ArgumentParser(
        description="Check version consistency across pyproject.toml, README, CHANGELOG, git tags, and release docs."
    )
    parser.add_argument(
        "--check", action="store_true", help="Exit non-zero on validation failure."
    )
    parser.add_argument(
        "--repo-root", type=Path, default=_REPO_ROOT, help="Repository root."
    )
    parser.add_argument(
        "--pyproject", type=Path, default=_DEFAULT_PYPROJECT, help="Path to pyproject.toml."
    )
    parser.add_argument(
        "--readme", type=Path, default=_DEFAULT_README, help="Path to README.md."
    )
    parser.add_argument(
        "--changelog", type=Path, default=_DEFAULT_CHANGELOG, help="Path to CHANGELOG.md."
    )
    parser.add_argument(
        "--releases-dir", type=Path, default=_DEFAULT_RELEASES_DIR, help="Path to docs/releases."
    )
    args = parser.parse_args(argv)

    result = validate_version_consistency(
        repo_root=args.repo_root,
        pyproject_path=args.pyproject,
        readme_path=args.readme,
        changelog_path=args.changelog,
        releases_dir=args.releases_dir,
    )
    result.emit()

    if result.is_success:
        print("Version consistency check passed.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())