# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHANGELOG = _REPO_ROOT / "CHANGELOG.md"
_DEFAULT_RELEASES_DIR = _REPO_ROOT / "docs" / "releases"

_CHANGELOG_VERSION_RE = re.compile(r"^## (\d+\.\d+\.\d+) - ")
_RELEASE_DOC_RE = re.compile(r"^v(\d+\.\d+\.\d+)\.md$")
_TITLE_RE = re.compile(r"^# workflow-cookbook v(\d+\.\d+\.\d+)$")
_TAG_RE = re.compile(r"^v(\d+\.\d+\.\d+)$")


@dataclass
class ValidationResult:
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


def load_changelog_versions(changelog_path: Path) -> list[str]:
    versions: list[str] = []
    for line in changelog_path.read_text(encoding="utf-8").splitlines():
        match = _CHANGELOG_VERSION_RE.match(line.strip())
        if match:
            versions.append(match.group(1))
    return versions


def load_release_doc_versions(releases_dir: Path) -> dict[str, Path]:
    versions: dict[str, Path] = {}
    for path in sorted(releases_dir.glob("v*.md")):
        match = _RELEASE_DOC_RE.match(path.name)
        if match:
            versions[match.group(1)] = path
    return versions


def load_release_doc_title_version(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _TITLE_RE.match(stripped)
        return match.group(1) if match else None
    return None


def _parse_tag_versions(lines: Iterable[str]) -> set[str]:
    versions: set[str] = set()
    for raw in lines:
        name = raw.strip().split("/")[-1]
        match = _TAG_RE.match(name)
        if match:
            versions.add(match.group(1))
    return versions


def load_git_tag_versions(repo_root: Path) -> set[str]:
    commands = (
        ["git", "ls-remote", "--tags", "origin"],
        ["git", "tag"],
    )
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        versions = _parse_tag_versions(completed.stdout.splitlines())
        if versions:
            return versions
    return set()


def load_published_release_versions(repo: str, token: str | None) -> set[str]:
    if not token:
        return set()
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases?per_page=100",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Failed to fetch GitHub releases for {repo}: HTTP {exc.code}") from exc

    versions: set[str] = set()
    for release in payload:
        if release.get("draft"):
            continue
        tag_name = str(release.get("tag_name", "")).strip()
        match = _TAG_RE.match(tag_name)
        if match:
            versions.add(match.group(1))
    return versions


def validate_release_evidence(
    *,
    repo_root: Path,
    changelog_path: Path,
    releases_dir: Path,
    github_repo: str | None = None,
    github_token: str | None = None,
) -> ValidationResult:
    result = ValidationResult()

    changelog_versions = load_changelog_versions(changelog_path)
    release_doc_versions = load_release_doc_versions(releases_dir)
    tag_versions = load_git_tag_versions(repo_root)

    if not changelog_versions:
        result.errors.append(f"No released versions were found in {changelog_path}.")
        return result

    if not release_doc_versions:
        result.errors.append(f"No release note docs were found under {releases_dir}.")

    changelog_set = set(changelog_versions)
    release_doc_set = set(release_doc_versions)

    for version in changelog_versions:
        if version not in release_doc_set:
            result.errors.append(
                f"CHANGELOG version {version} is missing docs/releases/v{version}.md."
            )

    for version, path in release_doc_versions.items():
        title_version = load_release_doc_title_version(path)
        if title_version != version:
            result.errors.append(
                f"{path} title does not match filename version v{version}."
            )
        if version not in changelog_set:
            result.errors.append(f"Release note {path.name} is not present in CHANGELOG.md.")

    if not tag_versions:
        result.warnings.append("No git tags were discovered; tag alignment could not be verified.")
    else:
        for version in sorted(tag_versions):
            if version not in changelog_set:
                result.errors.append(f"Git tag v{version} is not present in CHANGELOG.md.")
            if version not in release_doc_set:
                result.errors.append(f"Git tag v{version} is missing docs/releases/v{version}.md.")

    if github_repo:
        try:
            published_versions = load_published_release_versions(github_repo, github_token)
        except RuntimeError as exc:
            result.errors.append(str(exc))
        else:
            if not published_versions and not github_token:
                result.warnings.append(
                    "GitHub release alignment was skipped because no GitHub token was provided."
                )
            for version in sorted(published_versions):
                if version not in tag_versions:
                    result.errors.append(f"Published GitHub release v{version} is missing git tag v{version}.")
                if version not in changelog_set:
                    result.errors.append(f"Published GitHub release v{version} is not present in CHANGELOG.md.")
                if version not in release_doc_set:
                    result.errors.append(
                        f"Published GitHub release v{version} is missing docs/releases/v{version}.md."
                    )

    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate release evidence across CHANGELOG, docs/releases, git tags, and optional GitHub releases."
    )
    parser.add_argument("--check", action="store_true", help="Exit non-zero when validation fails.")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT, help="Repository root.")
    parser.add_argument("--changelog", type=Path, default=_DEFAULT_CHANGELOG, help="Path to CHANGELOG.md.")
    parser.add_argument("--releases-dir", type=Path, default=_DEFAULT_RELEASES_DIR, help="Path to docs/releases.")
    parser.add_argument("--github-repo", help="Repository in owner/name form for GitHub release verification.")
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="Environment variable that stores the GitHub token for release verification.",
    )
    args = parser.parse_args(argv)

    token = None
    if args.github_repo:
        token = os.environ.get(args.token_env)

    result = validate_release_evidence(
        repo_root=args.repo_root,
        changelog_path=args.changelog,
        releases_dir=args.releases_dir,
        github_repo=args.github_repo,
        github_token=token,
    )
    result.emit()
    if result.is_success:
        print("Release evidence matches CHANGELOG, docs/releases, git tags, and GitHub releases.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
