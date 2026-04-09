#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Generate release notes from CHANGELOG and docs/releases/.

Creates a summary of releases with major changes.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHANGELOG_PATH = _REPO_ROOT / "CHANGELOG.md"
_RELEASES_DIR = _REPO_ROOT / "docs" / "releases"


@dataclass
class ReleaseInfo:
    version: str
    date: str
    changes: list[str]
    file_path: Path | None = None


def _parse_changelog(content: str) -> list[ReleaseInfo]:
    """Parse CHANGELOG.md for release entries."""
    releases = []

    # Pattern: ## [version] or ## version
    pattern = r"##\s*\[?([v]?\d+\.\d+\.\d+)\]?\s*-?\s*(\d{4}-\d{2}-\d{2})?"

    current_version = None
    current_date = ""
    current_changes: list[str] = []

    for line in content.splitlines():
        match = re.match(pattern, line)
        if match:
            # Save previous release
            if current_version:
                releases.append(ReleaseInfo(
                    version=current_version,
                    date=current_date,
                    changes=current_changes,
                ))

            current_version = match.group(1)
            current_date = match.group(2) or ""
            current_changes = []
        elif current_version:
            # Collect changes
            stripped = line.strip()
            if stripped.startswith("- ") or stripped.startswith("* "):
                # Clean up change text
                change = stripped[2:].strip()
                # Remove leading number like "0075:"
                change = re.sub(r"^\d+:\s*", "", change)
                if change:
                    current_changes.append(change)

    # Save last release
    if current_version:
        releases.append(ReleaseInfo(
            version=current_version,
            date=current_date,
            changes=current_changes,
        ))

    return releases


def scan_releases_dir(releases_dir: Path) -> list[ReleaseInfo]:
    """Scan docs/releases/ for individual release files."""
    releases = []
    if not releases_dir.exists():
        return releases

    for release_file in releases_dir.glob("v*.md"):
        version = release_file.stem
        content = release_file.read_text(encoding="utf-8")

        # Extract date from front matter or content
        date = ""
        date_match = re.search(r"date:\s*(\d{4}-\d{2}-\d{2})", content)
        if date_match:
            date = date_match.group(1)

        # Extract changes
        changes = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("- ") or stripped.startswith("* "):
                changes.append(stripped[2:].strip())

        releases.append(ReleaseInfo(
            version=version,
            date=date,
            changes=changes,
            file_path=release_file,
        ))

    return releases


def generate_release_notes(
    changelog_releases: list[ReleaseInfo],
    file_releases: list[ReleaseInfo],
) -> str:
    """Generate consolidated release notes."""
    lines = [
        "# Release Notes",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d')}",
        "",
    ]

    # Merge and sort
    all_releases = {}

    for r in changelog_releases:
        all_releases[r.version] = r

    for r in file_releases:
        if r.version not in all_releases:
            all_releases[r.version] = r
        else:
            # Merge changes
            existing = all_releases[r.version]
            existing.changes.extend(r.changes)
            if not existing.date and r.date:
                existing.date = r.date
            existing.file_path = r.file_path

    # Sort by version (descending)
    sorted_versions = sorted(
        all_releases.keys(),
        key=lambda v: [int(x) for x in v.lstrip("v").split(".")],
        reverse=True,
    )

    # Output
    for version in sorted_versions:
        release = all_releases[version]
        lines.append(f"## {version}")
        if release.date:
            lines.append(f"Released: {release.date}")
        lines.append("")

        if release.changes:
            for change in release.changes[:10]:  # Limit to top 10
                lines.append(f"- {change}")
            if len(release.changes) > 10:
                lines.append(f"- ... and {len(release.changes) - 10} more")
        else:
            lines.append("No changes recorded.")

        lines.append("")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate release notes summary."
    )
    parser.add_argument(
        "--changelog",
        type=Path,
        default=_CHANGELOG_PATH,
        help="Path to CHANGELOG.md.",
    )
    parser.add_argument(
        "--releases-dir",
        type=Path,
        default=_RELEASES_DIR,
        help="Directory containing release files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (default: stdout).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Show only the latest release.",
    )

    args = parser.parse_args(argv)

    changelog_content = ""
    if args.changelog.exists():
        changelog_content = args.changelog.read_text(encoding="utf-8")

    changelog_releases = _parse_changelog(changelog_content)
    file_releases = scan_releases_dir(args.releases_dir)

    if args.latest and changelog_releases:
        # Only output latest
        latest = changelog_releases[0]
        output = f"# {latest.version}\n\n"
        if latest.date:
            output += f"Released: {latest.date}\n\n"
        for change in latest.changes[:5]:
            output += f"- {change}\n"
    else:
        output = generate_release_notes(changelog_releases, file_releases)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"Wrote release notes to {args.output}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())