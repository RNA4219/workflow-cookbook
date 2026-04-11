#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Generate Acceptance Index with status summary and release mapping.

Scans docs/acceptance/ directory and generates INDEX.md with:
- Status summary (approved/rejected/draft counts)
- Table of all acceptance records with release linkage
- Release mapping section showing acceptance-to-release relationships
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ACCEPTANCE_DIR = _REPO_ROOT / "docs" / "acceptance"
_INDEX_PATH = _ACCEPTANCE_DIR / "INDEX.md"
_CHANGELOG_PATH = _REPO_ROOT / "CHANGELOG.md"


@dataclass
class AcceptanceInfo:
    acceptance_id: str
    task_id: str
    intent_id: str
    status: str
    reviewed_at: str
    file_path: Path
    release: str = "unlinked"


@dataclass
class ReleaseInfo:
    version: str
    date: datetime


def _parse_front_matter(content: str) -> dict[str, Any]:
    """Parse YAML front matter from markdown content."""
    if not content.startswith("---"):
        return {}

    end_match = re.search(r"\n---\s*$|\n---\s*\n", content[3:])
    if not end_match:
        return {}

    front_matter = content[3 : end_match.start() + 3]
    result: dict[str, Any] = {}

    for line in front_matter.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2:
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
        result[key] = value

    return result


def _parse_changelog_releases(changelog_path: Path) -> list[ReleaseInfo]:
    """Parse CHANGELOG.md and extract release versions with dates."""
    releases = []
    if not changelog_path.exists():
        return releases

    content = changelog_path.read_text(encoding="utf-8")
    # Pattern: ## X.Y.Z - YYYY-MM-DD or ## [X.Y.Z] - YYYY-MM-DD
    pattern = r"##\s*\[?(\d+\.\d+\.\d+)\]?\s*-\s*(\d{4}-\d{2}-\d{2})"

    for match in re.finditer(pattern, content):
        version = match.group(1)
        date_str = match.group(2)
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            releases.append(ReleaseInfo(version=version, date=date))
        except ValueError:
            continue

    # Sort by date descending (newest first)
    releases.sort(key=lambda r: r.date, reverse=True)
    return releases


def _map_acceptance_to_release(
    reviewed_at: str, releases: list[ReleaseInfo]
) -> str:
    """Map an acceptance date to the most likely release version."""
    if not reviewed_at or not releases:
        return "unlinked"

    try:
        acc_date = datetime.strptime(reviewed_at, "%Y-%m-%d")
    except ValueError:
        return "unlinked"

    # Find the release that matches the acceptance date
    # Acceptance on the same day as release = belongs to that release
    # Acceptance after release but before next release = belongs to that release
    for i, rel in enumerate(releases):
        # Same day mapping
        if acc_date.date() == rel.date.date():
            return rel.version
        # After this release but before the previous one (if exists)
        if i + 1 < len(releases):
            prev_rel = releases[i + 1]
            if acc_date.date() < rel.date.date() and acc_date.date() >= prev_rel.date.date():
                return prev_rel.version
        # After the newest release (unreleased / next version)
        if acc_date.date() > rel.date.date():
            return "unreleased"

    return "unlinked"


def scan_acceptances(
    acceptance_dir: Path, releases: list[ReleaseInfo] | None = None
) -> list[AcceptanceInfo]:
    """Scan all acceptance files and map to releases."""
    acceptances = []
    if not acceptance_dir.exists():
        return acceptances

    for acc_file in acceptance_dir.glob("AC-*.md"):
        content = acc_file.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)

        acceptance_id = fm.get("acceptance_id", "")
        if not acceptance_id:
            continue

        reviewed_at = fm.get("reviewed_at", fm.get("reviewed_by", ""))
        release = _map_acceptance_to_release(reviewed_at, releases or [])

        acceptances.append(AcceptanceInfo(
            acceptance_id=acceptance_id,
            task_id=fm.get("task_id", ""),
            intent_id=fm.get("intent_id", ""),
            status=fm.get("status", "unknown").lower(),
            reviewed_at=reviewed_at,
            file_path=acc_file,
            release=release,
        ))

    return acceptances


def generate_index_markdown(
    acceptances: list[AcceptanceInfo], releases: list[ReleaseInfo]
) -> str:
    """Generate INDEX.md content with summary, table, and release mapping.

    Links are relative to docs/acceptance/INDEX.md location.
    """
    lines = ["# Acceptance Index", ""]

    # Gate Hardening Notes (preserve existing section)
    lines.append("## Gate Hardening Notes")
    lines.append("")
    lines.append("- **Gate Hardening** (RG-001〜RG-006): AC-20260411-01〜03 で完了")
    lines.append("  - AC-20260411-01: RG-002〜RG-006 (docs/Birdseye)")
    lines.append("  - AC-20260411-02: RG-001 (metrics gate必須化)")
    lines.append("  - AC-20260411-03: RG-002/RG-003 最終調整")
    lines.append("")

    # Status summary
    status_counts = Counter(a.status for a in acceptances)
    total = len(acceptances)

    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count | Percentage |")
    lines.append("| --- | --- | --- |")
    for status in ["approved", "rejected", "draft", "unknown"]:
        count = status_counts.get(status, 0)
        pct = f"{count / total * 100:.1f}%" if total > 0 else "0%"
        lines.append(f"| {status} | {count} | {pct} |")
    lines.append(f"| **Total** | **{total}** | **100%** |")
    lines.append("")

    # Release mapping section
    release_counts: dict[str, list[AcceptanceInfo]] = {}
    for acc in acceptances:
        if acc.release not in release_counts:
            release_counts[acc.release] = []
        release_counts[acc.release].append(acc)

    lines.append("## Release Mapping")
    lines.append("")
    lines.append("| Release | Acceptances | Docs |")
    lines.append("| --- | --- | --- |")

    # Show releases with their acceptances
    # Links are relative to docs/acceptance/INDEX.md -> ../releases/
    for rel in releases:
        accs_for_rel = release_counts.get(rel.version, [])
        acc_list = ", ".join(a.acceptance_id for a in accs_for_rel) or "-"
        rel_doc_path = _REPO_ROOT / "docs" / "releases" / f"v{rel.version}.md"
        rel_doc_link = f"[v{rel.version}](../releases/v{rel.version}.md)" if rel_doc_path.exists() else rel.version
        lines.append(f"| {rel_doc_link} | {acc_list} | {rel.date.strftime('%Y-%m-%d')} |")

    # Show unlinked/unreleased acceptances
    for release_key in ["unlinked", "unreleased"]:
        if release_key in release_counts and release_counts[release_key]:
            acc_list = ", ".join(a.acceptance_id for a in release_counts[release_key])
            lines.append(f"| {release_key} | {acc_list} | - |")

    lines.append("")

    # Records table with release column
    lines.append("## Records")
    lines.append("")
    lines.append("| Acceptance | Task | Intent | Status | Reviewed | Release | File |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    # Sort by acceptance_id descending (newest first)
    sorted_acceptances = sorted(acceptances, key=lambda a: a.acceptance_id, reverse=True)

    for acc in sorted_acceptances:
        # File link: same directory, just filename
        file_link = f"[{acc.file_path.name}]({acc.file_path.name})"
        release_display = acc.release
        if acc.release not in ["unlinked", "unreleased"]:
            release_display = f"[v{acc.release}](../releases/v{acc.release}.md)"
        lines.append(
            f"| {acc.acceptance_id} | {acc.task_id} | {acc.intent_id} | "
            f"{acc.status} | {acc.reviewed_at} | {release_display} | {file_link} |"
        )

    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Acceptance Index with status summary and release mapping."
    )
    parser.add_argument(
        "--acceptance-dir",
        type=Path,
        default=_ACCEPTANCE_DIR,
        help="Directory containing Acceptance Record files.",
    )
    parser.add_argument(
        "--changelog",
        type=Path,
        default=_CHANGELOG_PATH,
        help="Path to CHANGELOG.md for release mapping.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_INDEX_PATH,
        help="Output path for INDEX.md.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print to stdout instead of file.",
    )

    args = parser.parse_args(argv)

    releases = _parse_changelog_releases(args.changelog)
    acceptances = scan_acceptances(args.acceptance_dir, releases)
    markdown = generate_index_markdown(acceptances, releases)

    if args.print:
        print(markdown)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote acceptance index to {args.output} ({len(acceptances)} records, {len(releases)} releases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())