#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Extract upstream diff changes for weekly review.

Reads UPSTREAM.md and UPSTREAM_WEEKLY_LOG.md, extracts changes since last review.
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
_UPSTREAM_MD = _REPO_ROOT / "docs" / "UPSTREAM.md"
_WEEKLY_LOG = _REPO_ROOT / "docs" / "UPSTREAM_WEEKLY_LOG.md"


@dataclass
class UpstreamChange:
    date: str
    source: str
    description: str
    impact: str


def _parse_upstream_md(content: str) -> list[UpstreamChange]:
    """Parse UPSTREAM.md for change entries."""
    changes = []

    # Pattern: ## YYYY-MM-DD - Source
    # followed by description lines
    pattern = r"##\s+(\d{4}-\d{2}-\d{2})\s*-\s*([^\n]+)\n([\s\S]*?)(?=##\s+\d{4}|$)"

    for match in re.finditer(pattern, content):
        date = match.group(1)
        source = match.group(2).strip()
        description = match.group(3).strip()

        # Extract impact if present
        impact = "unknown"
        impact_match = re.search(r"\*\*Impact:\*\*\s*([^\n]+)", description)
        if impact_match:
            impact = impact_match.group(1).strip()

        changes.append(UpstreamChange(
            date=date,
            source=source,
            description=description,
            impact=impact,
        ))

    return changes


def _parse_weekly_log(content: str) -> dict[str, str]:
    """Parse UPSTREAM_WEEKLY_LOG.md for last reviewed dates."""
    reviewed = {}

    # Pattern: | source | date | ...
    for line in content.splitlines():
        if line.startswith("|") and "Source" not in line and "---" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 2:
                source = parts[0]
                date = parts[1]
                reviewed[source] = date

    return reviewed


def extract_new_changes(
    changes: list[UpstreamChange],
    reviewed: dict[str, str],
) -> list[UpstreamChange]:
    """Extract changes newer than last reviewed date."""
    new_changes = []

    for change in changes:
        last_reviewed = reviewed.get(change.source)
        if not last_reviewed or change.date > last_reviewed:
            new_changes.append(change)

    return sorted(new_changes, key=lambda c: c.date, reverse=True)


def generate_weekly_update(new_changes: list[UpstreamChange]) -> str:
    """Generate markdown for weekly update."""
    lines = [
        f"# Upstream Weekly Log - {datetime.now().strftime('%Y-%m-%d')}",
        "",
    ]

    if not new_changes:
        lines.append("No new upstream changes detected.")
        return "\n".join(lines)

    lines.append(f"## New Changes ({len(new_changes)})")
    lines.append("")

    for change in new_changes:
        lines.append(f"### {change.date} - {change.source}")
        lines.append("")
        lines.append(change.description)
        lines.append("")
        lines.append(f"**Impact:** {change.impact}")
        lines.append("")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract upstream diff changes for weekly review."
    )
    parser.add_argument(
        "--upstream-md",
        type=Path,
        default=_UPSTREAM_MD,
        help="Path to UPSTREAM.md.",
    )
    parser.add_argument(
        "--weekly-log",
        type=Path,
        default=_WEEKLY_LOG,
        help="Path to UPSTREAM_WEEKLY_LOG.md.",
    )
    parser.add_argument(
        "--update-log",
        action="store_true",
        help="Update weekly log with new changes.",
    )
    parser.add_argument(
        "--print-changes",
        action="store_true",
        help="Print new changes to stdout.",
    )

    args = parser.parse_args(argv)

    # Parse files
    upstream_content = ""
    if args.upstream_md.exists():
        upstream_content = args.upstream_md.read_text(encoding="utf-8")

    weekly_content = ""
    if args.weekly_log.exists():
        weekly_content = args.weekly_log.read_text(encoding="utf-8")

    changes = _parse_upstream_md(upstream_content)
    reviewed = _parse_weekly_log(weekly_content)
    new_changes = extract_new_changes(changes, reviewed)

    if args.print_changes or not args.update_log:
        if new_changes:
            print(f"New upstream changes: {len(new_changes)}")
            for change in new_changes:
                print(f"  - {change.date} {change.source}: {change.impact}")
        else:
            print("No new upstream changes detected.")
        print()

    if args.update_log and new_changes:
        update = generate_weekly_update(new_changes)

        # Prepend to existing log
        if args.weekly_log.exists():
            existing = args.weekly_log.read_text(encoding="utf-8")
            new_content = update + "\n\n---\n\n" + existing
        else:
            new_content = update

        args.weekly_log.parent.mkdir(parents=True, exist_ok=True)
        args.weekly_log.write_text(new_content, encoding="utf-8")
        print(f"Updated {args.weekly_log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())