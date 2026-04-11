#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Check docs review due dates.

Scans all markdown files with front matter and checks if review is overdue.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXCLUDED_DIRS = {"_old_docs", "memx_spec_v3", "artifacts", "datasets"}
_EXCLUDED_FILES = {"README.md", "CHANGELOG.md", "LICENSE", "SECURITY.md", "CODE_OF_CONDUCT.md"}


@dataclass
class DocReviewStatus:
    file_path: Path
    rel_path: str
    owner: str | None
    status: str | None
    last_reviewed: str | None
    next_review_due: str | None
    days_until_review: int | None
    days_overdue: int | None


def _parse_front_matter(content: str) -> dict[str, Any]:
    """Parse YAML front matter."""
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
        # Remove quotes if present
        if len(value) >= 2:
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
        result[key] = value

    return result


def _scan_docs(root: Path) -> list[DocReviewStatus]:
    """Scan all markdown files for review due dates."""
    results = []
    today = datetime.now()

    for md_file in root.rglob("*.md"):
        # Skip excluded directories
        rel_path = md_file.relative_to(root)
        if any(part in _EXCLUDED_DIRS for part in rel_path.parts):
            continue
        # Skip excluded files
        if md_file.name in _EXCLUDED_FILES:
            continue
        # Skip acceptance records (they have different review cycle)
        if md_file.name.startswith("AC-"):
            continue

        content = md_file.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)

        # Only check files that have next_review_due field
        next_review_due = fm.get("next_review_due")
        if not next_review_due:
            continue

        last_reviewed = fm.get("last_reviewed_at") or fm.get("last_reviewed")

        days_until_review = None
        days_overdue = None

        try:
            due_date = datetime.strptime(next_review_due, "%Y-%m-%d")
            delta = due_date - today
            if delta.days < 0:
                days_overdue = abs(delta.days)
            else:
                days_until_review = delta.days
        except ValueError:
            pass

        results.append(DocReviewStatus(
            file_path=md_file,
            rel_path=str(rel_path),
            owner=fm.get("owner"),
            status=fm.get("status"),
            last_reviewed=last_reviewed,
            next_review_due=next_review_due,
            days_until_review=days_until_review,
            days_overdue=days_overdue,
        ))

    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check docs review due dates."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root to scan.",
    )
    parser.add_argument(
        "--max-days-overdue",
        type=int,
        default=30,
        help="Maximum days overdue before critical error.",
    )
    parser.add_argument(
        "--warn-days",
        type=int,
        default=7,
        help="Days until review to show as warning.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with error if any docs are critically overdue.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON.",
    )

    args = parser.parse_args(argv)

    results = _scan_docs(args.root)

    if args.json:
        import json
        data = []
        for r in results:
            data.append({
                "file": r.rel_path,
                "owner": r.owner,
                "status": r.status,
                "last_reviewed": r.last_reviewed,
                "next_review_due": r.next_review_due,
                "days_until_review": r.days_until_review,
                "days_overdue": r.days_overdue,
            })
        print(json.dumps(data, indent=2))
        return 0

    # Separate into categories
    overdue_critical = []
    overdue_warn = []
    upcoming = []
    ok = []

    for r in results:
        if r.days_overdue is not None:
            if r.days_overdue > args.max_days_overdue:
                overdue_critical.append(r)
            else:
                overdue_warn.append(r)
        elif r.days_until_review is not None and r.days_until_review <= args.warn_days:
            upcoming.append(r)
        else:
            ok.append(r)

    print(f"Docs scanned: {len(results)}")
    print()

    if overdue_critical:
        print("## CRITICAL - Review overdue > {} days".format(args.max_days_overdue))
        print()
        for r in overdue_critical:
            print(f"- {r.rel_path}")
            print(f"  - Owner: {r.owner or 'N/A'}")
            print(f"  - Last reviewed: {r.last_reviewed or 'N/A'}")
            print(f"  - Due: {r.next_review_due}")
            print(f"  - Days overdue: {r.days_overdue}")
        print()

    if overdue_warn:
        print("## OVERDUE - Review needed")
        print()
        for r in overdue_warn:
            print(f"- {r.rel_path}")
            print(f"  - Owner: {r.owner or 'N/A'}")
            print(f"  - Due: {r.next_review_due}")
            print(f"  - Days overdue: {r.days_overdue}")
        print()

    if upcoming:
        print("## UPCOMING - Review within {} days".format(args.warn_days))
        print()
        for r in upcoming:
            print(f"- {r.rel_path}")
            print(f"  - Due: {r.next_review_due}")
            print(f"  - Days until review: {r.days_until_review}")
        print()

    if ok:
        print(f"## OK - {len(ok)} docs within review schedule")
        print()

    if args.check and overdue_critical:
        print(
            f"ERROR: {len(overdue_critical)} docs are critically overdue",
            file=sys.stderr
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())