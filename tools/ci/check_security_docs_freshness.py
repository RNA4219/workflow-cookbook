#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Check security docs freshness.

Validates that security docs are reviewed according to schedule.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SECURITY_DIR = _REPO_ROOT / "docs" / "security"
_RELEASES_DIR = _REPO_ROOT / "docs" / "releases"
_GOVERNANCE_DIR = _REPO_ROOT / "governance"


@dataclass
class DocFreshness:
    file_path: Path
    last_reviewed: str | None
    next_review_due: str | None
    days_overdue: int | None
    releases_since_review: int


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
        if len(value) >= 2:
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
        result[key] = value

    return result


def _count_releases_since_date(releases_dir: Path, since_date: str) -> int:
    """Count releases since a given date."""
    if not releases_dir.exists():
        return 0

    count = 0
    try:
        since = datetime.strptime(since_date, "%Y-%m-%d")
    except ValueError:
        return 0

    for release_file in releases_dir.glob("v*.md"):
        content = release_file.read_text(encoding="utf-8")
        date_match = re.search(r"date:\s*(\d{4}-\d{2}-\d{2})", content)
        if date_match:
            try:
                release_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                if release_date > since:
                    count += 1
            except ValueError:
                continue

    return count


def check_security_docs(
    security_dir: Path,
    releases_dir: Path,
    max_days_overdue: int = 90,
) -> list[DocFreshness]:
    """Check freshness of security docs."""
    results = []

    if not security_dir.exists():
        return results

    today = datetime.now()

    for doc_file in security_dir.glob("*.md"):
        content = doc_file.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)

        last_reviewed = fm.get("last_reviewed_at") or fm.get("last_reviewed")
        next_review_due = fm.get("next_review_due")

        days_overdue = None
        if next_review_due:
            try:
                due_date = datetime.strptime(next_review_due, "%Y-%m-%d")
                if today > due_date:
                    days_overdue = (today - due_date).days
            except ValueError:
                pass

        releases_since = 0
        if last_reviewed:
            releases_since = _count_releases_since_date(releases_dir, last_reviewed)

        results.append(DocFreshness(
            file_path=doc_file,
            last_reviewed=last_reviewed,
            next_review_due=next_review_due,
            days_overdue=days_overdue,
            releases_since_review=releases_since,
        ))

    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check security docs freshness."
    )
    parser.add_argument(
        "--security-dir",
        type=Path,
        default=_SECURITY_DIR,
        help="Security docs directory.",
    )
    parser.add_argument(
        "--releases-dir",
        type=Path,
        default=_RELEASES_DIR,
        help="Releases directory.",
    )
    parser.add_argument(
        "--max-days-overdue",
        type=int,
        default=90,
        help="Maximum days overdue before warning.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with error if any docs are overdue.",
    )

    args = parser.parse_args(argv)

    results = check_security_docs(
        args.security_dir,
        args.releases_dir,
        args.max_days_overdue,
    )

    errors = 0

    print(f"Security docs scanned: {len(results)}")
    print()

    for result in results:
        status = "OK"
        if result.days_overdue is not None:
            if result.days_overdue > args.max_days_overdue:
                status = "CRITICAL"
                errors += 1
            elif result.days_overdue > 0:
                status = "OVERDUE"

        print(f"{result.file_path.name}:")
        print(f"  Last reviewed: {result.last_reviewed or 'N/A'}")
        print(f"  Next review due: {result.next_review_due or 'N/A'}")
        print(f"  Releases since review: {result.releases_since_review}")
        if result.days_overdue is not None:
            print(f"  Days overdue: {result.days_overdue}")
        print(f"  Status: {status}")
        print()

    if args.check and errors:
        print(f"ERROR: {errors} security docs are critically overdue", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())