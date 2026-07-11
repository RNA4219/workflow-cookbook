#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Check dependency exceptions registry for expired or overdue exceptions."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def parse_front_matter(content: str) -> dict[str, Any]:
    """Parse YAML front matter from markdown content."""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    fm_content = content[3:end]
    try:
        payload = yaml.safe_load(fm_content) or {}
        return payload if isinstance(payload, dict) else {}
    except yaml.YAMLError:
        return {}


def parse_exception_blocks(content: str) -> list[dict[str, Any]]:
    """Parse exception blocks from dependency_exceptions.md."""
    exceptions: list[dict[str, Any]] = []
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("### EXC-"):
            exc: dict[str, Any] = {
                "id": line.split(":")[0].replace("### ", "").strip(),
                "lines": [line],
            }
            i += 1
            while i < len(lines) and not lines[i].startswith("### EXC-"):
                exc["lines"].append(lines[i])
                i += 1
            exceptions.append(exc)
        else:
            i += 1

    # Extract fields from each exception block
    parsed: list[dict[str, Any]] = []
    for exc in exceptions:
        data = {"id": exc["id"]}
        for line in exc["lines"]:
            if line.startswith("- **期限**"):
                date_str = line.split(":", 1)[1].strip()
                try:
                    data["expires_at"] = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    pass
            elif line.startswith("- **再評価日**"):
                date_str = line.split(":", 1)[1].strip()
                try:
                    data["re_eval_at"] = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    pass
        parsed.append(data)

    return parsed


def check_exceptions(
    exceptions_path: Path,
    max_overdue_days: int = 30,
) -> tuple[bool, list[str]]:
    """Check dependency exceptions for expired or overdue entries."""
    issues = []

    if not exceptions_path.exists():
        issues.append(f"Exceptions registry not found: {exceptions_path}")
        return False, issues

    content = exceptions_path.read_text(encoding="utf-8")
    fm = parse_front_matter(content)

    # Check next_review_due in front matter
    if "next_review_due" in fm:
        review_due_val = fm["next_review_due"]
        if isinstance(review_due_val, datetime):
            review_due = review_due_val
        elif isinstance(review_due_val, str):
            review_due = datetime.strptime(review_due_val, "%Y-%m-%d")
        else:
            # datetime.date from YAML
            review_due = datetime.combine(review_due_val, datetime.min.time())
        overdue_days = (datetime.now() - review_due).days
        if overdue_days > max_overdue_days:
            issues.append(
                f"Registry review overdue: {overdue_days} days (max: {max_overdue_days})"
            )
        elif overdue_days > 0:
            issues.append(f"Registry review due: {review_due.strftime('%Y-%m-%d')}")

    # Check individual exceptions
    exceptions = parse_exception_blocks(content)

    now = datetime.now()
    for exc in exceptions:
        if "expires_at" in exc:
            if exc["expires_at"] < now:
                issues.append(f"{exc['id']}: expired at {exc['expires_at'].strftime('%Y-%m-%d')}")
            elif (exc["expires_at"] - now).days < 14:
                issues.append(
                    f"{exc['id']}: expires soon ({exc['expires_at'].strftime('%Y-%m-%d')})"
                )
        if "re_eval_at" in exc:
            if exc["re_eval_at"] < now:
                issues.append(f"{exc['id']}: re-evaluation overdue")

    if not exceptions:
        issues.append("No exceptions registered")

    # If only warning-level issues, return success
    critical_issues = [
        i for i in issues
        if "expired" in i or "overdue" in i.lower() and "days" in i
    ]
    return len(critical_issues) == 0, issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check dependency exceptions registry"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run check mode (exit 1 on failure)",
    )
    parser.add_argument(
        "--exceptions-path",
        type=Path,
        default=Path("docs/security/dependency_exceptions.md"),
        help="Path to dependency exceptions registry",
    )
    parser.add_argument(
        "--max-overdue-days",
        type=int,
        default=30,
        help="Maximum days overdue before failure (default: 30)",
    )
    args = parser.parse_args()

    success, issues = check_exceptions(
        args.exceptions_path,
        args.max_overdue_days,
    )

    if issues:
        for issue in issues:
            print(issue)

    if args.check and not success:
        print("Dependency exceptions check failed")
        return 1

    print("Dependency exceptions check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())