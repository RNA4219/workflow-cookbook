from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .models import DocReviewStatus

_EXCLUDED_DIRS = {"_old_docs", "memx_spec_v3", "artifacts", "datasets", "examples"}
_TERMINAL_STATUSES = {"done", "completed", "resolved", "approved", "deployed", "rolled_back"}
_EXCLUDED_FILES = {"README.md", "CHANGELOG.md", "LICENSE", "SECURITY.md", "CODE_OF_CONDUCT.md"}


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
        if len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]
        result[key] = value

    return result


def _scan_docs(root: Path, *, today: date | None = None) -> list[DocReviewStatus]:
    """Scan all markdown files for review due dates."""
    results: list[DocReviewStatus] = []
    effective_today = today or date.today()

    for md_file in root.rglob("*.md"):
        rel_path = md_file.relative_to(root)
        if any(part in _EXCLUDED_DIRS for part in rel_path.parts):
            continue
        if md_file.name in _EXCLUDED_FILES or md_file.name.startswith("AC-"):
            continue

        content = md_file.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)
        status = str(fm.get("status", "")).split("#", 1)[0].strip().lower()
        if status in _TERMINAL_STATUSES:
            continue

        next_review_due = fm.get("next_review_due")
        if not next_review_due:
            continue

        last_reviewed = fm.get("last_reviewed_at") or fm.get("last_reviewed")
        days_until_review: int | None = None
        days_overdue: int | None = None
        try:
            due_date = datetime.strptime(next_review_due, "%Y-%m-%d").date()
            delta = due_date - effective_today
            if delta.days < 0:
                days_overdue = abs(delta.days)
            else:
                days_until_review = delta.days
        except ValueError:
            pass

        results.append(
            DocReviewStatus(
                file_path=md_file,
                rel_path=str(rel_path),
                owner=fm.get("owner"),
                status=fm.get("status"),
                last_reviewed=last_reviewed,
                next_review_due=next_review_due,
                days_until_review=days_until_review,
                days_overdue=days_overdue,
            )
        )

    return results


def _categorize(
    results: list[DocReviewStatus],
    *,
    max_days_overdue: int,
    warn_days: int,
) -> tuple[list[DocReviewStatus], list[DocReviewStatus], list[DocReviewStatus], list[DocReviewStatus]]:
    overdue_critical: list[DocReviewStatus] = []
    overdue_warn: list[DocReviewStatus] = []
    upcoming: list[DocReviewStatus] = []
    ok: list[DocReviewStatus] = []

    for result in results:
        if result.days_overdue is not None:
            if result.days_overdue > max_days_overdue:
                overdue_critical.append(result)
            else:
                overdue_warn.append(result)
        elif result.days_until_review is not None and result.days_until_review <= warn_days:
            upcoming.append(result)
        else:
            ok.append(result)
    return overdue_critical, overdue_warn, upcoming, ok
