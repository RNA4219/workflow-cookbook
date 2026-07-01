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
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.rel_path,
            "owner": self.owner,
            "status": self.status,
            "last_reviewed": self.last_reviewed,
            "next_review_due": self.next_review_due,
            "days_until_review": self.days_until_review,
            "days_overdue": self.days_overdue,
        }


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


def _scan_docs(root: Path, *, today: date | None = None) -> list[DocReviewStatus]:
    """Scan all markdown files for review due dates."""
    results = []
    effective_today = today or date.today()

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
            due_date = datetime.strptime(next_review_due, "%Y-%m-%d").date()
            delta = due_date - effective_today
            if delta.days < 0:
                days_overdue = abs(delta.days)
            else:
                days_until_review = delta.days
        except ValueError:
            # Invalid date format in next_review_due, skip calculation
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


def _categorize(
    results: list[DocReviewStatus],
    *,
    max_days_overdue: int,
    warn_days: int,
) -> tuple[list[DocReviewStatus], list[DocReviewStatus], list[DocReviewStatus], list[DocReviewStatus]]:
    overdue_critical = []
    overdue_warn = []
    upcoming = []
    ok = []

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


def _owner_summary(
    *,
    critical: list[DocReviewStatus],
    overdue: list[DocReviewStatus],
    upcoming: list[DocReviewStatus],
    ok: list[DocReviewStatus],
) -> dict[str, Any]:
    owners: dict[str, dict[str, Any]] = {}
    for category, items in (
        ("critical", critical),
        ("overdue", overdue),
        ("upcoming", upcoming),
        ("ok", ok),
    ):
        for item in items:
            owner = item.owner or "unowned"
            bucket = owners.setdefault(
                owner,
                {"critical": [], "overdue": [], "upcoming": [], "ok": [], "counts": {}},
            )
            bucket[category].append(item.to_dict())

    for bucket in owners.values():
        bucket["counts"] = {
            "critical": len(bucket["critical"]),
            "overdue": len(bucket["overdue"]),
            "upcoming": len(bucket["upcoming"]),
            "ok": len(bucket["ok"]),
        }
    return {"owners": owners}


def _build_nudges(
    *,
    critical: list[DocReviewStatus],
    overdue: list[DocReviewStatus],
    upcoming: list[DocReviewStatus],
    today: date,
    max_days_overdue: int,
) -> list[dict[str, Any]]:
    nudges: list[dict[str, Any]] = []
    for index, item in enumerate([*critical, *overdue, *upcoming], start=1):
        if item.days_overdue is not None and item.days_overdue > max_days_overdue:
            priority = "high"
            stale_days = item.days_overdue
            reason = f"{item.rel_path} review is {item.days_overdue} days overdue"
        elif item.days_overdue is not None:
            priority = "medium"
            stale_days = item.days_overdue
            reason = f"{item.rel_path} review is {item.days_overdue} days overdue"
        else:
            priority = "low"
            stale_days = 0
            reason = f"{item.rel_path} review is due in {item.days_until_review} days"
        expires_at = (
            f"{item.next_review_due}T00:00:00Z"
            if item.next_review_due
            else f"{today.isoformat()}T00:00:00Z"
        )
        nudges.append(
            {
                "$schema": "./schemas/periodic-nudge.schema.json",
                "nudge_id": f"NUDGE-DOCS-REVIEW-{today:%Y%m%d}-{index:03d}",
                "reason": reason,
                "target_kind": "docs",
                "target_ref": item.rel_path,
                "suggested_action": (
                    "Review the document, confirm whether its operational guidance is still current, "
                    "then update last_reviewed_at and next_review_due."
                ),
                "created_at": f"{today.isoformat()}T00:00:00Z",
                "priority": priority,
                "stale_days": stale_days,
                "threshold_days": max_days_overdue,
                "category": "review_due",
                "blocking": False,
                "expires_at": expires_at,
                "schema_version": "1.0.0",
            }
        )
    return nudges


def _review_update_plan(
    *,
    critical: list[DocReviewStatus],
    overdue: list[DocReviewStatus],
    today: date,
    review_window_days: int,
) -> dict[str, Any]:
    next_review_due = today + timedelta(days=review_window_days)
    updates = []
    for item in [*critical, *overdue]:
        updates.append(
            {
                "file": item.rel_path,
                "owner": item.owner,
                "current": {
                    "last_reviewed_at": item.last_reviewed,
                    "next_review_due": item.next_review_due,
                },
                "suggested": {
                    "last_reviewed_at": today.isoformat(),
                    "next_review_due": next_review_due.isoformat(),
                },
                "note": "Apply only after a human or agent review confirms the document is still current.",
            }
        )
    return {
        "generated_at": today.isoformat(),
        "review_window_days": review_window_days,
        "updates": updates,
    }


def _render_task_seed(
    *,
    owner_summary: dict[str, Any],
    today: date,
    max_days_overdue: int,
    warn_days: int,
) -> str:
    owners = owner_summary["owners"]
    task_id = f"{today:%Y%m%d}-docs-review"
    next_review_due = today + timedelta(days=30)
    lines = [
        "---",
        f"task_id: {task_id}",
        "intent_id: INT-DOCS-REVIEW-AUTOMATION",
        "owner: docs-core",
        "status: planned",
        f"last_reviewed_at: {today.isoformat()}",
        f"next_review_due: {next_review_due.isoformat()}",
        "---",
        "",
        f"# Docs Review Due Remediation {today.isoformat()}",
        "",
        "## Objective",
        "",
        "Review overdue workflow-cookbook documents by owner and refresh review metadata only after confirming each document remains current.",
        "",
        "## Scope",
        "",
        "- In: Markdown documents with `next_review_due` found by `check_docs_review_due.py`",
        "- Out: Acceptance records excluded by the checker and content rewrites unrelated to review freshness",
        "",
        "## Requirements",
        "",
        f"- Triage documents more than {max_days_overdue} days overdue first.",
        f"- Include upcoming reviews within {warn_days} days in owner planning.",
        "- Produce a review log or acceptance note for any large owner bucket before updating front matter.",
        "- Keep `last_reviewed_at` and `next_review_due` updates separate from unrelated content changes.",
        "",
        "## Owner Buckets",
        "",
        "| Owner | Critical | Overdue | Upcoming | OK |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for owner, bucket in sorted(
        owners.items(),
        key=lambda item: (
            -int(item[1]["counts"]["critical"]),
            -int(item[1]["counts"]["overdue"]),
            item[0],
        ),
    ):
        counts = bucket["counts"]
        lines.append(
            f"| {owner} | {counts['critical']} | {counts['overdue']} | {counts['upcoming']} | {counts['ok']} |"
        )
    lines.extend(
        [
            "",
            "## Local Commands",
            "",
            "```bash",
            "python tools/ci/check_docs_review_due.py --check --max-days-overdue 30",
            "python tools/ci/check_docs_review_due.py --owner-summary-json .ga/docs-review-owner-summary.json",
            "python tools/ci/check_docs_review_due.py --nudge-output .ga/docs-review-nudges.json",
            "python tools/ci/check_docs_review_due.py --review-update-plan-output .ga/docs-review-update-plan.json",
            "```",
            "",
            "## Deliverables",
            "",
            "- Owner-wise review status summary",
            "- PeriodicNudge JSON for overdue or upcoming review items",
            "- Review update plan with suggested front matter dates",
            "- Follow-up PR or acceptance record linking the reviewed documents",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_today(raw: str | None) -> date:
    if raw is None:
        return date.today()
    return datetime.strptime(raw, "%Y-%m-%d").date()


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
    parser.add_argument(
        "--today",
        help="Override today's date in YYYY-MM-DD format for reproducible reports.",
    )
    parser.add_argument(
        "--review-window-days",
        type=int,
        default=30,
        help="Days to add when suggesting next_review_due in the review update plan.",
    )
    parser.add_argument(
        "--owner-summary-json",
        type=Path,
        help="Write owner-grouped review summary JSON.",
    )
    parser.add_argument(
        "--task-seed-output",
        type=Path,
        help="Write a Task Seed draft for overdue document review remediation.",
    )
    parser.add_argument(
        "--nudge-output",
        type=Path,
        help="Write PeriodicNudge JSON array for overdue and upcoming reviews.",
    )
    parser.add_argument(
        "--review-update-plan-output",
        type=Path,
        help="Write a JSON plan with suggested review metadata updates.",
    )

    args = parser.parse_args(argv)

    today = _parse_today(args.today)
    results = _scan_docs(args.root, today=today)
    overdue_critical, overdue_warn, upcoming, ok = _categorize(
        results,
        max_days_overdue=args.max_days_overdue,
        warn_days=args.warn_days,
    )
    owner_summary = _owner_summary(
        critical=overdue_critical,
        overdue=overdue_warn,
        upcoming=upcoming,
        ok=ok,
    )

    if args.owner_summary_json:
        _write_json(args.owner_summary_json, owner_summary)
    if args.nudge_output:
        _write_json(
            args.nudge_output,
            _build_nudges(
                critical=overdue_critical,
                overdue=overdue_warn,
                upcoming=upcoming,
                today=today,
                max_days_overdue=args.max_days_overdue,
            ),
        )
    if args.review_update_plan_output:
        _write_json(
            args.review_update_plan_output,
            _review_update_plan(
                critical=overdue_critical,
                overdue=overdue_warn,
                today=today,
                review_window_days=args.review_window_days,
            ),
        )
    if args.task_seed_output:
        args.task_seed_output.parent.mkdir(parents=True, exist_ok=True)
        args.task_seed_output.write_text(
            _render_task_seed(
                owner_summary=owner_summary,
                today=today,
                max_days_overdue=args.max_days_overdue,
                warn_days=args.warn_days,
            ),
            encoding="utf-8",
        )

    if args.json:
        data = [r.to_dict() for r in results]
        print(json.dumps(data, indent=2))
        return 0

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
