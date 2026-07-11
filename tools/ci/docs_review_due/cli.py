from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

from .artifacts import _build_nudges, _owner_summary, _render_task_seed, _review_update_plan, _write_json
from .models import DocReviewStatus
from .scanner import _categorize, _scan_docs

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_today(raw: str | None) -> date:
    if raw is None:
        return date.today()
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _render_report(
    *,
    results: list[DocReviewStatus],
    overdue_critical: list[DocReviewStatus],
    overdue_warn: list[DocReviewStatus],
    upcoming: list[DocReviewStatus],
    ok: list[DocReviewStatus],
    max_days_overdue: int,
    warn_days: int,
) -> None:
    print(f"Docs scanned: {len(results)}")
    print()

    if overdue_critical:
        print(f"## CRITICAL - Review overdue > {max_days_overdue} days")
        print()
        for result in overdue_critical:
            print(f"- {result.rel_path}")
            print(f"  - Owner: {result.owner or 'N/A'}")
            print(f"  - Last reviewed: {result.last_reviewed or 'N/A'}")
            print(f"  - Due: {result.next_review_due}")
            print(f"  - Days overdue: {result.days_overdue}")
        print()

    if overdue_warn:
        print("## OVERDUE - Review needed")
        print()
        for result in overdue_warn:
            print(f"- {result.rel_path}")
            print(f"  - Owner: {result.owner or 'N/A'}")
            print(f"  - Due: {result.next_review_due}")
            print(f"  - Days overdue: {result.days_overdue}")
        print()

    if upcoming:
        print(f"## UPCOMING - Review within {warn_days} days")
        print()
        for result in upcoming:
            print(f"- {result.rel_path}")
            print(f"  - Due: {result.next_review_due}")
            print(f"  - Days until review: {result.days_until_review}")
        print()

    if ok:
        print(f"## OK - {len(ok)} docs within review schedule")
        print()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check docs review due dates.")
    parser.add_argument("--root", type=Path, default=_REPO_ROOT, help="Repository root to scan.")
    parser.add_argument(
        "--max-days-overdue",
        type=int,
        default=30,
        help="Maximum days overdue before critical error.",
    )
    parser.add_argument("--warn-days", type=int, default=7, help="Days until review to show as warning.")
    parser.add_argument("--check", action="store_true", help="Exit with error if any docs are critically overdue.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    parser.add_argument("--today", help="Override today's date in YYYY-MM-DD format for reproducible reports.")
    parser.add_argument(
        "--review-window-days",
        type=int,
        default=30,
        help="Days to add when suggesting next_review_due in the review update plan.",
    )
    parser.add_argument("--owner-summary-json", type=Path, help="Write owner-grouped review summary JSON.")
    parser.add_argument("--task-seed-output", type=Path, help="Write a Task Seed draft for overdue document review remediation.")
    parser.add_argument("--nudge-output", type=Path, help="Write PeriodicNudge JSON array for overdue and upcoming reviews.")
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
        print(json.dumps([result.to_dict() for result in results], indent=2))
        return 0

    _render_report(
        results=results,
        overdue_critical=overdue_critical,
        overdue_warn=overdue_warn,
        upcoming=upcoming,
        ok=ok,
        max_days_overdue=args.max_days_overdue,
        warn_days=args.warn_days,
    )
    if args.check and overdue_critical:
        print(f"ERROR: {len(overdue_critical)} docs are critically overdue", file=sys.stderr)
        return 1
    return 0
