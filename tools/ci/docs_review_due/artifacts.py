from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .models import DocReviewStatus


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
