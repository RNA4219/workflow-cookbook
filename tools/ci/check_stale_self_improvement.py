#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Check for stale reflections and generate nudges.

Identifies reflections that haven been reviewed or updated beyond threshold
and generates nudge records for periodic review.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REFLECTIONS_DIR = _REPO_ROOT / ".workflow-cache" / "reflections"
_NUDGE_DIR = _REPO_ROOT / ".workflow-cache" / "nudges"
_SKILL_DRAFTS_DIR = _REPO_ROOT / ".workflow-cache" / "skill-drafts"


def parse_front_matter(content: str) -> dict[str, Any]:
    """Parse YAML front matter from markdown content."""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    fm_content = content[3:end]
    try:
        return yaml.safe_load(fm_content) or {}
    except yaml.YAMLError:
        return {}


def load_json_reflection(path: Path) -> dict[str, Any] | None:
    """Load JSON reflection file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def check_stale_reflections(
    max_age_days: int = 30,
    reflections_dir: Path = _REFLECTIONS_DIR,
    repo_root: Path = _REPO_ROOT,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Check for stale reflections and return nudges."""
    nudges: list[dict[str, Any]] = []
    warnings: list[str] = []

    if not reflections_dir.exists():
        warnings.append(f"Reflections directory not found: {reflections_dir}")
        return nudges, warnings

    now = datetime.now()
    threshold = now - timedelta(days=max_age_days)

    for reflection_file in reflections_dir.glob("*.json"):
        reflection = load_json_reflection(reflection_file)
        if reflection is None:
            warnings.append(f"Invalid reflection file: {reflection_file}")
            continue

        created_at_str = reflection.get("created_at", "")
        if not created_at_str:
            warnings.append(f"Reflection missing created_at: {reflection_file}")
            continue

        try:
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        except ValueError:
            warnings.append(f"Invalid created_at format: {reflection_file}")
            continue

        age_days = (now - created_at).days

        if created_at < threshold:
            # Handle relative_to for paths outside repo_root
            try:
                target_ref = str(reflection_file.relative_to(repo_root))
            except ValueError:
                target_ref = str(reflection_file)

            nudge = {
                "nudge_id": f"NUDGE-{reflection_file.stem}-{now.strftime('%Y%m%d')}",
                "reason": f"Reflection {reflection_file.stem} is {age_days} days old (threshold: {max_age_days})",
                "target_kind": "reflection",
                "target_ref": target_ref,
                "suggested_action": "Review and update or archive reflection",
                "created_at": now.isoformat(),
                "priority": "medium" if age_days < 60 else "high",
                "stale_days": age_days,
                "threshold_days": max_age_days,
                "category": "stale_content",
                "blocking": False,
            }
            nudges.append(nudge)

    return nudges, warnings


def check_stale_skill_drafts(
    max_age_days: int = 14,
    skill_drafts_dir: Path = _SKILL_DRAFTS_DIR,
    repo_root: Path = _REPO_ROOT,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Check for stale skill drafts in review state."""
    nudges: list[dict[str, Any]] = []
    warnings: list[str] = []

    if not skill_drafts_dir.exists():
        return nudges, warnings

    now = datetime.now()
    threshold = now - timedelta(days=max_age_days)

    for draft_file in skill_drafts_dir.glob("*.json"):
        draft = load_json_reflection(draft_file)
        if draft is None:
            continue

        review_state = draft.get("review_state", "")
        if review_state not in ("draft", "review"):
            continue

        created_at_str = draft.get("created_at", "")
        if not created_at_str:
            continue

        try:
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        age_days = (now - created_at).days

        if created_at < threshold:
            # Handle relative_to for paths outside repo_root
            try:
                target_ref = str(draft_file.relative_to(repo_root))
            except ValueError:
                target_ref = str(draft_file)

            nudge = {
                "nudge_id": f"NUDGE-{draft_file.stem}-{now.strftime('%Y%m%d')}",
                "reason": f"Skill draft {draft_file.stem} in {review_state} state for {age_days} days",
                "target_kind": "skill_draft",
                "target_ref": target_ref,
                "suggested_action": f"Move to {review_state == 'draft' and 'review' or 'approved/rejected'} state",
                "created_at": now.isoformat(),
                "priority": "medium",
                "stale_days": age_days,
                "threshold_days": max_age_days,
                "category": "unfinished_task",
                "blocking": False,
            }
            nudges.append(nudge)

    return nudges, warnings


def write_nudges(nudges: list[dict[str, Any]], output_dir: Path) -> int:
    """Write nudge records to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for nudge in nudges:
        nudge_path = output_dir / f"{nudge['nudge_id']}.json"
        nudge_path.write_text(
            json.dumps(nudge, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check for stale reflections and skill drafts, generate nudges"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run check mode (exit 1 on stale content)",
    )
    parser.add_argument(
        "--reflection-max-age",
        type=int,
        default=30,
        help="Max age in days for reflections before nudge (default: 30)",
    )
    parser.add_argument(
        "--skill-draft-max-age",
        type=int,
        default=14,
        help="Max age in days for skill drafts in review state (default: 14)",
    )
    parser.add_argument(
        "--write-nudges",
        action="store_true",
        help="Write nudge records to .workflow-cache/nudges/",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = parser.parse_args()

    reflection_nudges, reflection_warnings = check_stale_reflections(
        max_age_days=args.reflection_max_age,
    )
    skill_nudges, skill_warnings = check_stale_skill_drafts(
        max_age_days=args.skill_draft_max_age,
    )

    all_nudges = reflection_nudges + skill_nudges
    all_warnings = reflection_warnings + skill_warnings

    if args.write_nudges and all_nudges:
        written = write_nudges(all_nudges, _NUDGE_DIR)
        print(f"Wrote {written} nudge records to {_NUDGE_DIR}")

    if args.json:
        output = {
            "nudges": all_nudges,
            "warnings": all_warnings,
            "summary": {
                "reflection_nudges": len(reflection_nudges),
                "skill_draft_nudges": len(skill_nudges),
                "total_nudges": len(all_nudges),
                "warnings": len(all_warnings),
            },
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        if all_warnings:
            for w in all_warnings:
                print(f"WARNING: {w}")
        if all_nudges:
            for n in all_nudges:
                print(f"NUDGE: {n['nudge_id']} - {n['reason']}")
        else:
            print("No stale reflections or skill drafts detected")

    if args.check and all_nudges:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())