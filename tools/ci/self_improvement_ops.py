# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REFLECTIONS_DIR = _REPO_ROOT / ".workflow-cache" / "reflections"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load_json_files(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            loaded.setdefault("_source_path", str(path))
            records.append(loaded)
    return records


def _is_reviewed(record: dict[str, Any]) -> bool:
    state = str(record.get("review_state") or record.get("status") or "").lower()
    if state in {"approved", "reviewed", "accepted"}:
        return True
    return record.get("reviewed") is True


def export_curated_memory(
    reflections: Iterable[dict[str, Any]],
    *,
    include_unreviewed: bool = False,
) -> dict[str, Any]:
    """Export reviewed reflection lessons/actions without raw transcript content."""
    items: list[dict[str, Any]] = []
    skipped = 0
    for reflection in reflections:
        if not include_unreviewed and not _is_reviewed(reflection):
            skipped += 1
            continue
        lessons = [
            lesson for lesson in reflection.get("lessons", [])
            if isinstance(lesson, dict) and lesson.get("actionable", True)
        ]
        next_actions = [
            action for action in reflection.get("next_actions", [])
            if isinstance(action, dict)
        ]
        items.append({
            "session_id": reflection.get("session_id"),
            "task_id": reflection.get("task_id"),
            "objective": reflection.get("objective"),
            "lessons": lessons,
            "next_actions": next_actions,
            "sources": reflection.get("sources", []),
        })
    return {
        "kind": "CuratedMemorySnapshot",
        "schema_version": "1.0.0",
        "created_at": _now(),
        "review_policy": "reviewed_only" if not include_unreviewed else "include_unreviewed",
        "items": items,
        "summary": {"included": len(items), "skipped_unreviewed": skipped},
    }


def generate_skill_draft(reflection: dict[str, Any], *, author: str = "workflow-cookbook") -> dict[str, Any]:
    """Generate a review-only SkillDraftRecord from one ReflectionSummary."""
    session_id = str(reflection.get("session_id") or "unknown-session")
    actions = [
        str(item.get("action"))
        for item in reflection.get("next_actions", [])
        if isinstance(item, dict) and item.get("action")
    ]
    lessons = [
        str(item.get("observation"))
        for item in reflection.get("lessons", [])
        if isinstance(item, dict) and item.get("observation")
    ]
    steps = actions or lessons or ["Review the reflection and convert repeated workflow into a reusable skill."]
    return {
        "draft_id": f"SKILL-DRAFT-{session_id}",
        "source_session_id": session_id,
        "title": f"Skill draft from {session_id}",
        "problem": str(reflection.get("objective") or "Reusable workflow improvement candidate"),
        "proposed_steps": [
            {"order": index + 1, "step": step, "rationale": "Derived from reviewed reflection"}
            for index, step in enumerate(steps)
        ],
        "review_state": "draft",
        "linked_task_id": reflection.get("task_id"),
        "linked_acceptance_ids": [
            source.get("ref")
            for source in reflection.get("sources", [])
            if isinstance(source, dict) and source.get("type") == "acceptance"
        ],
        "linked_evidence_ids": [
            source.get("ref")
            for source in reflection.get("sources", [])
            if isinstance(source, dict) and source.get("type") == "evidence"
        ],
        "author": author,
        "created_at": _now(),
        "updated_at": _now(),
        "schema_version": "1.0.0",
    }


def build_recall_response(
    query: str,
    records: Iterable[dict[str, Any]],
    *,
    source_type: str = "reflection",
    limit: int = 5,
) -> dict[str, Any]:
    """Build a RecallResponse by matching query terms against summarized records."""
    terms = {term.lower() for term in query.split() if term.strip()}
    hits: list[dict[str, Any]] = []
    for record in records:
        haystack = json.dumps(record, ensure_ascii=False).lower()
        matched = sorted(term for term in terms if term in haystack)
        if not matched:
            continue
        source_id = str(record.get("session_id") or record.get("acceptance_id") or record.get("_source_path"))
        excerpt = str(record.get("objective") or record.get("summary") or record.get("title") or source_id)
        hits.append({
            "source_type": source_type,
            "source_id": source_id,
            "excerpt": excerpt[:240],
            "reason": f"Matched terms: {', '.join(matched)}",
            "relevance_score": min(1.0, len(matched) / max(len(terms), 1)),
            "created_at": record.get("created_at"),
        })
    hits.sort(key=lambda item: item.get("relevance_score", 0), reverse=True)
    selected = hits[:limit]
    return {
        "query": query,
        "summary": f"Found {len(hits)} related record(s); returning {len(selected)}.",
        "hits": selected,
        "stale": {"classification": "unknown", "evaluated_at": _now()},
        "total_hits": len(hits),
        "search_backend": "local-json-summary",
        "created_at": _now(),
        "schema_version": "1.0.0",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Operate optional adaptive improvement artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-memory")
    export_parser.add_argument("--reflections-dir", type=Path, default=_REFLECTIONS_DIR)
    export_parser.add_argument("--include-unreviewed", action="store_true")
    export_parser.add_argument("--output", type=Path)

    draft_parser = subparsers.add_parser("generate-skill-draft")
    draft_parser.add_argument("--reflection-json", type=Path, required=True)
    draft_parser.add_argument("--author", default="workflow-cookbook")
    draft_parser.add_argument("--output", type=Path)

    recall_parser = subparsers.add_parser("build-recall")
    recall_parser.add_argument("--query", required=True)
    recall_parser.add_argument("--reflections-dir", type=Path, default=_REFLECTIONS_DIR)
    recall_parser.add_argument("--limit", type=int, default=5)
    recall_parser.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    if args.command == "export-memory":
        payload = export_curated_memory(
            _load_json_files(args.reflections_dir),
            include_unreviewed=args.include_unreviewed,
        )
    elif args.command == "generate-skill-draft":
        payload = generate_skill_draft(
            json.loads(args.reflection_json.read_text(encoding="utf-8")),
            author=args.author,
        )
    else:
        payload = build_recall_response(
            args.query,
            _load_json_files(args.reflections_dir),
            limit=args.limit,
        )

    content = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
