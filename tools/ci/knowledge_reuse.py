#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Knowledge Reuse CLI.

Provides cross-referencing between releases, acceptances, incidents,
and related documentation for knowledge reuse.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RELEASES_DIR = _REPO_ROOT / "docs" / "releases"
_ACCEPTANCE_DIR = _REPO_ROOT / "docs" / "acceptance"
_INCIDENTS_DIR = _REPO_ROOT / "docs"
_CACHE_DIR = _REPO_ROOT / ".workflow-cache"


@dataclass
class KnowledgeEntry:
    entry_type: str  # release, acceptance, incident
    entry_id: str
    title: str
    date: str
    related_entries: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    file_path: str = ""


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


def _extract_title(content: str) -> str:
    """Extract title from markdown."""
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled"


def scan_releases() -> list[KnowledgeEntry]:
    """Scan release files."""
    entries = []
    if not _RELEASES_DIR.exists():
        return entries

    for release_file in _RELEASES_DIR.glob("v*.md"):
        content = release_file.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)
        title = _extract_title(content)

        entries.append(KnowledgeEntry(
            entry_type="release",
            entry_id=release_file.stem,
            title=title,
            date=fm.get("date", ""),
            file_path=str(release_file.relative_to(_REPO_ROOT)),
        ))

    return entries


def scan_acceptances() -> list[KnowledgeEntry]:
    """Scan acceptance files."""
    entries = []
    if not _ACCEPTANCE_DIR.exists():
        return entries

    for acc_file in _ACCEPTANCE_DIR.glob("AC-*.md"):
        content = acc_file.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)

        acceptance_id = fm.get("acceptance_id", "")
        if not acceptance_id:
            continue

        entries.append(KnowledgeEntry(
            entry_type="acceptance",
            entry_id=acceptance_id,
            title=f"Acceptance: {acceptance_id}",
            date=fm.get("reviewed_at", ""),
            related_entries=[fm.get("task_id", "")] if fm.get("task_id") else [],
            file_path=str(acc_file.relative_to(_REPO_ROOT)),
        ))

    return entries


def scan_incidents() -> list[KnowledgeEntry]:
    """Scan incident files."""
    entries = []

    for incident_file in _INCIDENTS_DIR.glob("IN-*.md"):
        content = incident_file.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)
        title = _extract_title(content)

        entries.append(KnowledgeEntry(
            entry_type="incident",
            entry_id=incident_file.stem,
            title=title,
            date=fm.get("date", fm.get("last_reviewed_at", "")),
            tags=fm.get("tags", "").split(",") if fm.get("tags") else [],
            file_path=str(incident_file.relative_to(_REPO_ROOT)),
        ))

    return entries


def build_knowledge_index() -> dict[str, Any]:
    """Build cross-referenced knowledge index."""
    releases = scan_releases()
    acceptances = scan_acceptances()
    incidents = scan_incidents()

    # Build task-to-acceptance mapping
    task_to_acceptance: dict[str, str] = {}
    for acc in acceptances:
        for related in acc.related_entries:
            task_to_acceptance[related] = acc.entry_id

    # Build index
    index = {
        "entries": [],
        "by_type": {
            "release": [],
            "acceptance": [],
            "incident": [],
        },
        "by_tag": {},
    }

    for entry in releases + acceptances + incidents:
        entry_dict = {
            "type": entry.entry_type,
            "id": entry.entry_id,
            "title": entry.title,
            "date": entry.date,
            "related": entry.related_entries,
            "tags": entry.tags,
            "path": entry.file_path,
        }
        index["entries"].append(entry_dict)
        index["by_type"][entry.entry_type].append(entry.entry_id)

        for tag in entry.tags:
            index["by_tag"].setdefault(tag.strip(), []).append(entry.entry_id)

    return index


def search_knowledge(index: dict[str, Any], query: str) -> list[dict[str, Any]]:
    """Search knowledge index."""
    results = []
    query_lower = query.lower()

    for entry in index["entries"]:
        if query_lower in entry["title"].lower():
            results.append(entry)
        elif query_lower in entry["id"].lower():
            results.append(entry)
        elif any(query_lower in tag.lower() for tag in entry["tags"]):
            results.append(entry)

    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Knowledge Reuse CLI for cross-referencing releases, acceptances, incidents."
    )
    parser.add_argument(
        "command",
        choices=["index", "search", "list"],
        help="Command to execute.",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="",
        help="Search query.",
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        choices=["release", "acceptance", "incident", "all"],
        default="all",
        help="Filter by entry type.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for index command.",
    )

    args = parser.parse_args(argv)

    if args.command == "index":
        index = build_knowledge_index()

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote knowledge index to {args.output}")
        else:
            print(json.dumps(index, ensure_ascii=False, indent=2))

    elif args.command == "search":
        if not args.query:
            print("ERROR: --query required for search", file=sys.stderr)
            return 1

        index = build_knowledge_index()
        results = search_knowledge(index, args.query)

        print(f"Found {len(results)} results for '{args.query}':")
        for entry in results:
            print(f"  [{entry['type']}] {entry['id']}: {entry['title']}")

    elif args.command == "list":
        index = build_knowledge_index()

        entry_type = args.type if args.type != "all" else None

        for entry in index["entries"]:
            if entry_type and entry["type"] != entry_type:
                continue
            print(f"[{entry['type']}] {entry['id']}: {entry['title']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())