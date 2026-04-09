#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Generate Acceptance Index with status summary.

Scans docs/acceptance/ directory and generates INDEX.md with:
- Status summary (approved/rejected/draft counts)
- Table of all acceptance records
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ACCEPTANCE_DIR = _REPO_ROOT / "docs" / "acceptance"
_INDEX_PATH = _ACCEPTANCE_DIR / "INDEX.md"


@dataclass
class AcceptanceInfo:
    acceptance_id: str
    task_id: str
    intent_id: str
    status: str
    reviewed_at: str
    file_path: Path


def _parse_front_matter(content: str) -> dict[str, Any]:
    """Parse YAML front matter from markdown content."""
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


def scan_acceptances(acceptance_dir: Path) -> list[AcceptanceInfo]:
    """Scan all acceptance files."""
    acceptances = []
    if not acceptance_dir.exists():
        return acceptances

    for acc_file in acceptance_dir.glob("AC-*.md"):
        content = acc_file.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)

        acceptance_id = fm.get("acceptance_id", "")
        if not acceptance_id:
            continue

        acceptances.append(AcceptanceInfo(
            acceptance_id=acceptance_id,
            task_id=fm.get("task_id", ""),
            intent_id=fm.get("intent_id", ""),
            status=fm.get("status", "unknown").lower(),
            reviewed_at=fm.get("reviewed_at", fm.get("reviewed_by", "")),
            file_path=acc_file,
        ))

    return acceptances


def generate_index_markdown(acceptances: list[AcceptanceInfo]) -> str:
    """Generate INDEX.md content with summary and table."""
    lines = ["# Acceptance Index", ""]

    # Status summary
    status_counts = Counter(a.status for a in acceptances)
    total = len(acceptances)

    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Status | Count | Percentage |")
    lines.append("|---|---|---|")
    for status in ["approved", "rejected", "draft", "unknown"]:
        count = status_counts.get(status, 0)
        pct = f"{count / total * 100:.1f}%" if total > 0 else "0%"
        lines.append(f"| {status} | {count} | {pct} |")
    lines.append(f"| **Total** | **{total}** | **100%** |")
    lines.append("")

    # Table
    lines.append("## Records")
    lines.append("")
    lines.append("| Acceptance | Task | Intent | Status | Reviewed | File |")
    lines.append("|---|---|---|---|---|---|")

    # Sort by acceptance_id descending (newest first)
    sorted_acceptances = sorted(acceptances, key=lambda a: a.acceptance_id, reverse=True)

    for acc in sorted_acceptances:
        rel_path = acc.file_path.relative_to(_REPO_ROOT)
        lines.append(
            f"| {acc.acceptance_id} | {acc.task_id} | {acc.intent_id} | "
            f"{acc.status} | {acc.reviewed_at} | [{acc.file_path.name}]({rel_path}) |"
        )

    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Acceptance Index with status summary."
    )
    parser.add_argument(
        "--acceptance-dir",
        type=Path,
        default=_ACCEPTANCE_DIR,
        help="Directory containing Acceptance Record files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_INDEX_PATH,
        help="Output path for INDEX.md.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print to stdout instead of file.",
    )

    args = parser.parse_args(argv)

    acceptances = scan_acceptances(args.acceptance_dir)
    markdown = generate_index_markdown(acceptances)

    if args.print:
        print(markdown)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote acceptance index to {args.output} ({len(acceptances)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())