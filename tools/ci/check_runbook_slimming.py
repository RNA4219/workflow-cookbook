# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""RUNBOOK slimming checker.

Detects when RUNBOOK accumulates completed details that should be in
docs/completion-record.md instead.

Checks:
- Completed table growth (warning)
- Missing canonical link in completed entries (warning)
- Duplicate completion detail (warning)
- Current-ops exception (pass)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Thresholds
MAX_COMPLETED_TABLE_ROWS = 10  # warning threshold for completed tables
MIN_LINK_PATTERN = re.compile(r"\[.*?\]\(.*?\)")  # markdown link pattern


def _extract_tables(content: str) -> list[tuple[int, int, list[str]]]:
    """Extract markdown tables with their line ranges and rows.

    Excludes tables inside code blocks (``` delimited).

    Returns list of (start_line, end_line, rows) tuples.
    """
    tables: list[tuple[int, int, list[str]]] = []
    lines = content.split("\n")
    in_table = False
    in_code_block = False
    table_start = 0
    table_rows: list[str] = []

    for i, line in enumerate(lines):
        # Track code block boundaries
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            # End any ongoing table when entering code block
            if in_code_block and in_table and table_rows:
                tables.append((table_start, i, table_rows))
                in_table = False
                table_rows = []
            continue

        # Skip content inside code blocks
        if in_code_block:
            continue

        # Table row starts with | and contains at least one |
        if line.strip().startswith("|") and "|" in line.strip()[1:]:
            if not in_table:
                in_table = True
                table_start = i
                table_rows = []
            table_rows.append(line)
        else:
            if in_table and table_rows:
                # End of table
                tables.append((table_start, i, table_rows))
                in_table = False
                table_rows = []

    # Handle table at end of file
    if in_table and table_rows:
        tables.append((table_start, len(lines), table_rows))

    return tables


def _is_completed_table(rows: list[str], completion_record_path: Path | None) -> bool:
    """Determine if a table contains completed items.

    Looks for indicators like:
    - "完了" (completed) in cells
    - Dates with completion markers
    - Links to acceptance records
    - Links to completion record
    """
    completion_keywords = [
        "完了",
        "completed",
        "done",
        "済",
        "済み",
        "✓",
        "✅",
        "[x]",
    ]

    # If table links to completion-record, it's likely a completed summary
    if completion_record_path:
        for row in rows:
            if "completion-record" in row.lower():
                return True

    for row in rows:
        for keyword in completion_keywords:
            if keyword in row.lower():
                return True

    return False


def _has_canonical_link(rows: list[str]) -> bool:
    """Check if table rows contain links to canonical sources.

    Canonical sources: docs/tasks/, docs/acceptance/, docs/releases/,
    CHANGELOG.md, completion-record.md
    """
    canonical_patterns = [
        r"docs/tasks/",
        r"docs/acceptance/",
        r"docs/releases/",
        r"CHANGELOG\.md",
        r"completion-record\.md",
    ]

    for row in rows:
        for pattern in canonical_patterns:
            if re.search(pattern, row):
                return True

    return False


def _extract_completed_sections(content: str) -> list[tuple[int, str]]:
    """Extract sections that might contain completed items.

    Returns list of (line_number, section_title) tuples.
    """
    sections: list[tuple[int, str]] = []
    lines = content.split("\n")

    completion_section_patterns = [
        r"## 完了",
        r"## Completed",
        r"## Done",
        r"## History",
        r"## 履歴",
        r"## 実施履歴",
        r"## Implementation History",
    ]

    for i, line in enumerate(lines):
        for pattern in completion_section_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                sections.append((i, line.strip()))
                break

    return sections


def _check_duplicate_detail(
    runbook_content: str,
    completion_content: str | None,
) -> list[str]:
    """Check for duplicate details between RUNBOOK and completion-record.

    Returns list of warning messages for duplicates found.
    """
    warnings: list[str] = []

    if not completion_content:
        return warnings

    # Look for identical table rows
    runbook_tables = _extract_tables(runbook_content)
    completion_tables = _extract_tables(completion_content)

    for rb_start, rb_end, rb_rows in runbook_tables:
        for c_start, c_end, c_rows in completion_tables:
            # Compare row content (excluding separator lines)
            rb_data_rows = [r for r in rb_rows if not re.match(r"^\|[-:]+\|", r.strip())]
            c_data_rows = [r for r in c_rows if not re.match(r"^\|[-:]+\|", r.strip())]

            # Check for substantial overlap
            if len(rb_data_rows) > 2 and len(c_data_rows) > 2:
                overlap = sum(
                    1 for rb_row in rb_data_rows
                    if any(rb_row.strip() == c_row.strip() for c_row in c_data_rows)
                )
                if overlap >= min(len(rb_data_rows), len(c_data_rows)) * 0.5:
                    warnings.append(
                        f"Duplicate table content detected at RUNBOOK line {rb_start + 1}. "
                        "Move completed details to docs/completion-record.md."
                    )

    return warnings


def _is_current_ops_exception(rows: list[str], section_title: str | None) -> bool:
    """Check if this is a current operations exception.

    Current ops exceptions are:
    - Operational readiness backlog items
    - Current rollback/retry procedures
    - Active incident response steps
    """
    exception_patterns = [
        r"Operational Readiness",
        r"Rollback",
        r"Retry",
        r"インシデント",
        r"Incident",
        r"現在",
        r"Current",
        r"未解決",
        r"Open",
    ]

    if section_title:
        for pattern in exception_patterns:
            if re.search(pattern, section_title, re.IGNORECASE):
                return True

    # Check first column for operational keywords
    for row in rows[:3]:  # Check header and first data row
        for pattern in exception_patterns:
            if re.search(pattern, row, re.IGNORECASE):
                return True

    return False


def check_runbook_slimming(repo_root: Path) -> dict[str, list[str]]:
    """Check RUNBOOK for slimming violations.

    Returns dict mapping file paths to list of warning messages.
    """
    warnings: dict[str, list[str]] = {}

    runbook_path = repo_root / "RUNBOOK.md"
    if not runbook_path.exists():
        warnings[runbook_path] = ["RUNBOOK.md not found"]
        return warnings

    runbook_content = runbook_path.read_text(encoding="utf-8")

    completion_path = repo_root / "docs" / "completion-record.md"
    completion_content = None
    if completion_path.exists():
        completion_content = completion_path.read_text(encoding="utf-8")

    # Extract tables and check for completed tables
    tables = _extract_tables(runbook_content)
    for start, end, rows in tables:
        # Skip small tables (likely headers/links)
        data_rows = [r for r in rows if not re.match(r"^\|[-:]+\|", r.strip())]
        if len(data_rows) < 3:
            continue

        # Check if this is a completed table
        if _is_completed_table(rows, completion_path):
            # Check for current ops exception
            section_title = None
            sections = _extract_completed_sections(runbook_content)
            for sec_line, sec_title in sections:
                if start >= sec_line:
                    section_title = sec_title

            if _is_current_ops_exception(rows, section_title):
                continue  # Pass - current ops exception

            # Check table size
            if len(data_rows) > MAX_COMPLETED_TABLE_ROWS:
                warnings.setdefault(runbook_path, []).append(
                    f"Large completed table ({len(data_rows)} rows) at line {start + 1}. "
                    "Consider moving to docs/completion-record.md."
                )

            # Check for canonical links
            if not _has_canonical_link(rows):
                warnings.setdefault(runbook_path, []).append(
                    f"Completed table at line {start + 1} lacks canonical links "
                    "(docs/tasks/, docs/acceptance/, docs/releases/, CHANGELOG.md, "
                    "completion-record.md)."
                )

    # Check for duplicate details
    duplicate_warnings = _check_duplicate_detail(runbook_content, completion_content)
    if duplicate_warnings:
        warnings.setdefault(runbook_path, []).extend(duplicate_warnings)

    return warnings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check RUNBOOK.md for slimming violations."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run checks and report warnings.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output warnings as JSON.",
    )
    args = parser.parse_args(argv)

    if not args.check:
        print("Use --check to run RUNBOOK slimming checks.")
        return 0

    warnings = check_runbook_slimming(args.repo_root)

    if args.json:
        import json
        result = {
            str(path): messages for path, messages in warnings.items()
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if warnings:
        for path, messages in warnings.items():
            for message in messages:
                print(f"Warning: {path}: {message}", file=sys.stderr)
        return 1

    print("RUNBOOK slimming checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())