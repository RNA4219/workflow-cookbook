# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Task / Acceptance / Completion trace checker.

Validates the completion trace chain:
- docs/tasks/*.md → docs/acceptance/*.md → docs/completion-record.md → CHANGELOG.md

Checks:
- Done task without acceptance or exception reason (fail/warning)
- Completion without source link (fail)
- Acceptance orphan (warning)
- Changelog drift (warning)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_front_matter(content: str) -> dict[str, str]:
    """Parse YAML front matter from markdown content."""
    if not content.startswith("---"):
        return {}

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}

    front_matter = parts[1].strip()
    result: dict[str, str] = {}

    for line in front_matter.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()

    return result


def _extract_links(content: str) -> list[str]:
    """Extract markdown links from content."""
    pattern = re.compile(r"\[.*?\]\((.*?)\)")
    return [m.group(1) for m in pattern.finditer(content)]


def _get_task_files(docs_tasks_dir: Path) -> list[Path]:
    """Get all task seed files."""
    if not docs_tasks_dir.exists():
        return []
    return list(docs_tasks_dir.glob("task-*.md"))


def _get_acceptance_files(docs_acceptance_dir: Path) -> list[Path]:
    """Get all acceptance record files."""
    if not docs_acceptance_dir.exists():
        return []
    return list(docs_acceptance_dir.glob("AC-*.md"))


def _get_task_ids_from_acceptance(acceptance_files: list[Path]) -> dict[str, list[Path]]:
    """Map task_id to acceptance files that reference it."""
    task_to_acceptance: dict[str, list[Path]] = {}

    for acc_path in acceptance_files:
        content = acc_path.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)
        task_id = fm.get("task_id", "")
        if task_id:
            task_to_acceptance.setdefault(task_id, []).append(acc_path)

    return task_to_acceptance


def _get_referenced_acceptance_ids(task_files: list[Path]) -> dict[str, Path]:
    """Map acceptance_id to task files that reference it."""
    acceptance_to_task: dict[str, Path] = {}

    for task_path in task_files:
        content = task_path.read_text(encoding="utf-8")
        links = _extract_links(content)
        for link in links:
            if "acceptance/" in link and "AC-" in link:
                # Extract acceptance ID from link
                match = re.search(r"AC-\d{8}-\d{2}", link)
                if match:
                    acceptance_to_task.setdefault(match.group(), task_path)

    return acceptance_to_task


def check_done_tasks_have_acceptance(
    task_files: list[Path],
    task_to_acceptance: dict[str, list[Path]],
) -> list[str]:
    """Check that done tasks have acceptance records or exception reasons.

    Returns list of error/warning messages.
    """
    errors: list[str] = []

    for task_path in task_files:
        content = task_path.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)
        status = fm.get("status", "").strip().lower()

        if status == "done":
            task_id = fm.get("task_id", task_path.stem)

            # Check for acceptance record
            acceptance_ids = task_to_acceptance.get(task_id, [])

            # Check for exception reason in content
            has_exception_reason = False
            exception_keywords = [
                "acceptance 不要",
                "no acceptance required",
                "minor change",
                "小変更",
                "例外理由",
                "exception reason",
            ]
            for keyword in exception_keywords:
                if keyword.lower() in content.lower():
                    has_exception_reason = True
                    break

            # Check for acceptance links in content
            has_acceptance_link = False
            links = _extract_links(content)
            for link in links:
                if "acceptance/" in link and "AC-" in link:
                    has_acceptance_link = True
                    break

            if not acceptance_ids and not has_exception_reason and not has_acceptance_link:
                errors.append(
                    f"Done task '{task_id}' ({task_path.name}) lacks acceptance record "
                    "or exception reason."
                )

    return errors


def check_completion_record_has_source_links(
    completion_path: Path,
) -> list[str]:
    """Check that completion record items have source links.

    Returns list of error messages.
    """
    errors: list[str] = []

    if not completion_path.exists():
        errors.append(f"Completion record not found: {completion_path}")
        return errors

    content = completion_path.read_text(encoding="utf-8")

    # Find completion entries (tables with date/theme headers)
    # Each entry should have links to canonical sources

    # Pattern: date-based completion entries
    date_pattern = re.compile(r"## \d{4}-\d{2}-\d{2}")
    entries = date_pattern.finditer(content)

    canonical_patterns = [
        r"docs/tasks/",
        r"docs/acceptance/",
        r"docs/releases/",
        r"CHANGELOG\.md",
    ]

    for entry_match in entries:
        start = entry_match.start()
        # Find the next ## or end of file
        next_section = content.find("## ", start + 1)
        if next_section == -1:
            next_section = len(content)

        entry_content = content[start:next_section]

        # Check if entry has canonical links
        has_canonical_link = False
        for pattern in canonical_patterns:
            if re.search(pattern, entry_content):
                has_canonical_link = True
                break

        if not has_canonical_link:
            entry_date = entry_match.group().replace("## ", "")
            errors.append(
                f"Completion entry '{entry_date}' lacks canonical source link "
                "(docs/tasks/, docs/acceptance/, docs/releases/, CHANGELOG.md)."
            )

    return errors


def check_acceptance_orphans(
    acceptance_files: list[Path],
    task_files: list[Path],
    completion_path: Path,
) -> list[str]:
    """Check for acceptance records not referenced by any task or completion.

    Returns list of warning messages.
    """
    warnings: list[str] = []

    # Collect all referenced acceptance IDs
    referenced_ids: set[str] = set()

    # From task content links
    acceptance_to_task = _get_referenced_acceptance_ids(task_files)
    referenced_ids.update(acceptance_to_task.keys())

    # From task front matter: find tasks that reference acceptance via task_id
    # Build mapping: task_id -> task file
    task_ids_to_task: dict[str, Path] = {}
    for task_path in task_files:
        content = task_path.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)
        task_id = fm.get("task_id", "")
        if task_id:
            task_ids_to_task[task_id] = task_path

    # From acceptance front matter: if acceptance's task_id matches a real task,
    # the acceptance is referenced
    for acc_path in acceptance_files:
        content = acc_path.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)
        acc_task_id = fm.get("task_id", "")
        if acc_task_id and acc_task_id in task_ids_to_task:
            # This acceptance is referenced by a task via task_id matching
            acc_id = fm.get("acceptance_id", "")
            if acc_id:
                referenced_ids.add(acc_id)

    # From completion record
    if completion_path.exists():
        completion_content = completion_path.read_text(encoding="utf-8")
        completion_links = _extract_links(completion_content)
        for link in completion_links:
            match = re.search(r"AC-\d{8}-\d{2}", link)
            if match:
                referenced_ids.add(match.group())

    # Check each acceptance file
    for acc_path in acceptance_files:
        content = acc_path.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)
        acc_id = fm.get("acceptance_id", "")

        if acc_id and acc_id not in referenced_ids:
            # Skip INDEX.md, README.md, ACCEPTANCE_TEMPLATE.md
            if acc_path.name in ("INDEX.md", "README.md", "ACCEPTANCE_TEMPLATE.md"):
                continue
            warnings.append(
                f"Acceptance record '{acc_id}' ({acc_path.name}) is orphaned - "
                "not referenced by any task or completion record."
            )

    return warnings


def check_changelog_drift(
    changelog_path: Path,
    completion_path: Path,
) -> list[str]:
    """Check for changelog entries not traceable from completion record.

    Returns list of warning messages.
    """
    warnings: list[str] = []

    if not changelog_path.exists():
        warnings.append(f"CHANGELOG.md not found: {changelog_path}")
        return warnings

    if not completion_path.exists():
        # Already reported by completion record check
        return warnings

    changelog_content = changelog_path.read_text(encoding="utf-8")
    completion_content = completion_path.read_text(encoding="utf-8")

    # Find task/issue references in changelog (pattern: 0001, 00XX, etc.)
    changelog_refs: set[str] = set()
    ref_pattern = re.compile(r"\b\d{4}:\s")  # Pattern like "0001:"
    for match in ref_pattern.finditer(changelog_content):
        ref = match.group().strip(": ")
        changelog_refs.add(ref)

    # Find task references in completion record
    completion_refs: set[str] = set()
    for link in _extract_links(completion_content):
        if "docs/tasks/" in link:
            match = re.search(r"task-.*?-\d{8}", link)
            if match:
                completion_refs.add(match.group())

    # Note: This is a simplified check. Real drift detection would need
    # more sophisticated parsing of changelog entries and completion themes.
    # For now, just warn if changelog has many entries not reflected in completion.

    # Check if "[Unreleased]" has items that might need completion entries
    unreleased_match = re.search(r"## Unreleased.*?(## \d+\.\d+)", changelog_content)
    if unreleased_match:
        unreleased_content = unreleased_match.group(0)
        # Count significant entries (lines with "###" headers)
        entry_count = len(re.findall(r"### ", unreleased_content))

        # Check if completion record has recent entries
        recent_completion_pattern = re.compile(r"## \d{4}-\d{2}-\d{2}")
        completion_dates = recent_completion_pattern.findall(completion_content)

        # If unreleased has many entries but completion has few recent entries, warn
        if entry_count > 3 and len(completion_dates) < 2:
            warnings.append(
                "CHANGELOG [Unreleased] has entries that may not be reflected "
                "in completion-record.md. Consider adding completion entries."
            )

    return warnings


def check_completion_trace(
    repo_root: Path,
    *,
    require_acceptance_for_done: bool = False,
) -> dict[str, list[str]]:
    """Run all completion trace checks.

    Args:
        repo_root: Repository root path.
        require_acceptance_for_done: If True, treat done tasks without acceptance
            as errors. If False (default), treat as warnings.

    Returns dict with 'errors' and 'warnings' lists.
    """
    result: dict[str, list[str]] = {
        "errors": [],
        "warnings": [],
    }

    docs_tasks_dir = repo_root / "docs" / "tasks"
    docs_acceptance_dir = repo_root / "docs" / "acceptance"
    completion_path = repo_root / "docs" / "completion-record.md"
    changelog_path = repo_root / "CHANGELOG.md"

    task_files = _get_task_files(docs_tasks_dir)
    acceptance_files = _get_acceptance_files(docs_acceptance_dir)
    task_to_acceptance = _get_task_ids_from_acceptance(acceptance_files)

    # Check: Done tasks have acceptance
    done_errors = check_done_tasks_have_acceptance(task_files, task_to_acceptance)
    if require_acceptance_for_done:
        result["errors"].extend(done_errors)
    else:
        # Without --require-acceptance-for-done, done task issues are warnings
        result["warnings"].extend(done_errors)

    # Check: Completion record has source links
    completion_errors = check_completion_record_has_source_links(completion_path)
    result["errors"].extend(completion_errors)

    # Check: Acceptance orphans
    orphan_warnings = check_acceptance_orphans(
        acceptance_files, task_files, completion_path
    )
    result["warnings"].extend(orphan_warnings)

    # Check: Changelog drift
    drift_warnings = check_changelog_drift(changelog_path, completion_path)
    result["warnings"].extend(drift_warnings)

    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check Task/Acceptance/Completion trace chain."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run completion trace checks.",
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
        help="Output results as JSON.",
    )
    parser.add_argument(
        "--require-acceptance-for-done",
        action="store_true",
        help="Treat done tasks without acceptance as errors (not warnings).",
    )
    args = parser.parse_args(argv)

    if not args.check:
        print("Use --check to run completion trace checks.")
        return 0

    result = check_completion_trace(
        args.repo_root,
        require_acceptance_for_done=args.require_acceptance_for_done,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if not result["errors"] else 1

    for error in result["errors"]:
        print(f"Error: {error}", file=sys.stderr)

    for warning in result["warnings"]:
        print(f"Warning: {warning}", file=sys.stderr)

    if result["errors"]:
        return 1

    print("Task/Acceptance/Completion trace checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())