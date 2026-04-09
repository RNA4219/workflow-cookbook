#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Generate index for ADR and addenda directories.

Scans docs/ADR/ and docs/addenda/, generates INDEX.md files with:
- Table of all documents with titles and last reviewed dates
- Link validation
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DocInfo:
    file_path: Path
    title: str
    adr_id: str | None
    status: str | None
    last_reviewed: str | None
    broken_links: list[str]


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


def _extract_title(content: str) -> str:
    """Extract title from markdown content."""
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled"


def _check_links(content: str, file_path: Path, repo_root: Path) -> list[str]:
    """Check for broken markdown links."""
    broken = []
    # Find markdown links [text](path)
    pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    for match in re.finditer(pattern, content):
        link_text = match.group(1)
        link_path = match.group(2)

        # Skip external links
        if link_path.startswith("http://") or link_path.startswith("https://"):
            continue

        # Resolve relative path
        if link_path.startswith("/"):
            target = repo_root / link_path[1:]
        else:
            target = file_path.parent / link_path

        # Normalize and check
        try:
            target = target.resolve()
            if not target.exists():
                broken.append(f"{link_text} -> {link_path}")
        except Exception:
            broken.append(f"{link_text} -> {link_path}")

    return broken


def scan_directory(dir_path: Path, repo_root: Path) -> list[DocInfo]:
    """Scan a directory for markdown documents."""
    docs = []
    if not dir_path.exists():
        return docs

    for md_file in sorted(dir_path.glob("*.md")):
        if md_file.name in ("INDEX.md", "README.md"):
            continue

        content = md_file.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)
        title = _extract_title(content)
        broken_links = _check_links(content, md_file, repo_root)

        docs.append(DocInfo(
            file_path=md_file,
            title=title,
            adr_id=fm.get("adr_id") or fm.get("id"),
            status=fm.get("status"),
            last_reviewed=fm.get("last_reviewed_at") or fm.get("last_reviewed"),
            broken_links=broken_links,
        ))

    return docs


def generate_index_markdown(docs: list[DocInfo], title: str, rel_prefix: str) -> str:
    """Generate INDEX.md content."""
    lines = [f"# {title} Index", ""]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append(f"Total: {len(docs)} documents")
    lines.append("")

    if not docs:
        lines.append("No documents found.")
        return "\n".join(lines)

    # Table
    lines.append("| File | Title | Status | Last Reviewed |")
    lines.append("|---|---|---|---|")

    for doc in docs:
        rel_path = doc.file_path.name
        status = doc.status or "-"
        reviewed = doc.last_reviewed or "-"
        lines.append(f"| [{rel_path}]({rel_path}) | {doc.title} | {status} | {reviewed} |")

    lines.append("")

    # Broken links report
    broken_count = sum(len(d.broken_links) for d in docs)
    if broken_count > 0:
        lines.append("## Broken Links")
        lines.append("")
        for doc in docs:
            if doc.broken_links:
                lines.append(f"### {doc.file_path.name}")
                for link in doc.broken_links:
                    lines.append(f"- {link}")
                lines.append("")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate index for ADR and addenda directories."
    )
    parser.add_argument(
        "--adr-dir",
        type=Path,
        default=_REPO_ROOT / "docs" / "ADR",
        help="ADR directory path.",
    )
    parser.add_argument(
        "--addenda-dir",
        type=Path,
        default=_REPO_ROOT / "docs" / "addenda",
        help="Addenda directory path.",
    )
    parser.add_argument(
        "--check-links",
        action="store_true",
        help="Check for broken links.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated indices without writing.",
    )

    args = parser.parse_args(argv)

    errors = 0

    # ADR
    adr_docs = scan_directory(args.adr_dir, _REPO_ROOT)
    adr_index = generate_index_markdown(adr_docs, "ADR", "../")
    if args.dry_run:
        print(f"=== {args.adr_dir}/INDEX.md ===")
        print(adr_index)
        print()
    else:
        adr_index_path = args.adr_dir / "INDEX.md"
        adr_index_path.write_text(adr_index, encoding="utf-8")
        print(f"Wrote {adr_index_path} ({len(adr_docs)} documents)")

    broken_adr = sum(len(d.broken_links) for d in adr_docs)
    if broken_adr > 0:
        print(f"WARNING: {broken_adr} broken links in ADR", file=sys.stderr)
        if args.check_links:
            errors = 1

    # Addenda
    addenda_docs = scan_directory(args.addenda_dir, _REPO_ROOT)
    addenda_index = generate_index_markdown(addenda_docs, "Addenda", "../")
    if args.dry_run:
        print(f"=== {args.addenda_dir}/INDEX.md ===")
        print(addenda_index)
    else:
        addenda_index_path = args.addenda_dir / "INDEX.md"
        addenda_index_path.write_text(addenda_index, encoding="utf-8")
        print(f"Wrote {addenda_index_path} ({len(addenda_docs)} documents)")

    broken_addenda = sum(len(d.broken_links) for d in addenda_docs)
    if broken_addenda > 0:
        print(f"WARNING: {broken_addenda} broken links in addenda", file=sys.stderr)
        if args.check_links:
            errors = 1

    return errors


if __name__ == "__main__":
    raise SystemExit(main())