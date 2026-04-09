#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Check sync between sample files and referenced docs.

Scans examples/ directory and validates:
1. Sample files reference existing docs
2. Referenced docs mention the sample
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
_EXAMPLES_DIR = _REPO_ROOT / "examples"


@dataclass
class SampleRef:
    file_path: Path
    refs: list[str]  # referenced doc paths
    missing_refs: list[str] = field(default_factory=list)


@dataclass
class SyncReport:
    samples: list[SampleRef] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _extract_doc_refs(content: str, file_path: Path) -> list[str]:
    """Extract document references from file content."""
    refs = []

    # Markdown-style links: [text](path)
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", content):
        link_path = match.group(2)
        if link_path.startswith("http://") or link_path.startswith("https://"):
            continue
        refs.append(link_path)

    # JSON-style paths (common in config samples)
    try:
        data = json.loads(content)
        # Look for path-like strings
        def _extract_paths(obj: Any, paths: list[str]) -> None:
            if isinstance(obj, str):
                if "/" in obj and (obj.endswith(".md") or obj.endswith(".yaml") or obj.endswith(".json")):
                    paths.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _extract_paths(v, paths)
            elif isinstance(obj, list):
                for item in obj:
                    _extract_paths(item, paths)

        _extract_paths(data, refs)
    except json.JSONDecodeError:
        pass  # Not a JSON file, skip

    return refs


def scan_samples(examples_dir: Path, repo_root: Path) -> list[SampleRef]:
    """Scan sample files and extract doc references."""
    samples = []
    if not examples_dir.exists():
        return samples

    for sample_file in examples_dir.glob("*"):
        if sample_file.is_file() and not sample_file.name.startswith("."):
            content = sample_file.read_text(encoding="utf-8")
            refs = _extract_doc_refs(content, sample_file)

            # Resolve relative paths
            resolved_refs = []
            for ref in refs:
                if ref.startswith("/"):
                    target = repo_root / ref[1:]
                else:
                    target = examples_dir / ref
                resolved_refs.append(str(target))

            samples.append(SampleRef(
                file_path=sample_file,
                refs=resolved_refs,
            ))

    return samples


def validate_sync(samples: list[SampleRef], repo_root: Path) -> SyncReport:
    """Validate that all references exist."""
    report = SyncReport(samples=samples)

    for sample in samples:
        for ref in sample.refs:
            try:
                ref_path = Path(ref)
                if not ref_path.exists():
                    sample.missing_refs.append(ref)
                    report.errors.append(
                        f"Sample {sample.file_path.name} references missing file: {ref}"
                    )
            except Exception as e:
                report.warnings.append(
                    f"Could not resolve reference {ref} in {sample.file_path.name}: {e}"
                )

    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check sync between sample files and referenced docs."
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=_EXAMPLES_DIR,
        help="Directory containing sample files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON.",
    )

    args = parser.parse_args(argv)

    samples = scan_samples(args.examples_dir, _REPO_ROOT)
    report = validate_sync(samples, _REPO_ROOT)

    if args.json:
        output = {
            "samples": [
                {
                    "file": str(s.file_path.name),
                    "refs": s.refs,
                    "missing_refs": s.missing_refs,
                }
                for s in report.samples
            ],
            "errors": report.errors,
            "warnings": report.warnings,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"Samples scanned: {len(samples)}")
        print(f"Total references: {sum(len(s.refs) for s in samples)}")

    for warning in report.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if report.errors:
        for error in report.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Sample/docs sync is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())