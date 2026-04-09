#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Generate Evidence Report linking acceptance records with evidence files.

Scans docs/acceptance/ and .workflow-cache/ for evidence files,
generates a summary report showing the relationship.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ACCEPTANCE_DIR = _REPO_ROOT / "docs" / "acceptance"
_EVIDENCE_DIR = _REPO_ROOT / ".workflow-cache"


@dataclass
class AcceptanceSummary:
    acceptance_id: str
    task_id: str
    status: str
    file_path: Path


@dataclass
class EvidenceSummary:
    evidence_id: str
    task_id: str
    timestamp: str
    model: str
    file_path: Path


@dataclass
class EvidenceReport:
    acceptances: list[AcceptanceSummary] = field(default_factory=list)
    evidences: list[EvidenceSummary] = field(default_factory=list)
    linked: list[dict[str, Any]] = field(default_factory=list)
    unlinked_acceptances: list[AcceptanceSummary] = field(default_factory=list)
    unlinked_evidences: list[EvidenceSummary] = field(default_factory=list)


def _parse_front_matter(content: str) -> dict[str, Any]:
    """Parse YAML front matter from markdown content."""
    import re
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


def scan_acceptances(acceptance_dir: Path) -> list[AcceptanceSummary]:
    """Scan acceptance records."""
    acceptances = []
    if not acceptance_dir.exists():
        return acceptances

    for acc_file in acceptance_dir.glob("AC-*.md"):
        content = acc_file.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)
        acceptance_id = fm.get("acceptance_id", "")
        if acceptance_id:
            acceptances.append(AcceptanceSummary(
                acceptance_id=acceptance_id,
                task_id=fm.get("task_id", ""),
                status=fm.get("status", "unknown"),
                file_path=acc_file,
            ))

    return acceptances


def scan_evidences(evidence_dir: Path) -> list[EvidenceSummary]:
    """Scan evidence files."""
    evidences = []
    if not evidence_dir.exists():
        return evidences

    for evidence_file in evidence_dir.glob("**/*.json"):
        try:
            data = json.loads(evidence_file.read_text(encoding="utf-8"))
            evidence_id = data.get("evidence_id", evidence_file.stem)
            evidences.append(EvidenceSummary(
                evidence_id=evidence_id,
                task_id=data.get("taskSeedId", data.get("task_id", "")),
                timestamp=data.get("startTime", data.get("timestamp", "")),
                model=data.get("model", "unknown"),
                file_path=evidence_file,
            ))
        except (json.JSONDecodeError, KeyError):
            continue

    return evidences


def generate_report(
    acceptances: list[AcceptanceSummary],
    evidences: list[EvidenceSummary],
) -> EvidenceReport:
    """Generate report linking acceptances and evidences."""
    report = EvidenceReport(
        acceptances=acceptances,
        evidences=evidences,
    )

    # Build lookup
    evidence_by_task: dict[str, list[EvidenceSummary]] = {}
    for ev in evidences:
        if ev.task_id:
            evidence_by_task.setdefault(ev.task_id, []).append(ev)

    # Link by task_id
    linked_task_ids: set[str] = set()
    for acc in acceptances:
        if acc.task_id and acc.task_id in evidence_by_task:
            for ev in evidence_by_task[acc.task_id]:
                report.linked.append({
                    "acceptance_id": acc.acceptance_id,
                    "task_id": acc.task_id,
                    "evidence_id": ev.evidence_id,
                    "evidence_file": str(ev.file_path.relative_to(_REPO_ROOT)),
                })
            linked_task_ids.add(acc.task_id)
        else:
            report.unlinked_acceptances.append(acc)

    for ev in evidences:
        if ev.task_id not in linked_task_ids:
            report.unlinked_evidences.append(ev)

    return report


def format_markdown_report(report: EvidenceReport) -> str:
    """Format report as markdown."""
    lines = [
        "# Evidence Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"- Acceptance Records: {len(report.acceptances)}",
        f"- Evidence Files: {len(report.evidences)}",
        f"- Linked: {len(report.linked)}",
        f"- Unlinked Acceptances: {len(report.unlinked_acceptances)}",
        f"- Unlinked Evidences: {len(report.unlinked_evidences)}",
        "",
    ]

    if report.linked:
        lines.append("## Linked Records")
        lines.append("")
        lines.append("| Acceptance | Task | Evidence |")
        lines.append("|---|---|---|")
        for link in report.linked:
            lines.append(
                f"| {link['acceptance_id']} | {link['task_id']} | {link['evidence_id']} |"
            )
        lines.append("")

    if report.unlinked_acceptances:
        lines.append("## Unlinked Acceptances")
        lines.append("")
        lines.append("| Acceptance | Task | Status |")
        lines.append("|---|---|---|")
        for acc in report.unlinked_acceptances:
            lines.append(f"| {acc.acceptance_id} | {acc.task_id} | {acc.status} |")
        lines.append("")

    if report.unlinked_evidences:
        lines.append("## Unlinked Evidences")
        lines.append("")
        lines.append("| Evidence | Task | Model |")
        lines.append("|---|---|---|")
        for ev in report.unlinked_evidences:
            lines.append(f"| {ev.evidence_id} | {ev.task_id} | {ev.model} |")
        lines.append("")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Evidence Report linking acceptances and evidence files."
    )
    parser.add_argument(
        "--acceptance-dir",
        type=Path,
        default=_ACCEPTANCE_DIR,
        help="Directory containing acceptance records.",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=_EVIDENCE_DIR,
        help="Directory containing evidence files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown file (default: stdout).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of markdown.",
    )

    args = parser.parse_args(argv)

    acceptances = scan_acceptances(args.acceptance_dir)
    evidences = scan_evidences(args.evidence_dir)
    report = generate_report(acceptances, evidences)

    if args.json:
        output = {
            "acceptances": [
                {"id": a.acceptance_id, "task_id": a.task_id, "status": a.status}
                for a in report.acceptances
            ],
            "evidences": [
                {"id": e.evidence_id, "task_id": e.task_id, "model": e.model}
                for e in report.evidences
            ],
            "linked": report.linked,
            "unlinked_acceptances": [
                {"id": a.acceptance_id, "task_id": a.task_id}
                for a in report.unlinked_acceptances
            ],
            "unlinked_evidences": [
                {"id": e.evidence_id, "task_id": e.task_id}
                for e in report.unlinked_evidences
            ],
        }
        content = json.dumps(output, ensure_ascii=False, indent=2)
    else:
        content = format_markdown_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Wrote report to {args.output}")
    else:
        print(content)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())