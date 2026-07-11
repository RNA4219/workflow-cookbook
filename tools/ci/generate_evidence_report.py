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
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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
class ReleaseSummary:
    release_id: str
    title: str
    status: str
    file_path: Path
    references: list[str] = field(default_factory=list)


@dataclass
class EvidenceReport:
    acceptances: list[AcceptanceSummary] = field(default_factory=list)
    evidences: list[EvidenceSummary] = field(default_factory=list)
    releases: list[ReleaseSummary] = field(default_factory=list)
    linked: list[dict[str, Any]] = field(default_factory=list)
    release_links: list[dict[str, Any]] = field(default_factory=list)
    unlinked_acceptances: list[AcceptanceSummary] = field(default_factory=list)
    unlinked_evidences: list[EvidenceSummary] = field(default_factory=list)
    security: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    readiness: dict[str, Any] = field(default_factory=dict)


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
    acceptances: list[AcceptanceSummary] = []
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
    evidences: list[EvidenceSummary] = []
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


def _extract_markdown_refs(content: str) -> list[str]:
    refs: list[str] = []
    for marker in ("AC-", "EV-", "task-"):
        start = 0
        while True:
            index = content.find(marker, start)
            if index == -1:
                break
            end = index
            while end < len(content) and (
                content[end].isalnum() or content[end] in "-_./"
            ):
                end += 1
            refs.append(content[index:end].rstrip("./"))
            start = end
    return sorted(set(refs))


def scan_releases(release_dir: Path) -> list[ReleaseSummary]:
    """Scan release records and collect lightweight traceability references."""
    releases: list[ReleaseSummary] = []
    if not release_dir.exists():
        return releases

    for release_file in sorted(release_dir.glob("*.md")):
        content = release_file.read_text(encoding="utf-8")
        fm = _parse_front_matter(content)
        title = "Untitled"
        for line in content.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        releases.append(ReleaseSummary(
            release_id=str(fm.get("release_id") or fm.get("id") or release_file.stem),
            title=title,
            status=str(fm.get("status") or "unknown"),
            file_path=release_file,
            references=_extract_markdown_refs(content),
        ))

    return releases


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "missing", "path": str(path)}
    except json.JSONDecodeError as exc:
        return {"status": "invalid", "path": str(path), "error": str(exc)}
    if isinstance(loaded, dict):
        return loaded
    return {"status": "invalid", "path": str(path), "error": "JSON root is not an object"}


def _readiness_status(report: EvidenceReport) -> dict[str, Any]:
    findings: list[str] = []
    if report.unlinked_acceptances:
        findings.append(f"{len(report.unlinked_acceptances)} acceptance record(s) have no evidence link")
    if report.unlinked_evidences:
        findings.append(f"{len(report.unlinked_evidences)} evidence file(s) have no acceptance link")
    if report.releases and not report.release_links:
        findings.append("release records do not reference scanned acceptance/evidence IDs")
    if report.security:
        security_errors = report.security.get("errors")
        if isinstance(security_errors, list) and security_errors:
            findings.append(f"security posture has {len(security_errors)} error(s)")
        if report.security.get("status") in {"missing", "invalid"}:
            findings.append("security posture input is missing or invalid")
    if report.metrics:
        threshold_errors = report.metrics.get("errors")
        if isinstance(threshold_errors, list) and threshold_errors:
            findings.append(f"metrics threshold report has {len(threshold_errors)} error(s)")
        if report.metrics.get("status") in {"missing", "invalid"}:
            findings.append("metrics input is missing or invalid")

    return {
        "status": "ready" if not findings else "needs_attention",
        "findings": findings,
        "release_count": len(report.releases),
        "acceptance_count": len(report.acceptances),
        "evidence_count": len(report.evidences),
    }


def generate_report(
    acceptances: list[AcceptanceSummary],
    evidences: list[EvidenceSummary],
    releases: list[ReleaseSummary] | None = None,
    security: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> EvidenceReport:
    """Generate report linking acceptances and evidences."""
    report = EvidenceReport(
        acceptances=acceptances,
        evidences=evidences,
        releases=releases or [],
        security=security,
        metrics=metrics,
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

    acceptance_ids = {acc.acceptance_id for acc in acceptances if acc.acceptance_id}
    evidence_ids = {ev.evidence_id for ev in evidences if ev.evidence_id}
    for release in report.releases:
        for ref in release.references:
            if ref in acceptance_ids or ref in evidence_ids:
                report.release_links.append({
                    "release_id": release.release_id,
                    "reference": ref,
                    "release_file": str(release.file_path.relative_to(_REPO_ROOT)),
                })

    report.readiness = _readiness_status(report)
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
        f"- Release Records: {len(report.releases)}",
        f"- Linked: {len(report.linked)}",
        f"- Release Links: {len(report.release_links)}",
        f"- Unlinked Acceptances: {len(report.unlinked_acceptances)}",
        f"- Unlinked Evidences: {len(report.unlinked_evidences)}",
        f"- Readiness: {report.readiness.get('status', 'unknown')}",
        "",
    ]
    if report.readiness.get("findings"):
        lines.append("## Readiness Findings")
        lines.append("")
        for finding in report.readiness["findings"]:
            lines.append(f"- {finding}")
        lines.append("")

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

    if report.releases:
        lines.append("## Release Records")
        lines.append("")
        lines.append("| Release | Status | References |")
        lines.append("|---|---|---|")
        for release in report.releases:
            refs = ", ".join(release.references) if release.references else "-"
            lines.append(f"| {release.release_id} | {release.status} | {refs} |")
        lines.append("")

    if report.security is not None or report.metrics is not None:
        lines.append("## Gate Inputs")
        lines.append("")
        if report.security is not None:
            errors = report.security.get("errors", [])
            lines.append(f"- Security errors: {len(errors) if isinstance(errors, list) else 'n/a'}")
        if report.metrics is not None:
            errors = report.metrics.get("errors", [])
            lines.append(f"- Metrics errors: {len(errors) if isinstance(errors, list) else 'n/a'}")
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
        "--release-dir",
        type=Path,
        default=_REPO_ROOT / "docs" / "releases",
        help="Directory containing release records.",
    )
    parser.add_argument(
        "--security-json",
        type=Path,
        help="Optional security posture JSON exported by check_security_posture.py.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        help="Optional metrics threshold JSON or qa metrics JSON.",
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
    releases = scan_releases(args.release_dir)
    report = generate_report(
        acceptances,
        evidences,
        releases=releases,
        security=_load_optional_json(args.security_json),
        metrics=_load_optional_json(args.metrics_json),
    )

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
            "releases": [
                {
                    "id": r.release_id,
                    "title": r.title,
                    "status": r.status,
                    "references": r.references,
                }
                for r in report.releases
            ],
            "linked": report.linked,
            "release_links": report.release_links,
            "unlinked_acceptances": [
                {"id": a.acceptance_id, "task_id": a.task_id}
                for a in report.unlinked_acceptances
            ],
            "unlinked_evidences": [
                {"id": e.evidence_id, "task_id": e.task_id}
                for e in report.unlinked_evidences
            ],
            "security": report.security,
            "metrics": report.metrics,
            "readiness": report.readiness,
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
