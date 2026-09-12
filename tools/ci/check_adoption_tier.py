#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Assess workflow-cookbook adoption tiers for downstream repositories."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_ROOT = ROOT / "templates"


@dataclass(frozen=True)
class TierDefinition:
    tier: int
    name: str
    purpose: str
    required_paths: tuple[str, ...]


TIERS: tuple[TierDefinition, ...] = (
    TierDefinition(0, "Minimal", "Entry point only", ("README.md",)),
    TierDefinition(1, "Foundation", "AI navigation and scope definition", ("HUB.codex.md", "BLUEPRINT.md")),
    TierDefinition(2, "Operational", "Execution, guardrails, and acceptance baseline", ("RUNBOOK.md", "GUARDRAILS.md", "EVALUATION.md")),
    TierDefinition(
        3,
        "Full",
        "Task, acceptance, and Birdseye traceability",
        (
            "docs/acceptance",
            "docs/tasks",
            "docs/birdseye/index.json",
            "docs/birdseye/hot.json",
            "docs/birdseye/caps",
        ),
    ),
)

TEMPLATE_TARGETS: dict[str, str] = {
    "HUB.codex.md.template": "HUB.codex.md",
    "BLUEPRINT.md.template": "BLUEPRINT.md",
    "RUNBOOK.md.template": "RUNBOOK.md",
    "GUARDRAILS.md.template": "GUARDRAILS.md",
    "EVALUATION.md.template": "EVALUATION.md",
}
DIRECTORY_CONTENT = {"docs/acceptance": ".md", "docs/tasks": ".md", "docs/birdseye/caps": ".json"}
JSON_ROLES = {"docs/birdseye/index.json", "docs/birdseye/hot.json", "docs/birdseye/caps"}


def _content_error(path: Path, role: str) -> str | None:
    if path.is_symlink() or not path.is_file():
        return "not_regular_file"
    try:
        content = path.read_text(encoding="utf-8-sig")
        if not content.strip():
            return "empty"
        if role not in JSON_ROLES:
            return None
        value = json.loads(content)
        if not isinstance(value, dict):
            return "invalid_json_structure"
        if role == "docs/birdseye/index.json":
            nodes = value.get("nodes")
            valid = isinstance(nodes, dict) and bool(nodes) and all(
                isinstance(node, dict) and bool(node) for node in nodes.values()
            )
        elif role == "docs/birdseye/hot.json":
            nodes = value.get("nodes")
            valid = isinstance(nodes, list) and bool(nodes) and all(
                isinstance(node, dict) and isinstance(node.get("id"), str) and bool(node["id"].strip())
                for node in nodes
            )
        else:
            valid = all(isinstance(value.get(key), str) and bool(value[key].strip()) for key in ("id", "summary"))
        return None if valid else "invalid_json_structure"
    except (OSError, UnicodeError):
        return "unreadable"
    except ValueError:
        return "invalid_json"


def _parse_scalar(value: str) -> str:
    """Read one-line version scalars without treating quoted hashes as comments.

    Unsupported YAML forms are unknown, never compared as literal versions.
    Double-quoted values use JSON-compatible escaping; single quotes use YAML's
    doubled quote escaping. This checker remains usable without dependencies.
    """
    rendered = value.strip()
    if not rendered:
        return ""
    if rendered[0] in {"'", '"'}:
        match = re.fullmatch(r'''("(?:[^"\\]|\\.)*"|'(?:[^']|'')*')(?:[ \t]+#.*)?''', rendered)
        if not match:
            return ""
        quoted = match[1]
        if quoted[0] == "'":
            single_quoted = quoted[1:-1].replace("''", "'")
            return single_quoted if single_quoted.strip() else ""
        try:
            decoded = json.loads(quoted)
        except ValueError:
            return ""
        return decoded if isinstance(decoded, str) and decoded.strip() else ""
    rendered = re.split(r"\s+#", rendered, maxsplit=1)[0].strip()
    if rendered.startswith(tuple("#[]{}&*!|>%@`")) or rendered.lower() in ("null", "~") or re.search(r":\s", rendered):
        return ""
    return rendered


def _parse_front_matter(content: str) -> dict[str, str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = next((index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
    if end is None:
        return {}
    values: dict[str, str] = {}
    for line in lines[1:end]:
        if not line or line[0].isspace() or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if not re.fullmatch(r"[\w-]+", key):
            continue
        values[key] = "" if key in values else _parse_scalar(value)
    return values


def _path_status(repo: Path, rel_path: str) -> dict[str, Any]:
    target = repo / rel_path
    expected_kind = "dir" if rel_path in DIRECTORY_CONTENT else "file"
    error: str | None
    try:
        exists = target.exists()
        kind = "dir" if target.is_dir() else "file" if target.is_file() else "other"
        if not exists:
            error = "missing"
        elif target.is_symlink() or kind != expected_kind:
            error = "wrong_kind"
        elif expected_kind == "file":
            error = _content_error(target, rel_path)
        else:
            members = sorted(
                (member for member in target.iterdir() if member.suffix.casefold() == DIRECTORY_CONTENT[rel_path]),
                key=lambda member: member.name,
            )
            error = "empty" if not members else None
            for member in members:
                member_error = _content_error(member, rel_path)
                if member_error:
                    error = f"{member.name}: {member_error}"
                    break
    except OSError:
        exists, kind, error = False, "unknown", "unreadable"
    return {
        "path": rel_path,
        "exists": exists,
        "kind": kind,
        "expected_kind": expected_kind,
        "valid": error is None,
        "reason": error,
    }


def _cumulative_required_paths(tier: int) -> list[str]:
    paths: list[str] = []
    for definition in TIERS:
        if definition.tier > tier:
            break
        paths.extend(definition.required_paths)
    return paths


def _highest_complete_tier(repo: Path, statuses: Mapping[str, dict[str, Any]] | None = None) -> int:
    if statuses is None:
        statuses = {path: _path_status(repo, path) for path in _cumulative_required_paths(3)}
    current_tier = -1
    for definition in TIERS:
        required = _cumulative_required_paths(definition.tier)
        if all(statuses[rel_path]["valid"] for rel_path in required):
            current_tier = definition.tier
    return current_tier


def _template_drift(repo: Path, template_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for template_name, target_name in TEMPLATE_TARGETS.items():
        template_path = template_root / template_name
        target_path = repo / target_name
        if not target_path.exists() and not target_path.is_symlink():
            continue
        template_version = target_version = None
        reason = None
        try:
            if not template_path.is_file() or not target_path.is_file():
                reason = "missing_or_invalid_file"
            else:
                template_version = _parse_front_matter(template_path.read_text(encoding="utf-8-sig")).get("template_version")
                target_version = _parse_front_matter(target_path.read_text(encoding="utf-8-sig")).get("template_version")
                if not template_version or not target_version:
                    reason = "missing_version"
        except (OSError, UnicodeError):
            reason = "unreadable"
        status = "unknown" if reason else "current" if template_version == target_version else "drifted"
        checks.append(
            {
                "path": target_name,
                "template": template_name,
                "template_version": template_version,
                "document_template_version": target_version,
                "drifted": status == "drifted",
                "status": status,
                "reason": reason,
            }
        )
    return checks


def assess_repo(repo: Path, *, check_drift: bool = False, template_root: Path = DEFAULT_TEMPLATE_ROOT) -> dict[str, Any]:
    target = repo.expanduser().resolve()
    path_statuses = {path: _path_status(target, path) for path in _cumulative_required_paths(3)}
    current_tier = _highest_complete_tier(target, path_statuses)
    tier_name = TIERS[current_tier].name if current_tier >= 0 else "Unclassified"
    purpose = TIERS[current_tier].purpose if current_tier >= 0 else "README.md is missing or invalid"
    next_tier = current_tier + 1 if current_tier + 1 < len(TIERS) else None
    required_by_tier: dict[str, list[dict[str, Any]]] = {}
    for definition in TIERS:
        required_by_tier[str(definition.tier)] = [path_statuses[rel_path] for rel_path in definition.required_paths]

    missing_for_next: list[str] = []
    if next_tier is not None:
        missing_for_next = [
            rel_path
            for rel_path in _cumulative_required_paths(next_tier)
            if not path_statuses[rel_path]["exists"]
        ]

    drift_checks = _template_drift(target, template_root) if check_drift else []
    drifted = any(check["drifted"] for check in drift_checks)
    drift_status = "not_checked"
    if check_drift:
        drift_status = "drifted" if drifted else "unknown" if not drift_checks or any(
            check["status"] == "unknown" for check in drift_checks
        ) else "current"
    return {
        "repo": str(target),
        "current_tier": current_tier,
        "current_tier_name": tier_name,
        "purpose": purpose,
        "next_tier": next_tier,
        "missing_for_next_tier": missing_for_next,
        "invalid_for_next_tier": [path_statuses[path] for path in _cumulative_required_paths(next_tier)
                                  if path_statuses[path]["exists"] and not path_statuses[path]["valid"]] if next_tier is not None else [],
        "operational_compliance": "not_evaluated",
        "required_by_tier": required_by_tier,
        "drift_checks": drift_checks,
        "drifted": drifted,
        "drift_status": drift_status,
    }


def _load_repo_list(path: Path) -> list[Path]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("repo-list JSON must be a non-empty array")
    repos: list[Path] = []
    for index, item in enumerate(payload):
        value = item.get("repo") if isinstance(item, Mapping) else item
        if not isinstance(value, str) or not value.strip() or "\0" in value:
            raise ValueError(f"repo-list item {index} must be a non-empty path string or an object with repo")
        repos.append(Path(value))
    return repos


def _render_text(result: dict[str, Any]) -> str:
    lines = [
        f"Repo: {result['repo']}",
        f"Current Tier: {result['current_tier']} ({result['current_tier_name']})",
        "",
    ]
    for tier, statuses in result["required_by_tier"].items():
        rendered = ", ".join(
            f"{'OK' if item['valid'] else item['reason']} {item['path']}" for item in statuses
        )
        lines.append(f"Tier {tier}: {rendered}")
    if result["missing_for_next_tier"]:
        lines.extend(["", "Recommendation:"])
        for path in result["missing_for_next_tier"]:
            lines.append(f"- Add {path}")
    for item in result["invalid_for_next_tier"]:
        lines.append(f"- Repair {item['path']}: {item['reason']}")
    lines.append(f"Operational compliance: {result['operational_compliance']}")
    lines.append(f"Template drift: {result['drift_status']}")
    if result["drift_checks"]:
        lines.extend(["", "Template drift:"])
        for check in result["drift_checks"]:
            state = check["status"].upper()
            lines.append(
                f"- {state} {check['path']} "
                f"(doc={check['document_template_version'] or 'n/a'}, template={check['template_version'] or 'n/a'})"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assess workflow-cookbook adoption tier.")
    parser.add_argument("--repo", type=Path, help="Repository to assess.")
    parser.add_argument("--repo-list", type=Path, help="JSON array of repositories to assess.")
    parser.add_argument("--template-root", type=Path, default=DEFAULT_TEMPLATE_ROOT)
    parser.add_argument("--min-tier", type=int, choices=[0, 1, 2, 3], help="Minimum tier required when --check is set.")
    parser.add_argument("--check", action="store_true", help="Exit non-zero when min tier or drift checks fail.")
    parser.add_argument("--check-drift", action="store_true", help="Compare document template_version with templates.")
    parser.add_argument("--json", action="store_true", help="Output JSON.")
    args = parser.parse_args(argv)

    if args.repo is None and args.repo_list is None:
        parser.error("provide --repo or --repo-list")

    try:
        repos = _load_repo_list(args.repo_list) if args.repo_list else [args.repo]
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    results = [
        assess_repo(repo, check_drift=args.check_drift, template_root=args.template_root)
        for repo in repos
        if repo is not None
    ]

    if args.json:
        payload: Any = results[0] if len(results) == 1 else results
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("\n\n".join(_render_text(result) for result in results))

    failed = False
    if args.check:
        if args.min_tier is not None:
            failed = any(int(result["current_tier"]) < args.min_tier for result in results)
        if args.check_drift:
            failed = failed or any(result["drift_status"] != "current" for result in results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
