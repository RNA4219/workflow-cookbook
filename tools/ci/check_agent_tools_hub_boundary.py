# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Agent-tools-hub boundary checker.

Validates that workflow-cookbook/HUB.codex.md does not duplicate
agent-tools-hub responsibilities.

Checks:
- HUB.codex.md does not duplicate Agent_tools repo routing table (warning)
- Cross-repo repo selection references point to agent-tools-hub (warning)
- Birdseye/Task/Acceptance/CI/Evidence procedures pass
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Patterns that indicate Agent_tools routing duplication
ROUTING_DUPLICATION_PATTERNS = [
    # Repo selection patterns
    r"repo 選定",
    r"repo selection",
    r"どの repo を使う",
    r"which repo to use",
    r"repo map",
    # Skill routing patterns
    r"Skill routing",
    r"Skill selection",
    r"どの Skill を",
    r"which Skill",
    # Cross-repo entry patterns (when not limited to plugin/connection)
    r"複数 repo の入口",
    r"multi-repo entry",
    r"Agent_tools 全体",
    r"Agent_tools overall",
    # Routing table indicators
    r"routing table",
    r"routing policy",
]

# Patterns that should reference agent-tools-hub instead
SHOULD_REFERENCE_HUB_PATTERNS = [
    r"Agent_tools 全体",
    r"複数 repo の使い分け",
    r"repo 選び方",
    r"Skill の選び方",
]

# Allowed patterns that are legitimate workflow-cookbook content
ALLOWED_WORKFLOW_COOKBOOK_PATTERNS = [
    # Plugin/connection contexts are allowed
    r"plugin config",
    r"plugin 設定",
    r"Evidence 連携",
    r"Acceptance 連携",
    r"Task state",
    r"cross-repo plugin",
    # Workflow cookbook internal procedures
    r"Birdseye",
    r"Task Seed",
    r"Acceptance",
    r"CI",
    r"Evidence 手順",
    r"release/security",
    r"self-improvement loop",
]


def _extract_sections(content: str) -> dict[str, str]:
    """Extract markdown sections by header.

    Returns dict mapping section title to section content.
    """
    sections: dict[str, str] = {}
    lines = content.split("\n")
    current_section = ""
    current_content: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_section:
                sections[current_section] = "\n".join(current_content)
            current_section = line.strip()
            current_content = []
        else:
            current_content.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_content)

    return sections


def _has_agent_tools_hub_reference(content: str) -> bool:
    """Check if content references agent-tools-hub (not just mentions Agent_tools).

    Returns True only if there's an actual reference/link to agent-tools-hub,
    not just a mention of Agent_tools in general.
    """
    # Patterns that indicate actual reference to agent-tools-hub
    reference_patterns = [
        r"agent-tools-hub",
        r"../Agent_tools/README",
        r"../Agent_tools/HUB",
        r"Agent_tools/README\.md",
        r"Agent_tools/HUB\.codex",
        r"see agent-tools-hub",
        r"refer to agent-tools-hub",
        r"参照.*agent-tools-hub",
        r"agent-tools-hub.*参照",
        r"参照.*Agent_tools",
        r"Agent_tools.*を参照",
        r"see.*Agent_tools/README",
    ]
    for pattern in reference_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False


def _check_routing_duplication(hub_content: str) -> list[str]:
    """Check for Agent_tools routing duplication.

    Returns list of warning messages.
    """
    warnings: list[str] = []

    sections = _extract_sections(hub_content)

    for section_title, section_content in sections.items():
        # Skip allowed workflow cookbook sections
        is_allowed = False
        for allowed_pattern in ALLOWED_WORKFLOW_COOKBOOK_PATTERNS:
            if re.search(allowed_pattern, section_title, re.IGNORECASE):
                is_allowed = True
                break
            if re.search(allowed_pattern, section_content, re.IGNORECASE):
                # If section discusses allowed topics, check proportion
                allowed_mentions = len(
                    re.findall(allowed_pattern, section_content, re.IGNORECASE)
                )
                routing_mentions = 0
                for routing_pattern in ROUTING_DUPLICATION_PATTERNS:
                    routing_mentions += len(
                        re.findall(routing_pattern, section_content, re.IGNORECASE)
                    )
                if allowed_mentions > routing_mentions:
                    is_allowed = True
                    break

        if is_allowed:
            continue

        # Check for routing duplication patterns
        for pattern in ROUTING_DUPLICATION_PATTERNS:
            matches = re.findall(pattern, section_content, re.IGNORECASE)
            if matches:
                # Check if section has agent-tools-hub reference
                if not _has_agent_tools_hub_reference(section_content):
                    warnings.append(
                        f"Section '{section_title}' contains Agent_tools routing "
                        f"content ('{pattern}') without referencing agent-tools-hub. "
                        "Routing responsibilities belong to agent-tools-hub."
                    )
                    break  # One warning per section

    return warnings


def _check_hub_references(hub_content: str) -> list[str]:
    """Check that cross-repo references point to agent-tools-hub.

    Returns list of warning messages.
    """
    warnings: list[str] = []

    sections = _extract_sections(hub_content)

    for section_title, section_content in sections.items():
        # Check for patterns that should reference agent-tools-hub
        for pattern in SHOULD_REFERENCE_HUB_PATTERNS:
            matches = re.findall(pattern, section_content, re.IGNORECASE)
            if matches:
                # Verify agent-tools-hub is referenced nearby
                if not _has_agent_tools_hub_reference(section_content):
                    warnings.append(
                        f"Section '{section_title}' discusses cross-repo selection "
                        f"('{pattern}') but does not reference agent-tools-hub. "
                        "Refer to agent-tools-hub for repo/Skill selection guidance."
                    )
                    break  # One warning per pattern per section

    return warnings


# Expected procedure keywords (English and Japanese)
EXPECTED_PROCEDURES = [
    ("Birdseye", r"Birdseye"),
    ("Task Seed", r"Task Seed"),
    ("Acceptance", r"(Acceptance|検収)"),
    ("CI", r"CI"),
    ("Evidence", r"(Evidence|証跡)"),
]


def _check_procedure_sections(hub_content: str) -> list[str]:
    """Verify Birdseye/Task/Acceptance/CI/Evidence procedures are present.

    Returns list of warning messages if key procedures are missing.
    """
    warnings: list[str] = []

    for english_kw, search_pattern in EXPECTED_PROCEDURES:
        if not re.search(search_pattern, hub_content, re.IGNORECASE):
            warnings.append(
                f"HUB.codex.md lacks reference to '{english_kw}' procedures. "
                "Workflow cookbook should cover these operational areas."
            )

    return warnings


def check_agent_tools_hub_boundary(repo_root: Path) -> dict[str, list[str]]:
    """Run all agent-tools-hub boundary checks.

    Returns dict with 'errors' and 'warnings' lists.
    """
    result: dict[str, list[str]] = {
        "errors": [],
        "warnings": [],
    }

    hub_path = repo_root / "HUB.codex.md"
    if not hub_path.exists():
        result["errors"].append(f"HUB.codex.md not found: {hub_path}")
        return result

    hub_content = hub_path.read_text(encoding="utf-8")

    # Check routing duplication
    routing_warnings = _check_routing_duplication(hub_content)
    result["warnings"].extend(routing_warnings)

    # Check hub references
    ref_warnings = _check_hub_references(hub_content)
    result["warnings"].extend(ref_warnings)

    # Check procedure sections
    procedure_warnings = _check_procedure_sections(hub_content)
    result["warnings"].extend(procedure_warnings)

    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check agent-tools-hub boundary compliance."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run boundary checks.",
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
    args = parser.parse_args(argv)

    if not args.check:
        print("Use --check to run agent-tools-hub boundary checks.")
        return 0

    result = check_agent_tools_hub_boundary(args.repo_root)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if not result["errors"] else 1

    for error in result["errors"]:
        print(f"Error: {error}", file=sys.stderr)

    for warning in result["warnings"]:
        print(f"Warning: {warning}", file=sys.stderr)

    if result["errors"]:
        return 1

    if result["warnings"]:
        print("Agent-tools-hub boundary checks passed with warnings.")
        return 0

    print("Agent-tools-hub boundary checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())