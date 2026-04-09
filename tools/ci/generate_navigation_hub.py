#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Generate navigation hub for different use cases.

Creates a single page with entry points for:
- Task operations
- Acceptance operations
- Security operations
- Plugin operations
- CI operations
- Birdseye operations
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS_DIR = _REPO_ROOT / "docs"


NAV_SECTIONS = [
    {
        "title": "Task Operations",
        "description": "Task Seed creation, tracking, and completion",
        "entries": [
            ("Task Seed Template", "TASK.codex.md", "Template for creating new task seeds"),
            ("Tasks Directory", "docs/tasks/", "Active and completed tasks"),
            ("Tasks Guide", "docs/TASKS.md", "Task operations guide"),
        ],
    },
    {
        "title": "Acceptance Operations",
        "description": "Verification and acceptance workflow",
        "entries": [
            ("Acceptance Directory", "docs/acceptance/", "Acceptance records"),
            ("Acceptance Index", "docs/acceptance/INDEX.md", "Summary of all acceptances"),
            ("Acceptance Template", "docs/acceptance/ACCEPTANCE_TEMPLATE.md", "Template for acceptance records"),
            ("Acceptance README", "docs/acceptance/README.md", "Acceptance workflow guide"),
        ],
    },
    {
        "title": "Security Operations",
        "description": "Security review and compliance",
        "entries": [
            ("Security Review Checklist", "docs/security/Security_Review_Checklist.md", "Security review items"),
            ("SAC Guidelines", "docs/security/SAC.md", "Security architecture guidelines"),
            ("Security Privacy Guide", "docs/addenda/G_Security_Privacy.md", "Security/privacy operations"),
        ],
    },
    {
        "title": "Plugin Operations",
        "description": "Cross-repo plugin integration",
        "entries": [
            ("Plugin README", "tools/workflow_plugins/README.md", "Plugin host documentation"),
            ("Plugin Interfaces", "tools/workflow_plugins/interfaces.py", "Plugin capability interfaces"),
            ("Sample Config", "examples/workflow_plugins.cross_repo.sample.json", "Cross-repo plugin sample"),
            ("Config Schema", "schemas/workflow-plugin-config.schema.json", "Plugin config schema"),
        ],
    },
    {
        "title": "CI Operations",
        "description": "Continuous integration and governance",
        "entries": [
            ("CI Config", "docs/ci-config.md", "CI gate configuration"),
            ("Phased Rollout", "docs/ci_phased_rollout_requirements.md", "Rollout requirements"),
            ("Governance Policy", "governance/policy.yaml", "Policy definitions"),
            ("Checklist", "CHECKLISTS.md", "Release checklist"),
        ],
    },
    {
        "title": "Birdseye Operations",
        "description": "Document dependency mapping",
        "entries": [
            ("Birdseye README", "docs/BIRDSEYE.md", "Birdseye documentation"),
            ("Index JSON", "docs/birdseye/index.json", "Node index"),
            ("Codemap README", "tools/codemap/README.md", "Codemap tool documentation"),
        ],
    },
    {
        "title": "Evidence Operations",
        "description": "LLM behavior tracking",
        "entries": [
            ("Evidence Bridge", "tools/protocols/README.md", "Evidence protocol documentation"),
            ("Sample Config", "examples/inference_plugins.agent_protocol.sample.json", "Evidence plugin sample"),
            ("Agent Protocols Reference", "skills/workflow-agent-evidence/references/agent-protocols.md", "Evidence fields reference"),
        ],
    },
    {
        "title": "Reference Documentation",
        "description": "Core documentation",
        "entries": [
            ("Blueprint", "BLUEPRINT.md", "Requirements and constraints"),
            ("Runbook", "RUNBOOK.md", "Execution procedures"),
            ("Evaluation", "EVALUATION.md", "Acceptance criteria"),
            ("Guardrails", "GUARDRAILS.md", "Operational boundaries"),
            ("HUB", "HUB.codex.md", "Agent navigation hub"),
        ],
    },
]


def generate_hub() -> str:
    """Generate navigation hub markdown."""
    lines = [
        "# Navigation Hub",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "Quick entry points for common operations.",
        "",
    ]

    for section in NAV_SECTIONS:
        lines.append(f"## {section['title']}")
        lines.append("")
        lines.append(f"{section['description']}")
        lines.append("")
        lines.append("| Entry | Path | Description |")
        lines.append("|-------|------|-------------|")

        for name, path, desc in section["entries"]:
            lines.append(f"| [{name}]({path}) | `{path}` | {desc} |")

        lines.append("")

    # Quick commands section
    lines.append("## Quick Commands")
    lines.append("")
    lines.append("```sh")
    lines.append("# Birdseye update")
    lines.append("python tools/codemap/update.py --since --emit index+caps")
    lines.append("")
    lines.append("# Run tests")
    lines.append("uv run pytest tests/ -q")
    lines.append("")
    lines.append("# Check CI gates")
    lines.append("python tools/ci/check_ci_gate_matrix.py")
    lines.append("")
    lines.append("# Generate acceptance index")
    lines.append("python tools/ci/generate_acceptance_index_standalone.py")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate navigation hub."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DOCS_DIR / "NAVIGATION.md",
        help="Output file path.",
    )

    args = parser.parse_args(argv)

    content = generate_hub()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote navigation hub to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())