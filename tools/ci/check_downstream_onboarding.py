# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.ci.check_adoption_tier import _load_repo_list, assess_repo

ROOT = Path(__file__).resolve().parents[2]

CI_SIGNALS: Mapping[str, tuple[str, ...]] = {
    "acceptance_index": ("generate_acceptance_index", "docs/acceptance"),
    "branch_protection": ("check_branch_protection", "branch protection"),
    "ci_gate_matrix": ("check_ci_gate_matrix", "ci-config"),
    "security_posture": ("check_security_posture", "security posture"),
}


def _workflow_text(repo: Path) -> str:
    workflows_dir = repo / ".github" / "workflows"
    if not workflows_dir.exists():
        return ""
    parts: list[str] = []
    for path in sorted(workflows_dir.iterdir()):
        if path.suffix.casefold() in {".yml", ".yaml"} and path.is_file():
            parts.append(path.read_text(encoding="utf-8-sig"))
    return "\n".join(parts)


def assess_downstream_repo(repo: Path, *, min_tier: int = 2) -> dict[str, Any]:
    target = repo.expanduser().resolve()
    tier = assess_repo(target)
    workflow_text = _workflow_text(target)
    signals: dict[str, bool] = {}
    missing: list[str] = []
    for name, markers in CI_SIGNALS.items():
        present = any(marker in workflow_text for marker in markers)
        signals[name] = present
        if not present:
            missing.append(name)

    recommendations: list[str] = []
    if int(tier["current_tier"]) < min_tier:
        recommendations.append(f"Raise adoption tier to at least Tier {min_tier}.")
    if missing:
        recommendations.append(f"Add CI/onboarding signals: {', '.join(missing)}.")
    if int(tier["current_tier"]) >= 3 and not missing:
        recommendations.append("Downstream repo has full adoption docs and onboarding CI signals.")

    return {
        "repo": str(target),
        "adoption": tier,
        "ci_signals": signals,
        "missing_ci_signals": missing,
        "status": "ready" if int(tier["current_tier"]) >= min_tier and not missing else "needs_work",
        "recommendations": recommendations,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assess downstream workflow-cookbook onboarding readiness.")
    parser.add_argument("--repo", type=Path, help="Repository to assess.")
    parser.add_argument("--repo-list", type=Path, help="JSON array of repositories to assess.")
    parser.add_argument("--min-tier", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.repo is None and args.repo_list is None:
        parser.error("provide --repo or --repo-list")
    try:
        repos = _load_repo_list(args.repo_list) if args.repo_list else [args.repo]
        reports = [assess_downstream_repo(repo, min_tier=args.min_tier) for repo in repos if repo is not None]
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload: Any = reports[0] if len(reports) == 1 else reports
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for report in reports:
            print(f"{report['repo']}: {report['status']}")
            for recommendation in report["recommendations"]:
                print(f"- {recommendation}")

    return 1 if args.check and any(report["status"] != "ready" for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
