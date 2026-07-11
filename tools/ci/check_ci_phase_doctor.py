# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tools.ci.check_branch_protection import load_policy_required_jobs

_REPO_ROOT = Path(__file__).resolve().parents[2]

PHASE_REQUIREMENTS: Mapping[int, tuple[str, ...]] = {
    0: ("governance-gate", "docs-gate"),
    1: ("governance-gate", "python-ci"),
    2: ("governance-gate", "python-ci", "docs-gate"),
    3: ("governance-gate", "python-ci", "security-ci", "docs-gate"),
}

PHASE_WORKFLOWS: Mapping[int, tuple[str, ...]] = {
    0: ("governance-gate.yml", "markdown.yml"),
    1: ("governance-gate.yml", "test.yml"),
    2: ("governance-gate.yml", "test.yml", "markdown.yml"),
    3: ("governance-gate.yml", "test.yml", "security.yml", "markdown.yml"),
}


@dataclass
class PhaseDoctorReport:
    current_phase: int
    required_jobs: list[str]
    workflows: list[str]
    missing_for_current: list[str] = field(default_factory=list)
    missing_for_next: list[str] = field(default_factory=list)
    optional_candidates: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "current_phase": self.current_phase,
            "required_jobs": self.required_jobs,
            "workflows": self.workflows,
            "missing_for_current": self.missing_for_current,
            "missing_for_next": self.missing_for_next,
            "optional_candidates": self.optional_candidates,
            "recommendations": self.recommendations,
        }


def _list_workflows(workflows_dir: Path) -> list[str]:
    if not workflows_dir.exists():
        return []
    return sorted(path.name for path in workflows_dir.glob("*.yml"))


def _phase_supported(phase: int, required_jobs: set[str], workflows: set[str]) -> bool:
    expected_jobs = set(PHASE_REQUIREMENTS[phase])
    expected_workflows = set(PHASE_WORKFLOWS[phase])
    return expected_jobs.issubset(required_jobs) and expected_workflows.issubset(workflows)


def evaluate_ci_phase(
    *,
    required_jobs: Sequence[str],
    workflows: Sequence[str],
    ci_config_text: str = "",
) -> PhaseDoctorReport:
    required_job_set = set(required_jobs)
    workflow_set = set(workflows)
    current_phase = -1
    for phase in sorted(PHASE_REQUIREMENTS):
        if _phase_supported(phase, required_job_set, workflow_set):
            current_phase = phase

    if current_phase < 0:
        current_phase = 0
    missing_for_current = sorted(
        (set(PHASE_REQUIREMENTS[current_phase]) - required_job_set)
        | (set(PHASE_WORKFLOWS[current_phase]) - workflow_set)
    )
    next_phase = current_phase + 1 if current_phase + 1 in PHASE_REQUIREMENTS else None
    missing_for_next: list[str] = []
    if next_phase is not None:
        missing_for_next = sorted(
            (set(PHASE_REQUIREMENTS[next_phase]) - required_job_set)
            | (set(PHASE_WORKFLOWS[next_phase]) - workflow_set)
        )

    optional_candidates: list[str] = []
    if "codeql.yml" in workflow_set and "codeql" not in required_job_set:
        optional_candidates.append("codeql.yml is present but remains optional")
    if "metrics-gate" in ci_config_text and "metrics-gate" not in required_job_set:
        optional_candidates.append("metrics-gate is documented but not required in policy")

    recommendations: list[str] = []
    if missing_for_current:
        recommendations.append("Repair current phase coverage before promotion.")
    elif missing_for_next:
        recommendations.append(f"To promote to Phase {next_phase}, add or document: {', '.join(missing_for_next)}.")
    else:
        recommendations.append("Current required jobs and workflows satisfy the highest defined phase.")
    if any("codeql" in item for item in optional_candidates):
        recommendations.append(
            "Define CodeQL promotion criteria before making codeql.yml a Phase 3 required check."
        )

    return PhaseDoctorReport(
        current_phase=current_phase,
        required_jobs=sorted(required_jobs),
        workflows=sorted(workflows),
        missing_for_current=missing_for_current,
        missing_for_next=missing_for_next,
        optional_candidates=optional_candidates,
        recommendations=recommendations,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose CI rollout phase and next promotion candidates.")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--policy", type=Path, default=_REPO_ROOT / "governance" / "policy.yaml")
    parser.add_argument("--ci-config", type=Path, default=_REPO_ROOT / "docs" / "ci-config.md")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    parser.add_argument("--check", action="store_true", help="Fail when current phase coverage is incomplete.")
    args = parser.parse_args(argv)

    workflows = _list_workflows(args.repo_root / ".github" / "workflows")
    ci_config_text = args.ci_config.read_text(encoding="utf-8") if args.ci_config.exists() else ""
    report = evaluate_ci_phase(
        required_jobs=load_policy_required_jobs(args.policy),
        workflows=workflows,
        ci_config_text=ci_config_text,
    )

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"CI phase: Phase {report.current_phase}")
        for recommendation in report.recommendations:
            print(f"- {recommendation}")

    return 1 if args.check and report.missing_for_current else 0


if __name__ == "__main__":
    raise SystemExit(main())
