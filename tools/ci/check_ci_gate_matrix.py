# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.ci.check_branch_protection import LOGICAL_TO_REPO_CHECK, load_policy_required_jobs


_DEFAULT_POLICY = _REPO_ROOT / "governance" / "policy.yaml"
_DEFAULT_CI_CONFIG = _REPO_ROOT / "docs" / "ci-config.md"

LOGICAL_TO_WORKFLOW = {
    "governance-gate": ".github/workflows/governance-gate.yml",
    "python-ci": ".github/workflows/tests.yml",
    "security-ci": ".github/workflows/security.yml",
}


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        return not self.errors

    def emit(self) -> None:
        for message in self.errors:
            print(message, file=sys.stderr)
        for message in self.warnings:
            print(message, file=sys.stderr)


def validate_ci_gate_matrix(
    *,
    repo_root: Path,
    policy_path: Path,
    ci_config_path: Path,
) -> ValidationResult:
    result = ValidationResult()
    required_jobs = load_policy_required_jobs(policy_path)
    ci_config_text = ci_config_path.read_text(encoding="utf-8")

    for logical_id in required_jobs:
        workflow_rel = LOGICAL_TO_WORKFLOW.get(logical_id)
        concrete_check = LOGICAL_TO_REPO_CHECK.get(logical_id)

        if workflow_rel is None:
            result.errors.append(f"No workflow mapping is defined for logical gate ID '{logical_id}'.")
            continue
        if concrete_check is None:
            result.errors.append(f"No concrete check mapping is defined for logical gate ID '{logical_id}'.")
            continue

        workflow_path = repo_root / workflow_rel
        if not workflow_path.is_file():
            result.errors.append(
                f"Workflow file for logical gate ID '{logical_id}' is missing: {workflow_rel}"
            )
            continue

        workflow_text = workflow_path.read_text(encoding="utf-8")
        if concrete_check not in workflow_text:
            result.errors.append(
                f"Workflow '{workflow_rel}' does not mention expected check/job '{concrete_check}' for '{logical_id}'."
            )

        if logical_id not in ci_config_text:
            result.errors.append(f"docs/ci-config.md does not mention logical gate ID '{logical_id}'.")
        if workflow_rel not in ci_config_text:
            result.errors.append(f"docs/ci-config.md does not mention workflow path '{workflow_rel}'.")
        if concrete_check not in ci_config_text:
            result.errors.append(
                f"docs/ci-config.md does not mention concrete check/job '{concrete_check}' for '{logical_id}'."
            )

    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate governance/policy.yaml logical gate IDs against workflows and docs/ci-config.md."
    )
    parser.add_argument("--policy", type=Path, default=_DEFAULT_POLICY, help="Path to governance/policy.yaml.")
    parser.add_argument(
        "--ci-config",
        type=Path,
        default=_DEFAULT_CI_CONFIG,
        help="Path to docs/ci-config.md.",
    )
    args = parser.parse_args(argv)

    result = validate_ci_gate_matrix(
        repo_root=_REPO_ROOT,
        policy_path=args.policy,
        ci_config_path=args.ci_config,
    )
    result.emit()
    if result.is_success:
        print("CI gate matrix matches governance/policy.yaml, workflows, and docs/ci-config.md.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
