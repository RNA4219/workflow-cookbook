# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 RNA4219

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_POLICY = _REPO_ROOT / "governance" / "policy.yaml"

# Logical gate IDs in governance/policy.yaml map to concrete GitHub check names
# for this repository. Downstream repositories may use different concrete names.
LOGICAL_TO_REPO_CHECK = {
    "governance-gate": "governance",
    "python-ci": "pytest",
    "security-ci": "security-ci",
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


def load_policy_required_jobs(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    required_jobs: list[str] = []
    in_ci_block = False
    in_required_jobs = False
    ci_indent = None
    required_indent = None

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))

        if stripped == "ci:":
            in_ci_block = True
            in_required_jobs = False
            ci_indent = indent
            continue

        if in_ci_block and ci_indent is not None and indent <= ci_indent:
            in_ci_block = False
            in_required_jobs = False

        if not in_ci_block:
            continue

        if stripped == "required_jobs:":
            in_required_jobs = True
            required_indent = indent
            continue

        if in_required_jobs and required_indent is not None and indent <= required_indent:
            in_required_jobs = False

        if in_required_jobs and stripped.startswith("- "):
            required_jobs.append(stripped[2:].strip())

    return required_jobs


def extract_required_check_names(payload: Mapping[str, object]) -> set[str]:
    required_status_checks = payload.get("required_status_checks")
    if not isinstance(required_status_checks, Mapping):
        return set()

    names: set[str] = set()

    contexts = required_status_checks.get("contexts")
    if isinstance(contexts, Sequence) and not isinstance(contexts, (str, bytes)):
        for context in contexts:
            if isinstance(context, str) and context.strip():
                names.add(context.strip())

    checks = required_status_checks.get("checks")
    if isinstance(checks, Sequence) and not isinstance(checks, (str, bytes)):
        for item in checks:
            if isinstance(item, str) and item.strip():
                names.add(item.strip())
                continue
            if isinstance(item, Mapping):
                context = item.get("context")
                if isinstance(context, str) and context.strip():
                    names.add(context.strip())

    return names


def validate_branch_protection(
    payload: Mapping[str, object],
    *,
    required_jobs: Iterable[str],
    logical_to_check: Mapping[str, str] = LOGICAL_TO_REPO_CHECK,
) -> ValidationResult:
    result = ValidationResult()
    configured_checks = extract_required_check_names(payload)
    if not configured_checks:
        result.errors.append("Branch protection payload does not include required status checks.")
        return result

    for logical_id in required_jobs:
        concrete_check = logical_to_check.get(logical_id)
        if concrete_check is None:
            result.warnings.append(
                f"No concrete check mapping is defined for logical gate ID '{logical_id}'."
            )
            continue
        if concrete_check not in configured_checks:
            result.errors.append(
                f"Missing protected check for logical gate ID '{logical_id}': expected '{concrete_check}'."
            )

    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a GitHub branch protection export against governance/policy.yaml."
    )
    parser.add_argument(
        "--protection-json",
        type=Path,
        required=True,
        help="Path to a JSON file exported from the GitHub branch protection API.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=_DEFAULT_POLICY,
        help="Path to governance/policy.yaml.",
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.protection_json.read_text(encoding="utf-8"))
    required_jobs = load_policy_required_jobs(args.policy)
    result = validate_branch_protection(payload, required_jobs=required_jobs)
    result.emit()
    if result.is_success:
        print(
            "Branch protection matches governance/policy.yaml logical gate IDs.",
            file=sys.stdout,
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
