from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ci.check_branch_protection import (  # noqa: E402
    build_task_seed,
    build_weekly_audit_report,
    build_weekly_nudge,
    extract_required_check_names,
    load_policy_required_jobs,
    main,
    validate_branch_protection,
)


def test_load_policy_required_jobs_extracts_ci_block(tmp_path: Path) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        """
self_modification:
  forbidden_paths:
    - "/docs/**"
ci:
  required_jobs:
    - governance-gate
    - security-ci
    - python-ci
slo:
  lead_time_p95_hours: 24
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert load_policy_required_jobs(policy) == [
        "governance-gate",
        "security-ci",
        "python-ci",
    ]


def test_extract_required_check_names_supports_contexts_and_checks() -> None:
    payload = {
        "required_status_checks": {
            "contexts": ["governance", "unit"],
            "checks": [{"context": "Bandit"}, {"context": "Dependency Audit & SBOM"}],
        }
    }

    assert extract_required_check_names(payload) == {"governance", "unit", "Bandit", "Dependency Audit & SBOM"}


def test_validate_branch_protection_accepts_expected_repo_checks() -> None:
    payload = {
        "required_status_checks": {
            "contexts": [
                "governance",
                "unit",
                "Allowlist Guard",
                "Semgrep",
                "Bandit",
                "Gitleaks",
                "Dependency Audit & SBOM",
            ],
        }
    }

    result = validate_branch_protection(
        payload,
        required_jobs=["governance-gate", "python-ci", "security-ci"],
    )

    assert result.errors == []
    assert result.warnings == []


def test_validate_branch_protection_reports_missing_check() -> None:
    payload = {
        "required_status_checks": {
            "contexts": ["governance", "unit", "Allowlist Guard", "Semgrep", "Bandit", "Gitleaks"],
        }
    }

    result = validate_branch_protection(
        payload,
        required_jobs=["governance-gate", "python-ci", "security-ci"],
    )

    assert result.errors == [
        "Missing protected checks for logical gate ID 'security-ci': expected 'Dependency Audit & SBOM'."
    ]


def test_validate_branch_protection_warns_for_unmapped_logical_id() -> None:
    payload = {
        "required_status_checks": {
            "contexts": ["governance"],
        }
    }

    result = validate_branch_protection(
        payload,
        required_jobs=["governance-gate", "future-gate"],
    )

    assert result.errors == []
    assert result.warnings == [
        "No concrete check mapping is defined for logical gate ID 'future-gate'."
    ]


def test_build_weekly_branch_protection_artifacts() -> None:
    payload = {"required_status_checks": {"contexts": ["governance"]}}
    result = validate_branch_protection(payload, required_jobs=["governance-gate"])

    report = build_weekly_audit_report(
        payload=payload,
        required_jobs=["governance-gate"],
        result=result,
    )
    nudge = build_weekly_nudge(report)
    task_seed = build_task_seed(report)

    assert report["result"]["status"] == "pass"
    assert nudge["target_kind"] == "branch_protection"
    assert "Task Seed" in task_seed


def test_main_returns_success_for_matching_payload(tmp_path: Path, capsys) -> None:
    payload_path = tmp_path / "protection.json"
    payload_path.write_text(
        json.dumps(
            {
                "required_status_checks": {
                    "contexts": [
                        "governance",
                        "unit",
                        "Allowlist Guard",
                        "Semgrep",
                        "Bandit",
                        "Gitleaks",
                        "Dependency Audit & SBOM",
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """
ci:
  required_jobs:
    - governance-gate
    - python-ci
    - security-ci
""".strip()
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(("--protection-json", str(payload_path), "--policy", str(policy_path)))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "matches governance/policy.yaml" in captured.out
    assert captured.err == ""
