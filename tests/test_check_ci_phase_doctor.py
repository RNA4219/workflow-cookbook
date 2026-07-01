from __future__ import annotations

from tools.ci.check_ci_phase_doctor import evaluate_ci_phase


def test_evaluate_ci_phase_detects_phase_three_and_codeql_candidate() -> None:
    report = evaluate_ci_phase(
        required_jobs=["governance-gate", "python-ci", "security-ci", "docs-gate"],
        workflows=["governance-gate.yml", "test.yml", "security.yml", "markdown.yml", "codeql.yml"],
        ci_config_text="metrics-gate",
    )

    assert report.current_phase == 3
    assert "codeql.yml is present but remains optional" in report.optional_candidates
    assert "metrics-gate is documented but not required in policy" in report.optional_candidates


def test_evaluate_ci_phase_reports_next_phase_gaps() -> None:
    report = evaluate_ci_phase(
        required_jobs=["governance-gate", "python-ci"],
        workflows=["governance-gate.yml", "test.yml"],
    )

    assert report.current_phase == 1
    assert "docs-gate" in report.missing_for_next
    assert "markdown.yml" in report.missing_for_next
