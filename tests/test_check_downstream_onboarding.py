from __future__ import annotations

from pathlib import Path

from tools.ci.check_downstream_onboarding import assess_downstream_repo


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_downstream_onboarding_ready_repo(tmp_path: Path) -> None:
    for rel_path in (
        "README.md",
        "HUB.codex.md",
        "BLUEPRINT.md",
        "RUNBOOK.md",
        "GUARDRAILS.md",
        "EVALUATION.md",
        "docs/acceptance/README.md",
        "docs/tasks/task.md",
        "docs/birdseye/index.json",
        "docs/birdseye/hot.json",
        "docs/birdseye/caps/README.md.json",
    ):
        _write(tmp_path / rel_path, "{}")
    _write(
        tmp_path / ".github" / "workflows" / "workflow-cookbook.yml",
        "generate_acceptance_index\ncheck_branch_protection\ncheck_ci_gate_matrix\ncheck_security_posture",
    )

    report = assess_downstream_repo(tmp_path, min_tier=3)

    assert report["status"] == "ready"
    assert report["missing_ci_signals"] == []


def test_downstream_onboarding_reports_missing_ci(tmp_path: Path) -> None:
    _write(tmp_path / "README.md", "# Repo")

    report = assess_downstream_repo(tmp_path, min_tier=2)

    assert report["status"] == "needs_work"
    assert "acceptance_index" in report["missing_ci_signals"]
