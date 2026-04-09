from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ci.check_ci_gate_matrix import main, validate_ci_gate_matrix  # noqa: E402


def test_validate_ci_gate_matrix_pass(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / ".github" / "workflows").mkdir(parents=True)
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "governance").mkdir(parents=True)

    (repo_root / "governance" / "policy.yaml").write_text(
        "ci:\n  required_jobs:\n    - governance-gate\n    - python-ci\n    - security-ci\n",
        encoding="utf-8",
    )
    (repo_root / ".github" / "workflows" / "governance-gate.yml").write_text(
        "jobs:\n  governance:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    (repo_root / ".github" / "workflows" / "tests.yml").write_text(
        "jobs:\n  pytest:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    (repo_root / ".github" / "workflows" / "security.yml").write_text(
        "jobs:\n  security-ci:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "ci-config.md").write_text(
        "\n".join(
            [
                "| `required_jobs` の値 | この repo での正本 workflow | この repo で確認する job / check | 備考 |",
                "| --- | --- | --- | --- |",
                "| `governance-gate` | `.github/workflows/governance-gate.yml` | job `governance` | ok |",
                "| `python-ci` | `.github/workflows/tests.yml` | job `pytest` | ok |",
                "| `security-ci` | `.github/workflows/security.yml` | job `security-ci` | ok |",
            ]
        ),
        encoding="utf-8",
    )

    result = validate_ci_gate_matrix(
        repo_root=repo_root,
        policy_path=repo_root / "governance" / "policy.yaml",
        ci_config_path=repo_root / "docs" / "ci-config.md",
    )

    assert result.errors == []
    assert result.warnings == []


def test_validate_ci_gate_matrix_reports_missing_mapping(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / ".github" / "workflows").mkdir(parents=True)
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "governance").mkdir(parents=True)

    (repo_root / "governance" / "policy.yaml").write_text(
        "ci:\n  required_jobs:\n    - governance-gate\n    - python-ci\n",
        encoding="utf-8",
    )
    (repo_root / ".github" / "workflows" / "governance-gate.yml").write_text(
        "jobs:\n  governance:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    (repo_root / ".github" / "workflows" / "tests.yml").write_text(
        "jobs:\n  ci:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "ci-config.md").write_text(
        "| `governance-gate` | `.github/workflows/governance-gate.yml` | job `governance` | ok |\n",
        encoding="utf-8",
    )

    result = validate_ci_gate_matrix(
        repo_root=repo_root,
        policy_path=repo_root / "governance" / "policy.yaml",
        ci_config_path=repo_root / "docs" / "ci-config.md",
    )

    assert "docs/ci-config.md does not mention logical gate ID 'python-ci'." in result.errors
    assert "Workflow '.github/workflows/tests.yml' does not mention expected check/job 'pytest' for 'python-ci'." in result.errors


def test_main_returns_success_for_matching_repo(tmp_path: Path, monkeypatch, capsys) -> None:
    repo_root = tmp_path
    (repo_root / ".github" / "workflows").mkdir(parents=True)
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "governance").mkdir(parents=True)

    (repo_root / "governance" / "policy.yaml").write_text(
        "ci:\n  required_jobs:\n    - governance-gate\n",
        encoding="utf-8",
    )
    (repo_root / ".github" / "workflows" / "governance-gate.yml").write_text(
        "jobs:\n  governance:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "ci-config.md").write_text(
        "| `governance-gate` | `.github/workflows/governance-gate.yml` | job `governance` | ok |\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("tools.ci.check_ci_gate_matrix._REPO_ROOT", repo_root)
    exit_code = main(
        (
            "--policy",
            str(repo_root / "governance" / "policy.yaml"),
            "--ci-config",
            str(repo_root / "docs" / "ci-config.md"),
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "CI gate matrix matches" in captured.out
