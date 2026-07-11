from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ci.check_ci_gate_matrix import load_checker_stages, main, validate_ci_gate_matrix  # noqa: E402


def test_validate_ci_gate_matrix_pass(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / ".github" / "workflows").mkdir(parents=True)
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "governance").mkdir(parents=True)

    (repo_root / "governance" / "policy.yaml").write_text(
        "ci:\n  required_jobs:\n    - governance-gate\n    - python-ci\n    - security-ci\n    - docs-gate\n",
        encoding="utf-8",
    )
    (repo_root / ".github" / "workflows" / "governance-gate.yml").write_text(
        "jobs:\n  governance:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    (repo_root / ".github" / "workflows" / "test.yml").write_text(
        "jobs:\n  unit:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    (repo_root / ".github" / "workflows" / "security.yml").write_text(
        "jobs:\n  allowlist_guard:\n    name: Allowlist Guard\n  semgrep:\n    name: Semgrep\n  bandit:\n    name: Bandit\n  gitleaks:\n    name: Gitleaks\n  dep_audit:\n    name: Dependency Audit & SBOM\n",
        encoding="utf-8",
    )
    (repo_root / ".github" / "workflows" / "markdown.yml").write_text(
        "jobs:\n  docs-gate:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    (repo_root / "docs" / "ci-config.md").write_text(
        "\n".join(
            [
                "| `required_jobs` の値 | この repo での正本 workflow | この repo で確認する job / check | 備考 |",
                "| --- | --- | --- | --- |",
                "| `governance-gate` | `.github/workflows/governance-gate.yml` | job `governance` | ok |",
                "| `python-ci` | `.github/workflows/test.yml` | job `unit` | ok |",
                "| `security-ci` | `.github/workflows/security.yml` | `Allowlist Guard`, `Semgrep`, `Bandit`, `Gitleaks`, `Dependency Audit & SBOM` | ok |",
                "| `docs-gate` | `.github/workflows/markdown.yml` | job `docs-gate` | ok |",
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
    (repo_root / ".github" / "workflows" / "test.yml").write_text(
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
    assert "Workflow '.github/workflows/test.yml' does not mention expected check/job 'unit' for 'python-ci'." in result.errors


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


class TestCheckerStages:
    """Test checker_stages validation."""

    def test_load_checker_stages_from_policy(self, tmp_path: Path) -> None:
        policy = tmp_path / "policy.yaml"
        policy.write_text(
            "ci:\n  checker_stages:\n    RG-002: enforce\n    RG-003: warn\n",
            encoding="utf-8",
        )
        stages = load_checker_stages(policy)
        assert stages == {"RG-002": "enforce", "RG-003": "warn"}

    def test_load_checker_stages_empty_when_missing(self, tmp_path: Path) -> None:
        policy = tmp_path / "policy.yaml"
        policy.write_text("ci:\n  required_jobs:\n    - docs-gate\n", encoding="utf-8")
        stages = load_checker_stages(policy)
        assert stages == {}

    def test_validate_checker_stages_invalid_stage(self, tmp_path: Path) -> None:
        repo_root = tmp_path
        (repo_root / ".github" / "workflows").mkdir(parents=True)
        (repo_root / "docs").mkdir(parents=True)
        (repo_root / "governance").mkdir(parents=True)

        (repo_root / "governance" / "policy.yaml").write_text(
            "ci:\n  required_jobs:\n    - docs-gate\n  checker_stages:\n    RG-002: invalid_stage\n",
            encoding="utf-8",
        )
        (repo_root / ".github" / "workflows" / "markdown.yml").write_text(
            "jobs:\n  docs-gate:\n    runs-on: ubuntu-latest\n",
            encoding="utf-8",
        )
        (repo_root / "docs" / "ci-config.md").write_text(
            "| `docs-gate` | `.github/workflows/markdown.yml` | job `docs-gate` | RG-002 mentioned |\n",
            encoding="utf-8",
        )

        result = validate_ci_gate_matrix(
            repo_root=repo_root,
            policy_path=repo_root / "governance" / "policy.yaml",
            ci_config_path=repo_root / "docs" / "ci-config.md",
        )

        assert any("Invalid stage 'invalid_stage'" in e for e in result.errors)

    def test_validate_checker_stages_warns_missing_doc(self, tmp_path: Path) -> None:
        repo_root = tmp_path
        (repo_root / ".github" / "workflows").mkdir(parents=True)
        (repo_root / "docs").mkdir(parents=True)
        (repo_root / "governance").mkdir(parents=True)

        (repo_root / "governance" / "policy.yaml").write_text(
            "ci:\n  required_jobs:\n    - docs-gate\n  checker_stages:\n    RG-999: warn\n",
            encoding="utf-8",
        )
        (repo_root / ".github" / "workflows" / "markdown.yml").write_text(
            "jobs:\n  docs-gate:\n    runs-on: ubuntu-latest\n",
            encoding="utf-8",
        )
        # ci-config.md does NOT contain RG-999
        (repo_root / "docs" / "ci-config.md").write_text(
            "| docs-gate | workflow | docs-gate | other gate IDs |\n",
            encoding="utf-8",
        )

        result = validate_ci_gate_matrix(
            repo_root=repo_root,
            policy_path=repo_root / "governance" / "policy.yaml",
            ci_config_path=repo_root / "docs" / "ci-config.md",
        )

        # RG-999 should trigger warning because it's not in ci-config.md
        assert any("RG-999" in w for w in result.warnings)
