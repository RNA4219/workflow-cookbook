from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ci.check_downstream_onboarding import assess_downstream_repo

ROOT = Path(__file__).resolve().parents[1]


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
        _write(tmp_path / rel_path, "# Downstream documentation")
    _write(tmp_path / "docs/birdseye/index.json", '{"nodes": {"README.md": {"role": "overview"}}}')
    _write(tmp_path / "docs/birdseye/hot.json", '{"nodes": [{"id": "README.md"}]}')
    _write(tmp_path / "docs/birdseye/caps/README.md.json", '{"id": "README.md", "summary": "Entry point"}')
    _write(
        tmp_path / ".github" / "workflows" / "workflow-cookbook.yml",
        "generate_acceptance_index\ncheck_branch_protection\ncheck_ci_gate_matrix\ncheck_security_posture",
    )

    report = assess_downstream_repo(tmp_path, min_tier=3)

    assert report["status"] == "ready"
    assert report["missing_ci_signals"] == []
    _write(tmp_path / "docs/birdseye/index.json", "{}")
    assert assess_downstream_repo(tmp_path, min_tier=3)["status"] == "needs_work"


def test_downstream_onboarding_reports_missing_ci(tmp_path: Path) -> None:
    _write(tmp_path / "README.md", "# Repo")

    report = assess_downstream_repo(tmp_path, min_tier=2)

    assert report["status"] == "needs_work"
    assert "acceptance_index" in report["missing_ci_signals"]



def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.ci.check_downstream_onboarding", *args],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )


def _ready_repo(repo: Path, extension: str = "yml") -> None:
    for name in ("README.md", "HUB.codex.md", "BLUEPRINT.md", "RUNBOOK.md", "GUARDRAILS.md", "EVALUATION.md"):
        _write(repo / name, "# Documentation\n")
    _write(repo / ".github/workflows" / f"onboarding.{extension}",
           "name: Onboarding\non: push\njobs:\n  verify:\n    runs-on: ubuntu-latest\n    steps:\n"
           "      - run: |\n          generate_acceptance_index\n          check_branch_protection\n"
           "          check_ci_gate_matrix\n          check_security_posture\n")


@pytest.mark.parametrize("extension", ["yml", "yaml", "YML", "YAML"])
def test_both_workflow_extensions_are_detected(tmp_path: Path, extension: str) -> None:
    _ready_repo(tmp_path, extension)
    result = _run_cli("--repo", str(tmp_path), "--check", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ready"
    assert all(payload["ci_signals"].values())


@pytest.mark.parametrize("payload", [[], {}, None, [None], [{"path": "."}], [""], [" \t"], [{"repo": ""}], [{"repo": 1}]])
def test_empty_and_invalid_repo_list_fails(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "repos.json"
    _write(path, json.dumps(payload))
    result = _run_cli("--repo-list", str(path), "--check", "--json")
    assert result.returncode == 1
    assert result.stdout == ""
    assert "repo-list" in result.stderr
    assert "Traceback" not in result.stderr


def test_invalid_list_member_does_not_allow_partial_success(tmp_path: Path) -> None:
    repo = tmp_path / "ready"
    _ready_repo(repo)
    path = tmp_path / "repos.json"
    _write(path, json.dumps([str(repo), {"path": str(repo)}]))
    result = _run_cli("--repo-list", str(path), "--check", "--json")
    assert result.returncode == 1
    assert result.stdout == ""
    assert "repo-list item 1" in result.stderr


@pytest.mark.parametrize("contents", [b"{", b"\xff", None])
def test_unreadable_or_malformed_repo_list_has_clean_cli_error(tmp_path: Path, contents: bytes | None) -> None:
    path = tmp_path / "repos.json"
    if contents is not None:
        path.write_bytes(contents)
    result = _run_cli("--repo-list", str(path), "--check", "--json")
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip()
    assert "Traceback" not in result.stderr


def test_repo_list_assesses_all_valid_entries(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _ready_repo(first)
    _write(second / "README.md", "# Incomplete repo\n")
    path = tmp_path / "repos.json"
    _write(path, json.dumps([str(first), {"repo": str(second)}]))
    result = _run_cli("--repo-list", str(path), "--check", "--json")
    assert result.returncode == 1
    assert [item["status"] for item in json.loads(result.stdout)] == ["ready", "needs_work"]



def test_bom_repo_list_is_supported(tmp_path: Path) -> None:
    repo = tmp_path / "ready"
    _ready_repo(repo)
    path = tmp_path / "repos.json"
    path.write_text(json.dumps([{"repo": str(repo)}]), encoding="utf-8-sig")
    result = _run_cli("--repo-list", str(path), "--check", "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "ready"


def test_nul_repo_path_is_rejected_before_assessment(tmp_path: Path) -> None:
    path = tmp_path / "repos.json"
    _write(path, json.dumps(["invalid\0path"]))
    result = _run_cli("--repo-list", str(path), "--check", "--json")
    assert result.returncode == 1
    assert result.stdout == ""
    assert "repo-list item 0" in result.stderr
    assert "Traceback" not in result.stderr


def test_unreadable_workflow_has_clean_cli_error(tmp_path: Path) -> None:
    _ready_repo(tmp_path)
    (tmp_path / ".github/workflows/onboarding.yml").write_bytes(b"\xff")
    result = _run_cli("--repo", str(tmp_path), "--check", "--json")
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip()
    assert "Traceback" not in result.stderr
