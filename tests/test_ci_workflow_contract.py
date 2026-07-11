from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
USES_RE = re.compile(r"^\s*(?:-\s+)?uses:\s*([^@\s]+)@([^\s#]+)", re.MULTILINE)


def test_test_workflow_is_fail_closed() -> None:
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")

    forbidden = (
        "CI_STRICT",
        "pip install mypy || true",
        "pip install pytest || true",
        "if pytest ",
        "if mypy ",
        "if pipx run ruff ",
    )
    for marker in forbidden:
        assert marker not in workflow

    assert "unit:" in workflow
    assert "--cov=tools" in workflow
    assert "--cov=security_headers" in workflow
    assert "--cov-fail-under=80" in workflow
    assert "uv build" in workflow
    assert workflow.count("fetch-depth: 0") == 5


def test_metrics_workflow_does_not_present_fixture_as_real_evidence() -> None:
    workflow = (ROOT / ".github" / "workflows" / "markdown.yml").read_text(
        encoding="utf-8"
    )

    assert "metrics-contract-smoke:" in workflow
    assert "Generate qa-metrics.json" not in workflow
    assert "real - RG-001" not in workflow
    assert "check_version_consistency.py --check" in workflow
    assert "check_docs_review_due.py --check" in workflow


def test_external_actions_are_pinned_to_commit_sha() -> None:
    failures: list[str] = []
    for workflow in sorted((ROOT / ".github" / "workflows").rglob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        for action, ref in USES_RE.findall(text):
            if action.startswith("./"):
                continue
            if not SHA_RE.fullmatch(ref):
                failures.append(f"{workflow.relative_to(ROOT)}: {action}@{ref}")

    assert failures == []


def test_security_permissions_are_least_privilege() -> None:
    workflow = (ROOT / ".github" / "workflows" / "security.yml").read_text(
        encoding="utf-8"
    )

    assert "contents: read" in workflow
    assert "security-events: write" not in workflow
    assert "bandit==1.9.4" in workflow
    assert "pip-audit==2.10.0" in workflow
    assert "bandit -q -c pyproject.toml" in workflow


def test_repository_ignores_python_bytecode() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "__pycache__/" in ignore
    assert "*.py[cod]" in ignore
