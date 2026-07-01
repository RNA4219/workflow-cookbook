from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, content: str = "# Doc\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _front_matter(template_version: str) -> str:
    return "\n".join(
        [
            "---",
            "intent_id: INT-TEST",
            "owner: docs-core",
            "status: active",
            "last_reviewed_at: 2026-07-01",
            "next_review_due: 2026-08-01",
            f"template_version: {template_version}",
            "---",
            "",
            "# Doc",
            "",
        ]
    )


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.ci.check_adoption_tier", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_adoption_tier_detects_tier_two_and_missing_tier_three(tmp_path: Path) -> None:
    for name in ["README.md", "HUB.codex.md", "BLUEPRINT.md", "RUNBOOK.md", "GUARDRAILS.md", "EVALUATION.md"]:
        _write(tmp_path / name)

    result = _run_cli("--repo", str(tmp_path), "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["current_tier"] == 2
    assert "docs/tasks" in payload["missing_for_next_tier"]


def test_adoption_tier_check_fails_below_min_tier(tmp_path: Path) -> None:
    _write(tmp_path / "README.md")

    result = _run_cli("--repo", str(tmp_path), "--min-tier", "1", "--check")

    assert result.returncode == 1


def test_adoption_tier_detects_template_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    template_root = tmp_path / "templates"
    _write(repo / "README.md")
    _write(repo / "HUB.codex.md", _front_matter("0.9.0"))
    _write(repo / "BLUEPRINT.md")
    _write(template_root / "HUB.codex.md.template", _front_matter("1.0.0"))

    result = _run_cli(
        "--repo",
        str(repo),
        "--template-root",
        str(template_root),
        "--check-drift",
        "--check",
        "--json",
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["drifted"] is True
    assert payload["drift_checks"][0]["path"] == "HUB.codex.md"
