from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docs_review_due_compatibility_imports() -> None:
    from tools.ci import check_docs_review_due
    from tools.ci.docs_review_due import artifacts, models, scanner
    assert check_docs_review_due.DocReviewStatus is models.DocReviewStatus
    assert check_docs_review_due._scan_docs is scanner._scan_docs
    assert check_docs_review_due._build_nudges is artifacts._build_nudges


def _write_doc(
    path: Path, *, owner: str, due: str, reviewed: str = "2026-01-01", status: str = "active"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "---",
                "intent_id: INT-TEST",
                f"owner: {owner}",
                f"status: {status}",
                f"last_reviewed_at: {reviewed}",
                f"next_review_due: {due}",
                "---",
                "",
                "# Test Doc",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.ci.check_docs_review_due", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_docs_review_outputs_owner_summary_nudges_task_seed_and_update_plan(tmp_path: Path) -> None:
    _write_doc(tmp_path / "docs" / "critical.md", owner="docs-core", due="2026-05-01")
    _write_doc(tmp_path / "docs" / "upcoming.md", owner="ops-core", due="2026-07-03")

    owner_summary = tmp_path / "owner-summary.json"
    nudges = tmp_path / "nudges.json"
    task_seed = tmp_path / "task.md"
    update_plan = tmp_path / "update-plan.json"

    result = _run_cli(
        "--root",
        str(tmp_path),
        "--today",
        "2026-07-01",
        "--max-days-overdue",
        "30",
        "--warn-days",
        "7",
        "--owner-summary-json",
        str(owner_summary),
        "--nudge-output",
        str(nudges),
        "--task-seed-output",
        str(task_seed),
        "--review-update-plan-output",
        str(update_plan),
    )

    assert result.returncode == 0, result.stderr
    summary_payload = json.loads(owner_summary.read_text(encoding="utf-8"))
    assert summary_payload["owners"]["docs-core"]["counts"]["critical"] == 1
    assert summary_payload["owners"]["ops-core"]["counts"]["upcoming"] == 1

    nudge_payload = json.loads(nudges.read_text(encoding="utf-8"))
    assert [item["target_ref"].replace("\\", "/") for item in nudge_payload] == [
        "docs/critical.md",
        "docs/upcoming.md",
    ]
    assert nudge_payload[0]["priority"] == "high"
    assert nudge_payload[0]["category"] == "review_due"

    task_text = task_seed.read_text(encoding="utf-8")
    assert "Docs Review Due Remediation 2026-07-01" in task_text
    assert "| docs-core | 1 | 0 | 0 | 0 |" in task_text

    update_payload = json.loads(update_plan.read_text(encoding="utf-8"))
    assert update_payload["updates"][0]["suggested"]["last_reviewed_at"] == "2026-07-01"
    assert update_payload["updates"][0]["suggested"]["next_review_due"] == "2026-07-31"


def test_docs_review_check_fails_only_for_critical_overdue(tmp_path: Path) -> None:
    _write_doc(tmp_path / "docs" / "overdue.md", owner="docs-core", due="2026-06-15")

    result = _run_cli(
        "--root",
        str(tmp_path),
        "--today",
        "2026-07-01",
        "--max-days-overdue",
        "30",
        "--check",
    )

    assert result.returncode == 0, result.stderr

    result = _run_cli(
        "--root",
        str(tmp_path),
        "--today",
        "2026-07-30",
        "--max-days-overdue",
        "30",
        "--check",
    )

    assert result.returncode == 1
    assert "critically overdue" in result.stderr


def test_docs_review_ignores_terminal_records_and_examples(tmp_path: Path) -> None:
    _write_doc(
        tmp_path / "docs" / "completed.md",
        owner="docs-core",
        due="2025-01-01",
        status="completed",
    )
    _write_doc(
        tmp_path / "examples" / "fixture.md",
        owner="sample-author",
        due="2025-01-01",
    )

    result = _run_cli(
        "--root",
        str(tmp_path),
        "--today",
        "2026-07-11",
        "--check",
    )

    assert result.returncode == 0, result.stderr
    assert "Docs scanned: 0" in result.stdout
