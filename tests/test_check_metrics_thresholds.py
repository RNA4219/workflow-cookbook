from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.ci.check_metrics_thresholds", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_metrics_thresholds_pass_when_all_metrics_satisfy(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "checklist_compliance_rate": 96.0,
                "task_seed_cycle_time_minutes": 20.0,
                "birdseye_refresh_delay_minutes": 10.0,
                "review_latency": 2.0,
                "compress_ratio": 0.5,
                "semantic_retention": 0.9,
                "reopen_rate": 2.0,
                "spec_completeness": 95.0,
            }
        ),
        encoding="utf-8",
    )

    result = _run_cli("--check", "--metrics-json", str(metrics_path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["failures"] == []
    assert payload["warnings"] == []


def test_metrics_thresholds_fail_when_fail_rule_is_violated(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "checklist_compliance_rate": 80.0,
                "task_seed_cycle_time_minutes": 20.0,
                "birdseye_refresh_delay_minutes": 10.0,
                "review_latency": 2.0,
                "compress_ratio": 0.5,
                "semantic_retention": 0.9,
                "reopen_rate": 2.0,
                "spec_completeness": 95.0,
            }
        ),
        encoding="utf-8",
    )

    result = _run_cli("--check", "--metrics-json", str(metrics_path))

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["failures"] == [
        "checklist_compliance_rate: 80 percent does not satisfy min threshold (>= 95 percent)"
    ]


def test_metrics_thresholds_warn_without_failing_when_warn_rule_is_violated(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"birdseye_refresh_delay_minutes": 80.0}), encoding="utf-8")
    thresholds_path = tmp_path / "thresholds.yaml"
    thresholds_path.write_text(
        textwrap.dedent(
            """
            birdseye_refresh_delay_minutes:
              comparator: max
              threshold: 60
              level: warn
              unit: minutes
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    result = _run_cli(
        "--check",
        "--metrics-json",
        str(metrics_path),
        "--thresholds",
        str(thresholds_path),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["failures"] == []
    assert payload["warnings"] == [
        "birdseye_refresh_delay_minutes: 80 minutes does not satisfy max threshold (<= 60 minutes)"
    ]


def test_metrics_thresholds_fail_when_required_metric_missing(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"compress_ratio": 0.5}), encoding="utf-8")
    thresholds_path = tmp_path / "thresholds.yaml"
    thresholds_path.write_text(
        textwrap.dedent(
            """
            checklist_compliance_rate:
              comparator: min
              threshold: 95
              level: fail
              unit: percent
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    result = _run_cli(
        "--check",
        "--metrics-json",
        str(metrics_path),
        "--thresholds",
        str(thresholds_path),
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["failures"] == ["checklist_compliance_rate: metric is missing"]
