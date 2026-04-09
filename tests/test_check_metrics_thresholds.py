from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Import the module for direct testing
import tools.ci.check_metrics_thresholds as mt_module


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.ci.check_metrics_thresholds", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


# Direct function tests for coverage
class TestThresholdRule:
    def test_evaluate_returns_none_for_passing_min(self) -> None:
        rule = mt_module.ThresholdRule(
            metric="test_metric",
            comparator="min",
            threshold=95.0,
            level="fail",
            unit="percent",
        )
        result = rule.evaluate({"test_metric": 96.0})
        assert result is None

    def test_evaluate_returns_none_for_passing_max(self) -> None:
        rule = mt_module.ThresholdRule(
            metric="test_metric",
            comparator="max",
            threshold=60.0,
            level="warn",
            unit="minutes",
        )
        result = rule.evaluate({"test_metric": 50.0})
        assert result is None

    def test_evaluate_returns_message_for_failing(self) -> None:
        rule = mt_module.ThresholdRule(
            metric="test_metric",
            comparator="min",
            threshold=95.0,
            level="fail",
            unit="percent",
        )
        result = rule.evaluate({"test_metric": 80.0})
        assert result is not None
        assert "test_metric" in result
        assert "80" in result

    def test_evaluate_returns_missing_message(self) -> None:
        rule = mt_module.ThresholdRule(
            metric="missing_metric",
            comparator="min",
            threshold=95.0,
            level="fail",
        )
        result = rule.evaluate({"other_metric": 100.0})
        assert "missing" in result.lower()


class TestLoadThresholds:
    def test_loads_valid_thresholds(self, tmp_path: Path) -> None:
        thresholds_path = tmp_path / "thresholds.yaml"
        thresholds_path.write_text(
            textwrap.dedent("""
                test_metric:
                  comparator: min
                  threshold: 95
                  level: fail
                  unit: percent
            """).strip() + "\n",
            encoding="utf-8",
        )

        rules = mt_module._load_thresholds(thresholds_path)
        assert len(rules) == 1
        assert rules[0].metric == "test_metric"
        assert rules[0].threshold == 95.0

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(mt_module.MetricsThresholdError):
            mt_module._load_thresholds(tmp_path / "missing.yaml")

    def test_raises_for_invalid_comparator(self, tmp_path: Path) -> None:
        thresholds_path = tmp_path / "thresholds.yaml"
        thresholds_path.write_text(
            textwrap.dedent("""
                test_metric:
                  comparator: invalid
                  threshold: 95
            """).strip() + "\n",
            encoding="utf-8",
        )

        with pytest.raises(mt_module.MetricsThresholdError) as exc_info:
            mt_module._load_thresholds(thresholds_path)
        assert "comparator" in str(exc_info.value).lower()

    def test_raises_for_invalid_level(self, tmp_path: Path) -> None:
        thresholds_path = tmp_path / "thresholds.yaml"
        thresholds_path.write_text(
            textwrap.dedent("""
                test_metric:
                  comparator: min
                  threshold: 95
                  level: invalid
            """).strip() + "\n",
            encoding="utf-8",
        )

        with pytest.raises(mt_module.MetricsThresholdError) as exc_info:
            mt_module._load_thresholds(thresholds_path)
        assert "level" in str(exc_info.value).lower()


class TestLoadMetrics:
    def test_loads_valid_metrics(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(
            json.dumps({"test_metric": 95.0}),
            encoding="utf-8",
        )

        result = mt_module._load_metrics(metrics_path)
        assert result["test_metric"] == 95.0

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(mt_module.MetricsThresholdError):
            mt_module._load_metrics(tmp_path / "missing.json")

    def test_raises_for_invalid_json(self, tmp_path: Path) -> None:
        metrics_path = tmp_path / "invalid.json"
        metrics_path.write_text("not json", encoding="utf-8")

        with pytest.raises(mt_module.MetricsThresholdError):
            mt_module._load_metrics(metrics_path)


class TestEvaluateThresholds:
    def test_returns_failures_and_warnings(self) -> None:
        metrics = {"metric1": 80.0, "metric2": 70.0}
        rules = [
            mt_module.ThresholdRule("metric1", "min", 95.0, "fail", "percent"),
            mt_module.ThresholdRule("metric2", "min", 90.0, "warn", "percent"),
        ]

        failures, warnings = mt_module.evaluate_thresholds(metrics, rules)

        assert len(failures) == 1
        assert len(warnings) == 1

    def test_returns_empty_for_passing(self) -> None:
        metrics = {"metric1": 100.0}
        rules = [
            mt_module.ThresholdRule("metric1", "min", 95.0, "fail", "percent"),
        ]

        failures, warnings = mt_module.evaluate_thresholds(metrics, rules)

        assert failures == []
        assert warnings == []


# CLI tests (kept for integration testing)
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
