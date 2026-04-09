# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal env
    class _MiniYamlModule:
        @staticmethod
        def safe_load(content: str) -> dict[str, dict[str, object]]:
            result: dict[str, dict[str, object]] = {}
            current_key: str | None = None
            for raw_line in content.splitlines():
                line = raw_line.rstrip()
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if not line.startswith(" "):
                    key, _, _value = stripped.partition(":")
                    current_key = key.strip()
                    result[current_key] = {}
                    continue
                if current_key is None:
                    continue
                key, _, value = stripped.partition(":")
                parsed: object = value.strip()
                if parsed in {"fail", "warn", "min", "max"}:
                    pass
                else:
                    try:
                        parsed = float(parsed)
                    except ValueError:
                        parsed = parsed
                result[current_key][key.strip()] = parsed
            return result

    yaml = _MiniYamlModule()  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_JSON = ROOT / ".ga/qa-metrics.json"
DEFAULT_THRESHOLDS = ROOT / "governance/metrics_thresholds.yaml"


class MetricsThresholdError(RuntimeError):
    """Raised when metrics threshold validation could not run."""


@dataclass(frozen=True)
class ThresholdRule:
    metric: str
    comparator: str
    threshold: float
    level: str
    unit: str = ""

    def evaluate(self, metrics: Mapping[str, object]) -> str | None:
        if self.metric not in metrics:
            return f"{self.metric}: metric is missing"
        raw_value = metrics[self.metric]
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise MetricsThresholdError(
                f"{self.metric}: metric value must be numeric, got {raw_value!r}"
            ) from exc
        if self.comparator == "min":
            passed = value >= self.threshold
            expectation = f">= {self.threshold:g}"
        elif self.comparator == "max":
            passed = value <= self.threshold
            expectation = f"<= {self.threshold:g}"
        else:  # pragma: no cover - protected by parser validation
            raise MetricsThresholdError(f"{self.metric}: unknown comparator {self.comparator!r}")
        if passed:
            return None
        unit_suffix = f" {self.unit}" if self.unit else ""
        return (
            f"{self.metric}: {value:g}{unit_suffix} does not satisfy "
            f"{self.comparator} threshold ({expectation}{unit_suffix})"
        )


def _load_thresholds(path: Path) -> list[ThresholdRule]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MetricsThresholdError(f"Threshold config not found: {path}") from exc
    except Exception as exc:  # pragma: no cover - parser specific
        raise MetricsThresholdError(f"Failed to parse threshold config {path}: {exc}") from exc
    if raw is None or not isinstance(raw, Mapping):
        raise MetricsThresholdError(f"Threshold config must be a mapping: {path}")

    rules: list[ThresholdRule] = []
    for metric, config in raw.items():
        if not isinstance(config, Mapping):
            raise MetricsThresholdError(f"{metric}: threshold config must be a mapping")
        comparator = str(config.get("comparator", "")).strip()
        if comparator not in {"min", "max"}:
            raise MetricsThresholdError(f"{metric}: comparator must be 'min' or 'max'")
        level = str(config.get("level", "fail")).strip()
        if level not in {"fail", "warn"}:
            raise MetricsThresholdError(f"{metric}: level must be 'fail' or 'warn'")
        try:
            threshold = float(config["threshold"])
        except KeyError as exc:
            raise MetricsThresholdError(f"{metric}: threshold is required") from exc
        except (TypeError, ValueError) as exc:
            raise MetricsThresholdError(f"{metric}: threshold must be numeric") from exc
        unit = str(config.get("unit", "")).strip()
        rules.append(
            ThresholdRule(
                metric=str(metric),
                comparator=comparator,
                threshold=threshold,
                level=level,
                unit=unit,
            )
        )
    return rules


def _load_metrics(path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MetricsThresholdError(f"Metrics JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MetricsThresholdError(f"Metrics JSON is invalid: {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise MetricsThresholdError(f"Metrics JSON must be an object: {path}")
    return loaded


def evaluate_thresholds(
    metrics: Mapping[str, object], rules: list[ThresholdRule]
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    for rule in rules:
        result = rule.evaluate(metrics)
        if result is None:
            continue
        if rule.level == "warn":
            warnings.append(result)
        else:
            failures.append(result)
    return failures, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate collected QA metrics against governance thresholds"
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=DEFAULT_METRICS_JSON,
        help="Path to collected metrics JSON (default: .ga/qa-metrics.json)",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=DEFAULT_THRESHOLDS,
        help="Path to metrics threshold config",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate metrics and return non-zero when fail-level thresholds are violated",
    )
    args = parser.parse_args(argv)

    try:
        metrics = _load_metrics(args.metrics_json)
        rules = _load_thresholds(args.thresholds)
        failures, warnings = evaluate_thresholds(metrics, rules)
    except MetricsThresholdError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    summary = {
        "metrics_json": str(args.metrics_json),
        "thresholds": str(args.thresholds),
        "failures": failures,
        "warnings": warnings,
    }
    print(json.dumps(summary, ensure_ascii=False))
    if warnings:
        print("Threshold warnings:", file=sys.stderr)
        for warning in warnings:
            print(f"- {warning}", file=sys.stderr)
    if failures:
        print("Threshold failures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
