# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
CLI entry point for collect_metrics.

Argument parsing and main execution flow.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, cast

try:
    import yaml
except ModuleNotFoundError:
    class _MiniYamlModule:
        @staticmethod
        def safe_load(content: str) -> dict[str, str]:
            result: dict[str, str] = {}
            for line in content.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                key, _, value = stripped.partition(":")
                result[key.strip()] = value.strip()
            return result
    yaml = _MiniYamlModule()

from .security import MetricsCollectionError
from .helpers import coerce_optional_path
from .extractor import MetricExtractor
from .definitions import build_default_metric_registry
from .prometheus import load_prometheus, load_structured_log

LOGGER = logging.getLogger(__name__)

_METRICS_PATH_ENV = "GOVERNANCE_METRICS_PATH"
_DEFAULT_METRICS_PATH = Path(__file__).resolve().parents[3] / "governance/metrics.yaml"
_METRICS_URL_ENV = "GOVERNANCE_METRICS_URL"
_LOG_PATH_ENV = "GOVERNANCE_METRICS_LOG_PATH"
_PUSHGATEWAY_URL_ENV = "GOVERNANCE_PUSHGATEWAY_URL"
MISSING_SOURCE_ERROR = "No metrics input configured: provide --metrics-url or --log-path"


@dataclass(frozen=True)
class SuiteConfig:
    """Configuration preset for a metrics collection suite."""
    metrics_url: str | None = None
    log_path: str | None = None
    output: str | None = None
    pushgateway_url: str | None = None


SUITES: dict[str, SuiteConfig] = {
    "qa": SuiteConfig(output=".ga/qa-metrics.json"),
}


@dataclass(frozen=True)
class MetricsCollectionPlan:
    """Resolved plan for metrics collection."""
    metrics_url: str | None
    log_path: Path | None
    pushgateway_url: str | None

    @classmethod
    def from_namespace(
        cls,
        args: object,
        *,
        env: Mapping[str, str] | None = None,
        suites: Mapping[str, SuiteConfig] | None = None,
    ) -> "MetricsCollectionPlan":
        env = env or {}
        suites = suites or SUITES
        suite_name = getattr(args, "suite", None)
        suite = suites.get(suite_name) if suite_name else None
        metrics_url = cast(
            str | None,
            getattr(args, "metrics_url", None)
            or env.get(_METRICS_URL_ENV)
            or (suite.metrics_url if suite else None),
        )
        log_path = coerce_optional_path(
            getattr(args, "log_path", None)
            or env.get(_LOG_PATH_ENV)
            or (suite.log_path if suite else None)
        )
        pushgateway_url = cast(
            str | None,
            getattr(args, "pushgateway_url", None)
            or env.get(_PUSHGATEWAY_URL_ENV)
            or (suite.pushgateway_url if suite else None),
        )
        plan = cls(
            metrics_url=metrics_url,
            log_path=log_path,
            pushgateway_url=pushgateway_url,
        )
        if not plan.metrics_url and plan.log_path is None:
            raise MetricsCollectionError(MISSING_SOURCE_ERROR)
        return plan


@dataclass(frozen=True)
class MetricsRunner:
    """Runner for executing metrics collection."""
    plan: MetricsCollectionPlan
    output_path: Path | None
    extractor: MetricExtractor

    def resolve_sources(self) -> list[Mapping[str, float]]:
        sources: list[Mapping[str, float]] = []
        if self.plan.metrics_url:
            sources.append(load_prometheus(self.plan.metrics_url, self.extractor))
        if self.plan.log_path:
            sources.append(load_structured_log(self.plan.log_path, self.extractor))
        return sources

    def finalize(self, metrics: Mapping[str, float], keys: tuple[str, ...]) -> None:
        import urllib.request
        from .security import validate_url

        if self.plan.pushgateway_url:
            validate_url(self.plan.pushgateway_url, context="pushgateway_url")
            lines = [f"{key} {format(metrics[key], 'g')}" for key in keys if key in metrics]
            payload = ("\n".join(lines) + "\n").encode("utf-8")
            request = urllib.request.Request(self.plan.pushgateway_url, data=payload, method="PUT")
            request.add_header("Content-Type", "text/plain; version=0.0.4")
            try:
                with urllib.request.urlopen(request) as response:
                    response.read()
            except OSError as exc:
                raise MetricsCollectionError(
                    f"Failed to push metrics to PushGateway at {self.plan.pushgateway_url}: {exc}"
                ) from exc

        payload = json.dumps(metrics, ensure_ascii=False)
        sys.stdout.write(payload + "\n")
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(payload + "\n", encoding="utf-8")


@dataclass(frozen=True)
class LoadedMetrics:
    """Loaded metrics configuration."""
    extractor: MetricExtractor
    keys: tuple[str, ...]


@functools.lru_cache(maxsize=1)
def load_metric_config() -> LoadedMetrics:
    """Load metrics configuration from YAML file."""
    registry = build_default_metric_registry()
    override = os.environ.get(_METRICS_PATH_ENV)
    if override:
        candidate = Path(override)
        path = candidate if candidate.is_absolute() else (_DEFAULT_METRICS_PATH.parent / candidate)
    else:
        path = _DEFAULT_METRICS_PATH
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MetricsCollectionError(f"Metrics definition file not found: {path}") from exc
    try:
        loaded = yaml.safe_load(content)
    except Exception as exc:
        raise MetricsCollectionError(f"Failed to parse metrics definition from {path}: {exc}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise MetricsCollectionError(f"Metrics definition in {path} must be a mapping")
    keys: list[str] = []
    percentage_keys: list[str] = []
    definitions: list = []
    missing: list[str] = []
    for raw_key, raw_description in loaded.items():
        key = str(raw_key)
        keys.append(key)
        description = "" if raw_description is None else str(raw_description)
        if "(%)" in description:
            percentage_keys.append(key)
        definition = registry.get(key)
        if definition is None:
            missing.append(key)
        else:
            definitions.append(definition)
    if missing:
        raise MetricsCollectionError(
            "Missing metric extractor definitions for: " + ", ".join(sorted(missing))
        )
    extractor = MetricExtractor(tuple(definitions), percentage_keys=tuple(percentage_keys))
    return LoadedMetrics(extractor=extractor, keys=tuple(keys))


# Alias for backwards compatibility with tests
_load_metric_config = load_metric_config


def metric_keys() -> tuple[str, ...]:
    """Return ordered metric keys."""
    return load_metric_config().keys


def percentage_keys() -> tuple[str, ...]:
    """Return keys that should be formatted as percentages."""
    return load_metric_config().extractor.percentage_keys()


def collect_metrics(sources: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Collect and merge metrics from multiple sources."""
    if not sources:
        raise MetricsCollectionError(MISSING_SOURCE_ERROR)
    return load_metric_config().extractor.merge(sources)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Collect performance metrics for post-processing")
    parser.add_argument("--suite", choices=sorted(SUITES), help="Preset input/output configuration")
    parser.add_argument("--metrics-url", help="Prometheus metrics endpoint URL")
    parser.add_argument("--log-path", type=Path, help="Path to structured operations log")
    parser.add_argument("--output", type=Path, help="File path to write collected metrics JSON")
    parser.add_argument("--pushgateway-url", help="Prometheus PushGateway endpoint URL")
    args = parser.parse_args(argv)

    suite = SUITES.get(args.suite) if args.suite else None
    output_path = args.output if args.output is not None else (
        Path(suite.output) if suite and suite.output else None
    )
    try:
        plan = MetricsCollectionPlan.from_namespace(args, env=os.environ, suites=SUITES)
    except MetricsCollectionError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    loaded = load_metric_config()
    runner = MetricsRunner(plan, output_path, loaded.extractor)

    try:
        metrics = collect_metrics(runner.resolve_sources())
        runner.finalize(metrics, loaded.keys)
    except MetricsCollectionError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())