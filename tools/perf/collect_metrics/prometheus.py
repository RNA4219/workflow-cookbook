# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Prometheus metrics parsing and loading.

Fetches and parses metrics from Prometheus HTTP endpoint.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Mapping

from .security import MetricsCollectionError, validate_url
from .helpers import (
    coerce_float,
    parse_metric_name_and_labels,
    precision_mode_label_key,
)
from .extractor import MetricExtractor

LOGGER = logging.getLogger(__name__)


def parse_prometheus(text: str, extractor: MetricExtractor) -> dict[str, float]:
    """Parse Prometheus text format into metrics dict."""
    raw: dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        name_token = parts[0]
        raw_value = parts[-1]
        metric_name, labels = parse_metric_name_and_labels(name_token)
        value = coerce_float(raw_value)
        if value is None:
            continue
        raw[metric_name] = raw.get(metric_name, 0.0) + value
        precision_mode = labels.get("precision_mode") if labels else None
        if precision_mode:
            label_key = precision_mode_label_key(metric_name, precision_mode)
            raw[label_key] = raw.get(label_key, 0.0) + value

    metrics: dict[str, float] = {}
    extractor.capture_numeric(raw, metrics)
    return metrics


def load_prometheus(metrics_url: str, extractor: MetricExtractor) -> Mapping[str, float]:
    """Load metrics from Prometheus HTTP endpoint."""
    validate_url(metrics_url, context="metrics_url")
    try:
        with urllib.request.urlopen(metrics_url) as response:  # nosec B310  # URL validated by validate_url above, HTTPS enforced
            payload = response.read()
    except OSError as exc:
        raise MetricsCollectionError(f"Failed to read metrics from {metrics_url}: {exc}") from exc
    return parse_prometheus(payload.decode("utf-8"), extractor)


def load_structured_log(path: Path, extractor: MetricExtractor) -> Mapping[str, float]:
    """Load metrics from structured JSON log file."""
    if not path.exists():
        raise MetricsCollectionError(f"Structured log not found: {path}")
    metrics: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            extractor.capture_structured(parsed, metrics)
            statistics = parsed.get("statistics")
            if isinstance(statistics, Mapping):
                extractor.capture_structured(statistics, metrics)
            nested = parsed.get("metrics")
            if isinstance(nested, Mapping):
                extractor.capture_structured(nested, metrics, overwrite=True)
                statistics = nested.get("statistics")
                if isinstance(statistics, Mapping):
                    extractor.capture_structured(statistics, metrics, overwrite=True)
    return metrics


# Alias for backwards compatibility with tests
_parse_prometheus = parse_prometheus