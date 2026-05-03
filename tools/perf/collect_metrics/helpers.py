# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Metric extraction helper functions.

Utility functions for coercing values, deriving averages/ratios, and precision mode handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def coerce_float(value: object) -> float | None:
    """Convert value to float if possible."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def coerce_optional_path(value: object | None) -> Path | None:
    """Convert value to Path if possible."""
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    text = str(value)
    if not text:
        return None
    return Path(text)


def coerce_numeric_mapping(source: Mapping[str, object]) -> dict[str, float]:
    """Extract numeric values from a mapping."""
    numeric: dict[str, float] = {}
    for key, value in source.items():
        if isinstance(value, Mapping):
            continue
        coerced = coerce_float(value)
        if coerced is not None:
            numeric[key] = coerced
    return numeric


def lookup_nested_value(source: Mapping[str, object], path: Sequence[str]) -> object | None:
    """Look up nested value in mapping by path segments."""
    current: object = source
    for segment in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(segment)
    return current


OVERWRITE_KEYS: frozenset[str] = frozenset(
    {
        "checklist_compliance_rate",
        "review_latency",
        "compress_ratio",
        "semantic_retention",
        "reopen_rate",
        "spec_completeness",
    }
)


def derive_average(raw: Mapping[str, float], prefixes: Sequence[tuple[str, float]]) -> float | None:
    """Derive average from _sum/_count pairs."""
    for prefix, scale in prefixes:
        sum_key = f"{prefix}_sum"
        count_key = f"{prefix}_count"
        total = raw.get(sum_key)
        count = raw.get(count_key)
        if total is None or count in (None, 0.0):
            continue
        return (total / count) / scale
    return None


def derive_ratio_by_suffixes(
    raw: Mapping[str, float],
    numerator_suffixes: Sequence[str],
    denominator_suffixes: Sequence[str],
) -> float | None:
    """Derive ratio by matching numerator/denominator suffixes."""
    for numerator_suffix in numerator_suffixes:
        for name, value in raw.items():
            if not name.endswith(numerator_suffix):
                continue
            base = name[: -len(numerator_suffix)]
            numerator = value
            for suffix in denominator_suffixes:
                denominator_key = f"{base}{suffix}"
                denominator = raw.get(denominator_key)
                if denominator in (None, 0.0):
                    continue
                return numerator / denominator
    return None


def derive_checklist_compliance(raw: Mapping[str, float]) -> float | None:
    """Derive checklist compliance rate."""
    direct = raw.get("checklist_compliance_rate")
    if direct is not None:
        return direct
    suffix_pairs: Sequence[tuple[str, Sequence[str]]] = (
        ("_compliant_total", ("_total", "_count")),
        ("_compliant_count", ("_count", "_total")),
        ("_checked_total", ("_total", "_count")),
    )
    for numerator_suffix, denominator_suffixes in suffix_pairs:
        for name, value in raw.items():
            if not name.endswith(numerator_suffix):
                continue
            base = name[: -len(numerator_suffix)]
            numerator = value
            for suffix in denominator_suffixes:
                denominator_key = f"{base}{suffix}"
                denominator = raw.get(denominator_key)
                if denominator in (None, 0.0):
                    continue
                return numerator / denominator
    ratio = derive_ratio_by_suffixes(raw, ("_compliant",), ("_total", "_count", "_all"))
    if ratio is not None:
        return ratio
    return None


def derive_review_latency(raw: Mapping[str, float]) -> float | None:
    """Derive review latency from multiple prefix sources."""
    direct = raw.get("review_latency")
    if direct is not None:
        return direct
    REVIEW_LATENCY_AGGREGATE_PREFIXES = (
        ("workflow_review_latency_seconds", 3600.0),
        ("workflow_review_latency_minutes", 60.0),
        ("workflow_review_latency_hours", 1.0),
        ("review_latency_seconds", 3600.0),
        ("review_latency_minutes", 60.0),
        ("review_latency_hours", 1.0),
        ("legacy_review_latency_seconds", 3600.0),
        ("legacy_review_latency_minutes", 60.0),
        ("legacy_review_latency_hours", 1.0),
    )
    return derive_average(raw, REVIEW_LATENCY_AGGREGATE_PREFIXES)


def derive_task_seed_cycle_time_minutes(raw: Mapping[str, float]) -> float | None:
    """Derive task seed cycle time."""
    direct = raw.get("task_seed_cycle_time_minutes")
    if direct is not None:
        return direct
    prefixes: Sequence[tuple[str, float]] = (
        ("task_seed_cycle_time_seconds", 60.0),
        ("docops_task_seed_cycle_time_seconds", 60.0),
        ("task_seed_cycle_time_minutes", 1.0),
    )
    return derive_average(raw, prefixes)


def derive_birdseye_refresh_delay_minutes(raw: Mapping[str, float]) -> float | None:
    """Derive birdseye refresh delay."""
    direct = raw.get("birdseye_refresh_delay_minutes")
    if direct is not None:
        return direct
    prefixes: Sequence[tuple[str, float]] = (
        ("birdseye_refresh_delay_seconds", 60.0),
        ("docops_birdseye_refresh_delay_seconds", 60.0),
        ("birdseye_refresh_delay_minutes", 1.0),
    )
    return derive_average(raw, prefixes)


def extract_precision_mode_mapping(
    raw: Mapping[str, object], mode: str, nested_keys: Sequence[str]
) -> float | None:
    """Extract value for specific mode from precision mode mapping."""
    direct = coerce_float(raw.get(mode))
    if direct is not None:
        return direct
    for nested_key in nested_keys:
        nested = raw.get(nested_key)
        if isinstance(nested, Mapping):
            nested_value = coerce_float(nested.get(mode))
            if nested_value is not None:
                return nested_value
    return None


def extract_structured_precision_mode(
    source: Mapping[str, object], mode_keys: Sequence[str]
) -> str | None:
    """Extract active precision mode from structured source."""
    for key in mode_keys:
        direct = source.get(key)
        if isinstance(direct, str) and direct:
            return direct
        if "." in key:
            parts = key.split(".")
            nested = lookup_nested_value(source, parts)
            if isinstance(nested, str) and nested:
                return nested
    return None


def extract_numeric_precision_mode(
    source: Mapping[str, float], mode_metric_names: Sequence[str]
) -> str | None:
    """Extract active precision mode from numeric source."""
    best_mode: str | None = None
    best_value = float("-inf")
    for metric_name in mode_metric_names:
        prefix = f"{metric_name}|"
        for key, value in source.items():
            if not key.startswith(prefix):
                continue
            label = key[len(prefix):]
            label_key, _, label_value = label.partition("=")
            if not label_value:
                continue
            if label_key not in ("precision_mode", "mode"):
                continue
            if value > best_value:
                best_mode = label_value
                best_value = value
        if best_mode is not None:
            return best_mode
    return None


def precision_mode_label_key(metric_name: str, mode: str) -> str:
    """Generate label key for precision mode."""
    return f"{metric_name}|precision_mode={mode}"


def parse_metric_name_and_labels(name_token: str) -> tuple[str, dict[str, str]]:
    """Parse Prometheus metric name and labels."""
    if "{" not in name_token:
        return name_token, {}
    base, _, remainder = name_token.partition("{")
    labels: dict[str, str] = {}
    for chunk in remainder.rstrip("}").split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        key, _, raw_value = stripped.partition("=")
        if not raw_value:
            continue
        value = raw_value.strip().strip('"')
        labels[key.strip()] = value
    return base, labels


__all__ = [
    "coerce_float",
    "coerce_optional_path",
    "coerce_numeric_mapping",
    "lookup_nested_value",
    "OVERWRITE_KEYS",
    "derive_average",
    "derive_ratio_by_suffixes",
    "derive_checklist_compliance",
    "derive_review_latency",
    "derive_task_seed_cycle_time_minutes",
    "derive_birdseye_refresh_delay_minutes",
    "extract_precision_mode_mapping",
    "extract_structured_precision_mode",
    "extract_numeric_precision_mode",
    "precision_mode_label_key",
    "parse_metric_name_and_labels",
]