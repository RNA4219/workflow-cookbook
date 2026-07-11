# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Metric extraction rules.

Defines protocols and dataclasses for structured log and Prometheus metric extraction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from .helpers import (
    coerce_float,
    coerce_numeric_mapping,
    derive_average,
    derive_ratio_by_suffixes,
    extract_numeric_precision_mode,
    extract_precision_mode_mapping,
    extract_structured_precision_mode,
    precision_mode_label_key,
)


class StructuredRule(Protocol):
    """Protocol for structured log extraction rules."""

    @property
    def overwrite(self) -> bool: ...

    def extract(self, metric_key: str, source: Mapping[str, object]) -> float | None: ...


class NumericRule(Protocol):
    """Protocol for Prometheus numeric extraction rules."""

    def extract(self, metric_key: str, source: Mapping[str, float]) -> float | None: ...


@dataclass(frozen=True)
class DirectValueRule:
    """Extract direct value from structured source."""

    keys: tuple[str, ...] | None = None
    overwrite: bool = False

    def extract(self, metric_key: str, source: Mapping[str, object]) -> float | None:
        candidates = self.keys or (metric_key,)
        for key in candidates:
            value = coerce_float(source.get(key))
            if value is not None:
                return value
        return None


@dataclass(frozen=True)
class MappingRatioRule:
    """Extract ratio from nested mapping."""

    numerator_keys: tuple[str, ...]
    denominator_keys: tuple[str, ...]
    ratio_keys: tuple[str, ...] = ("ratio",)
    overwrite: bool = True

    def extract(self, metric_key: str, source: Mapping[str, object]) -> float | None:
        raw = source.get(metric_key)
        if not isinstance(raw, Mapping):
            return None
        numerator = None
        denominator = None
        for candidate in self.numerator_keys:
            numerator = coerce_float(raw.get(candidate))
            if numerator is not None:
                break
        for candidate in self.denominator_keys:
            denominator = coerce_float(raw.get(candidate))
            if denominator is not None and denominator != 0:
                break
        if numerator is not None and denominator is not None and denominator != 0:
            return numerator / denominator
        for candidate in self.ratio_keys:
            ratio = coerce_float(raw.get(candidate))
            if ratio is not None:
                return ratio
        return None


@dataclass(frozen=True)
class StructuredAverageRule:
    """Extract average from structured source using prefix matching."""

    prefixes: tuple[tuple[str, float], ...]
    overwrite: bool = False

    def extract(self, metric_key: str, source: Mapping[str, object]) -> float | None:
        numeric = coerce_numeric_mapping(source)
        if not numeric:
            return None
        return derive_average(numeric, self.prefixes)


@dataclass(frozen=True)
class DirectNumericRule:
    """Extract direct numeric value."""

    keys: tuple[str, ...] | None = None

    def extract(self, metric_key: str, source: Mapping[str, float]) -> float | None:
        candidates = self.keys or (metric_key,)
        for key in candidates:
            value = source.get(key)
            if value is not None:
                return value
        return None


@dataclass(frozen=True)
class NumericAverageRule:
    """Extract average from numeric source using prefix matching."""

    prefixes: tuple[tuple[str, float], ...]

    def extract(self, metric_key: str, source: Mapping[str, float]) -> float | None:
        return derive_average(source, self.prefixes)


@dataclass(frozen=True)
class SuffixRatioNumericRule:
    """Extract ratio by matching suffix patterns."""

    numerator_suffixes: tuple[str, ...]
    denominator_suffixes: tuple[str, ...]

    def extract(self, metric_key: str, source: Mapping[str, float]) -> float | None:
        return derive_ratio_by_suffixes(source, self.numerator_suffixes, self.denominator_suffixes)


@dataclass(frozen=True)
class NumericCallableRule:
    """Extract using custom callable function."""

    function: Callable[[Mapping[str, float]], float | None]

    def extract(self, metric_key: str, source: Mapping[str, float]) -> float | None:
        return self.function(source)


@dataclass(frozen=True)
class PrecisionModeStructuredRule:
    """Extract value for specific precision mode from structured source."""

    source_key: str
    mode: str
    nested_keys: tuple[str, ...] = ("rate", "modes")
    overwrite: bool = False

    def extract(self, metric_key: str, source: Mapping[str, object]) -> float | None:
        raw = source.get(self.source_key)
        if not isinstance(raw, Mapping):
            return None
        value = extract_precision_mode_mapping(raw, self.mode, self.nested_keys)
        return value


@dataclass(frozen=True)
class PrecisionModeNumericRule:
    """Extract value for specific precision mode from numeric source."""

    metric_name: str
    mode: str
    fallback_keys: tuple[str, ...] = ()

    def extract(self, metric_key: str, source: Mapping[str, float]) -> float | None:
        label_key = precision_mode_label_key(self.metric_name, self.mode)
        value = source.get(label_key)
        if value is not None:
            return value
        for key in self.fallback_keys:
            fallback = source.get(key)
            if fallback is not None:
                return fallback
        return None


@dataclass(frozen=True)
class PrecisionModeActiveStructuredRule:
    """Extract value for active precision mode from structured source."""

    source_key: str
    mode_keys: tuple[str, ...] = ("merge.precision_mode", "merge_precision_mode", "precision_mode")
    nested_keys: tuple[str, ...] = ("rate", "modes")
    overwrite: bool = False

    def extract(self, metric_key: str, source: Mapping[str, object]) -> float | None:
        mode = extract_structured_precision_mode(source, self.mode_keys)
        if mode is None:
            return None
        raw = source.get(self.source_key)
        if not isinstance(raw, Mapping):
            return None
        return extract_precision_mode_mapping(raw, mode, self.nested_keys)


@dataclass(frozen=True)
class PrecisionModeActiveNumericRule:
    """Extract value for active precision mode from numeric source."""

    metric_name: str
    mode_metric_names: tuple[str, ...] = ("merge.precision_mode", "merge_precision_mode")
    fallback_keys: tuple[str, ...] = ()

    def extract(self, metric_key: str, source: Mapping[str, float]) -> float | None:
        mode = extract_numeric_precision_mode(source, self.mode_metric_names)
        if mode is not None:
            label_key = precision_mode_label_key(self.metric_name, mode)
            value = source.get(label_key)
            if value is not None:
                return value
        for key in self.fallback_keys:
            fallback = source.get(key)
            if fallback is not None:
                return fallback
        return None


__all__ = [
    "StructuredRule",
    "NumericRule",
    "DirectValueRule",
    "MappingRatioRule",
    "StructuredAverageRule",
    "DirectNumericRule",
    "NumericAverageRule",
    "SuffixRatioNumericRule",
    "NumericCallableRule",
    "PrecisionModeStructuredRule",
    "PrecisionModeNumericRule",
    "PrecisionModeActiveStructuredRule",
    "PrecisionModeActiveNumericRule",
]
