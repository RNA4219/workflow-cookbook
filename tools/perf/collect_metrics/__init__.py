# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Metrics collection tool.

Collects metrics from Prometheus endpoint and/or structured logs.
"""

from __future__ import annotations

from .cli import (
    main,
    collect_metrics,
    metric_keys,
    percentage_keys,
    MetricsCollectionError,
    MetricsCollectionPlan,
    MetricsRunner,
    SuiteConfig,
    SUITES,
    _load_metric_config,
)
from .security import validate_url, MetricsCollectionError, _validate_url
from .rules import (
    NumericCallableRule,
    DirectValueRule,
    DirectNumericRule,
    MappingRatioRule,
    StructuredAverageRule,
    NumericAverageRule,
    SuffixRatioNumericRule,
    PrecisionModeStructuredRule,
    PrecisionModeNumericRule,
    PrecisionModeActiveStructuredRule,
    PrecisionModeActiveNumericRule,
)
from .helpers import (
    coerce_float,
    coerce_numeric_mapping,
    derive_average,
    derive_ratio_by_suffixes,
    derive_checklist_compliance,
    derive_review_latency,
    derive_task_seed_cycle_time_minutes,
    derive_birdseye_refresh_delay_minutes,
)
from .extractor import (
    MetricDefinition,
    MetricDefinitionRegistry,
    MetricExtractor,
)
from .definitions import build_default_metric_registry, BASE_METRIC_DEFINITIONS
from .prometheus import load_prometheus, load_structured_log, parse_prometheus, _parse_prometheus
from .cli import MISSING_SOURCE_ERROR

__all__ = [
    "main",
    "collect_metrics",
    "metric_keys",
    "percentage_keys",
    "MetricsCollectionError",
    "MetricsCollectionPlan",
    "MetricsRunner",
    "SuiteConfig",
    "SUITES",
    "validate_url",
    "_validate_url",
    "_load_metric_config",
    "MetricDefinition",
    "MetricDefinitionRegistry",
    "MetricExtractor",
    "NumericCallableRule",
    "DirectValueRule",
    "DirectNumericRule",
    "MappingRatioRule",
    "StructuredAverageRule",
    "NumericAverageRule",
    "SuffixRatioNumericRule",
    "PrecisionModeStructuredRule",
    "PrecisionModeNumericRule",
    "PrecisionModeActiveStructuredRule",
    "PrecisionModeActiveNumericRule",
    "coerce_float",
    "coerce_numeric_mapping",
    "derive_average",
    "derive_ratio_by_suffixes",
    "derive_checklist_compliance",
    "derive_review_latency",
    "derive_task_seed_cycle_time_minutes",
    "derive_birdseye_refresh_delay_minutes",
    "build_default_metric_registry",
    "BASE_METRIC_DEFINITIONS",
    "load_prometheus",
    "load_structured_log",
    "parse_prometheus",
    "_parse_prometheus",
    "MISSING_SOURCE_ERROR",
]