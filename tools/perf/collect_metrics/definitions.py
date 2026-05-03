# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""
Base metric definitions for governance metrics.

Defines all metric definitions used in collect_metrics.
"""

from __future__ import annotations

from .rules import (
    DirectValueRule,
    DirectNumericRule,
    MappingRatioRule,
    StructuredAverageRule,
    NumericAverageRule,
    NumericCallableRule,
    SuffixRatioNumericRule,
    PrecisionModeStructuredRule,
    PrecisionModeNumericRule,
    PrecisionModeActiveStructuredRule,
    PrecisionModeActiveNumericRule,
)
from .helpers import (
    derive_checklist_compliance,
    derive_review_latency,
    derive_task_seed_cycle_time_minutes,
    derive_birdseye_refresh_delay_minutes,
)
from .extractor import (
    MetricDefinition,
    MetricDefinitionRegistry,
)

# Prefix constants for average derivation
_TRIM_COMPRESS_PREFIXES = (
    ("trim_compress_ratio", 1.0),
    ("context_compression_ratio", 1.0),
)

_TRIM_SEMANTIC_PREFIXES = (
    ("trim_semantic_retention", 1.0),
    ("context_semantic_retention", 1.0),
)

_REVIEW_LATENCY_AGGREGATE_PREFIXES = (
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

BASE_METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        key="checklist_compliance_rate",
        structured_rules=(
            DirectValueRule(),
            MappingRatioRule(
                numerator_keys=("compliant", "checked", "passing", "numerator"),
                denominator_keys=("total", "denominator", "all", "overall"),
            ),
        ),
        numeric_rules=(
            DirectNumericRule(),
            NumericCallableRule(derive_checklist_compliance),
        ),
    ),
    MetricDefinition(
        key="task_seed_cycle_time_minutes",
        structured_rules=(DirectValueRule(),),
        numeric_rules=(
            DirectNumericRule(),
            NumericCallableRule(derive_task_seed_cycle_time_minutes),
        ),
    ),
    MetricDefinition(
        key="birdseye_refresh_delay_minutes",
        structured_rules=(DirectValueRule(),),
        numeric_rules=(
            DirectNumericRule(),
            NumericCallableRule(derive_birdseye_refresh_delay_minutes),
        ),
    ),
    MetricDefinition(
        key="merge_success_rate",
        structured_rules=(
            DirectValueRule(keys=("merge_success_rate", "merge.success.rate")),
            PrecisionModeActiveStructuredRule(source_key="merge.success.rate"),
        ),
        numeric_rules=(
            DirectNumericRule(keys=("merge_success_rate", "merge.success.rate")),
            PrecisionModeActiveNumericRule(
                metric_name="merge.success.rate",
                fallback_keys=("merge.success.rate", "merge_success_rate"),
            ),
        ),
        required=False,
    ),
    MetricDefinition(
        key="merge_conflict_rate",
        structured_rules=(
            DirectValueRule(keys=("merge_conflict_rate", "merge.conflict.rate")),
            PrecisionModeActiveStructuredRule(source_key="merge.conflict.rate"),
        ),
        numeric_rules=(
            DirectNumericRule(keys=("merge_conflict_rate", "merge.conflict.rate")),
            PrecisionModeActiveNumericRule(
                metric_name="merge.conflict.rate",
                fallback_keys=("merge.conflict.rate", "merge_conflict_rate"),
            ),
        ),
        required=False,
    ),
    MetricDefinition(
        key="merge_autosave_lag_ms",
        structured_rules=(
            DirectValueRule(keys=("merge_autosave_lag_ms", "merge.autosave.lag_ms")),
            PrecisionModeActiveStructuredRule(source_key="merge.autosave.lag_ms"),
        ),
        numeric_rules=(
            DirectNumericRule(keys=("merge_autosave_lag_ms", "merge.autosave.lag_ms")),
            PrecisionModeActiveNumericRule(
                metric_name="merge.autosave.lag_ms",
                fallback_keys=("merge.autosave.lag_ms", "merge_autosave_lag_ms"),
            ),
        ),
        required=False,
    ),
    MetricDefinition(
        key="review_latency",
        structured_rules=(
            DirectValueRule(keys=("workflow_review_latency", "review_latency")),
            StructuredAverageRule(
                prefixes=_REVIEW_LATENCY_AGGREGATE_PREFIXES,
                overwrite=True,
            ),
        ),
        numeric_rules=(
            DirectNumericRule(),
            NumericCallableRule(derive_review_latency),
        ),
    ),
    MetricDefinition(
        key="compress_ratio",
        structured_rules=(
            DirectValueRule(
                keys=(
                    "trim_compress_ratio_avg",
                    "trim_compress_ratio",
                    "compress_ratio",
                    "compression_ratio",
                )
            ),
            StructuredAverageRule(prefixes=_TRIM_COMPRESS_PREFIXES, overwrite=True),
        ),
        numeric_rules=(
            DirectNumericRule(
                keys=("trim_compress_ratio_avg", "trim_compress_ratio", "compress_ratio"),
            ),
            NumericAverageRule(prefixes=_TRIM_COMPRESS_PREFIXES),
        ),
    ),
    MetricDefinition(
        key="semantic_retention",
        structured_rules=(
            DirectValueRule(
                keys=(
                    "trim_semantic_retention_avg",
                    "trim_semantic_retention",
                    "semantic_retention",
                )
            ),
            StructuredAverageRule(prefixes=_TRIM_SEMANTIC_PREFIXES, overwrite=True),
        ),
        numeric_rules=(
            DirectNumericRule(
                keys=(
                    "trim_semantic_retention_avg",
                    "trim_semantic_retention",
                    "semantic_retention",
                ),
            ),
            NumericAverageRule(prefixes=_TRIM_SEMANTIC_PREFIXES),
        ),
    ),
    MetricDefinition(
        key="reopen_rate",
        structured_rules=(
            DirectValueRule(
                keys=(
                    "workflow_reopen_rate_avg",
                    "workflow_reopen_rate",
                    "docops_reopen_rate",
                    "reopen_rate",
                )
            ),
            MappingRatioRule(
                numerator_keys=("reopened", "reopens", "numerator"),
                denominator_keys=("total", "resolved", "all", "denominator"),
            ),
        ),
        numeric_rules=(
            DirectNumericRule(
                keys=(
                    "workflow_reopen_rate_avg",
                    "workflow_reopen_rate",
                    "docops_reopen_rate",
                    "reopen_rate",
                ),
            ),
            SuffixRatioNumericRule(
                numerator_suffixes=("_reopened", "_reopen"),
                denominator_suffixes=("_total", "_count", "_closed", "_all"),
            ),
            NumericAverageRule(
                prefixes=(
                    ("workflow_reopen_rate", 1.0),
                    ("workflow_reopen_rate_avg", 1.0),
                    ("docops_reopen_rate", 1.0),
                    ("reopen_rate", 1.0),
                )
            ),
        ),
    ),
    MetricDefinition(
        key="spec_completeness",
        structured_rules=(
            DirectValueRule(
                keys=(
                    "workflow_spec_completeness_ratio_avg",
                    "workflow_spec_completeness_avg",
                    "workflow_spec_completeness_ratio",
                    "workflow_spec_completeness",
                    "spec_completeness_ratio",
                    "spec_completeness",
                )
            ),
            MappingRatioRule(
                numerator_keys=("with_spec", "with_specs", "completed", "numerator"),
                denominator_keys=("total", "all", "denominator"),
            ),
        ),
        numeric_rules=(
            DirectNumericRule(
                keys=(
                    "workflow_spec_completeness_ratio_avg",
                    "workflow_spec_completeness_avg",
                    "workflow_spec_completeness_ratio",
                    "workflow_spec_completeness",
                    "spec_completeness_ratio",
                    "spec_completeness",
                ),
            ),
            SuffixRatioNumericRule(
                numerator_suffixes=("_with_spec", "_with_specs", "_completed"),
                denominator_suffixes=("_total", "_count", "_all"),
            ),
            NumericAverageRule(
                prefixes=(
                    ("workflow_spec_completeness_ratio", 1.0),
                    ("workflow_spec_completeness_ratio_avg", 1.0),
                    ("workflow_spec_completeness", 1.0),
                    ("spec_completeness_ratio", 1.0),
                    ("spec_completeness", 1.0),
                )
            ),
        ),
    ),
    MetricDefinition(
        key="merge_success_rate_baseline",
        structured_rules=(
            DirectValueRule(keys=("merge_success_rate_baseline", "merge.success.rate.baseline")),
            PrecisionModeStructuredRule(source_key="merge.success.rate", mode="baseline"),
        ),
        numeric_rules=(
            DirectNumericRule(keys=("merge_success_rate_baseline", "merge.success.rate.baseline")),
            PrecisionModeNumericRule(
                metric_name="merge.success.rate",
                mode="baseline",
                fallback_keys=("merge.success.rate.baseline",),
            ),
        ),
        required=False,
    ),
    MetricDefinition(
        key="merge_success_rate_strict",
        structured_rules=(
            DirectValueRule(keys=("merge_success_rate_strict", "merge.success.rate.strict")),
            PrecisionModeStructuredRule(source_key="merge.success.rate", mode="strict"),
        ),
        numeric_rules=(
            DirectNumericRule(keys=("merge_success_rate_strict", "merge.success.rate.strict")),
            PrecisionModeNumericRule(
                metric_name="merge.success.rate",
                mode="strict",
                fallback_keys=("merge.success.rate.strict",),
            ),
        ),
        required=False,
    ),
    MetricDefinition(
        key="merge_conflict_rate_baseline",
        structured_rules=(
            DirectValueRule(keys=("merge_conflict_rate_baseline", "merge.conflict.rate.baseline")),
            PrecisionModeStructuredRule(source_key="merge.conflict.rate", mode="baseline"),
        ),
        numeric_rules=(
            DirectNumericRule(keys=("merge_conflict_rate_baseline", "merge.conflict.rate.baseline")),
            PrecisionModeNumericRule(
                metric_name="merge.conflict.rate",
                mode="baseline",
                fallback_keys=("merge.conflict.rate.baseline",),
            ),
        ),
        required=False,
    ),
    MetricDefinition(
        key="merge_conflict_rate_strict",
        structured_rules=(
            DirectValueRule(keys=("merge_conflict_rate_strict", "merge.conflict.rate.strict")),
            PrecisionModeStructuredRule(source_key="merge.conflict.rate", mode="strict"),
        ),
        numeric_rules=(
            DirectNumericRule(keys=("merge_conflict_rate_strict", "merge.conflict.rate.strict")),
            PrecisionModeNumericRule(
                metric_name="merge.conflict.rate",
                mode="strict",
                fallback_keys=("merge.conflict.rate.strict",),
            ),
        ),
        required=False,
    ),
    MetricDefinition(
        key="merge_autosave_lag_ms_baseline",
        structured_rules=(
            DirectValueRule(keys=("merge_autosave_lag_ms_baseline", "merge.autosave.lag_ms.baseline")),
            PrecisionModeStructuredRule(source_key="merge.autosave.lag_ms", mode="baseline"),
        ),
        numeric_rules=(
            DirectNumericRule(keys=("merge_autosave_lag_ms_baseline", "merge.autosave.lag_ms.baseline")),
            PrecisionModeNumericRule(
                metric_name="merge.autosave.lag_ms",
                mode="baseline",
                fallback_keys=("merge.autosave.lag_ms.baseline",),
            ),
        ),
        required=False,
    ),
    MetricDefinition(
        key="merge_autosave_lag_ms_strict",
        structured_rules=(
            DirectValueRule(keys=("merge_autosave_lag_ms_strict", "merge.autosave.lag_ms.strict")),
            PrecisionModeStructuredRule(source_key="merge.autosave.lag_ms", mode="strict"),
        ),
        numeric_rules=(
            DirectNumericRule(keys=("merge_autosave_lag_ms_strict", "merge.autosave.lag_ms.strict")),
            PrecisionModeNumericRule(
                metric_name="merge.autosave.lag_ms",
                mode="strict",
                fallback_keys=("merge.autosave.lag_ms.strict",),
            ),
        ),
        required=False,
    ),
)


def build_default_metric_registry() -> MetricDefinitionRegistry:
    """Build registry with base metric definitions."""
    registry = MetricDefinitionRegistry()
    for definition in BASE_METRIC_DEFINITIONS:
        registry.register(definition)
    return registry