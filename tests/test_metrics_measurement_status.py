from __future__ import annotations

import pytest

from tools.ci.check_metrics_thresholds import MetricsThresholdError, ThresholdRule, evaluate_regressions
from tools.perf.collect_metrics.definitions import BASE_METRIC_DEFINITIONS
from tools.perf.collect_metrics.extractor import MetricExtractor
from tools.perf.collect_metrics.helpers import coerce_float
from tools.perf.context_trimmer import trim_messages


class Counter:
    def count_message(self, message):
        return 1

    def meta(self):
        return {"strategy": "fixture"}


def result(embedder=None):
    options = {"embedder": embedder} if embedder is not None else None
    return trim_messages(
        [{"role": "user", "content": "fixture"}],
        max_context_tokens=10,
        model="fixture",
        token_counter=Counter(),
        semantic_options=options,
    )


def broken_embedder(_):
    raise RuntimeError("fixture failure")


def test_missing_failed_and_measured_zero_remain_distinct():
    missing = result()["statistics"]
    failed = result(broken_embedder)["statistics"]
    measured = result(lambda _: [0.0, 0.0])["statistics"]
    assert missing["semantic_status"] == "not_measured"
    assert failed["semantic_status"] == "error"
    assert "semantic_retention" not in missing
    assert "semantic_retention" not in failed
    assert measured["semantic_status"] == "measured"
    assert measured["semantic_retention"] == 0.0
    assert failed["statistics_schema"] == "1.1"


@pytest.mark.parametrize("vector", [[], [float("nan")], [float("inf")]])
def test_invalid_embedding_is_not_a_numeric_score(vector):
    stats = result(lambda _: vector)["statistics"]
    assert stats["semantic_status"] == "error"
    assert "semantic_retention" not in stats


@pytest.mark.parametrize("status", ["error", "not_measured"])
def test_consumer_does_not_collect_legacy_placeholder_with_error_status(status):
    extractor = MetricExtractor(BASE_METRIC_DEFINITIONS, percentage_keys=())
    metrics = {}
    extractor.capture_structured({"semantic_retention": 0.0, "semantic_status": status}, metrics)
    assert "semantic_retention" not in metrics
    extractor.capture_structured({"semantic_retention": 0.0, "semantic_status": "measured"}, metrics)
    assert metrics["semantic_retention"] == 0.0


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "NaN"])
def test_invalid_numeric_input_is_not_a_metric(value):
    assert coerce_float(value) is None
    rule = ThresholdRule("metric", "min", 0.5, "fail")
    with pytest.raises(MetricsThresholdError):
        rule.evaluate({"metric": value})


def test_warn_proxy_does_not_become_fail_through_regression():
    rule = ThresholdRule("semantic_retention", "min", 0.85, "warn")
    assert evaluate_regressions({"semantic_retention": 0.2}, {"semantic_retention": 0.9}, [rule], tolerance=0.05) == []
