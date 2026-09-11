from copy import deepcopy
from datetime import UTC, datetime

import pytest

from tools.ci.check_gate_observations import summarize_observations

NOW = datetime(2026, 9, 10, tzinfo=UTC)
VERSIONS = {"model": "fixture", "runner": "fixture", "policy": "v1"}


def bundle(outcomes):
    return {
        "schema_version": "1.0",
        "gate_id": "RG-fixture",
        "observation_started_at": "2026-01-01T00:00:00Z",
        "versions": VERSIONS,
        "minimum_opportunities": 2,
        "records": [
            {
                "id": str(i),
                "timestamp": "2026-09-01T00:00:00Z",
                "outcome": outcome,
                "override": False,
                "versions": VERSIONS,
            }
            for i, outcome in enumerate(outcomes)
        ],
    }


def test_windows_and_distinct_detection_error_denominators():
    data = bundle(["true_positive", "true_positive", "false_positive", "true_negative"])
    windows = summarize_observations(data, as_of=NOW)["windows"]
    assert [w["days"] for w in windows] == [90, 180, 30]
    assert windows[0]["eligible_opportunities"] == 4
    assert windows[0]["detection_rate"] == 0.75
    assert windows[0]["false_positive_rate"] == 0.5
    assert windows[0]["rate_denominators"]["false_positive_rate"] == 2
    assert windows[0]["status"] == "ready_for_review"


def test_no_data_has_null_rates_not_zero():
    window = summarize_observations(bundle([]), as_of=NOW)["windows"][0]
    assert window["status"] == "insufficient_data"
    assert window["detection_rate"] is None
    assert window["false_positive_rate"] is None


def test_true_detections_do_not_become_false_positives():
    window = summarize_observations(bundle(["true_positive"] * 5), as_of=NOW)["windows"][0]
    assert window["detection_rate"] == 1
    assert window["false_positive_rate"] is None
    assert window["status"] == "ready_for_review"


def test_mixed_versions_are_excluded_and_not_pooled():
    data = deepcopy(bundle(["true_positive", "true_negative"]))
    data["records"][0]["versions"]["policy"] = "v2"
    data["versions"] = dict(VERSIONS)
    data["records"][1]["versions"] = dict(VERSIONS)
    window = summarize_observations(data, as_of=NOW)["windows"][0]
    assert window["eligible_opportunities"] == 1
    assert window["excluded_version_mismatch"] == 1
    assert window["status"] == "insufficient_data"


def test_short_period_and_errors_do_not_promote():
    data = bundle(["true_positive", "true_negative"])
    data["observation_started_at"] = "2026-08-01T00:00:00Z"
    assert summarize_observations(data, as_of=NOW)["windows"][0]["status"] == "insufficient_data"
    data["records"][1]["outcome"] = "error"
    data["records"][1]["override"] = True
    result = summarize_observations(data, as_of=NOW)
    assert result["decision"] == "owner_review_required"
    assert result["windows"][0]["status"] == "measurement_error"
    assert result["windows"][0]["overrides"] == 1


@pytest.mark.parametrize(
    "field,value",
    [("id", "0"), ("timestamp", "2027-01-01T00:00:00Z"), ("outcome", "unknown"), ("versions", {}), ("override", 1)],
)
def test_invalid_record_rejected(field, value):
    data = deepcopy(bundle(["true_positive", "true_negative"]))
    data["records"][1][field] = value
    with pytest.raises(ValueError):
        summarize_observations(data, as_of=NOW)
