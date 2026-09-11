"""観測期間・入力契約・CLIの回帰テスト（合成fixtureのみ）。"""

import json
import runpy
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tools.ci import check_gate_observations as checker

NOW = datetime(2026, 9, 10, tzinfo=UTC)


@pytest.fixture
def observations():
    versions = {"model": "fixture", "runner": "fixture", "policy": "v1"}
    return {
        "schema_version": "1.0",
        "gate_id": "RG-fixture",
        "observation_started_at": "2026-01-01T00:00:00Z",
        "versions": dict(versions),
        "minimum_opportunities": 1,
        "records": [
            {
                "id": "one",
                "timestamp": NOW.isoformat(),
                "outcome": "true_negative",
                "override": False,
                "versions": dict(versions),
            }
        ],
    }


@pytest.mark.parametrize("days", [30, 90, 180])
def test_window_excludes_cutoff_but_includes_next_instant_and_as_of(observations, days):
    cutoff = NOW - timedelta(days=days)
    observations["observation_started_at"] = cutoff.isoformat()
    original = observations["records"][0]
    observations["records"] = [
        {**original, "id": str(index), "timestamp": moment.isoformat()}
        for index, moment in enumerate([cutoff, cutoff + timedelta(microseconds=1), NOW])
    ]
    report = checker.summarize_observations(observations, as_of=NOW)
    window = next(item for item in report["windows"] if item["days"] == days)
    assert window["eligible_opportunities"] == 2
    assert window["counts"]["true_negative"] == 2
    assert window["status"] == "ready_for_review"
    assert report["windows_are_overlapping"] is True
    assert report["decision"] == "owner_review_required"


def test_equivalent_offset_timestamps_produce_same_observations(observations):
    expected = checker.summarize_observations(observations, as_of=NOW)
    observations["observation_started_at"] = "2026-01-01T09:00:00+09:00"
    observations["records"][0]["timestamp"] = "2026-09-09T20:00:00-04:00"
    assert checker.summarize_observations(observations, as_of=NOW) == expected


@pytest.mark.parametrize(
    "minimum,expected", [(1, "ready_for_review"), (2, "ready_for_review"), (3, "insufficient_data")]
)
def test_agreed_minimum_includes_exact_threshold(observations, minimum, expected):
    observations["minimum_opportunities"] = minimum
    observations["records"].append({**observations["records"][0], "id": "two"})
    report = checker.summarize_observations(observations, as_of=NOW)
    assert all(window["status"] == expected for window in report["windows"])


def test_false_negatives_and_errors_have_distinct_denominators(observations):
    original = observations["records"][0]
    observations["records"] = [
        {**original, "id": str(index), "outcome": outcome}
        for index, outcome in enumerate(["true_positive", "false_negative", "false_negative", "true_negative", "error"])
    ]
    window = checker.summarize_observations(observations, as_of=NOW)["windows"][0]
    assert window["status"] == "measurement_error"
    assert window["eligible_opportunities"] == 5
    assert window["rate_denominators"] == {"detection_rate": 4, "false_positive_rate": 1, "false_negative_rate": 3}
    assert window["detection_rate"] == 0.25
    assert window["false_positive_rate"] == 0
    assert window["false_negative_rate"] == pytest.approx(2 / 3)


def test_all_errors_leave_rates_unmeasured_even_below_minimum(observations):
    observations["minimum_opportunities"] = 2
    observations["records"][0]["outcome"] = "error"
    window = checker.summarize_observations(observations, as_of=NOW)["windows"][0]
    assert window["status"] == "measurement_error"
    assert all(window[name] is None for name in window["rate_denominators"])
    assert set(window["rate_denominators"].values()) == {0}


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("schema_version", "2.0", "schema_version"),
        ("gate_id", None, "gate_id"),
        ("gate_id", " ", "gate_id"),
        ("versions", [], "versions"),
        ("versions", {"model": "fixture", "runner": "fixture"}, "versions"),
        ("versions", {"model": "fixture", "runner": "fixture", "policy": " "}, "versions"),
        ("versions", {"model": "fixture", "runner": "fixture", "policy": 1}, "versions"),
        ("minimum_opportunities", True, "positive integer"),
        ("minimum_opportunities", 0, "positive integer"),
        ("minimum_opportunities", -1, "positive integer"),
        ("minimum_opportunities", 1.5, "positive integer"),
        ("minimum_opportunities", "2", "positive integer"),
        ("observation_started_at", None, "timestamp must be text"),
        ("observation_started_at", "2026-01-01T00:00:00", "timezone"),
        ("observation_started_at", "2026-09-11T00:00:00Z", "future"),
        ("records", {}, "array"),
        ("records", [None], "record must be an object"),
    ],
)
def test_invalid_bundle_fields_fail_with_reason(observations, field, value, message):
    observations[field] = value
    with pytest.raises(ValueError, match=message):
        checker.summarize_observations(observations, as_of=NOW)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("id", None, "record id"),
        ("id", " ", "record id"),
        ("timestamp", None, "timestamp must be text"),
        ("timestamp", "2026-09-01T00:00:00", "timezone"),
        ("timestamp", "2025-12-31T23:59:59Z", "outside the observation period"),
        ("outcome", [], "outcome is unknown"),
        ("versions", None, "versions are incomplete"),
        ("versions", {"model": "fixture", "runner": "fixture", "policy": ""}, "versions are incomplete"),
        ("versions", {"model": "fixture", "runner": "fixture", "policy": False}, "versions are incomplete"),
        ("override", "false", "override must be boolean"),
    ],
)
def test_invalid_record_types_and_bounds_fail_with_reason(observations, field, value, message):
    observations["records"][0][field] = value
    with pytest.raises(ValueError, match=message):
        checker.summarize_observations(observations, as_of=NOW)


def test_naive_as_of_is_rejected(observations):
    with pytest.raises(ValueError, match="as_of needs a timezone"):
        checker.summarize_observations(observations, as_of=NOW.replace(tzinfo=None))


def test_script_entrypoint_emits_json_without_modifying_input(observations, tmp_path, monkeypatch, capsys):
    source = tmp_path / "observations.json"
    original = json.dumps(observations, ensure_ascii=False).encode("utf-8")
    source.write_bytes(original)
    script = Path(checker.__file__)
    monkeypatch.setattr(sys, "argv", [str(script), "--observations", str(source), "--as-of", NOW.isoformat()])
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(script), run_name="__main__")
    assert exited.value.code == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert captured.err == ""
    assert report["gate_id"] == observations["gate_id"]
    assert report["decision"] == "owner_review_required"
    assert report["windows"][0]["detection_rate"] == 0
    assert report["windows"][0]["false_negative_rate"] is None
    assert source.read_bytes() == original


def test_cli_defaults_to_current_utc(observations, tmp_path, monkeypatch, capsys):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is UTC
            return NOW

    source = tmp_path / "observations.json"
    source.write_text(json.dumps(observations), encoding="utf-8")
    monkeypatch.setattr(checker, "datetime", FixedDateTime)
    assert checker.main(["--observations", str(source)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["as_of"] == NOW.isoformat()
    assert report["windows"][0]["eligible_opportunities"] == 1


@pytest.mark.parametrize(
    "content,message",
    [("{", "Expecting property name"), ("[]", "bundle must be an object"), ('{"schema_version":"1.0"}', "gate_id")],
)
def test_cli_rejects_invalid_json_or_bundle(tmp_path, capsys, content, message):
    source = tmp_path / "invalid.json"
    source.write_text(content, encoding="utf-8")
    assert checker.main(["--observations", str(source), "--as-of", NOW.isoformat()]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "invalid"
    assert message in report["error"]
    assert "windows" not in report


@pytest.mark.parametrize("as_of", ["not-a-date", "2026-09-10T00:00:00"])
def test_cli_rejects_invalid_as_of(observations, tmp_path, capsys, as_of):
    source = tmp_path / "observations.json"
    source.write_text(json.dumps(observations), encoding="utf-8")
    assert checker.main(["--observations", str(source), "--as-of", as_of]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "invalid"
    assert report["error"]
    assert "windows" not in report


def test_cli_reports_missing_file_as_invalid(tmp_path, capsys):
    assert checker.main(["--observations", str(tmp_path / "missing.json")]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "invalid"
    assert "missing.json" in report["error"]


def test_cli_requires_observations_argument(capsys):
    with pytest.raises(SystemExit) as exited:
        checker.main([])
    assert exited.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--observations" in captured.err
