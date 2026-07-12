from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker

from tools.ci.gate_effectiveness import analyze_gates

ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-07-12T12:00:00Z"


def _thresholds() -> dict:
    return yaml.safe_load((ROOT / "governance/self-improvement-gates.yaml").read_text(encoding="utf-8"))


def _bundle(*, hard_safety: bool = False, task_count: int = 20) -> dict:
    return {
        "schema_version": "self-improvement/v1",
        "generated_at": NOW,
        "task_count": task_count,
        "since": "2026-06-11T12:00:00Z",
        "until": NOW,
        "gate_catalog": [{
            "gate_id": "gate-a", "owner": "agent-gatefield",
            "policy_revision": "rev-1", "hard_safety": hard_safety,
        }],
        "gate_observations": [],
        "evidence_consumption_observations": [],
        "reflection_summaries": [],
    }


def _observation(
    index: int,
    *,
    decision: str = "pass",
    transition_changed: bool | str = False,
    override: str = "none",
) -> dict:
    return {
        "observation_id": f"OBS-{index}", "task_id": f"TASK-{index}",
        "occurred_at": "2026-07-01T12:00:00Z", "gate_id": "gate-a",
        "owner": "agent-gatefield", "policy_revision": "rev-1",
        "decision": decision, "effective_action": "continue",
        "transition_changed": transition_changed, "risk": "low", "override": override,
        "source_event_id": f"AUD-{index}",
    }


def test_unused_gate_requires_two_periods_before_archive_candidate() -> None:
    first = analyze_gates(_bundle(), _thresholds(), generated_at=NOW)
    second = analyze_gates(_bundle(), _thresholds(), previous_report=first, generated_at=NOW)
    assert first["gate_reports"][0]["action"] == "review"
    assert second["gate_reports"][0]["action"] == "archive_candidate"
    assert second["periodic_nudges"][0]["blocking"] is False


def test_hard_safety_gate_is_never_archive_candidate() -> None:
    first = analyze_gates(_bundle(hard_safety=True), _thresholds(), generated_at=NOW)
    second = analyze_gates(
        _bundle(hard_safety=True), _thresholds(), previous_report=first, generated_at=NOW
    )
    assert second["gate_reports"][0]["action"] == "review"


def test_short_period_and_legacy_unknown_are_insufficient_data() -> None:
    short = _bundle()
    short["since"] = "2026-07-01T12:00:00Z"
    short_report = analyze_gates(short, _thresholds(), generated_at=NOW)
    assert short_report["gate_reports"][0]["classification"] == "insufficient_data"
    assert short_report["periodic_nudges"] == []

    legacy = _bundle()
    legacy["gate_observations"] = [_observation(1, transition_changed="unknown", override="unknown")]
    legacy_report = analyze_gates(legacy, _thresholds(), generated_at=NOW)
    assert legacy_report["gate_reports"][0]["proposal_eligible"] is False


def test_no_effect_and_override_thresholds_generate_review_candidates_only() -> None:
    no_effect = _bundle()
    no_effect["since"] = "2026-07-01T12:00:00Z"
    no_effect["gate_observations"] = [_observation(index) for index in range(20)]
    no_effect_report = analyze_gates(no_effect, _thresholds(), generated_at=NOW)
    assert no_effect_report["gate_reports"][0]["action"] == "relax_candidate"

    override = _bundle()
    override["gate_observations"] = [
        _observation(index, override="applied" if index < 7 else "none") for index in range(10)
    ]
    override_report = analyze_gates(override, _thresholds(), generated_at=NOW)
    assert override_report["gate_reports"][0]["classification"] == "override_excessive"
    assert override_report["gate_reports"][0]["action"] == "review"
    allowed = {"keep", "review", "relax_candidate", "archive_candidate"}
    assert {item["action"] for item in no_effect_report["gate_reports"]} <= allowed


def test_evidence_read_rate_uses_only_explicit_acknowledgements() -> None:
    bundle = _bundle()
    generated = [{
        "observation_id": f"GEN-{index}", "task_id": f"TASK-{index}",
        "evidence_id": f"EV-{index}", "evidence_type": "acceptance",
        "occurred_at": "2026-07-01T12:00:00Z", "disposition": "generated",
        "source_event_id": f"GEN-{index}",
    } for index in range(10)]
    acknowledged = [{
        "observation_id": "ACK-0", "task_id": "TASK-0", "evidence_id": "EV-0",
        "evidence_type": "acceptance", "occurred_at": "2026-07-02T12:00:00Z",
        "disposition": "acknowledged", "reviewed_by": "reviewer", "source_event_id": "ACK-0",
    }]
    bundle["evidence_consumption_observations"] = [*generated, *acknowledged]
    item = analyze_gates(bundle, _thresholds(), generated_at=NOW)["evidence_reports"][0]
    assert item["metrics"]["acknowledged_count"] == 1
    assert item["metrics"]["ack_rate"] == 0.1
    assert item["classification"] == "unread_evidence"


def test_shipyard_golden_export_and_report_validate() -> None:
    bundle = json.loads(
        (ROOT / "examples/self-improvement-v1.shipyard-export.sample.json").read_text(encoding="utf-8")
    )
    bundle_schema = json.loads(
        (ROOT / "schemas/self-improvement/v1/improvement-observation-bundle.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(bundle_schema, format_checker=FormatChecker()).validate(bundle)
    report = analyze_gates(bundle, _thresholds(), generated_at=NOW)
    report_schema = json.loads(
        (ROOT / "schemas/self-improvement/v1/gate-effectiveness-report.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(report_schema, format_checker=FormatChecker()).validate(report)


def test_all_self_improvement_v1_schemas_are_valid() -> None:
    schema_dir = ROOT / "schemas/self-improvement/v1"
    for path in schema_dir.glob("*.schema.json"):
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))
