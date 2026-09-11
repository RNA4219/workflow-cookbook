import json
import time
from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest
from referencing import Registry, Resource

from tools.workflow_plugins.run_report import build_run_evidence_bundle, build_run_report, main
from tools.workflow_plugins.runtime import PluginPolicy, RunContext, WorkflowPluginRuntime
from tools.workflow_plugins.runtime_types import PluginTrace


@pytest.fixture
def measured():
    traces = [
        PluginTrace(
            plugin_name="fixture",
            capability="docs.resolve",
            method_name="resolve_docs",
            start_time=10,
            end_time=11,
            success=False,
            attempt=1,
            task_id="task",
            run_id="run",
            invocation_id="call",
            span_id="first",
        ),
        PluginTrace(
            plugin_name="fixture",
            capability="docs.resolve",
            method_name="resolve_docs",
            start_time=12,
            end_time=13,
            success=True,
            attempt=2,
            task_id="task",
            run_id="run",
            invocation_id="call",
            span_id="second",
        ),
    ]
    outcome = {
        "schema_version": "1.0",
        "task_id": "task",
        "run_id": "run",
        "acceptance_id": "AC-001",
        "accepted": True,
        "started_at": 9,
        "finished_at": 15,
        "input_tokens": None,
        "output_tokens": None,
        "cost": None,
        "cost_currency": None,
    }
    return traces, outcome


def test_evidence_identity_survives_trace_reordering(measured):
    traces, outcome = measured
    context = {"task_seed_id": "TS-001", "base_commit": "abcdef1", "head_commit": "abcdef2", "actor": "fixture"}
    original = build_run_evidence_bundle(traces, outcome, **context)
    reordered = build_run_evidence_bundle(list(reversed(traces)), outcome, **context)
    assert {item["span_id"]: item["evidence_id"] for item in original["links"]} == {
        item["span_id"]: item["evidence_id"] for item in reordered["links"]
    }


@pytest.mark.parametrize(
    "change,message",
    [
        ("version", "schema_version"),
        ("input_tokens", "explicit"),
        ("output_tokens", "explicit"),
        ("cost", "explicit"),
        ("empty_traces", "no correlated"),
        ("capability", "change capability"),
        ("timeout_success", "cannot succeed"),
    ],
)
def test_missing_or_inconsistent_observations_fail(measured, change, message):
    traces, outcome = measured
    if change == "version":
        outcome["schema_version"] = "2.0"
    elif change in ("input_tokens", "output_tokens", "cost"):
        outcome.pop(change)
    elif change == "empty_traces":
        traces = []
    elif change == "capability":
        traces[1].capability = "docs.ack"
    else:
        traces[1].timed_out = True
    with pytest.raises(ValueError, match=message):
        build_run_report(traces, outcome)


def test_measured_zero_cost_retains_currency_and_is_not_missing(measured):
    traces, outcome = measured
    outcome.update(cost=0, cost_currency="JPY", input_tokens=0, output_tokens=5)
    report = build_run_report(traces, outcome)
    assert report["usage"] == {"cost": 0, "cost_currency": "JPY", "input_tokens": 0, "output_tokens": 5}
    outcome["cost_currency"] = ""
    with pytest.raises(ValueError, match="cost_currency"):
        build_run_report(traces, outcome)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("task_seed_id", "task-1", "TS numeric"),
        ("base_commit", "abc", "commit references"),
        ("head_commit", "", "commit references"),
        ("actor", "", "actor"),
    ],
)
def test_evidence_context_must_identify_formal_task_and_revision(measured, field, value, message):
    traces, outcome = measured
    context = {"task_seed_id": "TS-001", "base_commit": "abc1234", "head_commit": "def5678", "actor": "fixture"}
    context[field] = value
    with pytest.raises(ValueError, match=message):
        build_run_evidence_bundle(traces, outcome, **context)


@pytest.mark.parametrize("payload", [{}, [1], ["trace"]])
def test_cli_rejects_nonobject_trace_array(measured, tmp_path, capsys, payload):
    _, outcome = measured
    traces_path, outcome_path = tmp_path / "traces.json", tmp_path / "outcome.json"
    traces_path.write_text(json.dumps(payload), encoding="utf-8")
    outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
    assert main(["--traces", str(traces_path), "--outcome", str(outcome_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "invalid" and "object array" in report["error"]
    assert "evidence" not in report


def test_cli_evidence_bundle_fingerprints_both_inputs(measured, tmp_path, capsys):
    from tools.evaluation.workflow import digest

    traces, outcome = measured
    paths = [tmp_path / name for name in ("traces.json", "outcome.json", "context.json")]
    context = {"task_seed_id": "TS-001", "base_commit": "abc1234", "head_commit": "def5678", "actor": "fixture"}
    for path, payload in zip(paths, ([t.to_dict() for t in traces], outcome, context), strict=True):
        path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["--traces", str(paths[0]), "--outcome", str(paths[1]), "--evidence-context", str(paths[2])]) == 0
    bundle = json.loads(capsys.readouterr().out)
    assert len(bundle["links"]) == len(bundle["evidence"]) == 2
    assert bundle["outcome_sha256"] == digest(paths[1])
    assert bundle["traces_sha256"] == digest(paths[0])
    assert bundle["report"]["acceptance_id"] == outcome["acceptance_id"]


def test_retry_failure_is_preserved_separately_from_final_acceptance(measured):
    traces, outcome = measured
    report = build_run_report(traces, outcome)
    assert report["accepted"] is True
    assert report["attempts"] == 2 and report["invocations"] == 1 and report["retries"] == 1
    assert report["capabilities"]["docs.resolve"]["failures"] == 1
    assert report["wall_time_ms"] == 6000
    assert report["tool_time_ms"] == 2000
    assert report["usage"]["input_tokens"] is None
    outcome["accepted"] = False
    assert build_run_report(traces, outcome)["accepted"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "other"),
        ("task_id", "other"),
        ("span_id", "second"),
        ("end_time", None),
        ("end_time", 8),
        ("start_time", -1),
        ("attempt", 3),
        ("attempt", True),
        ("timed_out", "false"),
    ],
)
def test_invalid_or_unrelated_trace_is_rejected(measured, field, value):
    traces, outcome = measured
    setattr(traces[0], field, value)
    with pytest.raises(ValueError):
        build_run_report(traces, outcome)


@pytest.mark.parametrize(
    "field,value",
    [
        ("accepted", 1),
        ("started_at", float("nan")),
        ("finished_at", 8),
        ("input_tokens", -1),
        ("output_tokens", True),
        ("cost", -1),
    ],
)
def test_invalid_outcome_is_rejected(measured, field, value):
    traces, outcome = measured
    outcome[field] = value
    with pytest.raises(ValueError):
        build_run_report(traces, outcome)


def test_runtime_automatically_correlates_explicit_run_context(tmp_path):
    class Plugin:
        capabilities = ("docs.resolve",)
        calls = 0

        def resolve_docs(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("fixture retry")
            return {"required": []}

    runtime = WorkflowPluginRuntime(
        [Plugin()],
        run_context=RunContext("task", "run"),
        default_policy=PluginPolicy(isolation_mode="inline", retry_count=1, retry_delay_seconds=0),
    )
    before = time.time()
    runtime.invoke_first("docs.resolve", repo_root=tmp_path, task_id="task")
    after = time.time()
    first, second = runtime.traces
    assert first.invocation_id == second.invocation_id
    assert first.span_id != second.span_id
    assert first.task_id == second.task_id == "task"
    assert first.run_id == second.run_id == "run"
    assert before <= first.start_time <= second.end_time <= after
    assert [trace.attempt for trace in runtime.traces] == [1, 2]


def test_legacy_trace_serialization_keeps_optional_context_absent():
    trace = PluginTrace(plugin_name="fixture", capability="docs.resolve", method_name="resolve_docs", start_time=0)
    assert "run_id" not in trace.to_dict()
    with pytest.raises(ValueError):
        RunContext("", "run")


def test_evidence_bundle_preserves_external_schema_and_correlates_outcome(measured):
    traces, outcome = measured
    bundle = build_run_evidence_bundle(
        traces,
        outcome,
        task_seed_id="TS-001",
        base_commit="abc1234",
        head_commit="def5678",
        actor="fixture",
    )
    schemas = Path(__file__).parent / "fixtures/agent-protocols"
    common = json.loads((schemas / "common.schema.json").read_text(encoding="utf-8"))
    schema = json.loads((schemas / "Evidence.schema.json").read_text(encoding="utf-8"))
    registry = Registry().with_resource(
        "https://agent-protocols/schemas/common.schema.json", Resource.from_contents(common)
    )
    validator = jsonschema.Draft202012Validator(schema, registry=registry)
    for entry in bundle["evidence"]:
        validator.validate(entry)
    assert [link["span_id"] for link in bundle["links"]] == ["first", "second"]
    assert bundle["links"][0]["acceptance_id"] == outcome["acceptance_id"]
    assert bundle["evidence"][0]["policyVerdict"] == "manual_review_required"
    assert bundle["report"]["accepted"] is True
    other_traces, other_outcome = deepcopy(measured)
    other_outcome["run_id"] = "different"
    for trace in other_traces:
        trace.run_id = "different"
    other = build_run_evidence_bundle(
        other_traces,
        other_outcome,
        task_seed_id="TS-001",
        base_commit="abc1234",
        head_commit="def5678",
        actor="fixture",
    )
    assert {e["id"] for e in bundle["evidence"]}.isdisjoint(e["id"] for e in other["evidence"])


def test_cli_fingerprints_input_and_preserves_rejected_acceptance(measured, tmp_path, capsys):
    traces, outcome = measured
    outcome["accepted"] = False
    trace_path, outcome_path = tmp_path / "traces.json", tmp_path / "outcome.json"
    trace_path.write_text(json.dumps([trace.to_dict() for trace in traces]), encoding="utf-8")
    outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
    assert main(["--traces", str(trace_path), "--outcome", str(outcome_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["accepted"] is False
    assert report["outcome_sha256"].startswith("sha256:")
    assert main(["--traces", str(tmp_path / "missing"), "--outcome", str(outcome_path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"
