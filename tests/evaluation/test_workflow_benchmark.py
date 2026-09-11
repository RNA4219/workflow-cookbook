import json
from pathlib import Path

import pytest

from tools.ci.check_evaluation_identity_manifest import validate_manifest
from tools.evaluation import workflow as benchmark


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("frozen fixture implementation", encoding="utf-8")
    dataset = {
        "schema_version": "1.0",
        "cases": [
            {"case_id": name, "split": split, "input": {}, "oracle": {}, "negative_control": control}
            for name, split, control in [("work", "test", False), ("control", "test", True), ("unused", "train", False)]
        ],
    }
    dataset_path = tmp_path / "dataset.json"
    write(dataset_path, dataset)
    manifest = json.loads(
        (Path(__file__).parents[2] / "templates/evaluation-measurement-manifest.template.json").read_text(
            encoding="utf-8"
        )
    )
    manifest.update(manifest_id="eval:unit", task_id="unit", owner="fixture", freeze_date="2026-09-10")
    manifest["evaluation"].update(
        dataset_sha256=benchmark.digest(dataset_path),
        primary_metric="success_rate",
        configuration={
            "benchmark": {
                "repeats": 2,
                "seed": 1,
                "reading_budget_bytes": 100,
                "output_budget_tokens": None,
                "runner_path": "source.txt",
            }
        },
    )
    for role in ("legacy", "candidate"):
        manifest["identity"][role].update(
            source_path="source.txt",
            source_sha256=benchmark.digest(source),
            input_sha256=benchmark.digest(dataset_path),
            source_revision="fixture",
        )
    manifest["identity"]["runner"]["sha256"] = benchmark.digest(source)
    manifest_path = tmp_path / "manifest.json"
    write(manifest_path, manifest)
    records = []
    for case in ("work", "control"):
        for repeat in range(2):
            for variant in ("legacy", "candidate"):
                records.append(
                    {
                        "case_id": case,
                        "repeat": repeat,
                        "variant": variant,
                        "run_id": f"{case}:{repeat}:{variant}",
                        "status": "completed",
                        "success": case == "work" and variant == "candidate",
                        "wall_time_ms": 8 if variant == "candidate" else 10,
                        "reading_bytes": 50,
                        "input_tokens": None,
                        "output_tokens": None,
                        "interventions": 0,
                        "artifact_path": "source.txt",
                        "artifact_sha256": benchmark.digest(source),
                    }
                )
    observations_path = tmp_path / "observations.json"
    write(
        observations_path,
        {
            "schema_version": "1.0",
            "manifest_sha256": benchmark.digest(manifest_path),
            "records": records,
        },
    )
    return manifest_path, dataset_path, observations_path


@pytest.mark.parametrize(
    "change,expected",
    [
        ("primary_metric", "primary_metric"),
        ("benchmark", "configuration.benchmark"),
        ("variant_input", "same dataset"),
    ],
)
def test_benchmark_specific_manifest_constraints(inputs, change, expected):
    manifest = benchmark.read_object(inputs[0])
    if change == "primary_metric":
        manifest["evaluation"]["primary_metric"] = "wall_time_ms"
    elif change == "benchmark":
        manifest["evaluation"]["configuration"]["benchmark"] = []
    else:
        manifest["identity"]["candidate"]["input_sha256"] = "sha256:" + "a" * 64
    assert not validate_manifest(manifest, stage="preflight").errors
    write(inputs[0], manifest)
    with pytest.raises(ValueError, match=expected):
        benchmark.load_inputs(*inputs[:2])


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_valid_game_manifest_is_outside_workflow_benchmark_scope(inputs, version):
    manifest = benchmark.read_object(inputs[0])
    manifest.update(schema_version=version, profile="game", gate="G50")
    manifest["evaluation"]["measurement_unit"] = "game"
    manifest["identity"].update(
        deck={"path": "deck.csv", "sha256": "sha256:" + "a" * 64},
        native_runtime={"id": "fixture", "sha256": "sha256:" + "b" * 64, "rng_control": "uncontrolled"},
    )
    manifest["evaluation"].update(
        opponent_set_id="fixture",
        opponents=["a", "b"],
        cohort_id="fixture",
        target_deck="a",
        starts=["first", "second"],
    )
    assert not validate_manifest(manifest, stage="preflight").errors
    write(inputs[0], manifest)
    with pytest.raises(ValueError, match="non-game 1.1"):
        benchmark.load_inputs(*inputs[:2])


@pytest.mark.parametrize(
    "shape,message",
    [
        ("object", "JSON object"),
        ("records", "array"),
        ("record", "object"),
        ("failure_success", "success=null"),
    ],
)
def test_malformed_observations_cannot_produce_report(inputs, shape, message):
    observed = benchmark.read_object(inputs[2])
    if shape == "object":
        observed = []
    elif shape == "records":
        observed["records"] = {}
    elif shape == "record":
        observed["records"][0] = []
    else:
        observed["records"][0].update(status="crash", success=True)
    write(inputs[2], observed)
    with pytest.raises(ValueError, match=message):
        benchmark.compare(*inputs)


def test_module_entrypoint_emits_replayable_schedule(inputs, monkeypatch, capsys):
    import runpy
    import sys

    monkeypatch.setattr(
        sys, "argv", ["evaluation", "schedule", "--manifest", str(inputs[0]), "--dataset", str(inputs[1])]
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("tools.evaluation", run_name="__main__")
    assert stopped.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    manifest, cases = benchmark.load_inputs(*inputs[:2])
    assert payload["manifest_sha256"] == benchmark.digest(inputs[0])
    assert payload["schedule"] == benchmark.schedule(manifest, cases)


@pytest.mark.parametrize("option", [None, "--observations", "--output-dir"])
def test_compare_requires_both_observations_and_destination(inputs, option, capsys, tmp_path):
    args = ["compare", "--manifest", str(inputs[0]), "--dataset", str(inputs[1])]
    if option:
        args += [option, str(inputs[2] if option == "--observations" else tmp_path / "output")]
    assert benchmark.main(args) == 1
    assert "requires --observations and --output-dir" in json.loads(capsys.readouterr().out)["error"]
    assert not (tmp_path / "output").exists()


def test_schedule_is_paired_reproducible_and_excludes_training(inputs):
    manifest, cases = benchmark.load_inputs(*inputs[:2])
    plan = benchmark.schedule(manifest, cases)
    assert plan == benchmark.schedule(manifest, cases)
    assert len(plan) == 8
    for old, new in zip(plan[::2], plan[1::2], strict=True):
        assert (old["case_id"], old["repeat"]) == (new["case_id"], new["repeat"])
        assert {old["variant"], new["variant"]} == {"legacy", "candidate"}
    assert {r["case_id"] for r in plan} == {"work", "control"}


@pytest.mark.parametrize(
    "change",
    [
        "manifest",
        "source",
        "dataset",
        "runner",
        "dataset_shape",
        "case_object",
        "duplicate_case",
        "split",
        "input_shape",
        "negative_flag",
        "no_control",
    ],
)
def test_frozen_inputs_and_dataset_contract_are_enforced(inputs, change):
    manifest = benchmark.read_object(inputs[0])
    dataset = benchmark.read_object(inputs[1])
    if change == "manifest":
        manifest["schema_version"] = "0"
    elif change in ("source", "runner"):
        if change == "source":
            manifest["identity"]["legacy"]["source_sha256"] = "sha256:" + "0" * 64
        else:
            manifest["identity"]["runner"]["sha256"] = "sha256:" + "0" * 64
    elif change == "dataset":
        manifest["evaluation"]["dataset_sha256"] = "sha256:" + "0" * 64
    else:
        if change == "dataset_shape":
            dataset["cases"] = {}
        elif change == "case_object":
            dataset["cases"][0] = "invalid"
        elif change == "duplicate_case":
            dataset["cases"].append(dict(dataset["cases"][0]))
        elif change == "split":
            dataset["cases"][0]["split"] = "validation"
        elif change == "input_shape":
            dataset["cases"][0]["input"] = []
        elif change == "negative_flag":
            dataset["cases"][0]["negative_control"] = 1
        else:
            for case in dataset["cases"]:
                case["negative_control"] = False
        write(inputs[1], dataset)
        sha = benchmark.digest(inputs[1])
        manifest["evaluation"]["dataset_sha256"] = sha
        for role in ("legacy", "candidate"):
            manifest["identity"][role]["input_sha256"] = sha
    write(inputs[0], manifest)
    with pytest.raises(ValueError):
        benchmark.load_inputs(inputs[0], inputs[1])


@pytest.mark.parametrize("field", ["success", "input_tokens", "output_tokens", "interventions"])
def test_unmeasured_values_must_be_explicit(inputs, field):
    observations = benchmark.read_object(inputs[2])
    record = observations["records"][0]
    record.update(status="error", success=None)
    record.pop(field)
    write(inputs[2], observations)
    with pytest.raises(ValueError, match="explicit"):
        benchmark.compare(*inputs)


def test_paired_improvement_and_missing_usage_are_distinct(inputs):
    _, report = benchmark.compare(*inputs)
    assert report["decision"] == "owner_review_required"
    assert report["records"] == 8
    assert report["success_rate_delta"] == 1
    assert report["independent_cases"] == 1
    assert report["paired_case_deltas"] == [{"case_id": "work", "success_rate_delta": 1, "wall_time_ms_delta": -2}]
    metric = report["variants"]["candidate"]["metrics"]["input_tokens"]
    assert metric["missing"] == 2 and metric["measured"] == 0 and metric["mean"] is None
    assert report["variants"]["candidate"]["metrics"]["interventions"]["mean"] == 0


@pytest.mark.parametrize("status", ["error", "timeout", "crash"])
def test_failed_observations_remain_in_denominator_and_postrun_rejects(inputs, tmp_path, status):
    observations = benchmark.read_object(inputs[2])
    record = observations["records"][1]
    record.update(status=status, success=None)
    write(inputs[2], observations)
    manifest, report = benchmark.compare(*inputs)
    assert report["decision"] == "measurement_error"
    assert report["variants"]["candidate"]["success_rate"] == 0.5
    assert report["failures"][status] == 1
    output = tmp_path / "failed-report"
    assert benchmark.publish(manifest, report, output)
    assert (output / "report.json").is_file()
    assert validate_manifest(benchmark.read_object(output / "completed-manifest.json"), stage="postrun").errors


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", ""),
        ("repeat", True),
        ("status", "unknown"),
        ("success", 1),
        ("wall_time_ms", -1),
        ("wall_time_ms", float("nan")),
        ("wall_time_ms", True),
        ("reading_bytes", 101),
        ("input_tokens", -1),
        ("interventions", False),
        ("artifact_sha256", "sha256:wrong"),
    ],
)
def test_invalid_record_fails_before_output(inputs, field, value):
    observations = benchmark.read_object(inputs[2])
    observations["records"][0][field] = value
    write(inputs[2], observations)
    with pytest.raises(ValueError):
        benchmark.compare(*inputs)


@pytest.mark.parametrize("change", ["duplicate", "missing", "train", "wrong_manifest"])
def test_incomplete_or_unmatched_runs_are_rejected(inputs, change):
    observations = benchmark.read_object(inputs[2])
    if change == "duplicate":
        observations["records"].append(dict(observations["records"][0]))
    elif change == "missing":
        observations["records"].pop()
    elif change == "train":
        observations["records"][0]["case_id"] = "unused"
    else:
        observations["manifest_sha256"] = "invalid"
    write(inputs[2], observations)
    with pytest.raises(ValueError):
        benchmark.compare(*inputs)


def test_negative_control_mismatch_prevents_comparison_success(inputs, tmp_path, capsys):
    observations = benchmark.read_object(inputs[2])
    observations["records"][-1]["success"] = True
    write(inputs[2], observations)
    args = [
        "compare",
        "--manifest",
        str(inputs[0]),
        "--dataset",
        str(inputs[1]),
        "--observations",
        str(inputs[2]),
        "--output-dir",
        str(tmp_path / "report"),
    ]
    assert benchmark.main(args) == 2
    assert json.loads(capsys.readouterr().out)["decision"] == "invalid_control"


def test_cli_preserves_frozen_inputs_and_publishes_hash_verified_manifest(inputs, tmp_path, capsys):
    original = [path.read_bytes() for path in inputs]
    args = [
        "compare",
        "--manifest",
        str(inputs[0]),
        "--dataset",
        str(inputs[1]),
        "--observations",
        str(inputs[2]),
        "--output-dir",
        str(tmp_path / "report"),
    ]
    assert benchmark.main(args) == 0
    assert json.loads(capsys.readouterr().out)["postrun_errors"] == []
    completed = benchmark.read_object(tmp_path / "report/completed-manifest.json")
    assert completed["outcome"]["artifact_sha256"] == benchmark.digest(tmp_path / "report/report.json")
    assert [path.read_bytes() for path in inputs] == original
    assert benchmark.main(args) == 1
    assert "already exists" in capsys.readouterr().out


@pytest.mark.parametrize("target", ["source.txt", "dataset.json"])
def test_changed_frozen_file_is_rejected(inputs, target):
    (inputs[0].parent / target).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        benchmark.load_inputs(*inputs[:2])


def test_configured_token_budget_requires_actual_measurement(inputs):
    manifest = benchmark.read_object(inputs[0])
    manifest["evaluation"]["configuration"]["benchmark"]["output_budget_tokens"] = 10
    write(inputs[0], manifest)
    observations = benchmark.read_object(inputs[2])
    observations["manifest_sha256"] = benchmark.digest(inputs[0])
    write(inputs[2], observations)
    with pytest.raises(ValueError, match="token budget"):
        benchmark.compare(*inputs)


def test_regression_and_measured_usage_are_reported(inputs):
    observations = benchmark.read_object(inputs[2])
    for record in observations["records"]:
        record["success"] = record["case_id"] == "work" and record["variant"] == "legacy"
        record["input_tokens"] = 7
    write(inputs[2], observations)
    _, report = benchmark.compare(*inputs)
    assert report["success_rate_delta"] == -1
    assert report["variants"]["candidate"]["metrics"]["input_tokens"]["sum"] == 14
    assert report["variants"]["candidate"]["metrics"]["input_tokens"]["p95"] == 7
