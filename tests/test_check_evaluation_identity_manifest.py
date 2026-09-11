from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tools.ci.check_evaluation_identity_manifest import main, validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def valid_manifest(*, completed: bool = False) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "manifest_id": "eval:20260715-01",
        "task_id": "20260715-01",
        "owner": "owner",
        "purpose": "comparison",
        "status": "frozen",
        "freeze_date": "2026-07-15",
        "gate": "G50",
        "hypothesis": "公開状態から検証可能な単一原因",
        "action_delta": "候補だけが切替先を変える",
        "negative_control": "条件外では選択を変えない",
        "identity": {
            "legacy": {
                "label": "legacy",
                "source_path": "legacy.py",
                "source_sha256": digest("legacy-source"),
                "input_sha256": digest("legacy-input"),
            },
            "candidate": {
                "label": "candidate",
                "source_path": "candidate.py",
                "source_sha256": digest("candidate-source"),
                "input_sha256": digest("candidate-input"),
            },
            "deck": {"path": "deck.csv", "sha256": digest("deck")},
            "runner": {"id": "standard-runner", "sha256": digest("runner")},
            "native_runtime": {
                "id": "cg.dll",
                "sha256": digest("native"),
                "rng_control": "uncontrolled",
            },
        },
        "evaluation": {
            "opponent_set_id": "sample4",
            "opponents": ["deck-a", "deck-b"],
            "cohort_id": "sample4",
            "target_deck": "deck-a",
            "starts": ["first", "second"],
        },
        "external_mutations": {"registry_frozen": True, "submission_frozen": True},
    }
    if completed:
        manifest["outcome"] = {
            "artifact_path": "artifacts/EV-20260715-01/report.json",
            "artifact_sha256": digest("artifact"),
            "records": 50,
            "total_games": 50,
            "fallback_count": 0,
            "illegal_action_count": 0,
            "crash_count": 0,
            "timeout_count": 0,
        }
    return manifest


def test_preflight_accepts_frozen_complete_identity() -> None:
    result = validate_manifest(valid_manifest(), stage="preflight")

    assert result.status == "ok"
    assert result.errors == []


def test_preflight_rejects_missing_action_delta_and_unfrozen_submission() -> None:
    manifest = valid_manifest()
    manifest.pop("action_delta")
    manifest["external_mutations"] = {"registry_frozen": True, "submission_frozen": False}

    result = validate_manifest(manifest, stage="preflight")

    assert result.status == "failed"
    assert result.errors == [
        "missing or invalid: action_delta",
        "external_mutations.submission_frozen must be true during evaluation",
    ]


def test_postrun_rejects_zero_game_and_failure_artifact() -> None:
    manifest = valid_manifest(completed=True)
    outcome = manifest["outcome"]
    assert isinstance(outcome, dict)
    outcome["records"] = 0
    outcome["total_games"] = 0
    outcome["fallback_count"] = 1

    result = validate_manifest(manifest, stage="postrun")

    assert result.status == "failed"
    assert result.errors == [
        "outcome.records must be an integer greater than zero",
        "outcome.total_games must be an integer greater than zero",
        "outcome.fallback_count must be zero",
    ]


def test_cli_emits_json_and_fails_closed_for_invalid_manifest(tmp_path: Path, capsys) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(valid_manifest()), encoding="utf-8")

    exit_code = main(["--manifest", str(manifest_path), "--stage", "postrun", "--check", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 1
    payload = json.loads(captured.out)
    assert payload["status"] == "failed"
    assert "outcome.records must be an integer greater than zero" in payload["errors"]


def test_agent_gate_and_template_expose_the_same_contract() -> None:
    agent_instructions = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    contract = (ROOT / "docs" / "contracts" / "evaluation-identity-contract.md").read_text(encoding="utf-8")
    template = json.loads(
        (ROOT / "templates" / "evaluation-identity-manifest.template.json").read_text(encoding="utf-8")
    )
    schema = json.loads((ROOT / "schemas" / "evaluation-identity-manifest.schema.json").read_text(encoding="utf-8"))

    assert "--stage preflight --check" in agent_instructions
    assert "--stage postrun --check" in agent_instructions
    assert "0試合artifactは即Reject" in contract
    assert template["external_mutations"] == {"registry_frozen": True, "submission_frozen": True}
    assert "action_delta" in schema["required"]


def measurement_manifest(profile: str = "document_retrieval", *, completed: bool = False) -> dict:
    manifest = valid_manifest(completed=completed)
    manifest.update(schema_version="1.1", profile=profile)
    manifest.pop("gate")
    identity = manifest["identity"]
    for role in ("deck", "native_runtime"):
        identity.pop(role)
    for role in ("legacy", "candidate"):
        identity[role]["source_revision"] = "test-revision"
    manifest["evaluation"] = {
        "dataset_id": "fixture-corpus",
        "dataset_sha256": digest("corpus"),
        "measurement_unit": {"document_retrieval": "query", "performance": "operation", "workflow": "task"}[profile],
        "primary_metric": "success_rate",
        "configuration": {"limit": 3},
        "model_version": "not_used:fixture",
        "policy_version": "fixture-v1",
    }
    if completed:
        manifest["status"] = "completed"
        manifest["outcome"] = {
            "artifact_path": "report.json",
            "artifact_sha256": digest("fixture-result"),
            "records": 3,
            "error_count": 0,
            "crash_count": 0,
            "timeout_count": 0,
            "metrics": {"success_rate": 2 / 3},
        }
    return manifest


@pytest.mark.parametrize("profile", ["document_retrieval", "performance", "workflow"])
@pytest.mark.parametrize("completed", [False, True])
def test_measurement_profiles_need_no_game_identity(profile, completed):
    manifest = measurement_manifest(profile, completed=completed)
    result = validate_manifest(manifest, stage="postrun" if completed else "preflight")
    assert result.errors == []
    schema = json.loads((ROOT / "schemas/evaluation-identity-manifest.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(manifest)
    assert "total_games" not in manifest.get("outcome", {})


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"profile": "unknown"}, "profile"),
        ({"profile": []}, "profile"),
        ({"purpose": {}}, "purpose"),
        ({"schema_version": "1.0"}, "legacy game"),
        ({"freeze_date": "YYYY-MM-DD"}, "calendar date"),
    ],
)
def test_invalid_profile_or_identity_fails_closed(change, expected):
    manifest = measurement_manifest()
    manifest.update(change)
    assert any(expected in error for error in validate_manifest(manifest, stage="preflight").errors)


@pytest.mark.parametrize(
    "field,value",
    [("records", 0), ("records", True), ("error_count", 1), ("crash_count", 1), ("timeout_count", 1), ("metrics", {})],
)
def test_empty_or_failed_measurement_is_not_success(field, value):
    manifest = measurement_manifest(completed=True)
    manifest["outcome"][field] = value
    assert validate_manifest(manifest, stage="postrun").status == "failed"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, None])
def test_metric_must_be_measured_finite_number(value):
    manifest = measurement_manifest(completed=True)
    manifest["outcome"]["metrics"]["success_rate"] = value
    assert validate_manifest(manifest, stage="postrun").status == "failed"


def test_new_profile_requires_revision_and_matching_unit():
    manifest = measurement_manifest()
    manifest["identity"]["candidate"].pop("source_revision")
    manifest["evaluation"]["measurement_unit"] = "game"
    errors = validate_manifest(manifest, stage="preflight").errors
    assert any("source_revision" in error for error in errors)
    assert any("must be query" in error for error in errors)


def test_cli_checks_actual_artifact_hash_and_missing_file(tmp_path, capsys):
    manifest = measurement_manifest(completed=True)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    command = ["--manifest", str(path), "--stage", "postrun", "--check", "--json"]
    assert main(command) == 1
    (tmp_path / "report.json").write_text("fixture-result", encoding="utf-8")
    assert main(command) == 0
    (tmp_path / "report.json").write_text("different-result", encoding="utf-8")
    assert main(command) == 1
    assert "does not match" in capsys.readouterr().out


def test_new_game_profile_preserves_game_requirements():
    manifest = valid_manifest(completed=True)
    common = measurement_manifest(completed=True)
    manifest.update(schema_version="1.1", profile="game", status="completed")
    for role in ("legacy", "candidate"):
        manifest["identity"][role]["source_revision"] = "test-revision"
    manifest["evaluation"].update(common["evaluation"], measurement_unit="game")
    manifest["outcome"]["metrics"] = {"success_rate": 0.5}
    assert validate_manifest(manifest, stage="postrun").status == "ok"
    schema = json.loads((ROOT / "schemas/evaluation-identity-manifest.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(manifest)
    manifest["outcome"]["total_games"] = 0
    manifest["identity"].pop("deck")
    errors = validate_manifest(manifest, stage="postrun").errors
    assert any("total_games" in error for error in errors)
    assert any("identity.deck" in error for error in errors)
