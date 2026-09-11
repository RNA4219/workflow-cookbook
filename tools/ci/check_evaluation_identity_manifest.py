# SPDX-License-Identifier: MIT
# Copyright 2026 RNA4219

"""Validate the identity and minimum outcome evidence of an evaluation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
VALID_GATES = {"G50", "G200", "G1000"}
VALID_PURPOSES = {"comparison", "experiment", "recovery", "promotion"}
VALID_RNG_CONTROL = {"controlled", "uncontrolled"}
PROFILE_UNITS = {"game": "game", "document_retrieval": "query", "performance": "operation", "workflow": "task"}
OUTCOME_COUNTS = (
    "fallback_count",
    "illegal_action_count",
    "crash_count",
    "timeout_count",
)


@dataclass(frozen=True)
class ValidationResult:
    errors: list[str]

    @property
    def status(self) -> str:
        return "ok" if not self.errors else "failed"


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _require_text(data: Mapping[str, Any], field: str, errors: list[str], *, label: str | None = None) -> None:
    if not isinstance(data.get(field), str) or not str(data[field]).strip():
        errors.append(f"missing or invalid: {label or field}")


def _require_sha256(data: Mapping[str, Any], field: str, errors: list[str], *, label: str | None = None) -> None:
    value = data.get(field)
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        errors.append(f"missing or invalid sha256: {label or field}")


def _validate_policy_identity(identity: Mapping[str, Any], role: str, errors: list[str]) -> None:
    entry = _as_mapping(identity.get(role))
    for field in ("label", "source_path"):
        _require_text(entry, field, errors, label=f"identity.{role}.{field}")
    for field in ("source_sha256", "input_sha256"):
        _require_sha256(entry, field, errors, label=f"identity.{role}.{field}")


def _profile(manifest: Mapping[str, Any]) -> str:
    value = manifest.get("profile", "game" if manifest.get("schema_version") == "1.0" else "")
    return value if isinstance(value, str) else ""


def _choice(value: object, choices: set[str]) -> bool:
    return isinstance(value, str) and value in choices


def _validate_measurement(manifest: Mapping[str, Any], errors: list[str]) -> None:
    try:
        date.fromisoformat(str(manifest.get("freeze_date", "")))
    except ValueError:
        errors.append("freeze_date must be an ISO calendar date")
    identity = _as_mapping(manifest.get("identity"))
    for role in ("legacy", "candidate"):
        _require_text(
            _as_mapping(identity.get(role)), "source_revision", errors, label=f"identity.{role}.source_revision"
        )
    evaluation = _as_mapping(manifest.get("evaluation"))
    for field in ("dataset_id", "primary_metric", "model_version", "policy_version"):
        _require_text(evaluation, field, errors, label=f"evaluation.{field}")
    _require_sha256(evaluation, "dataset_sha256", errors, label="evaluation.dataset_sha256")
    unit = PROFILE_UNITS.get(_profile(manifest))
    if unit is not None and evaluation.get("measurement_unit") != unit:
        errors.append(f"evaluation.measurement_unit must be {unit}")
    if not isinstance(evaluation.get("configuration"), Mapping):
        errors.append("evaluation.configuration must be an object")


def _validate_static_contract(manifest: Mapping[str, Any], errors: list[str]) -> None:
    for field in (
        "manifest_id",
        "task_id",
        "owner",
        "freeze_date",
        "hypothesis",
        "action_delta",
        "negative_control",
    ):
        _require_text(manifest, field, errors)
    if not _choice(manifest.get("schema_version"), {"1.0", "1.1"}):
        errors.append("schema_version must be 1.0 or 1.1")
    profile = _profile(manifest)
    if profile not in PROFILE_UNITS:
        errors.append("unknown or missing profile")
    if manifest.get("schema_version") == "1.0" and profile != "game":
        errors.append("schema_version 1.0 supports only the legacy game profile")
    if not isinstance(manifest.get("manifest_id"), str) or not str(manifest["manifest_id"]).startswith("eval:"):
        errors.append("manifest_id must start with eval:")
    if not _choice(manifest.get("purpose"), VALID_PURPOSES):
        errors.append("purpose must be comparison, experiment, recovery, or promotion")
    if profile == "game" and not _choice(manifest.get("gate"), VALID_GATES):
        errors.append("gate must be G50, G200, or G1000")

    identity = _as_mapping(manifest.get("identity"))
    _validate_policy_identity(identity, "legacy", errors)
    _validate_policy_identity(identity, "candidate", errors)
    roles = [("runner", "id")]
    if profile == "game":
        roles += [("deck", "path"), ("native_runtime", "id")]
    for role, text_field in roles:
        entry = _as_mapping(identity.get(role))
        _require_text(entry, text_field, errors, label=f"identity.{role}.{text_field}")
        _require_sha256(entry, "sha256", errors, label=f"identity.{role}.sha256")
    if profile == "game":
        native_runtime = _as_mapping(identity.get("native_runtime"))
        if not _choice(native_runtime.get("rng_control"), VALID_RNG_CONTROL):
            errors.append("identity.native_runtime.rng_control must be controlled or uncontrolled")
        evaluation = _as_mapping(manifest.get("evaluation"))
        for field in ("opponent_set_id", "cohort_id", "target_deck"):
            _require_text(evaluation, field, errors, label=f"evaluation.{field}")
        for field in ("opponents", "starts"):
            value = evaluation.get(field)
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(item, str) and item.strip() for item in value)
            ):
                errors.append(f"missing or invalid: evaluation.{field}")
    if manifest.get("schema_version") == "1.1":
        _validate_measurement(manifest, errors)

    external_mutations = _as_mapping(manifest.get("external_mutations"))
    for field in ("registry_frozen", "submission_frozen"):
        if external_mutations.get(field) is not True:
            errors.append(f"external_mutations.{field} must be true during evaluation")


def _validate_preflight(manifest: Mapping[str, Any], errors: list[str]) -> None:
    if manifest.get("status") != "frozen":
        errors.append("status must be frozen before an evaluation starts")


def _validate_postrun(manifest: Mapping[str, Any], errors: list[str]) -> None:
    outcome = _as_mapping(manifest.get("outcome"))
    _require_text(outcome, "artifact_path", errors, label="outcome.artifact_path")
    _require_sha256(outcome, "artifact_sha256", errors, label="outcome.artifact_sha256")
    game = _profile(manifest) == "game"
    for field in ("records", "total_games") if game else ("records",):
        value = outcome.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"outcome.{field} must be an integer greater than zero")
    for field in OUTCOME_COUNTS if game else ("error_count", "crash_count", "timeout_count"):
        value = outcome.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value != 0:
            errors.append(f"outcome.{field} must be zero")
    if manifest.get("schema_version") == "1.1":
        if manifest.get("status") != "completed":
            errors.append("status must be completed for postrun")
        metrics = _as_mapping(outcome.get("metrics"))
        primary = _as_mapping(manifest.get("evaluation")).get("primary_metric")
        if not isinstance(primary, str) or primary not in metrics:
            errors.append("outcome.metrics must include the primary metric")
        for name, value in metrics.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or (isinstance(value, float) and not math.isfinite(value))
            ):
                errors.append(f"outcome.metrics.{name} must be a finite number")


def validate_manifest(manifest: Mapping[str, Any], *, stage: str) -> ValidationResult:
    errors: list[str] = []
    _validate_static_contract(manifest, errors)
    if stage == "preflight":
        _validate_preflight(manifest, errors)
    elif stage == "postrun":
        _validate_postrun(manifest, errors)
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    return ValidationResult(errors=errors)


def _read_manifest(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("manifest must be a JSON object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an evaluation identity manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Evaluation manifest JSON path.")
    parser.add_argument("--stage", choices=("preflight", "postrun"), required=True)
    parser.add_argument("--check", action="store_true", help="Exit non-zero when validation fails.")
    parser.add_argument(
        "--artifact-root", type=Path, help="Base for relative artifact paths; defaults to the manifest directory."
    )
    parser.add_argument("--json", action="store_true", help="Emit the validation result as JSON.")
    args = parser.parse_args(argv)

    try:
        manifest = _read_manifest(args.manifest)
        result = validate_manifest(manifest, stage=args.stage)
        if (
            args.stage == "postrun"
            and not result.errors
            and (manifest.get("schema_version") == "1.1" or args.artifact_root)
        ):
            outcome = _as_mapping(manifest.get("outcome"))
            artifact = (args.artifact_root or args.manifest.parent) / str(outcome["artifact_path"])
            actual = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
            if actual != outcome["artifact_sha256"]:
                result.errors.append("outcome.artifact_sha256 does not match the artifact file")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    payload = {"status": result.status, "stage": args.stage, "errors": result.errors}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Evaluation identity manifest: {result.status}")
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
    return 1 if args.check and result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
