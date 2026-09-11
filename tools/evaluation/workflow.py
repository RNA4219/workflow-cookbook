"""凍結した課題と実測記録を対応付けて比較する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from tools.ci.check_evaluation_identity_manifest import validate_manifest

METRICS = ("wall_time_ms", "input_tokens", "output_tokens", "reading_bytes", "interventions")
STATUSES = {"completed", "error", "timeout", "crash"}


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def verified_file(base: Path, path: object, expected: object) -> Path:
    target = (base / text(path, "artifact path")).resolve(strict=True)
    if not target.is_file() or digest(target) != expected:
        raise ValueError(f"artifact hash mismatch: {target}")
    return target


def load_inputs(manifest_path: Path, dataset_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_object(manifest_path)
    validation = validate_manifest(manifest, stage="preflight")
    if validation.errors:
        raise ValueError("; ".join(validation.errors))
    if manifest["schema_version"] != "1.1" or manifest["profile"] == "game":
        raise ValueError("benchmark requires a non-game 1.1 manifest")
    evaluation = manifest["evaluation"]
    if evaluation["primary_metric"] != "success_rate":
        raise ValueError("primary_metric must be success_rate")
    config = evaluation["configuration"].get("benchmark")
    if not isinstance(config, dict):
        raise ValueError("configuration.benchmark is required")
    integer(config.get("repeats"), "repeats", minimum=1)
    integer(config.get("seed"), "seed")
    integer(config.get("reading_budget_bytes"), "reading_budget_bytes", minimum=1)
    if config.get("output_budget_tokens") is not None:
        integer(config["output_budget_tokens"], "output_budget_tokens", minimum=1)
    identity = manifest["identity"]
    dataset_hash = digest(dataset_path)
    if dataset_hash != evaluation["dataset_sha256"]:
        raise ValueError("dataset hash mismatch")
    for role in ("legacy", "candidate"):
        entry = identity[role]
        verified_file(manifest_path.parent, entry["source_path"], entry["source_sha256"])
        if entry["input_sha256"] != dataset_hash:
            raise ValueError("variants must use the same dataset")
    verified_file(manifest_path.parent, config.get("runner_path"), identity["runner"]["sha256"])
    dataset = read_object(dataset_path)
    if dataset.get("schema_version") != "1.0" or not isinstance(dataset.get("cases"), list):
        raise ValueError("dataset schema_version 1.0 and cases array required")
    seen: set[str] = set()
    cases: list[dict[str, Any]] = []
    for case in dataset["cases"]:
        if not isinstance(case, dict):
            raise ValueError("case must be an object")
        case_id = text(case.get("case_id"), "case_id")
        if case_id in seen:
            raise ValueError("duplicate case_id")
        seen.add(case_id)
        if case.get("split") not in ("train", "test"):
            raise ValueError("case split must be train or test")
        if not isinstance(case.get("input"), dict) or not isinstance(case.get("oracle"), dict):
            raise ValueError("case input and oracle must be objects")
        if not isinstance(case.get("negative_control"), bool):
            raise ValueError("case negative_control must be boolean")
        if case["split"] == "test":
            cases.append(case)
    if not any(c["negative_control"] for c in cases) or not any(not c["negative_control"] for c in cases):
        raise ValueError("test split needs a regular case and a negative control")
    return manifest, cases


def schedule(manifest: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    config = manifest["evaluation"]["configuration"]["benchmark"]
    rng = random.Random(config["seed"])  # nosec B311: reproducible benchmark order, no security use
    pairs = [(case["case_id"], repeat) for case in cases for repeat in range(config["repeats"])]
    rng.shuffle(pairs)
    output: list[dict[str, Any]] = []
    for case_id, repeat in pairs:
        variants = ["legacy", "candidate"]
        rng.shuffle(variants)
        output.extend({"case_id": case_id, "repeat": repeat, "variant": variant} for variant in variants)
    return output


def distribution(values: Sequence[float | int | None]) -> dict[str, Any]:
    measured = sorted(value for value in values if value is not None)
    return {
        "measured": len(measured),
        "missing": len(values) - len(measured),
        "sum": sum(measured) if measured else None,
        "mean": statistics.mean(measured) if measured else None,
        "median": statistics.median(measured) if measured else None,
        "p95": measured[math.ceil(0.95 * len(measured)) - 1] if measured else None,
    }


def compare(manifest_path: Path, dataset_path: Path, observations_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, cases = load_inputs(manifest_path, dataset_path)
    observed = read_object(observations_path)
    if observed.get("schema_version") != "1.0" or observed.get("manifest_sha256") != digest(manifest_path):
        raise ValueError("observations must identify the frozen manifest")
    records = observed.get("records")
    if not isinstance(records, list):
        raise ValueError("records must be an array")
    expected = {(r["case_id"], r["repeat"], r["variant"]) for r in schedule(manifest, cases)}
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    run_ids: set[str] = set()
    config = manifest["evaluation"]["configuration"]["benchmark"]
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("record must be an object")
        run_id = text(record.get("run_id"), "run_id")
        key = (
            text(record.get("case_id"), "case_id"),
            integer(record.get("repeat"), "repeat"),
            text(record.get("variant"), "variant"),
        )
        if run_id in run_ids or key in indexed or key not in expected:
            raise ValueError("duplicate or unexpected run")
        run_ids.add(run_id)
        if not isinstance(record.get("status"), str) or record["status"] not in STATUSES:
            raise ValueError("unknown run status")
        if "success" not in record:
            raise ValueError("success must be explicit (null for measurement failure)")
        if record["status"] == "completed":
            if not isinstance(record.get("success"), bool):
                raise ValueError("completed run success must be boolean")
        elif record.get("success") is not None:
            raise ValueError("measurement failures must have success=null")
        finite(record.get("wall_time_ms"), "wall_time_ms")
        used = integer(record.get("reading_bytes"), "reading_bytes")
        if used > config["reading_budget_bytes"]:
            raise ValueError("reading budget exceeded")
        for field in ("input_tokens", "output_tokens", "interventions"):
            if field not in record:
                raise ValueError(f"{field} must be explicit (null when unmeasured)")
            if record[field] is not None:
                integer(record[field], field)
        token_budget = config.get("output_budget_tokens")
        if token_budget is not None and (record["output_tokens"] is None or record["output_tokens"] > token_budget):
            raise ValueError("output token budget exceeded or unmeasured")
        verified_file(observations_path.parent, record.get("artifact_path"), record.get("artifact_sha256"))
        indexed[key] = record
    if set(indexed) != expected:
        raise ValueError("missing paired observations")
    ordinary = {case["case_id"] for case in cases if not case["negative_control"]}
    controls = {case["case_id"] for case in cases if case["negative_control"]}
    variants: dict[str, Any] = {}
    for variant in ("legacy", "candidate"):
        selected = [r for key, r in indexed.items() if key[0] in ordinary and key[2] == variant]
        successes = sum(r["status"] == "completed" and r["success"] for r in selected)
        variants[variant] = {
            "records": len(selected),
            "successes": successes,
            "success_rate": successes / len(selected),
            "metrics": {field: distribution([r[field] for r in selected]) for field in METRICS},
        }
    case_deltas = []
    invalid_controls = []
    for case in cases:
        success_deltas, time_deltas = [], []
        for repeat in range(config["repeats"]):
            old = indexed[(case["case_id"], repeat, "legacy")]
            new = indexed[(case["case_id"], repeat, "candidate")]
            old_success = old["status"] == "completed" and old["success"] is True
            new_success = new["status"] == "completed" and new["success"] is True
            if case["case_id"] in controls and old_success != new_success:
                invalid_controls.append({"case_id": case["case_id"], "repeat": repeat})
            success_deltas.append(int(new_success) - int(old_success))
            time_deltas.append(new["wall_time_ms"] - old["wall_time_ms"])
        if case["case_id"] in ordinary:
            case_deltas.append(
                {
                    "case_id": case["case_id"],
                    "success_rate_delta": statistics.mean(success_deltas),
                    "wall_time_ms_delta": statistics.mean(time_deltas),
                }
            )
    failures = {status: sum(r["status"] == status for r in records) for status in ("error", "timeout", "crash")}
    decision = (
        "measurement_error"
        if any(failures.values())
        else ("invalid_control" if invalid_controls else "owner_review_required")
    )
    report = {
        "schema_version": "1.0",
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": digest(manifest_path),
        "dataset_sha256": digest(dataset_path),
        "observations_sha256": digest(observations_path),
        "records": len(records),
        "decision": decision,
        "variants": variants,
        "failures": failures,
        "negative_control_mismatches": invalid_controls,
        "paired_case_deltas": case_deltas,
        "success_rate_delta": variants["candidate"]["success_rate"] - variants["legacy"]["success_rate"],
        "inference": "descriptive_only",
        "independent_cases": len(ordinary),
    }
    return manifest, report


def publish(manifest: Mapping[str, Any], report: Mapping[str, Any], output_dir: Path) -> list[str]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("output directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".wfc-comparison-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        report_path = stage / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        completed = deepcopy(dict(manifest))
        completed["status"] = "completed"
        completed["outcome"] = {
            "artifact_path": "report.json",
            "artifact_sha256": digest(report_path),
            "records": report["records"],
            "error_count": report["failures"]["error"],
            "crash_count": report["failures"]["crash"],
            "timeout_count": report["failures"]["timeout"],
            "metrics": {"success_rate": report["variants"]["candidate"]["success_rate"]},
        }
        errors = validate_manifest(completed, stage="postrun").errors
        (stage / "completed-manifest.json").write_text(
            json.dumps(completed, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (stage / "postrun.json").write_text(json.dumps({"errors": errors}) + "\n", encoding="utf-8")
        os.rename(stage, output_dir)
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("schedule", "compare"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "schedule":
            manifest, cases = load_inputs(args.manifest, args.dataset)
            result: dict[str, Any] = {"manifest_sha256": digest(args.manifest), "schedule": schedule(manifest, cases)}
            code = 0
        else:
            if args.observations is None or args.output_dir is None:
                raise ValueError("compare requires --observations and --output-dir")
            manifest, report = compare(args.manifest, args.dataset, args.observations)
            # Persist absolute identity paths when relocating the completed manifest.
            for role in ("legacy", "candidate"):
                manifest["identity"][role]["source_path"] = str(
                    (args.manifest.parent / manifest["identity"][role]["source_path"]).resolve()
                )
            benchmark = manifest["evaluation"]["configuration"]["benchmark"]
            benchmark["runner_path"] = str((args.manifest.parent / benchmark["runner_path"]).resolve())
            errors = publish(manifest, report, args.output_dir)
            result = {"output_dir": str(args.output_dir), "decision": report["decision"], "postrun_errors": errors}
            code = 0 if report["decision"] == "owner_review_required" and not errors else 2
    except (OSError, ValueError, TypeError) as exc:
        result, code = {"status": "invalid", "error": str(exc)}, 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return code
