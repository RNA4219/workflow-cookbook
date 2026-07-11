#!/usr/bin/env python
# SPDX-License-Identifier: MIT

"""Generate and validate a cross-repo five-tool validation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VALID_FINAL_VERDICTS = {"go", "conditional_go", "no_go", "needs_review", "disqualified"}
QEG_TO_FINAL = {
    "go": "go",
    "conditional_go": "conditional_go",
    "no_go": "no_go",
    "disqualified": "disqualified",
}


@dataclass
class ValidationResult:
    status: str
    errors: list[str]
    warnings: list[str]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object in " + str(path))
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(  # nosec B607  # executable is the fixed git command
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repo_snapshot(entry: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    repo = _resolve_path(str(entry["path"]), base_dir)
    status = _git(repo, "status", "--short", "--untracked-files=all").splitlines()
    upstream = ""
    try:
        upstream = _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    except subprocess.CalledProcessError:
        upstream = ""
    return {
        "tool": entry["tool"],
        "name": entry.get("name", entry["tool"]),
        "path": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "upstream": upstream,
        "dirty": bool(status),
        "dirty_count": len(status),
        "required": bool(entry.get("required", True)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _walk(value: Any) -> list[Any]:
    values = [value]
    if isinstance(value, dict):
        for child in value.values():
            values.extend(_walk(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_walk(child))
    return values


def _collect_by_key(value: Any, wanted: set[str]) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in wanted and isinstance(child, str):
                found.append(child)
            found.extend(_collect_by_key(child, wanted))
    elif isinstance(value, list):
        for child in value:
            found.extend(_collect_by_key(child, wanted))
    return sorted(set(found))


def _schema_version(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("schema_version", "schemaVersion", "qegVersion"):
        if key in payload:
            return str(payload[key])
    return None


def _policy_hashes(payload: Any) -> list[str]:
    return _collect_by_key(payload, {"policyHash", "policy_hash", "qeg_policy_hash_ref"})


def _verdicts(payload: Any) -> list[str]:
    return _collect_by_key(payload, {"verdict", "gate_verdict", "overall_assessment", "expected_verdict"})


def _trace_ids(payload: Any) -> dict[str, list[str]]:
    return {
        "requirements": _collect_by_key(payload, {"requirement_id", "requirementId"}),
        "evidence": _collect_by_key(payload, {"evidence_id", "evidenceId"}),
        "runs": _collect_by_key(payload, {"run_id", "runId"}),
        "nodes": _collect_by_key(payload, {"node_id", "nodeId"}),
    }


def artifact_snapshot(entry: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    path = _resolve_path(str(entry["path"]), base_dir)
    snapshot: dict[str, Any] = {
        "tool": entry["tool"],
        "role": entry["role"],
        "direction": entry.get("direction", "output"),
        "path": str(path),
        "required": bool(entry.get("required", True)),
        "exists": path.exists(),
    }
    if not path.exists():
        return snapshot
    snapshot["sha256"] = _sha256(path)
    snapshot["bytes"] = path.stat().st_size
    if path.suffix.lower() == ".json":
        payload = _read_json(path)
        snapshot["schema_version"] = _schema_version(payload)
        snapshot["policy_hashes"] = _policy_hashes(payload)
        snapshot["verdicts"] = _verdicts(payload)
        snapshot["trace_ids"] = _trace_ids(payload)
        snapshot["has_direct_gate_policy"] = any(
            isinstance(item, dict) and "gate_policy" in item for item in _walk(payload)
        )
    return snapshot


def generate_manifest(config_path: Path, base_dir: Path | None = None) -> dict[str, Any]:
    config = _read_json(config_path)
    base_dir = ROOT if base_dir is None else base_dir.resolve()
    repos = [repo_snapshot(entry, base_dir) for entry in config.get("repos", [])]
    artifacts = [artifact_snapshot(entry, base_dir) for entry in config.get("artifacts", [])]
    qeg_policy_hashes = sorted(
        {
            item
            for artifact in artifacts
            if artifact.get("tool") == "qeg"
            for item in artifact.get("policy_hashes", [])
            if item.startswith("sha256:")
        }
    )
    final_verdict = config.get("final_verdict") or infer_final_verdict(artifacts)
    return {
        "schema_version": "1.0",
        "manifest_id": config.get("manifest_id", "five-tool:run-manifest"),
        "run_id": config.get("run_id", config.get("manifest_id", "five-tool-run")),
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "chain": ["rand", "code-to-gate", "hate", "manual-bb", "qeg"],
        "target": config.get("target", {}),
        "repos": repos,
        "artifacts": artifacts,
        "qeg_policy_hash": config.get("qeg_policy_hash") or (qeg_policy_hashes[0] if qeg_policy_hashes else None),
        "final_verdict": final_verdict,
        "degraded": config.get("degraded", []),
        "skipped": config.get("skipped", []),
        "contract": {
            "direct_gate_policy_allowed_tools": ["qeg"],
            "manifest_config": str(config_path),
        },
    }


def infer_final_verdict(artifacts: list[dict[str, Any]]) -> str:
    for artifact in artifacts:
        if artifact.get("tool") != "qeg":
            continue
        for verdict in artifact.get("verdicts", []):
            if verdict in QEG_TO_FINAL:
                return QEG_TO_FINAL[verdict]
    return "needs_review"


def validate_manifest(manifest: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if manifest.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    if manifest.get("final_verdict") not in VALID_FINAL_VERDICTS:
        errors.append("final_verdict is missing or invalid")
    if not manifest.get("qeg_policy_hash"):
        errors.append("qeg_policy_hash is required")

    for repo in manifest.get("repos", []):
        if repo.get("required") and not repo.get("commit"):
            errors.append(f"required repo has no commit: {repo.get('tool')}")
        if repo.get("dirty"):
            warnings.append(f"repo is dirty: {repo.get('tool')} ({repo.get('dirty_count')} changes)")

    for artifact in manifest.get("artifacts", []):
        label = f"{artifact.get('tool')}:{artifact.get('role')}"
        if artifact.get("required") and not artifact.get("exists"):
            errors.append(f"required artifact missing: {label}")
            continue
        if not artifact.get("exists"):
            warnings.append(f"optional artifact missing: {label}")
            continue
        path = Path(str(artifact["path"]))
        current_hash = _sha256(path)
        if artifact.get("sha256") and artifact["sha256"] != current_hash:
            errors.append(f"artifact hash mismatch: {label}")
        if artifact.get("tool") != "qeg" and artifact.get("has_direct_gate_policy"):
            errors.append(f"non-QEG artifact carries direct gate_policy: {label}")
        trace_ids = artifact.get("trace_ids", {})
        if artifact.get("tool") in {"rand", "manual-bb", "qeg"} and not any(trace_ids.values()):
            warnings.append(f"artifact has no joinable trace ids: {label}")

    qeg_artifacts = [a for a in manifest.get("artifacts", []) if a.get("tool") == "qeg" and a.get("exists")]
    qeg_verdicts = {verdict for artifact in qeg_artifacts for verdict in artifact.get("verdicts", [])}
    if manifest.get("final_verdict") == "go" and qeg_verdicts.intersection({"no_go", "disqualified"}):
        errors.append("final_verdict go conflicts with QEG no_go/disqualified evidence")
    if manifest.get("qeg_policy_hash"):
        qeg_policy_hashes = {item for artifact in qeg_artifacts for item in artifact.get("policy_hashes", [])}
        if manifest["qeg_policy_hash"] not in qeg_policy_hashes:
            errors.append("qeg_policy_hash is not present in QEG artifacts")
    return ValidationResult(status="failed" if errors else "ok", errors=errors, warnings=warnings)


def _result_payload(result: ValidationResult) -> dict[str, Any]:
    return {"status": result.status, "errors": result.errors, "warnings": result.warnings}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate or validate a five-tool run manifest.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Base repository for relative config paths (default: current working directory).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--validate", action="store_true")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "generate":
        manifest = generate_manifest(args.config, args.repo_root)
        _write_json(args.out, manifest)
        if args.validate:
            result = validate_manifest(manifest)
            print(json.dumps(_result_payload(result), ensure_ascii=False, indent=2))
            return 0 if result.status == "ok" else 1
        print(str(args.out))
        return 0

    manifest = _read_json(args.manifest)
    result = validate_manifest(manifest)
    if args.json:
        print(json.dumps(_result_payload(result), ensure_ascii=False, indent=2))
    else:
        print(f"Five-tool manifest validation: {result.status}")
        for warning in result.warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
