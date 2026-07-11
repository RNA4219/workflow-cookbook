# SPDX-License-Identifier: MIT
# Copyright 2025 RNA4219

"""Trace serialization and agent-protocols Evidence projection."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .runtime_types import PluginTrace


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_stable_json(value).encode('utf-8')).hexdigest()}"


def _timestamp_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(value, tz=UTC).isoformat()


def _runtime_environment() -> dict[str, str]:
    return {
        "os": f"{platform.system()} {platform.release()}".strip(),
        "runtime": f"Python {platform.python_version()}",
        "containerImageDigest": "uncontainerized",
        "lockfileHash": _sha256_json("workflow-plugin-runtime"),
    }


def trace_payload(traces: Sequence[PluginTrace]) -> list[dict[str, Any]]:
    return [trace.to_dict() for trace in traces]


def write_trace_payload(path: str | Path, traces: Sequence[PluginTrace]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(trace_payload(traces), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_trace_evidence_payload(
    traces: Sequence[PluginTrace],
    *,
    task_seed_id: str,
    base_commit: str,
    head_commit: str,
    actor: str,
    evidence_id_prefix: str = "EV-WORKFLOW-PLUGIN",
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for index, trace in enumerate(traces, start=1):
        trace_data = trace.to_dict()
        start_time = _timestamp_from_epoch(trace.start_time)
        end_time = _timestamp_from_epoch(trace.end_time or trace.start_time)
        payload.append(
            {
                "schemaVersion": "1.0.0",
                "id": f"{evidence_id_prefix}-{index:03d}",
                "kind": "Evidence",
                "state": "Published",
                "version": 1,
                "createdAt": end_time,
                "updatedAt": end_time,
                "taskSeedId": task_seed_id,
                "baseCommit": base_commit,
                "headCommit": head_commit,
                "inputHash": _sha256_json(
                    {
                        "plugin": trace.plugin_name,
                        "capability": trace.capability,
                        "method": trace.method_name,
                        "attempt": trace.attempt,
                    }
                ),
                "outputHash": _sha256_json(
                    {
                        "success": trace.success,
                        "error": trace.error,
                        "resultSummary": trace.result_summary,
                    }
                ),
                "model": {
                    "name": "workflow-plugin-runtime",
                    "version": "unknown",
                    "parametersHash": _sha256_json(
                        {
                            "timeoutSeconds": trace.timeout_seconds,
                            "isolationMode": trace.isolation_mode,
                        }
                    ),
                },
                "tools": ["WorkflowPluginRuntime", trace.capability],
                "environment": _runtime_environment(),
                "staleStatus": {
                    "classification": "fresh",
                    "evaluatedAt": end_time,
                },
                "mergeResult": {"status": "not_applicable"},
                "startTime": start_time,
                "endTime": end_time,
                "actor": actor,
                "policyVerdict": "approved" if trace.success else "manual_review_required",
                "diffHash": _sha256_json(trace_data),
            }
        )
    return payload


def write_trace_evidence_jsonl(
    path: str | Path,
    traces: Sequence[PluginTrace],
    *,
    task_seed_id: str,
    base_commit: str,
    head_commit: str,
    actor: str,
    evidence_id_prefix: str = "EV-WORKFLOW-PLUGIN",
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    evidence = build_trace_evidence_payload(
        traces,
        task_seed_id=task_seed_id,
        base_commit=base_commit,
        head_commit=head_commit,
        actor=actor,
        evidence_id_prefix=evidence_id_prefix,
    )
    with target.open("w", encoding="utf-8") as handle:
        for entry in evidence:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
