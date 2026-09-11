"""PluginTraceと独立した最終検収を照合して集計する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.evaluation.workflow import digest, finite, integer, read_object, text

from .runtime_evidence import build_trace_evidence_payload
from .runtime_types import PluginTrace


def build_run_report(traces: Sequence[PluginTrace], outcome: Mapping[str, Any]) -> dict[str, Any]:
    if outcome.get("schema_version") != "1.0":
        raise ValueError("outcome schema_version 1.0 required")
    for field in ("task_id", "run_id", "acceptance_id"):
        text(outcome.get(field), field)
    if not isinstance(outcome.get("accepted"), bool):
        raise ValueError("accepted must be a boolean from the acceptance grader")
    start = finite(outcome.get("started_at"), "started_at")
    end = finite(outcome.get("finished_at"), "finished_at")
    if end < start:
        raise ValueError("finished_at is before started_at")
    for field in ("input_tokens", "output_tokens", "cost"):
        if field not in outcome:
            raise ValueError(f"{field} must be explicit; null when unmeasured")
        if outcome[field] is not None:
            if field == "cost":
                finite(outcome[field], field)
                text(outcome.get("cost_currency"), "cost_currency")
            else:
                integer(outcome[field], field)
    if not traces:
        raise ValueError("no correlated traces")
    seen: set[str] = set()
    invocations: dict[str, list[int]] = defaultdict(list)
    invocation_capabilities: dict[str, str] = {}
    capabilities: dict[str, dict[str, Any]] = {}
    for trace in traces:
        if trace.task_id != outcome["task_id"] or trace.run_id != outcome["run_id"]:
            raise ValueError("trace task/run does not match outcome")
        span = text(trace.span_id, "span_id")
        invocation = text(trace.invocation_id, "invocation_id")
        if span in seen:
            raise ValueError("duplicate span")
        seen.add(span)
        attempt = integer(trace.attempt, "attempt", minimum=1)
        invocations[invocation].append(attempt)
        capability = text(trace.capability, "capability")
        if invocation_capabilities.setdefault(invocation, capability) != capability:
            raise ValueError("one invocation cannot change capability")
        trace_start = finite(trace.start_time, "trace start")
        trace_end = finite(trace.end_time, "trace end")
        if not start <= trace_start <= trace_end <= end:
            raise ValueError("trace outside run time interval")
        if not isinstance(trace.success, bool) or not isinstance(trace.timed_out, bool):
            raise ValueError("trace success and timed_out must be boolean")
        if trace.timed_out and trace.success:
            raise ValueError("timed out trace cannot succeed")
        item = capabilities.setdefault(capability, {"attempts": 0, "failures": 0, "timeouts": 0, "tool_time_ms": 0.0})
        item["attempts"] += 1
        item["failures"] += not trace.success
        item["timeouts"] += trace.timed_out
        item["tool_time_ms"] += (trace_end - trace_start) * 1000
    for attempts in invocations.values():
        if sorted(attempts) != list(range(1, len(attempts) + 1)):
            raise ValueError("missing or duplicated retry attempt")
    return {
        "schema_version": "1.0",
        "task_id": outcome["task_id"],
        "run_id": outcome["run_id"],
        "acceptance_id": outcome["acceptance_id"],
        "accepted": outcome["accepted"],
        "wall_time_ms": (end - start) * 1000,
        "attempts": len(traces),
        "invocations": len(invocations),
        "retries": len(traces) - len(invocations),
        "capabilities": capabilities,
        "tool_time_ms": sum(item["tool_time_ms"] for item in capabilities.values()),
        "usage": {field: outcome.get(field) for field in ("input_tokens", "output_tokens", "cost", "cost_currency")},
        "acceptance_authority": "supplied_grader_outcome",
    }


def build_run_evidence_bundle(
    traces: Sequence[PluginTrace],
    outcome: Mapping[str, Any],
    *,
    task_seed_id: str,
    base_commit: str,
    head_commit: str,
    actor: str,
) -> dict[str, Any]:
    report = build_run_report(traces, outcome)
    if not re.fullmatch(r"TS-[0-9]{3,}", task_seed_id):
        raise ValueError("formal Evidence requires TS numeric task_seed_id")
    if min(len(base_commit), len(head_commit)) < 7:
        raise ValueError("commit references must contain at least 7 characters")
    text(actor, "actor")
    evidence = build_trace_evidence_payload(
        traces,
        task_seed_id=task_seed_id,
        base_commit=base_commit,
        head_commit=head_commit,
        actor=actor,
        evidence_id_prefix="EV",
    )
    for trace, entry in zip(traces, evidence, strict=True):
        identity = json.dumps([outcome["task_id"], outcome["run_id"], trace.span_id], ensure_ascii=False)
        entry["id"] = "EV-" + str(int.from_bytes(hashlib.sha256(identity.encode()).digest(), "big"))
    return {
        "schema_version": "1.0",
        "report": report,
        "evidence": evidence,
        "links": [
            {
                "evidence_id": entry["id"],
                "span_id": trace.span_id,
                "run_id": trace.run_id,
                "task_id": trace.task_id,
                "acceptance_id": outcome["acceptance_id"],
            }
            for trace, entry in zip(traces, evidence, strict=True)
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--outcome", type=Path, required=True)
    parser.add_argument("--evidence-context", type=Path)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.traces.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise ValueError("traces must be an object array")
        traces = [PluginTrace(**{k: v for k, v in item.items() if k != "duration_seconds"}) for item in payload]
        outcome = read_object(args.outcome)
        if args.evidence_context:
            context = read_object(args.evidence_context)
            report = build_run_evidence_bundle(traces, outcome, **context)
        else:
            report = build_run_report(traces, outcome)
        report["outcome_sha256"] = digest(args.outcome)
        report["traces_sha256"] = digest(args.traces)
        code = 0
    except (OSError, ValueError, TypeError) as exc:
        report, code = {"status": "invalid", "error": str(exc)}, 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
