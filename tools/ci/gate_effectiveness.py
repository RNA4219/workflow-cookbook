# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any


def analyze_gates(
    bundle: dict[str, Any],
    thresholds: dict[str, Any],
    *,
    previous_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a review-only GateEffectivenessReport from self-improvement/v1."""
    now = generated_at or _now()
    since, until, days = _observation_period(bundle)
    task_count = _as_int(bundle.get("task_count"), default=0)
    observations = [item for item in bundle.get("gate_observations", []) if isinstance(item, dict)]
    by_gate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        by_gate[str(observation.get("gate_id") or "unknown")].append(observation)

    previous_items = {
        str(item.get("target_ref")): item
        for item in (previous_report or {}).get("gate_reports", [])
        if isinstance(item, dict)
    }
    gate_reports: list[dict[str, Any]] = []
    for entry in bundle.get("gate_catalog", []):
        if not isinstance(entry, dict):
            continue
        gate_id = str(entry.get("gate_id") or "unknown")
        gate_observations = by_gate.get(gate_id, [])
        report = _analyze_gate(
            entry,
            gate_observations,
            task_count=task_count,
            observation_days=days,
            thresholds=thresholds,
            previous=previous_items.get(gate_id),
        )
        gate_reports.append(report)

    evidence_reports = _analyze_evidence(bundle, thresholds)
    proposals = [
        item
        for item in [*gate_reports, *evidence_reports]
        if item["proposal_eligible"] and item["action"] != "keep"
    ]
    return {
        "schema_version": "self-improvement/v1",
        "report_id": f"GER-{_compact_timestamp(now)}",
        "generated_at": now,
        "observation_period": {"since": since, "until": until, "days": days},
        "gate_reports": gate_reports,
        "evidence_reports": evidence_reports,
        "periodic_nudges": [_to_nudge(item, now) for item in proposals],
    }


def _analyze_gate(
    entry: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    task_count: int,
    observation_days: float,
    thresholds: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    gate_id = str(entry.get("gate_id") or "unknown")
    evaluations = len(observations)
    decisions = Counter(str(item.get("decision") or "unknown") for item in observations)
    effect_count = sum(
        1
        for item in observations
        if item.get("decision") in {"hold", "block"} or item.get("transition_changed") is True
    )
    override_count = sum(1 for item in observations if item.get("override") == "applied")
    unknown_count = sum(
        1
        for item in observations
        if item.get("decision") == "unknown"
        or item.get("transition_changed") == "unknown"
        or item.get("override") == "unknown"
    )
    override_ratio = override_count / evaluations if evaluations else 0.0
    unused = thresholds["unused_gate"]
    no_effect = thresholds["no_effect"]
    overrides = thresholds["override_excessive"]

    classification = "healthy"
    action = "keep"
    eligible = True
    reasons: list[str] = []
    if unknown_count > 0:
        classification, eligible = "insufficient_data", False
        reasons.append("legacy observations contain unknown decision fields")
    elif evaluations <= _as_int(unused["maximum_evaluations"]):
        if observation_days < _as_float(unused["observation_days"]):
            classification, eligible = "insufficient_data", False
            reasons.append("observation period is shorter than the unused Gate threshold")
        elif task_count >= _as_int(unused["minimum_tasks"]):
            classification, action = "unused", "review"
            reasons.append("no evaluations were observed despite sufficient task volume")
        else:
            classification, eligible = "insufficient_data", False
            reasons.append("task volume is below the unused Gate threshold")
    elif (
        override_count >= _as_int(overrides["minimum_overrides"])
        and override_ratio >= _as_float(overrides["minimum_ratio"])
    ):
        classification, action = "override_excessive", "review"
        reasons.append("override count and ratio exceed configured thresholds")
    elif evaluations >= _as_int(no_effect["minimum_evaluations"]):
        if effect_count <= _as_int(no_effect["maximum_effective_decisions"]):
            classification, action = "no_effect", "relax_candidate"
            reasons.append("no hold, block, or transition-changing result was observed")
        else:
            reasons.append("Gate changed at least one effective outcome")
    else:
        classification, eligible = "insufficient_data", False
        reasons.append("evaluation count is below the effectiveness threshold")

    consecutive = _consecutive_periods(classification, previous)
    hard_safety = entry.get("hard_safety") is True
    required_periods = _as_int(thresholds["archive_candidate"]["consecutive_periods"])
    if classification == "unused" and consecutive >= required_periods:
        if hard_safety:
            action = "review"
            reasons.append("hard-safety Gate is excluded from archive candidates")
        else:
            action = "archive_candidate"
            reasons.append("the same unused result continued across observation periods")

    return {
        "target_kind": "gate",
        "target_ref": gate_id,
        "owner": str(entry.get("owner") or "unknown"),
        "policy_revision": str(entry.get("policy_revision") or "unknown"),
        "hard_safety": hard_safety,
        "classification": classification,
        "action": action,
        "proposal_eligible": eligible,
        "consecutive_periods": consecutive,
        "metrics": {
            "task_count": task_count,
            "evaluation_count": evaluations,
            "decision_counts": dict(sorted(decisions.items())),
            "effective_decision_count": effect_count,
            "override_count": override_count,
            "override_ratio": round(override_ratio, 6),
            "unknown_observation_count": unknown_count,
        },
        "reasons": reasons,
    }


def _analyze_evidence(
    bundle: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in bundle.get("evidence_consumption_observations", []):
        if isinstance(item, dict):
            by_type[str(item.get("evidence_type") or "unknown")].append(item)
    reports: list[dict[str, Any]] = []
    config = thresholds["unread_evidence"]
    for evidence_type, observations in sorted(by_type.items()):
        generated = {
            str(item.get("evidence_id"))
            for item in observations
            if item.get("disposition") == "generated"
        }
        acknowledged = {
            str(item.get("evidence_id"))
            for item in observations
            if item.get("disposition") == "acknowledged"
            and str(item.get("evidence_id")) in generated
        }
        ack_rate = len(acknowledged) / len(generated) if generated else 0.0
        if len(generated) < _as_int(config["minimum_generated"]):
            classification, action, eligible = "insufficient_data", "keep", False
            reasons = ["generated Evidence count is below the configured threshold"]
        elif ack_rate <= _as_float(config["maximum_ack_rate"]):
            classification, action, eligible = "unread_evidence", "review", True
            reasons = ["explicit Evidence acknowledgement rate is at or below the configured threshold"]
        else:
            classification, action, eligible = "healthy", "keep", True
            reasons = ["explicit Evidence acknowledgement rate is above the configured threshold"]
        reports.append({
            "target_kind": "evidence_type",
            "target_ref": evidence_type,
            "classification": classification,
            "action": action,
            "proposal_eligible": eligible,
            "metrics": {
                "generated_count": len(generated),
                "acknowledged_count": len(acknowledged),
                "ack_rate": round(ack_rate, 6),
            },
            "reasons": reasons,
        })
    return reports


def _observation_period(bundle: dict[str, Any]) -> tuple[str, str, float]:
    timestamps = [
        str(item.get("occurred_at"))
        for key in ("gate_observations", "evidence_consumption_observations")
        for item in bundle.get(key, [])
        if isinstance(item, dict) and item.get("occurred_at")
    ]
    since = str(bundle.get("since") or (min(timestamps) if timestamps else bundle["generated_at"]))
    until = str(bundle.get("until") or (max(timestamps) if timestamps else bundle["generated_at"]))
    start = _parse_datetime(since)
    end = _parse_datetime(until)
    return since, until, max(0.0, (end - start).total_seconds() / 86400)


def _consecutive_periods(classification: str, previous: dict[str, Any] | None) -> int:
    if previous and previous.get("classification") == classification:
        return _as_int(previous.get("consecutive_periods"), default=1) + 1
    return 1


def _to_nudge(item: dict[str, Any], created_at: str) -> dict[str, Any]:
    target_kind = str(item["target_kind"])
    target_ref = str(item["target_ref"])
    return {
        "nudge_id": f"NUDGE-{target_kind}-{target_ref}-{_compact_timestamp(created_at)}",
        "reason": "; ".join(item["reasons"]),
        "target_kind": target_kind,
        "target_ref": target_ref,
        "suggested_action": item["action"],
        "created_at": created_at,
        "priority": "high" if item["action"] == "archive_candidate" else "medium",
        "category": "other",
        "blocking": False,
        "schema_version": "1.0.0",
    }


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _compact_timestamp(value: str) -> str:
    return "".join(character for character in value if character.isdigit())[:14]


def _as_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
