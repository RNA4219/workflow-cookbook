# SPDX-License-Identifier: MIT
"""Gateの観測機会を90/180/30日で集計する。stageの変更は行わない。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

WINDOWS = (90, 180, 30)
OUTCOMES = {"true_positive", "false_positive", "true_negative", "false_negative", "error"}
VERSION_KEYS = {"model", "runner", "policy"}


def timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be text")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamp needs a timezone")
    return result.astimezone(UTC)


def summarize_observations(bundle: Mapping[str, Any], *, as_of: datetime) -> dict[str, Any]:
    if as_of.tzinfo is None:
        raise ValueError("as_of needs a timezone")
    if (
        bundle.get("schema_version") != "1.0"
        or not isinstance(bundle.get("gate_id"), str)
        or not bundle["gate_id"].strip()
    ):
        raise ValueError("schema_version 1.0 and gate_id are required")
    versions = bundle.get("versions")
    if (
        not isinstance(versions, Mapping)
        or set(versions) != VERSION_KEYS
        or not all(isinstance(v, str) and v.strip() for v in versions.values())
    ):
        raise ValueError("versions must identify model, runner and policy")
    minimum = bundle.get("minimum_opportunities")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise ValueError("minimum_opportunities must be an agreed positive integer")
    started = timestamp(bundle.get("observation_started_at"))
    if started > as_of:
        raise ValueError("observation_started_at is in the future")
    records = bundle.get("records")
    if not isinstance(records, list):
        raise ValueError("records must be an array")
    seen = set()
    parsed = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("record must be an object")
        identity = record.get("id")
        if not isinstance(identity, str) or not identity.strip() or identity in seen:
            raise ValueError("record id is missing or duplicated")
        seen.add(identity)
        moment = timestamp(record.get("timestamp"))
        if not started <= moment <= as_of:
            raise ValueError("record timestamp is outside the observation period")
        if not isinstance(record.get("outcome"), str) or record["outcome"] not in OUTCOMES:
            raise ValueError("record outcome is unknown")
        if not isinstance(record.get("override"), bool):
            raise ValueError("record override must be boolean")
        record_versions = record.get("versions")
        if (
            not isinstance(record_versions, Mapping)
            or set(record_versions) != VERSION_KEYS
            or not all(isinstance(v, str) and v.strip() for v in record_versions.values())
        ):
            raise ValueError("record versions are incomplete")
        parsed.append((moment, record))
    windows = []
    for days in WINDOWS:
        cutoff = as_of - timedelta(days=days)
        selected = [r for moment, r in parsed if moment > cutoff]
        eligible = [r for r in selected if r["versions"] == versions]
        counts = Counter(r["outcome"] for r in eligible)
        n = len(eligible)
        negatives = counts["false_positive"] + counts["true_negative"]
        positives = counts["true_positive"] + counts["false_negative"]
        classified = n - counts["error"]
        status = "ready_for_review"
        if started > cutoff or n < minimum:
            status = "insufficient_data"
        if counts["error"]:
            status = "measurement_error"
        windows.append(
            {
                "days": days,
                "status": status,
                "eligible_opportunities": n,
                "excluded_version_mismatch": len(selected) - n,
                "counts": {key: counts[key] for key in sorted(OUTCOMES)},
                "overrides": sum(r["override"] for r in eligible),
                "detection_rate": (counts["true_positive"] + counts["false_positive"]) / classified
                if classified
                else None,
                "false_positive_rate": counts["false_positive"] / negatives if negatives else None,
                "false_negative_rate": counts["false_negative"] / positives if positives else None,
                "rate_denominators": {
                    "detection_rate": classified,
                    "false_positive_rate": negatives,
                    "false_negative_rate": positives,
                },
            }
        )
    return {
        "schema_version": "1.0",
        "gate_id": bundle["gate_id"],
        "as_of": as_of.isoformat(),
        "versions": dict(versions),
        "minimum_opportunities": minimum,
        "windows": windows,
        "decision": "owner_review_required",
        "windows_are_overlapping": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--as-of", help="ISO timestamp; defaults to current UTC")
    args = parser.parse_args(argv)
    try:
        bundle = json.loads(args.observations.read_text(encoding="utf-8"))
        if not isinstance(bundle, Mapping):
            raise ValueError("observation bundle must be an object")
        report = summarize_observations(bundle, as_of=timestamp(args.as_of) if args.as_of else datetime.now(UTC))
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
