from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = (
    "id",
    "taskSeedId",
    "baseCommit",
    "headCommit",
    "actor",
    "policyVerdict",
    "startTime",
    "endTime",
)


def load_evidence_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    entries: list[dict[str, Any]] = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        entries.append(json.loads(raw))
    return entries


def summarize_evidence(entries: list[dict[str, Any]]) -> dict[str, Any]:
    missing: list[str] = []
    for entry in entries:
        for key in REQUIRED_KEYS:
            if key not in entry:
                missing.append(f"{entry.get('id', '<unknown>')}:{key}")
    return {
        "count": len(entries),
        "actors": sorted({str(entry.get("actor", "")) for entry in entries if entry.get("actor")}),
        "policy_verdicts": sorted(
            {str(entry.get("policyVerdict", "")) for entry in entries if entry.get("policyVerdict")}
        ),
        "missing_required_fields": missing,
    }


def main() -> None:
    sample_path = (
        Path(__file__).resolve().parents[2]
        / "agent-protocols"
        / "examples"
        / "evidence.sample.json"
    )
    evidence_jsonl = Path("evidence.sample.jsonl")
    if not evidence_jsonl.exists():
        evidence_jsonl.write_text(sample_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    report = summarize_evidence(load_evidence_jsonl(evidence_jsonl))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
