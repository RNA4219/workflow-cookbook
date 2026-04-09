from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.perf.structured_logger import InferenceLogRecord
from tools.protocols.evidence_bridge import (
    AgentProtocolContextExtractor,
    AgentProtocolEvidenceBridge,
    AgentProtocolEvidenceError,
    AgentProtocolEvidenceMapper,
    AgentProtocolEvidencePlugin,
    JsonLinesEvidenceWriter,
)


def _hash_json(value: object) -> str:
    normalized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def test_evidence_bridge_builds_schema_compatible_payload(tmp_path: Path) -> None:
    bridge = AgentProtocolEvidenceBridge(repo_root=tmp_path)
    record = InferenceLogRecord(
        logger="workflow.metrics",
        event="inference",
        level="INFO",
        timestamp="2026-04-09T10:00:00+00:00",
        inference_id="run-100",
        model="gpt-5.4",
        prompt={"messages": [{"role": "user", "content": "Ping"}]},
        response={"content": "Pong", "finish_reason": "stop"},
        extra={
            "agent_protocol": {
                "evidence_id": "EV-100",
                "task_seed_id": "TS-100",
                "base_commit": "abc1234",
                "head_commit": "def5678",
                "actor": "codex",
            }
        },
    )

    evidence = bridge.build_inference_evidence(record)

    assert evidence is not None
    assert evidence["schemaVersion"] == "1.0.0"
    assert evidence["kind"] == "Evidence"
    assert evidence["state"] == "Published"
    assert evidence["version"] == 1
    assert evidence["id"] == "EV-100"
    assert evidence["taskSeedId"] == "TS-100"
    assert evidence["baseCommit"] == "abc1234"
    assert evidence["headCommit"] == "def5678"
    assert evidence["actor"] == "codex"
    assert evidence["startTime"] == "2026-04-09T10:00:00+00:00"
    assert evidence["endTime"] == "2026-04-09T10:00:00+00:00"
    assert evidence["inputHash"] == _hash_json({"messages": [{"role": "user", "content": "Ping"}]})
    assert evidence["outputHash"] == _hash_json({"content": "Pong", "finish_reason": "stop"})
    assert evidence["diffHash"] == _hash_json("")
    assert evidence["model"]["name"] == "gpt-5.4"
    assert evidence["model"]["version"] == "unknown"
    assert evidence["tools"] == ["StructuredLogger"]
    assert evidence["environment"]["containerImageDigest"] == "uncontainerized"
    assert evidence["environment"]["runtime"].startswith("Python ")
    assert evidence["staleStatus"] == {
        "classification": "fresh",
        "evaluatedAt": "2026-04-09T10:00:00+00:00",
    }
    assert evidence["mergeResult"] == {"status": "not_applicable"}
    assert evidence["policyVerdict"] == "manual_review_required"


def test_evidence_bridge_uses_optional_context_and_lockfile(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("locked-deps", encoding="utf-8")
    bridge = AgentProtocolEvidenceBridge(repo_root=tmp_path)
    record = InferenceLogRecord(
        logger="workflow.metrics",
        event="inference",
        level="INFO",
        timestamp="2026-04-09T10:30:00+00:00",
        inference_id="run-200",
        model="gpt-5.4",
        prompt={"messages": [{"role": "user", "content": "Plan"}]},
        response={"content": "Done"},
        extra={
            "agent_protocol": {
                "evidence_id": "EV-200",
                "task_seed_id": "TS-200",
                "base_commit": "1111111",
                "head_commit": "2222222",
                "actor": "codex",
                "start_time": "2026-04-09T10:00:00Z",
                "model_version": "2026-04-01",
                "parameters": {"temperature": 0.2},
                "tools": ["StructuredLogger", "Edit"],
                "policy_verdict": "approved",
                "stale_status": {
                    "classification": "soft_stale",
                    "evaluated_at": "2026-04-09T10:15:00Z",
                    "reason": "repo changed",
                },
                "merge_result": {
                    "status": "merged",
                    "merged_at": "2026-04-09T10:20:00Z",
                    "strategy": "squash",
                },
                "diff": {"files": ["docs/spec.md"]},
                "approvals_snapshot": [
                    {
                        "role": "project_lead",
                        "actor_id": "lead-1",
                        "decision": "approved",
                        "decided_at": "2026-04-09T10:25:00Z",
                    }
                ],
            }
        },
    )

    evidence = bridge.build_inference_evidence(record)

    assert evidence is not None
    assert evidence["startTime"] == "2026-04-09T10:00:00+00:00"
    assert evidence["model"]["version"] == "2026-04-01"
    assert evidence["model"]["parametersHash"] == _hash_json({"temperature": 0.2})
    assert evidence["tools"] == ["StructuredLogger", "Edit"]
    assert evidence["staleStatus"]["classification"] == "soft_stale"
    assert evidence["staleStatus"]["reason"] == "repo changed"
    assert evidence["mergeResult"]["status"] == "merged"
    assert evidence["mergeResult"]["mergedAt"] == "2026-04-09T10:20:00+00:00"
    assert evidence["mergeResult"]["strategy"] == "squash"
    assert evidence["policyVerdict"] == "approved"
    assert evidence["diffHash"] == _hash_json({"files": ["docs/spec.md"]})
    assert evidence["approvalsSnapshot"][0]["actorId"] == "lead-1"
    assert evidence["environment"]["lockfileHash"] == (
        f"sha256:{hashlib.sha256('locked-deps'.encode('utf-8')).hexdigest()}"
    )


def test_evidence_bridge_returns_none_without_agent_protocol_context(tmp_path: Path) -> None:
    bridge = AgentProtocolEvidenceBridge(repo_root=tmp_path)
    record = InferenceLogRecord(
        logger="workflow.metrics",
        event="inference",
        level="INFO",
        timestamp="2026-04-09T10:00:00+00:00",
        model="gpt-5.4",
        prompt={"messages": []},
        response={"content": "noop"},
        extra={"run": "plain-log"},
    )

    assert bridge.build_inference_evidence(record) is None


def test_evidence_bridge_rejects_incomplete_context(tmp_path: Path) -> None:
    bridge = AgentProtocolEvidenceBridge(repo_root=tmp_path)
    record = InferenceLogRecord(
        logger="workflow.metrics",
        event="inference",
        level="INFO",
        timestamp="2026-04-09T10:00:00+00:00",
        model="gpt-5.4",
        prompt={"messages": []},
        response={"content": "noop"},
        extra={
            "agent_protocol": {
                "evidence_id": "EV-300",
                "base_commit": "abc1234",
                "head_commit": "def5678",
                "actor": "codex",
            }
        },
    )

    with pytest.raises(AgentProtocolEvidenceError):
        bridge.build_inference_evidence(record)


class StubEnvironmentResolver:
    def resolve(self) -> dict[str, str]:
        return {
            "os": "TestOS",
            "runtime": "TestRuntime",
            "containerImageDigest": "sha256:test-container",
            "lockfileHash": "sha256:test-lockfile",
        }


def test_evidence_mapper_allows_custom_context_key_and_environment(tmp_path: Path) -> None:
    mapper = AgentProtocolEvidenceMapper(
        context_extractor=AgentProtocolContextExtractor(context_key="plugin_context"),
        environment_resolver=StubEnvironmentResolver(),
    )
    record = InferenceLogRecord(
        logger="workflow.metrics",
        event="inference",
        level="INFO",
        timestamp="2026-04-09T10:00:00+00:00",
        model="gpt-5.4",
        prompt={"messages": [{"role": "user", "content": "hello"}]},
        response={"content": "world"},
        extra={
            "plugin_context": {
                "evidence_id": "EV-400",
                "task_seed_id": "TS-400",
                "base_commit": "abc1234",
                "head_commit": "def5678",
                "actor": "custom-plugin",
            }
        },
    )

    evidence = mapper.map_record(record)

    assert evidence is not None
    assert evidence["id"] == "EV-400"
    assert evidence["environment"] == {
        "os": "TestOS",
        "runtime": "TestRuntime",
        "containerImageDigest": "sha256:test-container",
        "lockfileHash": "sha256:test-lockfile",
    }


def test_evidence_plugin_writes_jsonl_via_writer(tmp_path: Path) -> None:
    output_path = tmp_path / "plugin-evidence.jsonl"
    plugin = AgentProtocolEvidencePlugin(
        writer=JsonLinesEvidenceWriter(path=output_path),
        mapper=AgentProtocolEvidenceMapper(
            environment_resolver=StubEnvironmentResolver(),
        ),
    )
    record = InferenceLogRecord(
        logger="workflow.metrics",
        event="inference",
        level="INFO",
        timestamp="2026-04-09T10:00:00+00:00",
        model="gpt-5.4",
        prompt={"messages": [{"role": "user", "content": "hello"}]},
        response={"content": "world"},
        extra={
            "agent_protocol": {
                "evidence_id": "EV-500",
                "task_seed_id": "TS-500",
                "base_commit": "abc1234",
                "head_commit": "def5678",
                "actor": "plugin",
            }
        },
    )

    plugin.handle_inference(record)

    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["id"] == "EV-500"
    assert payload["environment"]["runtime"] == "TestRuntime"
