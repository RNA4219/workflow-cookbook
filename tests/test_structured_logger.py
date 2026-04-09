from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.perf.structured_logger import InferenceLogRecord, StructuredLogger
from tools.protocols.evidence_bridge import AgentProtocolEvidenceFileSink


class CapturePlugin:
    def __init__(self) -> None:
        self.records: list[InferenceLogRecord] = []

    def handle_inference(self, record: InferenceLogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "chainlit.log"


def test_structured_logger_writes_inference_records(log_path: Path) -> None:
    logger = StructuredLogger(name="workflow.metrics", path=log_path)

    logger.inference(
        inference_id="run-42",
        model="gpt-4.1-mini",
        prompt={"messages": [{"role": "user", "content": "Ping"}]},
        response={"content": "Pong", "finish_reason": "stop"},
        metrics={
            "semantic_retention": 0.82,
            "spec_completeness": {"with_spec": 91, "total": 100},
        },
        tags=("qa", "integration"),
        extra={"run": "test"},
    )

    contents = log_path.read_text(encoding="utf-8").splitlines()
    assert len(contents) == 1

    payload = json.loads(contents[0])
    assert payload["logger"] == "workflow.metrics"
    assert payload["event"] == "inference"
    assert payload["level"] == "INFO"
    assert payload["metrics"] == {
        "semantic_retention": 0.82,
        "spec_completeness": {"with_spec": 91, "total": 100},
    }
    assert payload["prompt"] == {"messages": [{"role": "user", "content": "Ping"}]}
    assert payload["response"] == {"content": "Pong", "finish_reason": "stop"}
    assert payload["tags"] == ["qa", "integration"]
    assert payload["extra"] == {"run": "test"}
    assert payload["inference_id"] == "run-42"
    assert payload["model"] == "gpt-4.1-mini"
    assert isinstance(payload["timestamp"], str)

    # Chainlit のログ収集と互換なメトリクス構造を確認
    assert payload["metrics"]["spec_completeness"]["with_spec"] == 91
    assert payload["metrics"]["spec_completeness"]["total"] == 100


def test_structured_logger_emits_agent_protocol_evidence(tmp_path: Path, log_path: Path) -> None:
    evidence_path = tmp_path / "evidence.jsonl"
    logger = StructuredLogger(
        name="workflow.metrics",
        path=log_path,
        evidence_sink=AgentProtocolEvidenceFileSink(
            path=evidence_path,
            repo_root=tmp_path,
        ),
    )

    logger.inference(
        inference_id="run-99",
        model="gpt-5.4",
        prompt={"messages": [{"role": "user", "content": "Track me"}]},
        response={"content": "Tracked"},
        extra={
            "agent_protocol": {
                "evidence_id": "EV-999",
                "task_seed_id": "TS-999",
                "base_commit": "abc1234",
                "head_commit": "def5678",
                "actor": "codex",
                "tools": ["StructuredLogger", "Shell"],
            }
        },
    )

    log_payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8").splitlines()[0])

    assert log_payload["inference_id"] == "run-99"
    assert evidence_payload["id"] == "EV-999"
    assert evidence_payload["taskSeedId"] == "TS-999"
    assert evidence_payload["model"]["name"] == "gpt-5.4"
    assert evidence_payload["tools"] == ["StructuredLogger", "Shell"]


def test_structured_logger_supports_generic_plugins(log_path: Path) -> None:
    plugin = CapturePlugin()
    logger = StructuredLogger(
        name="workflow.metrics",
        path=log_path,
        plugins=[plugin],
    )

    logger.inference(
        inference_id="run-plugin",
        model="gpt-5.4-mini",
        prompt={"messages": [{"role": "user", "content": "hello"}]},
        response={"content": "world"},
    )

    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["inference_id"] == "run-plugin"
    assert len(plugin.records) == 1
    assert plugin.records[0].model == "gpt-5.4-mini"
