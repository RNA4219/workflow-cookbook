from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.perf.structured_logger import StructuredLogger
from tools.protocols.plugin_loader import (
    InferencePluginLoadError,
    InferencePluginSpec,
    instantiate_inference_plugins,
)


def test_instantiate_plugins_loads_factory_from_import_string(tmp_path: Path) -> None:
    specs = [
        InferencePluginSpec(
            factory="tools.protocols.evidence_bridge:create_agent_protocol_evidence_plugin",
            options={
                "path": str(tmp_path / "evidence.jsonl"),
                "repo_root": str(tmp_path),
            },
        )
    ]

    plugins = instantiate_inference_plugins(specs)

    assert len(plugins) == 1
    assert hasattr(plugins[0], "handle_inference")


def test_instantiate_plugins_skips_disabled_specs(tmp_path: Path) -> None:
    specs = [
        InferencePluginSpec(
            factory="tools.protocols.evidence_bridge:create_agent_protocol_evidence_plugin",
            options={
                "path": str(tmp_path / "evidence.jsonl"),
                "repo_root": str(tmp_path),
            },
            enabled=False,
        )
    ]

    assert instantiate_inference_plugins(specs) == []


def test_instantiate_plugins_rejects_invalid_factory() -> None:
    specs = [InferencePluginSpec(factory="tools.protocols.evidence_bridge:not_found")]

    with pytest.raises(InferencePluginLoadError):
        instantiate_inference_plugins(specs)


def test_structured_logger_from_plugin_specs_builds_logger_with_plugins(tmp_path: Path) -> None:
    stream = io.StringIO()
    evidence_path = tmp_path / "evidence.jsonl"
    logger = StructuredLogger.from_plugin_specs(
        name="workflow.metrics",
        stream=stream,
        plugin_specs=[
            InferencePluginSpec(
                factory="tools.protocols.evidence_bridge:create_agent_protocol_evidence_plugin",
                options={
                    "path": str(evidence_path),
                    "repo_root": str(tmp_path),
                },
            )
        ],
    )

    logger.inference(
        inference_id="run-config",
        model="gpt-5.4",
        prompt={"messages": [{"role": "user", "content": "hello"}]},
        response={"content": "world"},
        extra={
            "agent_protocol": {
                "evidence_id": "EV-600",
                "task_seed_id": "TS-600",
                "base_commit": "abc1234",
                "head_commit": "def5678",
                "actor": "config-plugin",
            }
        },
    )

    log_payload = json.loads(stream.getvalue().splitlines()[0])
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8").splitlines()[0])
    assert log_payload["inference_id"] == "run-config"
    assert evidence_payload["id"] == "EV-600"
    assert evidence_payload["actor"] == "config-plugin"
