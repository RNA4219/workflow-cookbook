from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.perf.structured_logger import StructuredLogger
from tools.protocols import plugin_config
from tools.protocols.plugin_config import (
    InferencePluginConfigError,
    load_inference_plugin_specs,
    load_inference_plugin_specs_from_mapping,
    load_inference_plugin_specs_from_path,
)


def test_load_plugin_specs_from_mapping_accepts_named_root() -> None:
    specs = load_inference_plugin_specs_from_mapping(
        {
            "inference_plugins": [
                {
                    "factory": "tools.protocols.evidence_bridge:create_agent_protocol_evidence_plugin",
                    "options": {"path": "evidence.jsonl", "repo_root": "."},
                }
            ]
        }
    )

    assert len(specs) == 1
    assert specs[0].factory == "tools.protocols.evidence_bridge:create_agent_protocol_evidence_plugin"
    assert specs[0].options == {"path": "evidence.jsonl", "repo_root": "."}
    assert specs[0].enabled is True


def test_load_plugin_specs_from_json_path(tmp_path: Path) -> None:
    config_path = tmp_path / "plugins.json"
    config_path.write_text(
        json.dumps(
            {
                "inference_plugins": [
                    {
                        "factory": "tools.protocols.evidence_bridge:create_agent_protocol_evidence_plugin",
                        "options": {
                            "path": str(tmp_path / "evidence.jsonl"),
                            "repo_root": str(tmp_path),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = load_inference_plugin_specs_from_path(config_path)

    assert len(specs) == 1
    assert specs[0].options == {
        "path": str(tmp_path / "evidence.jsonl"),
        "repo_root": str(tmp_path),
    }


def test_load_plugin_specs_from_yaml_path_uses_yaml_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "plugins.yaml"
    config_path.write_text("ignored: because-fake-loader\n", encoding="utf-8")
    monkeypatch.setattr(
        plugin_config,
        "yaml",
        SimpleNamespace(
            safe_load=lambda _raw: {
                "inference_plugins": [
                    {
                        "factory": "tools.protocols.evidence_bridge:create_agent_protocol_evidence_plugin",
                        "options": {"path": "evidence.jsonl", "repo_root": "."},
                    }
                ]
            }
        ),
        raising=False,
    )

    specs = load_inference_plugin_specs_from_path(config_path)

    assert len(specs) == 1
    assert specs[0].factory == "tools.protocols.evidence_bridge:create_agent_protocol_evidence_plugin"


def test_load_plugin_specs_from_yaml_path_fails_without_yaml_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "plugins.yaml"
    config_path.write_text("inference_plugins: []\n", encoding="utf-8")
    monkeypatch.setattr(plugin_config, "yaml", None, raising=False)

    with pytest.raises(InferencePluginConfigError):
        load_inference_plugin_specs_from_path(config_path)


def test_structured_logger_from_plugin_config_supports_mapping_source(tmp_path: Path) -> None:
    stream = io.StringIO()
    evidence_path = tmp_path / "evidence.jsonl"
    logger = StructuredLogger.from_plugin_config(
        name="workflow.metrics",
        stream=stream,
        plugin_config={
            "inference_plugins": [
                {
                    "factory": "tools.protocols.evidence_bridge:create_agent_protocol_evidence_plugin",
                    "options": {
                        "path": str(evidence_path),
                        "repo_root": str(tmp_path),
                    },
                }
            ]
        },
    )

    logger.inference(
        inference_id="run-config-map",
        model="gpt-5.4",
        prompt={"messages": [{"role": "user", "content": "hello"}]},
        response={"content": "world"},
        extra={
            "agent_protocol": {
                "evidence_id": "EV-700",
                "task_seed_id": "TS-700",
                "base_commit": "abc1234",
                "head_commit": "def5678",
                "actor": "mapping-config",
            }
        },
    )

    log_payload = json.loads(stream.getvalue().splitlines()[0])
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8").splitlines()[0])
    assert log_payload["inference_id"] == "run-config-map"
    assert evidence_payload["id"] == "EV-700"
    assert evidence_payload["actor"] == "mapping-config"


def test_load_plugin_specs_rejects_invalid_enabled_type() -> None:
    with pytest.raises(InferencePluginConfigError):
        load_inference_plugin_specs(
            {"inference_plugins": [{"factory": "x:y", "enabled": "yes"}]}
        )


def test_sample_plugin_config_is_loadable() -> None:
    sample_path = ROOT / "examples" / "inference_plugins.agent_protocol.sample.json"

    specs = load_inference_plugin_specs_from_path(sample_path)

    assert len(specs) == 1
    assert specs[0].factory == (
        "tools.protocols.evidence_bridge:create_agent_protocol_evidence_plugin"
    )
    assert specs[0].enabled is True


def test_plugin_config_schema_matches_sample_shape() -> None:
    schema_path = ROOT / "schemas" / "inference-plugin-config.schema.json"
    sample_path = ROOT / "examples" / "inference_plugins.agent_protocol.sample.json"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    sample = json.loads(sample_path.read_text(encoding="utf-8"))

    assert schema["$id"].endswith("/inference-plugin-config.schema.json")
    plugin_root = schema["$defs"]["pluginConfigRoot"]
    plugin_spec = schema["$defs"]["pluginSpec"]
    assert plugin_root["required"] == ["inference_plugins"]
    assert "factory" in plugin_spec["required"]
    assert plugin_spec["properties"]["enabled"]["type"] == "boolean"
    assert "inference_plugins" in sample
    assert isinstance(sample["inference_plugins"], list)
    assert sample["inference_plugins"][0]["factory"] == (
        "tools.protocols.evidence_bridge:create_agent_protocol_evidence_plugin"
    )
