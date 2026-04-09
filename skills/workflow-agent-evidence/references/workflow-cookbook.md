# workflow-cookbook Integration Notes

## Core Files

| File | Role |
|------|------|
| `tools/perf/structured_logger.py` | Generic LLM activity logger |
| `tools/protocols/evidence_bridge.py` | Evidence contract mapping |
| `tools/protocols/plugin_loader.py` | Plugin discovery and loading |
| `tools/protocols/plugin_config.py` | Configuration parsing |
| `tools/protocols/README.md` | Protocol documentation |

## Sample Files

| File | Role |
|------|------|
| `examples/inference_plugins.agent_protocol.sample.json` | Plugin config sample |
| `examples/agent_protocol_evidence_consumer.sample.py` | Consumer implementation sample |
| `schemas/inference-plugin-config.schema.json` | JSON schema for config validation |

## Validation Commands

```sh
# Core tests
uv run pytest tests/test_structured_logger.py -q
uv run pytest tests/test_agent_protocol_evidence.py -q
uv run pytest tests/test_plugin_loader.py -q
uv run pytest tests/test_plugin_config.py -q

# All related tests
uv run pytest tests/test_structured_logger.py tests/test_agent_protocol_evidence.py tests/test_plugin_loader.py tests/test_plugin_config.py -q

# Coverage gate
uv run pytest tests/ --cov=. --cov-fail-under=80 -q

# Config validation
python tools/workflow_plugins/validate_workflow_plugin_config.py --config examples/inference_plugins.agent_protocol.sample.json
```

## Architecture Checklist

- [ ] Keep `StructuredLogger` generic (no Evidence-specific logic)
- [ ] Keep contract-specific mapping inside `tools/protocols/`
- [ ] Plugin boundary at `handle_inference(record)`
- [ ] Update sample config, consumer sample, schema together
- [ ] Sync requirements/spec/design before implementation changes
- [ ] Run all related tests after changes

## Integration Pattern

```text
StructuredLogger
    ↓ record
handle_inference(record)
    ↓ mapping
EvidenceBridge
    ↓ output
Evidence record (agent-protocols schema)
```

## Related Docs

- `docs/requirements.md` — Evidence requirements
- `docs/spec.md` — Evidence specification
- `docs/design.md` — Architecture details
- `docs/interfaces.md` — Interface contracts
