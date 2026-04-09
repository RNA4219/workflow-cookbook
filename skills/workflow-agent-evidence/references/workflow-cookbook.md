workflow-cookbook integration notes

Key files
- tools/perf/structured_logger.py
- tools/protocols/evidence_bridge.py
- tools/protocols/plugin_loader.py
- tools/protocols/plugin_config.py
- tools/protocols/README.md
- examples/inference_plugins.agent_protocol.sample.json
- examples/agent_protocol_evidence_consumer.sample.py
- schemas/inference-plugin-config.schema.json

Typical validation
- uv run pytest tests/test_structured_logger.py -q
- uv run pytest tests/test_agent_protocol_evidence.py -q
- uv run pytest tests/test_plugin_loader.py -q
- uv run pytest tests/test_plugin_config.py -q

Checklist
- Keep StructuredLogger generic.
- Keep contract-specific mapping inside tools/protocols.
- Update sample config, consumer sample, schema, docs, and tests together.
- If requirements change, sync requirements/spec/design before implementation.
