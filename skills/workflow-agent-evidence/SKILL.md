---
name: workflow-agent-evidence
description: Integrate and maintain workflow-cookbook logging with agent-protocols Evidence records. Use when Codex needs to wire StructuredLogger plugins, tools/protocols/evidence_bridge.py, plugin_loader.py, plugin_config.py, sample config, consumer samples, schema alignment, related tests, or requirements/spec/design updates for Evidence tracking.
---

# Workflow Agent Evidence

Use this skill when connecting workflow-cookbook LLM activity logging to the
agent-protocols Evidence contract.

## Quick Start

1. If the target is workflow-cookbook, read `workflow-cookbook/README.md` and
   `workflow-cookbook/tools/protocols/README.md` first.
2. If the task touches Evidence fields or schema behavior, read
   `references/agent-protocols.md`.
3. Update logger integration points, plugin config, sample config, consumer
   sample, and tests together.
4. If behavior changes, sync requirements/spec/design before implementation.

## Standard Flow

### 1. Confirm the integration points

Prioritize these files:

- `workflow-cookbook/tools/perf/structured_logger.py`
- `workflow-cookbook/tools/protocols/evidence_bridge.py`
- `workflow-cookbook/tools/protocols/plugin_loader.py`
- `workflow-cookbook/tools/protocols/plugin_config.py`
- `workflow-cookbook/examples/inference_plugins.agent_protocol.sample.json`
- `workflow-cookbook/examples/agent_protocol_evidence_consumer.sample.py`

Read `references/workflow-cookbook.md` for details.

### 2. Preserve the architecture

- Keep `StructuredLogger` generic.
- Keep contract-specific mapping inside `tools/protocols/`.
- Keep the plugin boundary at `handle_inference(record)`.
- Update config loader, sample, schema, docs, and tests together.

### 3. Preserve the Evidence contract

Do not introduce missing required Evidence fields.
Read `references/agent-protocols.md` before changing contract-related logic.

### 4. Sync docs and tests

When implementation changes, update the relevant docs as needed:

- `docs/requirements.md`
- `docs/spec.md`
- `docs/design.md`
- `docs/interfaces.md`
- `RUNBOOK.md`
- `CHANGELOG.md`

### 5. Validate

At minimum, review these test targets:

- `tests/test_structured_logger.py`
- `tests/test_agent_protocol_evidence.py`
- `tests/test_plugin_loader.py`
- `tests/test_plugin_config.py`

Run Markdown lint when README or release docs change.

## References

- `references/workflow-cookbook.md`
  - integration paths, key files, validation commands
- `references/agent-protocols.md`
  - required Evidence fields and review points
