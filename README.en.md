# Workflow Cookbook / Workflow Operations Kit

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Language: [English](README.md) | [日本語](README.ja.md) | [简体中文](README.zh-CN.md)

Workflow Cookbook is a docs and runtime kit for workflow operations and
context engineering.
It bundles Birdseye/Codemap, Task Seeds, acceptance operations, reusable CI,
and plugin-based Evidence tracking.

<!-- LLM-BOOTSTRAP v1 -->
Recommended read order:

1. `docs/birdseye/index.json` for the lightweight node graph
2. `docs/birdseye/caps/<path>.json` for focused point reads

Focus procedure:

- Find node IDs for recently changed files within +/-2 hops from `index.json`
- Read only the matching `caps/*.json` files

<!-- /LLM-BOOTSTRAP -->

## What's Included

- Birdseye / Codemap for Markdown hub synchronization and dependency mapping
- Operational docs centered on `BLUEPRINT`, `RUNBOOK`, `EVALUATION`, and `CHECKLISTS`
- LLM behavior tracking through `StructuredLogger` plugins
- Sample config and consumer sample for the `agent-protocols` `Evidence` contract
- A workflow host that can connect `agent-taskstate` and `memx-resolver` as optional plugins
- Reusable CI / governance workflows and validation scripts

## Quick Start

1. Read the core docs:
   [`BLUEPRINT.md`](BLUEPRINT.md),
   [`RUNBOOK.md`](RUNBOOK.md),
   [`EVALUATION.md`](EVALUATION.md)
2. Refresh Birdseye:

   ```sh
   python tools/codemap/update.py --since --emit index+caps
   ```

3. Record an acceptance result:
   [`docs/acceptance/README.md`](docs/acceptance/README.md),
   [`docs/acceptance/ACCEPTANCE_TEMPLATE.md`](docs/acceptance/ACCEPTANCE_TEMPLATE.md)
4. Review the test quality baseline:
   [`docs/addenda/J_Test_Engineering.md`](docs/addenda/J_Test_Engineering.md)
5. Try Evidence tracking:
   [`tools/protocols/README.md`](tools/protocols/README.md),
   [`examples/inference_plugins.agent_protocol.sample.json`](examples/inference_plugins.agent_protocol.sample.json)
6. Try cross-repo plugins:
   [`tools/workflow_plugins/README.md`](tools/workflow_plugins/README.md),
   [`examples/workflow_plugins.cross_repo.sample.json`](examples/workflow_plugins.cross_repo.sample.json)
7. Validate plugin config:
   [`tools/workflow_plugins/validate_workflow_plugin_config.py`](tools/workflow_plugins/validate_workflow_plugin_config.py)

## Navigation

- Start here:
  [`BLUEPRINT.md`](BLUEPRINT.md),
  [`RUNBOOK.md`](RUNBOOK.md),
  [`EVALUATION.md`](EVALUATION.md)
- Birdseye / Codemap:
  [`docs/BIRDSEYE.md`](docs/BIRDSEYE.md),
  [`tools/codemap/README.md`](tools/codemap/README.md),
  [`HUB.codex.md`](HUB.codex.md)
- CI / Governance:
  [`CHECKLISTS.md`](CHECKLISTS.md),
  [`docs/ci-config.md`](docs/ci-config.md),
  [`docs/ci_phased_rollout_requirements.md`](docs/ci_phased_rollout_requirements.md)
- Quality baseline:
  [`docs/addenda/J_Test_Engineering.md`](docs/addenda/J_Test_Engineering.md),
  [`docs/acceptance/README.md`](docs/acceptance/README.md)

## Skills

- Evidence integration skill:
  [`skills/workflow-agent-evidence/SKILL.md`](skills/workflow-agent-evidence/SKILL.md)
- Agent metadata:
  [`skills/workflow-agent-evidence/agents/openai.yaml`](skills/workflow-agent-evidence/agents/openai.yaml),
  [`skills/workflow-agent-evidence/agents/claude.yaml`](skills/workflow-agent-evidence/agents/claude.yaml)
- Skill references:
  [`skills/workflow-agent-evidence/references/workflow-cookbook.md`](skills/workflow-agent-evidence/references/workflow-cookbook.md),
  [`skills/workflow-agent-evidence/references/agent-protocols.md`](skills/workflow-agent-evidence/references/agent-protocols.md)
- Protocol plugin guide:
  [`tools/protocols/README.md`](tools/protocols/README.md)

## Common Entry Points

### Birdseye / Codemap

```sh
python tools/codemap/update.py --since --emit index+caps
python tools/codemap/update.py --since --radius 1 --emit caps
python tools/codemap/update.py --targets docs/birdseye/index.json,docs/birdseye/hot.json --emit index+caps
```

### LLM Evidence Tracking

- Plugin API:
  [`tools/protocols/README.md`](tools/protocols/README.md)
- Sample config:
  [`examples/inference_plugins.agent_protocol.sample.json`](examples/inference_plugins.agent_protocol.sample.json)
- Consumer sample:
  [`examples/agent_protocol_evidence_consumer.sample.py`](examples/agent_protocol_evidence_consumer.sample.py)

### Task Operations

- Task Seed sample:
  [`examples/TASK.sample.md`](examples/TASK.sample.md)
- Operating constraints:
  [`GUARDRAILS.md`](GUARDRAILS.md),
  [`RUNBOOK.md`](RUNBOOK.md)
- Release-side docs:
  [`CHECKLISTS.md`](CHECKLISTS.md),
  [`CHANGELOG.md`](CHANGELOG.md),
  [`docs/acceptance/README.md`](docs/acceptance/README.md)

### Advanced: Cross-Repo Plugins

- Host / config:
  [`tools/workflow_plugins/README.md`](tools/workflow_plugins/README.md),
  [`examples/workflow_plugins.cross_repo.sample.json`](examples/workflow_plugins.cross_repo.sample.json),
  [`schemas/workflow-plugin-config.schema.json`](schemas/workflow-plugin-config.schema.json)
- Config validation:
  [`tools/workflow_plugins/validate_workflow_plugin_config.py`](tools/workflow_plugins/validate_workflow_plugin_config.py)
- Dispatcher / interfaces:
  [`tools/workflow_plugins/runtime.py`](tools/workflow_plugins/runtime.py),
  [`tools/workflow_plugins/interfaces.py`](tools/workflow_plugins/interfaces.py)
- Task / acceptance sync:
  [`tools/ci/check_task_acceptance_sync.py`](tools/ci/check_task_acceptance_sync.py),
  [`tools/ci/generate_acceptance_index.py`](tools/ci/generate_acceptance_index.py)
- Docs resolve / ack / stale:
  [`tools/context/workflow_docs.py`](tools/context/workflow_docs.py)

## Reusable CI

- Python CI:
  [`.github/workflows/reusable/python-ci.yml`](.github/workflows/reusable/python-ci.yml)
- Security CI:
  [`.github/workflows/reusable/security-ci.yml`](.github/workflows/reusable/security-ci.yml)
- Security posture:
  [`.github/workflows/security.yml`](.github/workflows/security.yml),
  [`tools/ci/check_security_posture.py`](tools/ci/check_security_posture.py)
- Release evidence:
  [`.github/workflows/release-evidence.yml`](.github/workflows/release-evidence.yml),
  [`tools/ci/check_release_evidence.py`](tools/ci/check_release_evidence.py)
- Cross-repo integration:
  [`.github/workflows/cross-repo-integration.yml`](.github/workflows/cross-repo-integration.yml)
- Governance gate:
  [`.github/workflows/governance-gate.yml`](.github/workflows/governance-gate.yml)

See [`docs/ci-config.md`](docs/ci-config.md) for downstream usage and required job
semantics. Use [`tools/ci/check_ci_gate_matrix.py`](tools/ci/check_ci_gate_matrix.py)
to validate gate alignment in this repo.

## Supporting Docs

- Requirements / spec / design:
  [`docs/requirements.md`](docs/requirements.md),
  [`docs/spec.md`](docs/spec.md),
  [`docs/design.md`](docs/design.md)
- Operational addenda:
  [`docs/ROADMAP_AND_SPECS.md`](docs/ROADMAP_AND_SPECS.md),
  [`docs/addenda/J_Test_Engineering.md`](docs/addenda/J_Test_Engineering.md),
  [`docs/addenda/N_Improvement_Backlog.md`](docs/addenda/N_Improvement_Backlog.md),
  [`docs/addenda/M_Versioning_Release.md`](docs/addenda/M_Versioning_Release.md)
- Security:
  [`docs/security/SAC.md`](docs/security/SAC.md),
  [`docs/security/Security_Review_Checklist.md`](docs/security/Security_Review_Checklist.md)

## License

MIT. Unless noted otherwise, files copied from this repo into other projects
remain under the MIT License.
