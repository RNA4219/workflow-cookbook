# Navigation Hub

Generated: 2026-04-10

Quick entry points for common operations.

## Task Operations

Task Seed creation, tracking, and completion

| Entry | Path | Description |
|-------|------|-------------|
| [Task Seed Template](TASK.codex.md) | `TASK.codex.md` | Template for creating new task seeds |
| [Tasks Directory](docs/tasks/) | `docs/tasks/` | Active and completed tasks |
| [Tasks Guide](docs/TASKS.md) | `docs/TASKS.md` | Task operations guide |

## Acceptance Operations

Verification and acceptance workflow

| Entry | Path | Description |
|-------|------|-------------|
| [Acceptance Directory](docs/acceptance/) | `docs/acceptance/` | Acceptance records |
| [Acceptance Index](docs/acceptance/INDEX.md) | `docs/acceptance/INDEX.md` | Summary of all acceptances |
| [Acceptance Template](docs/acceptance/ACCEPTANCE_TEMPLATE.md) | `docs/acceptance/ACCEPTANCE_TEMPLATE.md` | Template for acceptance records |
| [Acceptance README](docs/acceptance/README.md) | `docs/acceptance/README.md` | Acceptance workflow guide |

## Security Operations

Security review and compliance

| Entry | Path | Description |
|-------|------|-------------|
| [Security Review Checklist](docs/security/Security_Review_Checklist.md) | `docs/security/Security_Review_Checklist.md` | Security review items |
| [SAC Guidelines](docs/security/SAC.md) | `docs/security/SAC.md` | Security architecture guidelines |
| [Security Privacy Guide](docs/addenda/G_Security_Privacy.md) | `docs/addenda/G_Security_Privacy.md` | Security/privacy operations |

## Plugin Operations

Cross-repo plugin integration

| Entry | Path | Description |
|-------|------|-------------|
| [Plugin README](tools/workflow_plugins/README.md) | `tools/workflow_plugins/README.md` | Plugin host documentation |
| [Plugin Interfaces](tools/workflow_plugins/interfaces.py) | `tools/workflow_plugins/interfaces.py` | Plugin capability interfaces |
| [Sample Config](examples/workflow_plugins.cross_repo.sample.json) | `examples/workflow_plugins.cross_repo.sample.json` | Cross-repo plugin sample |
| [Config Schema](schemas/workflow-plugin-config.schema.json) | `schemas/workflow-plugin-config.schema.json` | Plugin config schema |

## CI Operations

Continuous integration and governance

| Entry | Path | Description |
|-------|------|-------------|
| [CI Config](docs/ci-config.md) | `docs/ci-config.md` | CI gate configuration |
| [Phased Rollout](docs/ci_phased_rollout_requirements.md) | `docs/ci_phased_rollout_requirements.md` | Rollout requirements |
| [Governance Policy](governance/policy.yaml) | `governance/policy.yaml` | Policy definitions |
| [Checklist](CHECKLISTS.md) | `CHECKLISTS.md` | Release checklist |

## Birdseye Operations

Document dependency mapping

| Entry | Path | Description |
|-------|------|-------------|
| [Birdseye README](docs/BIRDSEYE.md) | `docs/BIRDSEYE.md` | Birdseye documentation |
| [Index JSON](docs/birdseye/index.json) | `docs/birdseye/index.json` | Node index |
| [Codemap README](tools/codemap/README.md) | `tools/codemap/README.md` | Codemap tool documentation |

## Evidence Operations

LLM behavior tracking

| Entry | Path | Description |
|-------|------|-------------|
| [Evidence Bridge](tools/protocols/README.md) | `tools/protocols/README.md` | Evidence protocol documentation |
| [Sample Config](examples/inference_plugins.agent_protocol.sample.json) | `examples/inference_plugins.agent_protocol.sample.json` | Evidence plugin sample |
| [Agent Protocols Reference](skills/workflow-agent-evidence/references/agent-protocols.md) | `skills/workflow-agent-evidence/references/agent-protocols.md` | Evidence fields reference |

## Reference Documentation

Core documentation

| Entry | Path | Description |
|-------|------|-------------|
| [Blueprint](BLUEPRINT.md) | `BLUEPRINT.md` | Requirements and constraints |
| [Runbook](RUNBOOK.md) | `RUNBOOK.md` | Execution procedures |
| [Evaluation](EVALUATION.md) | `EVALUATION.md` | Acceptance criteria |
| [Guardrails](GUARDRAILS.md) | `GUARDRAILS.md` | Operational boundaries |
| [HUB](HUB.codex.md) | `HUB.codex.md` | Agent navigation hub |

## Quick Commands

```sh
# Birdseye update
python tools/codemap/update.py --since --emit index+caps

# Run tests
uv run pytest tests/ -q

# Check CI gates
python tools/ci/check_ci_gate_matrix.py

# Generate acceptance index
python tools/ci/generate_acceptance_index_standalone.py
```
