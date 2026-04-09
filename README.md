---
intent_id: DOC-README
owner: docs-core
status: active
last_reviewed_at: 2026-04-10
next_review_due: 2026-05-10
---

# Workflow Cookbook

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/RNA4219/workflow-cookbook/actions/workflows/test.yml/badge.svg)](https://github.com/RNA4219/workflow-cookbook/actions/workflows/test.yml)

<!-- SLO-BADGES -->
[![SLO: lead_time](https://img.shields.io/badge/Lead%20Time%20P95-1d-brightgreen)](https://img.shields.io/badge/Lead%20Time%20P95-1d-brightgreen)
[![SLO: mttr](https://img.shields.io/badge/MTTR%20P95-30m-brightgreen)](https://img.shields.io/badge/MTTR%20P95-30m-brightgreen)
[![SLO: change_failure_rate](https://img.shields.io/badge/Change%20Failure%20Rate-20%-yellow)](https://img.shields.io/badge/Change%20Failure%20Rate-20%-yellow)
<!-- /SLO-BADGES -->

Language: English | [日本語](README.ja.md) | [简体中文](README.zh-CN.md)

A docs and runtime kit for workflow operations and context engineering.
Integrates Birdseye/Codemap, Task Seeds, acceptance operations, reusable CI, and Evidence tracking.

---

## Overview

| Feature | Description |
|---------|-------------|
| **Birdseye / Codemap** | Auto-sync Markdown hubs and dependencies |
| **Task Seed** | Task definition templates and operational workflows |
| **Acceptance** | Acceptance records and quality gates |
| **CI / Governance** | Reusable workflows and policy validation |
| **Evidence** | LLM behavior tracking and `agent-protocols` integration |
| **Plugins** | Cross-repo integration and docs resolve |

<!-- LLM-BOOTSTRAP v1 -->
Recommended read order:

1. `docs/birdseye/index.json` — Node graph (lightweight)
2. `docs/birdseye/caps/<path>.json` — Point reads for needed nodes

Focus procedure:

- Find node IDs for recently changed files within +/-2 hops from `index.json`
- Read only the matching `caps/*.json` files

<!-- /LLM-BOOTSTRAP -->

---

## Quick Start

```sh
# 1. Update Birdseye
python tools/codemap/update.py --since --emit index+caps

# 2. Run tests
uv run pytest tests/ -q

# 3. Validate CI gates
python tools/ci/check_ci_gate_matrix.py

# 4. Check Birdseye freshness
python tools/ci/check_birdseye_freshness.py --check
```

---

## Documentation Guide

### Start Here

| File | Description |
|------|-------------|
| [`BLUEPRINT.md`](BLUEPRINT.md) | Requirements, constraints, background |
| [`RUNBOOK.md`](RUNBOOK.md) | Execution procedures, commands |
| [`EVALUATION.md`](EVALUATION.md) | Acceptance criteria, quality metrics |
| [`HUB.codex.md`](HUB.codex.md) | Agent-oriented hub |

### CI / Governance

| File | Description |
|------|-------------|
| [`CHECKLISTS.md`](CHECKLISTS.md) | Release checklist items |
| [`docs/ci-config.md`](docs/ci-config.md) | CI gates and job mapping |
| [`governance/policy.yaml`](governance/policy.yaml) | Self-modification bounds, SLOs |

### Operations

| File | Description |
|------|-------------|
| [`docs/acceptance/README.md`](docs/acceptance/README.md) | Acceptance record workflow |
| [`docs/addenda/J_Test_Engineering.md`](docs/addenda/J_Test_Engineering.md) | Test quality baseline |
| [`docs/addenda/O_Adaptive_Improvement_Loop.md`](docs/addenda/O_Adaptive_Improvement_Loop.md) | Adaptive improvement loop |

---

## Skills

| Skill | Description |
|-------|-------------|
| [`skills/workflow-agent-evidence/SKILL.md`](skills/workflow-agent-evidence/SKILL.md) | Evidence integration |
| [`skills/workflow-agent-evidence/agents/claude.yaml`](skills/workflow-agent-evidence/agents/claude.yaml) | Claude metadata |
| [`skills/workflow-agent-evidence/agents/openai.yaml`](skills/workflow-agent-evidence/agents/openai.yaml) | OpenAI metadata |

---

## Key Commands

### Birdseye / Codemap

```sh
# Full update
python tools/codemap/update.py --targets docs/birdseye/index.json,docs/birdseye/hot.json --emit index+caps

# Local update (radius 1)
python tools/codemap/update.py --since --radius 1 --emit caps

# Freshness check
python tools/ci/check_birdseye_freshness.py --check --max-verified-age-days 90
```

### Metrics / KPI

```sh
# QA metrics collection
python -m tools.perf.collect_metrics --suite qa --metrics-url <url> --log-path <path>

# Threshold validation
python tools/ci/check_metrics_thresholds.py --check --metrics-json .ga/qa-metrics.json
```

### Acceptance / Task

```sh
# Acceptance record validation
python tools/ci/check_acceptance.py --check

# Task/Acceptance sync check
python tools/ci/check_task_acceptance_sync.py --plugin-config examples/workflow_plugins.cross_repo.sample.json

# Generate acceptance index
python tools/ci/generate_acceptance_index.py --plugin-config examples/workflow_plugins.cross_repo.sample.json
```

### Security / Release

```sh
# Security posture check
python tools/ci/check_security_posture.py --check --github-repo owner/name

# Release evidence check
python tools/ci/check_release_evidence.py --check --github-repo owner/name

# Branch protection validation
python tools/ci/check_branch_protection.py --protection-json <json>
```

---

## CI Workflows

| Workflow | Description |
|----------|-------------|
| [`.github/workflows/test.yml`](.github/workflows/test.yml) | Tests + coverage |
| [`.github/workflows/governance-gate.yml`](.github/workflows/governance-gate.yml) | Policy validation |
| [`.github/workflows/security.yml`](.github/workflows/security.yml) | Security checks |
| [`.github/workflows/release-evidence.yml`](.github/workflows/release-evidence.yml) | Release evidence |
| [`.github/workflows/cross-repo-integration.yml`](.github/workflows/cross-repo-integration.yml) | Cross-repo integration |

Reusable workflows:

- [`.github/workflows/reusable/python-ci.yml`](.github/workflows/reusable/python-ci.yml)
- [`.github/workflows/reusable/security-ci.yml`](.github/workflows/reusable/security-ci.yml)

---

## Plugin Integration

### Evidence Plugin

```sh
# Config sample
examples/inference_plugins.agent_protocol.sample.json

# Consumer sample
examples/agent_protocol_evidence_consumer.sample.py

# Details
tools/protocols/README.md
```

### Cross-Repo Plugin

```sh
# Config sample
examples/workflow_plugins.cross_repo.sample.json

# Schema
schemas/workflow-plugin-config.schema.json

# Validation
python tools/workflow_plugins/validate_workflow_plugin_config.py --config examples/workflow_plugins.cross_repo.sample.json
```

---

## Supporting Docs

| Category | Files |
|----------|-------|
| Spec | [`docs/requirements.md`](docs/requirements.md), [`docs/spec.md`](docs/spec.md), [`docs/design.md`](docs/design.md) |
| Ops | [`docs/ROADMAP_AND_SPECS.md`](docs/ROADMAP_AND_SPECS.md), [`CHANGELOG.md`](CHANGELOG.md) |
| Security | [`docs/security/SAC.md`](docs/security/SAC.md), [`docs/security/Security_Review_Checklist.md`](docs/security/Security_Review_Checklist.md) |
| Extension | [`docs/addenda/N_Improvement_Backlog.md`](docs/addenda/N_Improvement_Backlog.md), [`docs/addenda/P_Expansion_Candidates.md`](docs/addenda/P_Expansion_Candidates.md) |

---

## License

MIT. Unless noted otherwise, files copied from this repo into other projects
remain under the MIT License.
