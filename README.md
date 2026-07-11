# Workflow Cookbook

[![Version](https://img.shields.io/badge/version-1.2.0-blue)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/RNA4219/workflow-cookbook/actions/workflows/test.yml/badge.svg)](https://github.com/RNA4219/workflow-cookbook/actions/workflows/test.yml)

<!-- SLO-BADGES -->
[![SLO: lead_time](https://img.shields.io/badge/Lead%20Time%20P95-1d-brightgreen)](https://img.shields.io/badge/Lead%20Time%20P95-1d-brightgreen)
[![SLO: mttr](https://img.shields.io/badge/MTTR%20P95-30m-brightgreen)](https://img.shields.io/badge/MTTR%20P95-30m-brightgreen)
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
python -m tools.codemap.update --since --emit index+caps

# 2. Run tests
uv run pytest tests/ -q

# 3. Validate CI gates
python tools/ci/check_ci_gate_matrix.py

# 4. Check Birdseye freshness
python tools/ci/check_birdseye_freshness.py --check
```

> **Windows users**: The `python` command may invoke the Windows Store stub.
> Use `py -3` or `uv run python` instead of `python` in the examples above.

### Install the public CLI package

The supported distribution path is a normal, non-editable install. Pin a commit for
reproducible automation:

```sh
python -m pip install "workflow-cookbook @ git+https://github.com/RNA4219/workflow-cookbook.git@<commit-sha>"
```

For a local checkout, build and install the wheel:

```sh
uv build
python -m pip install dist/workflow_cookbook-*.whl
```

The package exposes `wfc-governance-gate`, `wfc-collect-metrics`,
`wfc-codemap-update`, `wfc-context-pack`, and
`wfc-five-tool-manifest`. All entrypoints support `--help`; repository-aware
commands accept `--repo-root PATH`. Metrics configuration can be selected with
`wfc-collect-metrics --metrics-config PATH`, which takes precedence over
`WFC_METRICS_CONFIG` and the default `governance/metrics.yaml`.

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
python -m tools.codemap.update --targets docs/birdseye/index.json,docs/birdseye/hot.json --emit index+caps

# Local update (radius 1)
python -m tools.codemap.update --since --radius 1 --emit caps

# Freshness check
python tools/ci/check_birdseye_freshness.py --check --max-verified-age-days 90

# Freshness remediation suggestions
python tools/ci/check_birdseye_freshness.py --check --remediation-output .ga/birdseye-remediation.json

# CI phase rollout doctor
python tools/ci/check_ci_phase_doctor.py --json
```

### Metrics / KPI

```sh
# QA metrics collection
python -m tools.perf.collect_metrics --suite qa --metrics-url <url> --log-path <path>

# Threshold validation
python tools/ci/check_metrics_thresholds.py --check --metrics-json .ga/qa-metrics.json

# Regression check against previous-good metrics
python tools/ci/check_metrics_thresholds.py --check --metrics-json .ga/qa-metrics.json --baseline-json .ga/qa-metrics.previous.json
```

### Acceptance / Task

```sh
# Acceptance record validation
python tools/ci/check_acceptance.py --check

# Task/Acceptance sync check
python tools/ci/check_task_acceptance_sync.py --plugin-config examples/workflow_plugins.cross_repo.sample.json

# Strict mode: fail when done tasks do not have acceptance records
python tools/ci/check_task_acceptance_sync.py --plugin-config examples/workflow_plugins.cross_repo.sample.json --require-acceptance-for-done

# Generate acceptance index
python tools/ci/generate_acceptance_index.py --plugin-config examples/workflow_plugins.cross_repo.sample.json
```

### Security / Release

```sh
# Security posture check
python tools/ci/check_security_posture.py --check --github-repo owner/name

# Security posture snapshot and diff
python tools/ci/check_security_posture.py --export-json .ga/security-posture.json
python tools/ci/check_security_posture.py --baseline-json .ga/security-posture.previous.json --json

# Release evidence check
python tools/ci/check_release_evidence.py --check --github-repo owner/name

# Branch protection validation
python tools/ci/check_branch_protection.py --protection-json <json>

# Branch protection weekly audit artifacts
python tools/ci/check_branch_protection.py --protection-json <json> --report-output .ga/branch-protection-weekly.json --nudge-output .ga/branch-protection-nudge.json

# Security docs freshness check
python tools/ci/check_security_docs_freshness.py --check

# Sample/docs sync check
python tools/ci/check_sample_docs_sync.py --check

# Schema/sample/docs matrix check
python tools/ci/check_sample_docs_sync.py --matrix-check --json

# Docs review automation
python tools/ci/check_docs_review_due.py --owner-summary-json .ga/docs-review-owner-summary.json --nudge-output .ga/docs-review-nudges.json
```

### Evidence / Reporting

```sh
# Generate evidence report
python tools/ci/generate_evidence_report.py --output docs/evidence_report.md

# Generate release readiness report
python tools/ci/generate_evidence_report.py --security-json .ga/security-posture.json --metrics-json .ga/qa-metrics.json --output docs/release_readiness.md

# Generate acceptance index
python tools/ci/generate_acceptance_index_standalone.py --output docs/acceptance_index.md

# Extract upstream changes
python tools/ci/extract_upstream_changes.py --upstream-md docs/UPSTREAM.md --weekly-log docs/WEEKLY.md

# Export task state
python tools/ci/export_task_state.py --output task_state.json

# Assess downstream adoption tier
python tools/ci/check_adoption_tier.py --repo ../agent-taskstate --json

# Assess downstream onboarding readiness
python tools/ci/check_downstream_onboarding.py --repo ../agent-taskstate --json

# Adaptive improvement utilities
python tools/ci/self_improvement_ops.py export-memory --output .ga/curated-memory.json
python tools/ci/self_improvement_ops.py build-recall --query "release readiness" --output .ga/recall-response.json

# Five-tool validation manifest
python tools/ci/five_tool_manifest.py generate --config examples/five-tool-chain-manifest.sample.json --out docs/evidence/five-tool-validation-20260703/five-tool-run-manifest.json --validate
python tools/ci/five_tool_manifest.py validate --manifest docs/evidence/five-tool-validation-20260703/five-tool-run-manifest.json --json
```

---

## CI Workflows

| Workflow | Description |
|----------|-------------|
| [`.github/workflows/test.yml`](.github/workflows/test.yml) | Tests + coverage |
| [`.github/workflows/governance-gate.yml`](.github/workflows/governance-gate.yml) | Policy validation |
| [`.github/workflows/security.yml`](.github/workflows/security.yml) | Security checks (Bandit, Semgrep, Gitleaks, Dependency Audit) |
| [`.github/workflows/release-evidence.yml`](.github/workflows/release-evidence.yml) | Release evidence |
| [`.github/workflows/cross-repo-integration.yml`](.github/workflows/cross-repo-integration.yml) | Cross-repo integration |
| [`.github/workflows/docs-resolve-pr-gate.yml`](.github/workflows/docs-resolve-pr-gate.yml) | Docs resolve validation |

Reusable workflows:

- [`.github/workflows/reusable/python-ci.yml`](.github/workflows/reusable/python-ci.yml)

---

## Plugin Integration

### Connected Repositories

| Repository | Role | Integration Type |
|------------|------|------------------|
| [`agent-protocols`](https://github.com/RNA4219/agent-protocols) | Contract schemas (Evidence, TaskSeed, Acceptance, etc.) | Schema reference |
| [`agent-taskstate`](https://github.com/RNA4219/agent-taskstate) | Task state management, typed_ref, context bundle | Workflow plugin |
| [`memx-resolver`](https://github.com/RNA4219/memx-resolver) | Docs resolve, ack, stale check | Workflow plugin |
| [`RanD`](https://github.com/RNA4219/RanD) | Requirements audit packet | Five-tool validation manifest |
| [`code-to-gate`](https://github.com/RNA4219/code-to-gate) | Static gate signals | Five-tool validation manifest |
| [`harness-auto-test-evidence`](https://github.com/RNA4219/harness-auto-test-evidence) | Automated test evidence | Five-tool validation manifest |
| [`manual-bb-test-harness`](https://github.com/RNA4219/manual-bb-test-harness) | Manual black-box Go/No-Go brief | Five-tool validation manifest |
| [`quality-evidence-graph`](https://github.com/RNA4219/quality-evidence-graph) | QEG policyHash and final verdict | Five-tool validation manifest |

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
| Adoption | [`docs/adoption-tiers.md`](docs/adoption-tiers.md), [`docs/adoption-guide.md`](docs/adoption-guide.md), [`templates/`](templates/) |
| Schemas | [`schemas/`](schemas/) including `tool-request`, plugin config, and self-improvement DTO contracts |

---

## License

MIT. Unless noted otherwise, files copied from this repo into other projects
remain under the MIT License.
