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
<!-- /SLO-BADGES -->

语言：[English](README.md) | [日本語](README.ja.md) | 简体中文

面向工作流运维与 context engineering 的文档与运行时工具包。
整合 Birdseye/Codemap、Task Seed、验收运维、可复用 CI、Evidence 追踪。

---

## 概览

| 功能 | 说明 |
|------|------|
| **Birdseye / Codemap** | 自动同步 Markdown hub 与依赖关系 |
| **Task Seed** | 任务定义模板与运维流程 |
| **Acceptance** | 验收记录与质量门禁 |
| **CI / Governance** | 可复用 workflow 与 policy 校验 |
| **Evidence** | LLM 行为追踪与 `agent-protocols` 集成 |
| **Plugins** | 跨仓库集成与 docs resolve |

<!-- LLM-BOOTSTRAP v1 -->
建议阅读顺序：

1. `docs/birdseye/index.json` —— 节点一览（轻量）
2. `docs/birdseye/caps/<path>.json` —— 按需局部读取

聚焦步骤：

- 从 index.json 找出最近变更文件在 +/-2 hop 范围内的节点 ID
- 只读取对应的 caps/*.json 文件

<!-- /LLM-BOOTSTRAP -->

---

## 快速开始

```sh
# 1. 更新 Birdseye
python tools/codemap/update.py --since --emit index+caps

# 2. 运行测试
uv run pytest tests/ -q

# 3. 校验 CI gates
python tools/ci/check_ci_gate_matrix.py

# 4. 检查 Birdseye 新鲜度
python tools/ci/check_birdseye_freshness.py --check
```

---

## 文档导航

### 首先阅读

| 文件 | 说明 |
|------|------|
| [`BLUEPRINT.md`](BLUEPRINT.md) | 需求、约束、背景 |
| [`RUNBOOK.md`](RUNBOOK.md) | 执行步骤、命令 |
| [`EVALUATION.md`](EVALUATION.md) | 验收标准、质量指标 |
| [`HUB.codex.md`](HUB.codex.md) | Agent 导向 hub |

### CI / Governance

| 文件 | 说明 |
|------|------|
| [`CHECKLISTS.md`](CHECKLISTS.md) | 发布确认项 |
| [`docs/ci-config.md`](docs/ci-config.md) | CI gate 与 job mapping |
| [`governance/policy.yaml`](governance/policy.yaml) | 自改边界、SLO |

### 运维补充

| 文件 | 说明 |
|------|------|
| [`docs/acceptance/README.md`](docs/acceptance/README.md) | 验收记录运维 |
| [`docs/addenda/J_Test_Engineering.md`](docs/addenda/J_Test_Engineering.md) | 测试质量基线 |
| [`docs/addenda/O_Adaptive_Improvement_Loop.md`](docs/addenda/O_Adaptive_Improvement_Loop.md) | 自适应改进闭环 |

---

## Skills

| Skill | 说明 |
|-------|------|
| [`skills/workflow-agent-evidence/SKILL.md`](skills/workflow-agent-evidence/SKILL.md) | Evidence 集成 |
| [`skills/workflow-agent-evidence/agents/claude.yaml`](skills/workflow-agent-evidence/agents/claude.yaml) | Claude metadata |
| [`skills/workflow-agent-evidence/agents/openai.yaml`](skills/workflow-agent-evidence/agents/openai.yaml) | OpenAI metadata |

---

## 主要命令

### Birdseye / Codemap

```sh
# 全量更新
python tools/codemap/update.py --targets docs/birdseye/index.json,docs/birdseye/hot.json --emit index+caps

# 局部更新（radius 1）
python tools/codemap/update.py --since --radius 1 --emit caps

# 新鲜度检查
python tools/ci/check_birdseye_freshness.py --check --max-verified-age-days 90
```

### Metrics / KPI

```sh
# QA 指标采集
python -m tools.perf.collect_metrics --suite qa --metrics-url <url> --log-path <path>

# 阈值判定
python tools/ci/check_metrics_thresholds.py --check --metrics-json .ga/qa-metrics.json
```

### Acceptance / Task

```sh
# 验收记录校验
python tools/ci/check_acceptance.py --check

# Task/Acceptance 同步检查
python tools/ci/check_task_acceptance_sync.py --plugin-config examples/workflow_plugins.cross_repo.sample.json

# 生成验收 index
python tools/ci/generate_acceptance_index.py --plugin-config examples/workflow_plugins.cross_repo.sample.json
```

### Security / Release

```sh
# 安全 posture 检查
python tools/ci/check_security_posture.py --check --github-repo owner/name

# 发布证迹检查
python tools/ci/check_release_evidence.py --check --github-repo owner/name

# Branch protection 校验
python tools/ci/check_branch_protection.py --protection-json <json>

# 安全 docs 新鲜度检查
python tools/ci/check_security_docs_freshness.py --check

# sample/docs 同步检查
python tools/ci/check_sample_docs_sync.py --check
```

### Evidence / Report

```sh
# Evidence 报告生成
python tools/ci/generate_evidence_report.py --output docs/evidence_report.md

# 验收 index 生成
python tools/ci/generate_acceptance_index_standalone.py --output docs/acceptance_index.md

# Upstream 差分提取
python tools/ci/extract_upstream_changes.py --upstream-md docs/UPSTREAM.md --weekly-log docs/WEEKLY.md

# Task state 导出
python tools/ci/export_task_state.py --output task_state.json
```

---

## CI Workflows

| Workflow | 说明 |
|----------|------|
| [`.github/workflows/test.yml`](.github/workflows/test.yml) | 测试 + coverage |
| [`.github/workflows/governance-gate.yml`](.github/workflows/governance-gate.yml) | policy 校验 |
| [`.github/workflows/security.yml`](.github/workflows/security.yml) | 安全检查 (Bandit, Semgrep, Gitleaks, Dependency Audit) |
| [`.github/workflows/release-evidence.yml`](.github/workflows/release-evidence.yml) | 发布证迹 |
| [`.github/workflows/cross-repo-integration.yml`](.github/workflows/cross-repo-integration.yml) | 跨仓库集成 |
| [`.github/workflows/docs-resolve-pr-gate.yml`](.github/workflows/docs-resolve-pr-gate.yml) | docs resolve 校验 |

可复用 workflow：

- [`.github/workflows/reusable/python-ci.yml`](.github/workflows/reusable/python-ci.yml)

---

## Plugin 集成

### 关联仓库

| 仓库 | 角色 | 集成方式 |
|------|------|----------|
| [`agent-protocols`](https://github.com/RNA4219/agent-protocols) | 契约模式 (Evidence, TaskSeed, Acceptance 等) | 模式引用 |
| [`agent-taskstate`](https://github.com/RNA4219/agent-taskstate) | 任务状态管理、typed_ref、context bundle | Workflow plugin |
| [`memx-resolver`](https://github.com/RNA4219/memx-resolver) | docs resolve、ack、stale check | Workflow plugin |

### Evidence Plugin

```sh
# 配置示例
examples/inference_plugins.agent_protocol.sample.json

# 消费示例
examples/agent_protocol_evidence_consumer.sample.py

# 详情
tools/protocols/README.md
```

### Cross-Repo Plugin

```sh
# 配置示例
examples/workflow_plugins.cross_repo.sample.json

# Schema
schemas/workflow-plugin-config.schema.json

# 校验
python tools/workflow_plugins/validate_workflow_plugin_config.py --config examples/workflow_plugins.cross_repo.sample.json
```

---

## 支撑文档

| 分类 | 文件 |
|------|------|
| 规格 | [`docs/requirements.md`](docs/requirements.md), [`docs/spec.md`](docs/spec.md), [`docs/design.md`](docs/design.md) |
| 运维 | [`docs/ROADMAP_AND_SPECS.md`](docs/ROADMAP_AND_SPECS.md), [`CHANGELOG.md`](CHANGELOG.md) |
| 安全 | [`docs/security/SAC.md`](docs/security/SAC.md), [`docs/security/Security_Review_Checklist.md`](docs/security/Security_Review_Checklist.md) |
| 扩展 | [`docs/addenda/N_Improvement_Backlog.md`](docs/addenda/N_Improvement_Backlog.md), [`docs/addenda/P_Expansion_Candidates.md`](docs/addenda/P_Expansion_Candidates.md) |

---

## License

MIT。除非另有说明，从本仓库复制到其他项目中的文件仍遵循 MIT License。
