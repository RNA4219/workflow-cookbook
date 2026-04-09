# Workflow Cookbook / Workflow Operations Kit

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

语言：[English](README.md) | [日本語](README.ja.md) | 简体中文

Workflow Cookbook 是一个面向工作流运维与 context engineering 的文档与运行时工具包。
它整合了 Birdseye / Codemap、Task Seed、验收运维、可复用 CI，
基于插件的 Evidence 追踪能力，以及面向下游软件的、
与具体 repo 无关的自我改进闭环 blueprint。

<!-- LLM-BOOTSTRAP v1 -->
建议阅读顺序：

1. `docs/birdseye/index.json`，用于查看轻量级节点图
2. `docs/birdseye/caps/<path>.json`，用于按需进行局部读取

聚焦步骤：

- 从 `index.json` 找出最近变更文件在 +/-2 hop 范围内的节点 ID
- 只读取对应的 `caps/*.json` 文件

<!-- /LLM-BOOTSTRAP -->

## 包含内容

- 使用 Birdseye / Codemap 同步 Markdown hub 与依赖关系
- 以 `BLUEPRINT`、`RUNBOOK`、`EVALUATION`、`CHECKLISTS` 为核心的运维文档链路
- 基于 `StructuredLogger` 插件的 LLM 行为追踪
- 面向 `agent-protocols` `Evidence` 契约的 sample config 与 consumer sample
- 可将 `agent-taskstate` 与 `memx-resolver` 作为可选插件接入的 workflow host
- 可在发布后按需启用的、面向 reflection、recall、skill evolution、
  user/workspace model 的自我改进闭环补充设计
- 可复用的 CI / governance workflow 与校验脚本

## 快速开始

1. 先阅读核心文档：
   [`BLUEPRINT.md`](BLUEPRINT.md),
   [`RUNBOOK.md`](RUNBOOK.md),
   [`EVALUATION.md`](EVALUATION.md)
2. 刷新 Birdseye：

   ```sh
   python tools/codemap/update.py --since --emit index+caps
   ```

3. 记录验收结果：
   [`docs/acceptance/README.md`](docs/acceptance/README.md),
   [`docs/acceptance/ACCEPTANCE_TEMPLATE.md`](docs/acceptance/ACCEPTANCE_TEMPLATE.md)
4. 查看测试质量基线：
   [`docs/addenda/J_Test_Engineering.md`](docs/addenda/J_Test_Engineering.md)
5. 对已收集 KPI 执行阈值判定：
   [`tools/ci/check_metrics_thresholds.py`](tools/ci/check_metrics_thresholds.py),
   [`governance/metrics_thresholds.yaml`](governance/metrics_thresholds.yaml)
6. 检查 Birdseye 新鲜度：
   [`tools/ci/check_birdseye_freshness.py`](tools/ci/check_birdseye_freshness.py)
7. 试用 Evidence 追踪：
   [`tools/protocols/README.md`](tools/protocols/README.md),
   [`examples/inference_plugins.agent_protocol.sample.json`](examples/inference_plugins.agent_protocol.sample.json)
8. 试用跨仓库插件：
   [`tools/workflow_plugins/README.md`](tools/workflow_plugins/README.md),
   [`examples/workflow_plugins.cross_repo.sample.json`](examples/workflow_plugins.cross_repo.sample.json)
9. 校验插件配置：
   [`tools/workflow_plugins/validate_workflow_plugin_config.py`](tools/workflow_plugins/validate_workflow_plugin_config.py)

## 导航

- 首先阅读：
  [`BLUEPRINT.md`](BLUEPRINT.md),
  [`RUNBOOK.md`](RUNBOOK.md),
  [`EVALUATION.md`](EVALUATION.md)
- Birdseye / Codemap：
  [`docs/BIRDSEYE.md`](docs/BIRDSEYE.md),
  [`tools/codemap/README.md`](tools/codemap/README.md),
  [`HUB.codex.md`](HUB.codex.md)
- CI / Governance：
  [`CHECKLISTS.md`](CHECKLISTS.md),
  [`docs/ci-config.md`](docs/ci-config.md),
  [`docs/ci_phased_rollout_requirements.md`](docs/ci_phased_rollout_requirements.md)
- 质量基线：
  [`docs/addenda/J_Test_Engineering.md`](docs/addenda/J_Test_Engineering.md),
  [`docs/acceptance/README.md`](docs/acceptance/README.md),
  [`governance/metrics_thresholds.yaml`](governance/metrics_thresholds.yaml)

## Skills

- Evidence 集成 Skill：
  [`skills/workflow-agent-evidence/SKILL.md`](skills/workflow-agent-evidence/SKILL.md)
- Agent metadata：
  [`skills/workflow-agent-evidence/agents/openai.yaml`](skills/workflow-agent-evidence/agents/openai.yaml),
  [`skills/workflow-agent-evidence/agents/claude.yaml`](skills/workflow-agent-evidence/agents/claude.yaml)
- Skill 参考资料：
  [`skills/workflow-agent-evidence/references/workflow-cookbook.md`](skills/workflow-agent-evidence/references/workflow-cookbook.md),
  [`skills/workflow-agent-evidence/references/agent-protocols.md`](skills/workflow-agent-evidence/references/agent-protocols.md)
- Protocol 插件指南：
  [`tools/protocols/README.md`](tools/protocols/README.md)

## 常用入口

### Birdseye / Codemap

```sh
python tools/codemap/update.py --since --emit index+caps
python tools/codemap/update.py --since --radius 1 --emit caps
python tools/codemap/update.py --targets docs/birdseye/index.json,docs/birdseye/hot.json --emit index+caps
python tools/ci/check_birdseye_freshness.py --check
```

### Metrics / KPI 判定

```sh
python -m tools.perf.collect_metrics --suite qa --metrics-url <url> --log-path <path>
python tools/ci/check_metrics_thresholds.py --check --metrics-json .ga/qa-metrics.json
```

### LLM Evidence 追踪

- 插件 API：
  [`tools/protocols/README.md`](tools/protocols/README.md)
- sample config：
  [`examples/inference_plugins.agent_protocol.sample.json`](examples/inference_plugins.agent_protocol.sample.json)
- consumer sample：
  [`examples/agent_protocol_evidence_consumer.sample.py`](examples/agent_protocol_evidence_consumer.sample.py)

### 任务运维

- Task Seed sample：
  [`examples/TASK.sample.md`](examples/TASK.sample.md)
- 执行前提：
  [`GUARDRAILS.md`](GUARDRAILS.md),
  [`RUNBOOK.md`](RUNBOOK.md)
- 发布侧文档：
  [`CHECKLISTS.md`](CHECKLISTS.md),
  [`CHANGELOG.md`](CHANGELOG.md),
  [`docs/acceptance/README.md`](docs/acceptance/README.md)

### Advanced: Cross-Repo Plugins

- Host / config：
  [`tools/workflow_plugins/README.md`](tools/workflow_plugins/README.md),
  [`examples/workflow_plugins.cross_repo.sample.json`](examples/workflow_plugins.cross_repo.sample.json),
  [`schemas/workflow-plugin-config.schema.json`](schemas/workflow-plugin-config.schema.json)
- 配置校验：
  [`tools/workflow_plugins/validate_workflow_plugin_config.py`](tools/workflow_plugins/validate_workflow_plugin_config.py)
- Dispatcher / interfaces：
  [`tools/workflow_plugins/runtime.py`](tools/workflow_plugins/runtime.py),
  [`tools/workflow_plugins/interfaces.py`](tools/workflow_plugins/interfaces.py)
- task / acceptance sync：
  [`tools/ci/check_task_acceptance_sync.py`](tools/ci/check_task_acceptance_sync.py),
  [`tools/ci/generate_acceptance_index.py`](tools/ci/generate_acceptance_index.py)
- docs resolve / ack / stale：
  [`tools/context/workflow_docs.py`](tools/context/workflow_docs.py)

## 可复用 CI

- Python CI：
  [`.github/workflows/reusable/python-ci.yml`](.github/workflows/reusable/python-ci.yml)
- Security CI：
  [`.github/workflows/reusable/security-ci.yml`](.github/workflows/reusable/security-ci.yml)
- Security posture：
  [`.github/workflows/security.yml`](.github/workflows/security.yml),
  [`tools/ci/check_security_posture.py`](tools/ci/check_security_posture.py)
- Release evidence：
  [`.github/workflows/release-evidence.yml`](.github/workflows/release-evidence.yml),
  [`tools/ci/check_release_evidence.py`](tools/ci/check_release_evidence.py)
- Cross-repo integration：
  [`.github/workflows/cross-repo-integration.yml`](.github/workflows/cross-repo-integration.yml)
- Governance gate：
  [`.github/workflows/governance-gate.yml`](.github/workflows/governance-gate.yml)

关于下游仓库如何调用以及 required jobs 的语义，请参见
[`docs/ci-config.md`](docs/ci-config.md)。本仓库中的 gate 对齐可通过
[`tools/ci/check_ci_gate_matrix.py`](tools/ci/check_ci_gate_matrix.py) 进行校验。

## 支撑文档

- Requirements / spec / design：
  [`docs/requirements.md`](docs/requirements.md),
  [`docs/spec.md`](docs/spec.md),
  [`docs/design.md`](docs/design.md)
- 运维补充：
  [`docs/ROADMAP_AND_SPECS.md`](docs/ROADMAP_AND_SPECS.md),
  [`docs/addenda/J_Test_Engineering.md`](docs/addenda/J_Test_Engineering.md),
  [`docs/addenda/O_Adaptive_Improvement_Loop.md`](docs/addenda/O_Adaptive_Improvement_Loop.md),
  [`docs/addenda/N_Improvement_Backlog.md`](docs/addenda/N_Improvement_Backlog.md),
  [`docs/addenda/P_Expansion_Candidates.md`](docs/addenda/P_Expansion_Candidates.md),
  [`docs/addenda/M_Versioning_Release.md`](docs/addenda/M_Versioning_Release.md)
- 安全文档：
  [`docs/security/SAC.md`](docs/security/SAC.md),
  [`docs/security/Security_Review_Checklist.md`](docs/security/Security_Review_Checklist.md)

## License

MIT。除非另有说明，从本仓库复制到其他项目中的文件仍遵循 MIT License。
