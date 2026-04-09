# Workflow Cookbook / Workflow Operations Kit

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Language: [English](README.md) | 日本語 | [简体中文](README.zh-CN.md)

ワークフロー運用と context engineering のための docs / runtime kit です。Birdseye / Codemap、
Task Seed、acceptance 運用、reusable CI、plugin ベースの Evidence 追跡に加えて、
下流ソフトウェア向けの repo 非依存な自己改善ループ blueprint をまとめて扱います。

<!-- LLM-BOOTSTRAP v1 -->
読む順番:

1. docs/birdseye/index.json  …… ノード一覧・隣接関係（軽量）
2. docs/birdseye/caps/`<path>`.json …… 必要ノードだけ point read（個別カプセル）

フォーカス手順:

- 直近変更ファイル±2hopのノードIDを index.json から取得
- 対応する caps/*.json のみ読み込み

<!-- /LLM-BOOTSTRAP -->

## 何が入っているか

- Birdseye / Codemap による Markdown ハブと依存関係の同期
- `BLUEPRINT` / `RUNBOOK` / `EVALUATION` / `CHECKLISTS` を軸にした運用導線
- `StructuredLogger` plugin による LLM 行動追跡
- `agent-protocols` の `Evidence` 契約へ接続する sample config と consumer sample
- `agent-taskstate` / `memx-resolver` を optional plugin として接続する workflow host
- リリース後運用で任意に有効化できる、reflection / recall /
  skill evolution / user model 向け自己改善ループの補助設計
- reusable CI / Governance workflow と検証スクリプト

## クイックスタート

1. 全体像をつかむ:
   [`BLUEPRINT.md`](BLUEPRINT.md),
   [`RUNBOOK.md`](RUNBOOK.md),
   [`EVALUATION.md`](EVALUATION.md)
2. Birdseye を更新する:

   ```sh
   python tools/codemap/update.py --since --emit index+caps
   ```

3. 検収記録を残す:
   [`docs/acceptance/README.md`](docs/acceptance/README.md),
   [`docs/acceptance/ACCEPTANCE_TEMPLATE.md`](docs/acceptance/ACCEPTANCE_TEMPLATE.md)
4. テスト品質基準を確認する:
   [`docs/addenda/J_Test_Engineering.md`](docs/addenda/J_Test_Engineering.md)
5. 収集済み KPI を閾値判定する:
   [`tools/ci/check_metrics_thresholds.py`](tools/ci/check_metrics_thresholds.py),
   [`governance/metrics_thresholds.yaml`](governance/metrics_thresholds.yaml)
6. Birdseye の鮮度を確認する:
   [`tools/ci/check_birdseye_freshness.py`](tools/ci/check_birdseye_freshness.py)
7. LLM 行動追跡を試す:
   [`tools/protocols/README.md`](tools/protocols/README.md),
   [`examples/inference_plugins.agent_protocol.sample.json`](examples/inference_plugins.agent_protocol.sample.json)
8. cross-repo plugin を試す:
   [`tools/workflow_plugins/README.md`](tools/workflow_plugins/README.md),
   [`examples/workflow_plugins.cross_repo.sample.json`](examples/workflow_plugins.cross_repo.sample.json)
9. plugin config を検証する:
   [`tools/workflow_plugins/validate_workflow_plugin_config.py`](tools/workflow_plugins/validate_workflow_plugin_config.py)

## 導線

- まず読む:
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
- 品質基準:
  [`docs/addenda/J_Test_Engineering.md`](docs/addenda/J_Test_Engineering.md),
  [`docs/acceptance/README.md`](docs/acceptance/README.md),
  [`governance/metrics_thresholds.yaml`](governance/metrics_thresholds.yaml)

## Skills

- Evidence 連携 Skill:
  [`skills/workflow-agent-evidence/SKILL.md`](skills/workflow-agent-evidence/SKILL.md)
- Agent metadata:
  [`skills/workflow-agent-evidence/agents/openai.yaml`](skills/workflow-agent-evidence/agents/openai.yaml),
  [`skills/workflow-agent-evidence/agents/claude.yaml`](skills/workflow-agent-evidence/agents/claude.yaml)
- Skill の補助資料:
  [`skills/workflow-agent-evidence/references/workflow-cookbook.md`](skills/workflow-agent-evidence/references/workflow-cookbook.md),
  [`skills/workflow-agent-evidence/references/agent-protocols.md`](skills/workflow-agent-evidence/references/agent-protocols.md)
- Protocol plugin 導線:
  [`tools/protocols/README.md`](tools/protocols/README.md)

## よく使う入口

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

### LLM 行動追跡

- plugin API:
  [`tools/protocols/README.md`](tools/protocols/README.md)
- sample config:
  [`examples/inference_plugins.agent_protocol.sample.json`](examples/inference_plugins.agent_protocol.sample.json)
- consumer sample:
  [`examples/agent_protocol_evidence_consumer.sample.py`](examples/agent_protocol_evidence_consumer.sample.py)

### タスク運用

- task seed sample:
  [`examples/TASK.sample.md`](examples/TASK.sample.md)
- 実行前提:
  [`GUARDRAILS.md`](GUARDRAILS.md),
  [`RUNBOOK.md`](RUNBOOK.md)
- リリース反映:
  [`CHECKLISTS.md`](CHECKLISTS.md),
  [`CHANGELOG.md`](CHANGELOG.md),
  [`docs/acceptance/README.md`](docs/acceptance/README.md)

### Advanced: Cross-Repo Plugins

- host / config:
  [`tools/workflow_plugins/README.md`](tools/workflow_plugins/README.md),
  [`examples/workflow_plugins.cross_repo.sample.json`](examples/workflow_plugins.cross_repo.sample.json),
  [`schemas/workflow-plugin-config.schema.json`](schemas/workflow-plugin-config.schema.json)
- config validate:
  [`tools/workflow_plugins/validate_workflow_plugin_config.py`](tools/workflow_plugins/validate_workflow_plugin_config.py)
- dispatcher / interface:
  [`tools/workflow_plugins/runtime.py`](tools/workflow_plugins/runtime.py),
  [`tools/workflow_plugins/interfaces.py`](tools/workflow_plugins/interfaces.py)
- task / acceptance sync:
  [`tools/ci/check_task_acceptance_sync.py`](tools/ci/check_task_acceptance_sync.py),
  [`tools/ci/generate_acceptance_index.py`](tools/ci/generate_acceptance_index.py)
- docs resolve / ack / stale:
  [`tools/context/workflow_docs.py`](tools/context/workflow_docs.py)

## 再利用 CI

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

下流 repo からの呼び出し方法や required jobs の考え方は
[`docs/ci-config.md`](docs/ci-config.md) を参照してください。
本 repo の gate 整合は
[`tools/ci/check_ci_gate_matrix.py`](tools/ci/check_ci_gate_matrix.py) で検証できます。

## 補助ドキュメント

- 仕様・設計:
  [`docs/requirements.md`](docs/requirements.md),
  [`docs/spec.md`](docs/spec.md),
  [`docs/design.md`](docs/design.md)
- 運用補足:
  [`docs/ROADMAP_AND_SPECS.md`](docs/ROADMAP_AND_SPECS.md),
  [`docs/addenda/J_Test_Engineering.md`](docs/addenda/J_Test_Engineering.md),
  [`docs/addenda/O_Adaptive_Improvement_Loop.md`](docs/addenda/O_Adaptive_Improvement_Loop.md),
  [`docs/addenda/N_Improvement_Backlog.md`](docs/addenda/N_Improvement_Backlog.md),
  [`docs/addenda/P_Expansion_Candidates.md`](docs/addenda/P_Expansion_Candidates.md),
  [`docs/addenda/M_Versioning_Release.md`](docs/addenda/M_Versioning_Release.md)
- セキュリティ:
  [`docs/security/SAC.md`](docs/security/SAC.md),
  [`docs/security/Security_Review_Checklist.md`](docs/security/Security_Review_Checklist.md)

## License

MIT. Unless noted otherwise, files copied from this repo into other projects
remain under the MIT License.
