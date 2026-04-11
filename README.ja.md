---
intent_id: DOC-README
owner: docs-core
status: active
last_reviewed_at: 2026-04-11
next_review_due: 2026-05-11
---

# Workflow Cookbook

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/RNA4219/workflow-cookbook/actions/workflows/test.yml/badge.svg)](https://github.com/RNA4219/workflow-cookbook/actions/workflows/test.yml)

<!-- SLO-BADGES -->
[![SLO: lead_time](https://img.shields.io/badge/Lead%20Time%20P95-1d-brightgreen)](https://img.shields.io/badge/Lead%20Time%20P95-1d-brightgreen)
[![SLO: mttr](https://img.shields.io/badge/MTTR%20P95-30m-brightgreen)](https://img.shields.io/badge/MTTR%20P95-30m-brightgreen)
<!-- /SLO-BADGES -->

Language: [English](README.md) | 日本語 | [简体中文](README.zh-CN.md)

エージェント運用と context engineering のための docs / runtime kit。
Birdseye / Codemap、Task Seed、acceptance 運用、reusable CI、Evidence 追跡を統合。

---

## 概要

| 機能 | 内容 |
|------|------|
| **Birdseye / Codemap** | Markdown ハブと依存関係の自動同期 |
| **Task Seed** | タスク定義テンプレートと運用導線 |
| **Acceptance** | 検収記録と品質ゲート |
| **CI / Governance** | 再利用可能な workflow と policy 検証 |
| **Evidence** | LLM 行動追跡と `agent-protocols` 連携 |
| **Plugins** | cross-repo 連携と docs resolve |

<!-- LLM-BOOTSTRAP v1 -->
読む順番:

1. `docs/birdseye/index.json` …… ノード一覧（軽量）
2. `docs/birdseye/caps/<path>.json` …… 必要ノードだけ point read

フォーカス手順:

- 直近変更ファイル±2hop のノードIDを index.json から取得
- 対応する caps/*.json のみ読み込み

<!-- /LLM-BOOTSTRAP -->

---

## クイックスタート

```sh
# 1. Birdseye 更新
python tools/codemap/update.py --since --emit index+caps

# 2. テスト実行
uv run pytest tests/ -q

# 3. CI gate 検証
python tools/ci/check_ci_gate_matrix.py

# 4. Birdseye 鮮度確認
python tools/ci/check_birdseye_freshness.py --check
```

> **Windows ユーザー**: `python` コマンドが Windows Store stub を起動する場合があります。
> 上記例では `python` を `py -3` または `uv run python` に置き換えてください。

---

## ドキュメント導線

### 最初に読む

| ファイル | 内容 |
|----------|------|
| [`BLUEPRINT.md`](BLUEPRINT.md) | 要件・制約・背景 |
| [`RUNBOOK.md`](RUNBOOK.md) | 実行手順・コマンド |
| [`EVALUATION.md`](EVALUATION.md) | 受け入れ基準・品質指標 |
| [`HUB.codex.md`](HUB.codex.md) | エージェント向けハブ |

### CI / Governance

| ファイル | 内容 |
|----------|------|
| [`CHECKLISTS.md`](CHECKLISTS.md) | リリース確認項目 |
| [`docs/ci-config.md`](docs/ci-config.md) | CI gate と job mapping |
| [`governance/policy.yaml`](governance/policy.yaml) | 自己改変境界・SLO |

### 運用補足

| ファイル | 内容 |
|----------|------|
| [`docs/acceptance/README.md`](docs/acceptance/README.md) | 検収記録運用 |
| [`docs/addenda/J_Test_Engineering.md`](docs/addenda/J_Test_Engineering.md) | テスト品質基準 |
| [`docs/addenda/O_Adaptive_Improvement_Loop.md`](docs/addenda/O_Adaptive_Improvement_Loop.md) | 自己改善ループ |

---

## Skills

| Skill | 内容 |
|-------|------|
| [`skills/workflow-agent-evidence/SKILL.md`](skills/workflow-agent-evidence/SKILL.md) | Evidence 連携 |
| [`skills/workflow-agent-evidence/agents/claude.yaml`](skills/workflow-agent-evidence/agents/claude.yaml) | Claude metadata |
| [`skills/workflow-agent-evidence/agents/openai.yaml`](skills/workflow-agent-evidence/agents/openai.yaml) | OpenAI metadata |

---

## 主要コマンド

### Birdseye / Codemap

```sh
# 全体更新
python tools/codemap/update.py --targets docs/birdseye/index.json,docs/birdseye/hot.json --emit index+caps

# 局所更新（radius 1）
python tools/codemap/update.py --since --radius 1 --emit caps

# 鮮度確認
python tools/ci/check_birdseye_freshness.py --check --max-verified-age-days 90
```

### Metrics / KPI

```sh
# QA メトリクス収集
python -m tools.perf.collect_metrics --suite qa --metrics-url <url> --log-path <path>

# 閾値判定
python tools/ci/check_metrics_thresholds.py --check --metrics-json .ga/qa-metrics.json
```

### Acceptance / Task

```sh
# 検収記録検証
python tools/ci/check_acceptance.py --check

# Task/Acceptance 同期確認
python tools/ci/check_task_acceptance_sync.py --plugin-config examples/workflow_plugins.cross_repo.sample.json

# 検収 index 生成
python tools/ci/generate_acceptance_index.py --plugin-config examples/workflow_plugins.cross_repo.sample.json
```

### Security / Release

```sh
# セキュリティ posture 確認
python tools/ci/check_security_posture.py --check --github-repo owner/name

# リリース証跡確認
python tools/ci/check_release_evidence.py --check --github-repo owner/name

# Branch protection 検証
python tools/ci/check_branch_protection.py --protection-json <json>

# セキュリティ docs 鮮度確認
python tools/ci/check_security_docs_freshness.py --check

# sample/docs 同期確認
python tools/ci/check_sample_docs_sync.py --check
```

### Evidence / Report

```sh
# Evidence レポート生成
python tools/ci/generate_evidence_report.py --output docs/evidence_report.md

# 検収 index 生成
python tools/ci/generate_acceptance_index_standalone.py --output docs/acceptance_index.md

# Upstream 差分抽出
python tools/ci/extract_upstream_changes.py --upstream-md docs/UPSTREAM.md --weekly-log docs/WEEKLY.md

# Task state エクスポート
python tools/ci/export_task_state.py --output task_state.json
```

---

## CI Workflows

| Workflow | 内容 |
|----------|------|
| [`.github/workflows/test.yml`](.github/workflows/test.yml) | テスト + coverage |
| [`.github/workflows/governance-gate.yml`](.github/workflows/governance-gate.yml) | policy 検証 |
| [`.github/workflows/security.yml`](.github/workflows/security.yml) | セキュリティ確認 (Bandit, Semgrep, Gitleaks, Dependency Audit) |
| [`.github/workflows/release-evidence.yml`](.github/workflows/release-evidence.yml) | リリース証跡 |
| [`.github/workflows/cross-repo-integration.yml`](.github/workflows/cross-repo-integration.yml) | cross-repo 連携 |
| [`.github/workflows/docs-resolve-pr-gate.yml`](.github/workflows/docs-resolve-pr-gate.yml) | docs resolve 検証 |

再利用可能 workflow:

- [`.github/workflows/reusable/python-ci.yml`](.github/workflows/reusable/python-ci.yml)

---

## Plugin 連携

### 連携リポジトリ

| リポジトリ | 役割 | 連携方式 |
|------------|------|----------|
| [`agent-protocols`](https://github.com/RNA4219/agent-protocols) | 契約スキーマ (Evidence, TaskSeed, Acceptance 等) | スキーマ参照 |
| [`agent-taskstate`](https://github.com/RNA4219/agent-taskstate) | タスク状態管理、typed_ref、context bundle | Workflow plugin |
| [`memx-resolver`](https://github.com/RNA4219/memx-resolver) | docs resolve、ack、stale check | Workflow plugin |

### Evidence Plugin

```sh
# config sample
examples/inference_plugins.agent_protocol.sample.json

# consumer sample
examples/agent_protocol_evidence_consumer.sample.py

# 詳細
tools/protocols/README.md
```

### Cross-Repo Plugin

```sh
# config sample
examples/workflow_plugins.cross_repo.sample.json

# schema
schemas/workflow-plugin-config.schema.json

# 検証
python tools/workflow_plugins/validate_workflow_plugin_config.py --config examples/workflow_plugins.cross_repo.sample.json
```

---

## 補助ドキュメント

| 分類 | ファイル |
|------|----------|
| 仕様 | [`docs/requirements.md`](docs/requirements.md), [`docs/spec.md`](docs/spec.md), [`docs/design.md`](docs/design.md) |
| 運用 | [`docs/ROADMAP_AND_SPECS.md`](docs/ROADMAP_AND_SPECS.md), [`CHANGELOG.md`](CHANGELOG.md) |
| セキュリティ | [`docs/security/SAC.md`](docs/security/SAC.md), [`docs/security/Security_Review_Checklist.md`](docs/security/Security_Review_Checklist.md) |
| 拡張 | [`docs/addenda/N_Improvement_Backlog.md`](docs/addenda/N_Improvement_Backlog.md), [`docs/addenda/P_Expansion_Candidates.md`](docs/addenda/P_Expansion_Candidates.md) |

---

## License

MIT. Unless noted otherwise, files copied from this repo into other projects
remain under the MIT License.
