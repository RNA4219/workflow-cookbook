---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-04-09
next_review_due: 2026-05-09
---

# REQUIREMENTS

## 1. 目的

Workflow Cookbook は、AI エージェント運用とドキュメントガバナンスを一貫して扱うための基盤リポジトリである。  
本リポジトリは次を同時に満たさなければならない。

- 人間とエージェントが同じ入口から仕様・設計・実行手順へ辿れること
- Birdseye により最小読込で依存関係と更新対象を把握できること
- AutoSave / Merge / メトリクス収集などの参照実装を通じて、運用設計をコードとして検証できること
- sibling repository `../agent-protocols` の `Evidence` 契約へ接続し、
  LLM 実行の行動証跡を追跡できること
- CI / Security / Governance のテンプレートを派生リポジトリで再利用できること

## 2. 対象利用者

- Workflow Cookbook 自体を保守するメンテナ
- 派生リポジトリへテンプレートや運用ルールを導入する開発者
- Birdseye と Task Seed を使って作業する AI エージェント
- リリース判定、セキュリティレビュー、CI 運用を担う QA / Ops / Security 担当

## 3. スコープ

- 対象
  - `README.md` / `BLUEPRINT.md` / `RUNBOOK.md` / `EVALUATION.md` / `CHECKLISTS.md` / `HUB.codex.md` を中心とした運用ドキュメント群
  - `docs/birdseye/` と `tools/codemap/update.py` による Birdseye 生成・鮮度管理
  - `tools/autosave/` / `tools/merge/` / `tools/perf/` などの参照実装と検証コード
  - `agent-protocols` の `Evidence` schema へ変換する LLM 行動追跡ブリッジ
  - `.github/workflows/` と `governance/` による CI / Security / Governance テンプレート
- 非対象
  - 本番サービスのホスティングやデプロイ実体
  - 外部 SaaS の本番運用設定そのもの
  - Cookbook 外部リポジトリの個別要件定義

## 4. 機能要件

### 4.1 ドキュメントハブ

1. 主要ドキュメントは役割ごとに分離され、相互リンクで辿れること。
2. `README.md` は初動入口、`HUB.codex.md` はタスク分割入口、`RUNBOOK.md` は実行手順入口として機能すること。
3. `docs/requirements.md` / `docs/spec.md` / `docs/design.md` は互いに矛盾せず、レビュー時に参照可能であること。

### 4.2 Birdseye / Codemap

1. `docs/birdseye/index.json` はノード一覧・エッジ・カプセル参照を保持すること。
2. `docs/birdseye/caps/*.json` は各ノードの要約、依存、リスク、関連テストを保持すること。
3. `docs/birdseye/hot.json` は主要導線を即時参照できるホットリストとして維持されること。
4. `tools/codemap/update.py` は `--targets` / `--emit` / `--since` / `--radius` により Birdseye の局所更新と全体更新を制御できること。
5. Birdseye の `generated_at` は 5 桁ゼロ埋めの世代番号として一貫して扱われること。

### 4.3 参照実装

1. AutoSave 参照実装はロック検証、スナップショット単調性、ロールアウトガードを検証できること。
2. Merge 参照実装は `baseline` / `strict` の精度モードとロック協調を扱えること。
3. メトリクス収集系は構造化ログと Prometheus 由来データを統合できること。
4. `StructuredLogger` は任意のプラグインを介して `agent-protocols` の `Evidence`
   契約へ推論ログをミラーできること。
5. `Evidence` 連携は `extra.agent_protocol` コンテキストが無い場合に既存ログ出力を壊さず、
   コンテキストがある場合のみ schema 互換の証跡を生成できること。
6. plugin は import 文字列と options から構成でき、外部リポジトリが logger 本体を
   変更せずに差し込めること。
7. plugin は mapping または config file から宣言的に構成できること。
8. 参照実装は docs の契約と矛盾しない最小テストを伴うこと。
9. テスト実装工程は単体テスト・結合テスト・coverage 確認を標準とし、
   Python 系の既定 coverage 下限を 80% とすること。

### 4.4 CI / Governance テンプレート

1. 再利用可能な Python / Security CI ワークフローを提供すること。
2. `governance/policy.yaml` は required jobs と自己改変境界の基準点として機能すること。
3. CI 段階導入の方針は `docs/ci_phased_rollout_requirements.md` と workflow 群で追跡できること。
4. `docs/CONTRACTS.md` に定義された `.ga/qa-metrics.json` と `governance/predictor.yaml`
   の feature detection 契約が維持されること。

## 5. 非機能要件

- 保守性
  - 変更は最小差分で行い、ドキュメントと実装が乖離しないこと。
  - 派生リポジトリが再利用しやすいよう、公開テンプレートは後方互換を重視すること。
- 観測性
  - AutoSave / Merge / QA メトリクスは、少なくとも構造化ログまたは集約 JSON から検証可能であること。
  - LLM 実行証跡は `agent-protocols` の `Evidence` として再利用可能な JSON で追跡できること。
- セキュリティ
  - 機密情報はコミットしないこと。
  - Security review 導線と workflow の関係が docs 上で追跡できること。
- 可読性
  - エージェントが最小コンテキストで読めるよう、Birdseye とハブ文書を維持すること。

## 6. 受入条件

- Birdseye / Codemap の更新手順が `README.md`、`docs/BIRDSEYE.md`、`docs/birdseye/README.md`、`GUARDRAILS.md`、`RUNBOOK.md` で整合していること。
- `docs/requirements.md`、`docs/spec.md`、`docs/design.md`、`docs/CONTRACTS.md` がこの repo の実態に即した内容になっていること。
- `docs/CONTRACTS.md` に定義された `.ga/qa-metrics.json` と `governance/predictor.yaml` の feature detection 契約が維持されていること。
- `agent-protocols` の `Evidence` 契約に必要な最小フィールド
  （`taskSeedId`、`baseCommit`、`headCommit`、`actor`、`model`、`tools` など）が
  LLM 行動追跡ログから再構成できること。
- PR 本文に Priority Score（値と根拠）が記録されていること。
- `governance/policy.yaml` の `forbidden_paths` を無断で変更しないこと。
- インシデント発生時は `docs/IN-YYYYMMDD-XXX.md` を作成し、該当 PR および `RUNBOOK.md` から相互リンクすること。
- 最低限の回帰確認として、次のテストが通ること。
  - `tests/test_codemap_update.py`
  - `tests/autosave/test_project_lock_service.py`
  - `tests/merge/test_precision_mode_pipeline.py`
  - `tests/test_collect_metrics_cli.py`
  - `tests/perf/test_collect_metrics_autosave_merge.py`
  - `tests/test_structured_logger.py`
  - `tests/test_agent_protocol_evidence.py`
  - `tests/test_plugin_loader.py`
  - `tests/test_plugin_config.py`
- Python CI では `pytest --cov=. --cov-report=term-missing --cov-fail-under=80`
  を基準として coverage を確認すること。
- `CHANGELOG.md` の `[Unreleased]` に今回の差分が追記されていること。

## 7. トレーサビリティ

- 要件の入口: `docs/requirements.md`
- 仕様の入口: `docs/spec.md`
- 設計の入口: `docs/design.md`
- 外部契約: `docs/CONTRACTS.md`
- 実行手順: `RUNBOOK.md`
- 受入基準: `EVALUATION.md`
- 鮮度管理と最小読込: `GUARDRAILS.md` / `docs/BIRDSEYE.md`
- タスク分割: `HUB.codex.md` / `TASK.codex.md`
