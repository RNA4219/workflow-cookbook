---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-04-09
next_review_due: 2026-05-09
---

# SPEC

## 1. 目的

Workflow Cookbook は、QA / Governance-first の運用ドキュメント、Birdseye
資産、参照実装、CI / Governance テンプレートを一体で提供する基盤リポジトリである。
本仕様書は、このリポジトリが外部へ公開するふるまい、入出力、運用上の互換条件を定義する。

## 2. 適用範囲

- 対象
  - ルートドキュメント群
  - `docs/birdseye/` の生成物
  - `tools/codemap/update.py`
  - `tools/autosave/` / `tools/merge/` / `tools/perf/`
  - `.github/workflows/` と `governance/`
- 非対象
  - 本番ホスティング実体
  - 外部 SaaS の本番設定
  - Cookbook 外部リポジトリ固有の追加要件

## 3. 想定利用者

- メンテナ
  - repo の文書、生成物、参照実装を保守する。
- 派生リポジトリ導入者
  - reusable workflow、policy、運用文書を再利用する。
- AI エージェント
  - Birdseye とハブ文書を最小読込の起点として利用する。
- QA / Ops / Security 担当
  - 受入基準、KPI、セキュリティ導線、CI 段階導入を評価する。

## 4. 機能仕様

### 4.1 ドキュメントハブ

- ルート文書の責務は次のとおり固定する。
  - `README.md`: 初動入口
  - `HUB.codex.md`: タスク分割入口
  - `RUNBOOK.md`: 実行手順入口
  - `EVALUATION.md`: 受入基準入口
  - `CHECKLISTS.md`: リリースと衛生チェック入口
  - `GUARDRAILS.md`: 行動制約と鮮度管理入口
- `docs/requirements.md` / `docs/spec.md` / `docs/design.md` /
  `docs/CONTRACTS.md` は互いに矛盾してはならない。
- 仕様変更時は `CHANGELOG.md` の `[Unreleased]` に差分を記録する。

### 4.2 Birdseye / Codemap

#### 4.2.1 `docs/birdseye/index.json`

- 最低限次のトップレベルキーを保持する。
  - `generated_at`
  - `nodes`
  - `edges`
- `generated_at`
  - 5 桁ゼロ埋めの世代番号であること。
  - 例: `00025`
- `nodes`
  - キーはノード ID であること。
  - 値は少なくとも次を持つこと。
    - `role`
    - `caps`
    - `mtime`
- `edges`
  - 2 要素配列の配列であること。
  - 各要素は `["from", "to"]` の形式で依存関係を表すこと。

#### 4.2.2 `docs/birdseye/hot.json`

- 最低限次のトップレベルキーを保持する。
  - `generated_at`
  - `index_snapshot`
  - `refresh_command`
  - `curation_notes`
  - `nodes`
- `generated_at`
  - `index.json` と同じ更新サイクルの 5 桁ゼロ埋め世代番号であること。
- `nodes[*]`
  - 少なくとも次を持つこと。
    - `id`
    - `role`
    - `reason`
    - `caps`
    - `edges`
    - `last_verified_at`

#### 4.2.3 `docs/birdseye/caps/*.json`

- カプセルは point read 用の最小要約として振る舞う。
- 最低限次のキーを持つこと。
  - `id`
  - `role`
  - `summary`
  - `deps_in`
  - `deps_out`
  - `risks`
  - `tests`
  - `generated_at`
- 必要に応じて `public_api` を持てること。

#### 4.2.4 `codemap.update` CLI

- エントリポイントは `python tools/codemap/update.py` とする。
- 主要引数は次のとおり。
  - `--targets`
    - 明示ターゲットによる更新対象指定
  - `--emit`
    - `index` / `caps` / `index+caps`
  - `--since`
    - `git diff --name-only <ref>...HEAD` ベースの対象抽出
  - `--radius`
    - 依存 hop 数制御
- `--radius` の仕様は次のとおり。
  - 既定値は `2`
  - `0` は seed ノードのみ更新
  - `1` 以上は指定 hop 数まで近傍展開
  - 負数は CLI エラー
- ルートターゲットに `docs/birdseye/`、`index.json`、`hot.json`、`caps/`
  を含めた場合は、全カプセルを探索起点として扱うこと。
- `index.json` を更新する場合、`hot.json` も同じ更新サイクルへ揃えること。

### 4.3 AutoSave 参照実装

#### 4.3.1 入力

- `AutoSaveRequest` は少なくとも次のフィールドを持つ。
  - `project_id`
  - `snapshot_delta`
  - `lock_token`
  - `snapshot_id`
  - `timestamp`
  - `precision_mode`
  - `latency_ms`（任意）
  - `lock_wait_ms`（任意）

#### 4.3.2 出力

- `AutoSaveResult` は少なくとも次のフィールドを持つ。
  - `status`
  - `applied_snapshot_id`
  - `next_retry_at`
- `status` は少なくとも次を返せること。
  - `ok`
  - `skipped`

#### 4.3.3 検証ルール

- ロックトークンの検証を行うこと。
- snapshot ID の単調増加を検証すること。
- rollout gate と checklist 完了条件を評価できること。

#### 4.3.4 例外とテレメトリ

- 主な例外は次のとおり。
  - `LockTokenInvalidError`
  - `SnapshotOrderViolation`
- commit 時には `autosave.snapshot.commit` を emit できること。
- payload は少なくとも次を含められること。
  - `project_id`
  - `snapshot_id`
  - `precision_mode`
  - `latency_ms`（任意）
  - `lock_wait_ms`（任意）

### 4.4 Merge 参照実装

#### 4.4.1 入力

- `MergePipelineRequest` は少なくとも次のフィールドを持つ。
  - `project_id`
  - `request_id`
  - `merged_snapshot`
  - `last_applied_snapshot_id`
  - `lock_token`
  - `autosave_lag_ms`（任意）
  - `latency_ms`（任意）
  - `lock_wait_ms`（任意）
  - `precision_mode_override`（任意）

#### 4.4.2 出力

- `MergePipelineResult` は少なくとも次のフィールドを持つ。
  - `status`
  - `precision_mode`
  - `resolved_snapshot_id`
  - `lock_released`
- `status` は次のいずれかであること。
  - `merged`
  - `conflicted`
  - `rolled_back`

#### 4.4.3 精度モード

- `precision_mode` は `baseline` または `strict` とする。
- `strict` では valid な `lock_token` を必須とする。
- `baseline` では lock release を許可できること。

#### 4.4.4 テレメトリ

- Merge は `merge.pipeline.metrics` を emit できること。
- payload は少なくとも次を含められること。
  - `precision_mode`
  - `status`
  - `merge.success.rate`
  - `merge.conflict.rate`
  - `merge.autosave.lag_ms`
  - `lock_validated`
  - `resolved_snapshot_id`
  - `latency_ms`（任意）
  - `lock_wait_ms`（任意）

### 4.5 Metrics 収集 CLI

#### 4.5.1 入力ソース

- 収集元は次のいずれか、または両方を受け付けること。
  - `--metrics-url`
  - `--log-path`
- どちらも未指定の場合は `MetricsCollectionError` を返すこと。
- 既定エラーメッセージは
  `No metrics input configured: provide --metrics-url or --log-path`
  を用いること。

#### 4.5.2 suite

- `--suite qa` を提供すること。
- `qa` suite の既定出力先は `.ga/qa-metrics.json` とすること。

#### 4.5.3 出力

- 結果は標準出力へ JSON として出力すること。
- `output_path` がある場合はファイルにも書き出すこと。
- `--pushgateway-url` 指定時は PushGateway へ PUT 送信できること。

### 4.6 CI / Governance テンプレート

- `.github/workflows/reusable/*.yml` は `workflow_call` により派生リポジトリから再利用できること。
- `governance/policy.yaml` は少なくとも次の責務を持つこと。
  - 論理 gate ID としての `required_jobs` の基準
  - `forbidden_paths` の基準
- 論理 gate ID と GitHub 上の実 check 名の対応は `docs/ci-config.md` で管理すること。
- `docs/ci_phased_rollout_requirements.md` と workflow 群は、Phase 0〜3
  の段階導入方針を追跡できること。

### 4.7 外部契約

- `docs/CONTRACTS.md` の契約は feature detection で扱うこと。
- 少なくとも次を optional な外部入力として扱えること。
  - `.ga/qa-metrics.json`
  - `governance/predictor.yaml`
- これらが未提供でも Cookbook 側は正常動作しなければならない。

## 5. 互換性と変更管理

- ドキュメント、Birdseye 生成物、参照実装、CI テンプレートは相互に矛盾してはならない。
- 公開インターフェース変更時は、関連テストと関連文書を同時に更新すること。
- 変更履歴は `CHANGELOG.md` の `[Unreleased]` に追記すること。

## 6. 検証観点

- 文書整合
  - `requirements` / `spec` / `design` / `CONTRACTS` が矛盾しないこと。
- Birdseye 整合
  - `README.md`、`docs/BIRDSEYE.md`、`docs/birdseye/README.md`、
    `GUARDRAILS.md`、`RUNBOOK.md` の更新手順が一致すること。
  - `index.json.generated_at` と `hot.json.generated_at` が同じ更新サイクルであること。
- 代表テスト
  - `tests/test_codemap_update.py`
  - `tests/autosave/test_project_lock_service.py`
  - `tests/merge/test_precision_mode_pipeline.py`
  - `tests/test_collect_metrics_cli.py`
  - `tests/perf/test_collect_metrics_autosave_merge.py`

## 7. 関連資料

- 要件: `docs/requirements.md`
- 設計: `docs/design.md`
- 受入基準: `EVALUATION.md`
- 実行手順: `RUNBOOK.md`
- 境界一覧: `docs/interfaces.md`
- 外部契約: `docs/CONTRACTS.md`
