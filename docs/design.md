---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-04-09
next_review_due: 2026-05-09
---

# DESIGN

## 1. 設計目標

Workflow Cookbook は、単なる Markdown テンプレート集ではなく、
「運用ドキュメント」「最小読込用 Birdseye」「参照実装」「CI / Governance テンプレート」
を 1 つのリポジトリで同期させる構成を採る。

設計上の主眼は次の 4 点である。

1. 人間とエージェントが同じ入口から辿れること
2. Birdseye により局所更新と最小読込を両立すること
3. docs と code の契約をテストで裏付けること
4. 派生リポジトリが一部機能だけを再利用できること

## 2. 全体アーキテクチャ

本リポジトリは、次の 4 層を重ねる設計とする。

### 2.1 ドキュメント層

- 役割
  - 運用の正本と受入基準を提供する。
- 主な構成
  - `README.md`
  - `RUNBOOK.md`
  - `EVALUATION.md`
  - `CHECKLISTS.md`
  - `HUB.codex.md`
  - `GUARDRAILS.md`
  - `docs/requirements.md`
  - `docs/spec.md`
  - `docs/design.md`
  - `docs/CONTRACTS.md`
- 設計原則
  - 入口文書は責務を分ける。
  - 相互リンクで辿れることを優先し、重複記述は最小にする。
  - 受入条件は `requirements` と `EVALUATION` で同期する。

### 2.2 Birdseye 層

- 役割
  - 最小読込のための依存インデックスを提供する。
- 主な構成
  - `docs/birdseye/index.json`
  - `docs/birdseye/hot.json`
  - `docs/birdseye/caps/*.json`
  - `tools/codemap/update.py`
- 設計原則
  - `index.json` は軽量インデックスに徹する。
  - 詳細は `caps/*.json` へ分離する。
  - `hot.json` は即時参照のために curated subset を保持する。
  - `index.json.generated_at` と `hot.json.generated_at` は同一更新サイクルへ揃える。

### 2.3 参照実装層

- 役割
  - AutoSave / Merge / Metrics などの運用設計をコードで検証する。
- 主な構成
  - `tools/autosave/`
  - `tools/merge/`
  - `tools/perf/`
  - `tools/protocols/`
  - `tests/`
- 設計原則
  - 実サービス本体ではなく、契約検証のための最小実装を置く。
  - 主要な入出力型と例外は docs に先に現れるようにする。
  - テレメトリ名は docs と tests の両方で追跡可能にする。
  - 外部契約への接続は logger 本体と plugin / 変換ブリッジを分離し、未接続時の既存挙動を維持する。

### 2.4 CI / Governance 層

- 役割
  - 受入・安全・段階導入の基準をテンプレート化する。
- 主な構成
  - `.github/workflows/reusable/*.yml`
  - `.github/workflows/*.yml`
  - `governance/policy.yaml`
  - `governance/prioritization.yaml`
  - `governance/metrics.yaml`
  - optional: `governance/predictor.yaml`
- 設計原則
  - Cookbook 自身の検証導線と、派生リポジトリ向け reusable workflow を分ける。
  - `policy.yaml` は論理 gate ID としての required jobs、および `forbidden_paths`
    の基準点にする。
  - 論理 gate ID から GitHub 上の実 check 名への対応は `docs/ci-config.md` で管理する。
  - `predictor.yaml` は外部拡張が任意に持ち込む optional config として扱う。

## 3. コンポーネント責務

### 3.1 ドキュメントハブ

- `README.md`
  - 初動入口。
  - Birdseye 更新、Runbook、Checklist への導線をまとめる。
- `HUB.codex.md`
  - タスク分割入口。
  - Task Seed と依存関係の整理を担う。
- `RUNBOOK.md`
  - 実行順序、確認手順、障害時の対応を担う。
- `EVALUATION.md`
  - Acceptance Criteria、KPI、Verification Checklist を担う。
- `GUARDRAILS.md`
  - Birdseye 鮮度判定と最小読込ガードを担う。

### 3.2 `codemap.update`

- 責務
  - Birdseye 資産の再生成。
  - ターゲット解決。
  - focus node の近傍展開。
  - `index.json` / `hot.json` / `caps/*.json` の同期更新。
- 内部設計の要点
  - seed 解決と近傍展開は焦点解決ロジックに集約する。
  - `--radius` は探索範囲にのみ作用させる。
  - serial allocator で `generated_at` を一貫更新する。
- 出力責務
  - `index.json` と `hot.json` は同一世代へ揃える。
  - capsule は対象ノードだけを更新する。

### 3.3 AutoSave project lock service

- 責務
  - rollout gate の評価
  - lock token の検証
  - snapshot monotonicity の検証
  - commit テレメトリの発火
- 状態所有
  - 最終 snapshot ID は service 側が project 単位で保持する。
  - flag state は外部から注入する。
  - lock lifecycle 自体は `ProjectLockCoordinator` 側に委譲する。
- 失敗モデル
  - invalid token は retryable に扱う。
  - snapshot order violation は non-retryable として扱う。
  - rollout inactive 時は `skipped` を返す。

### 3.4 Merge pipeline

- 責務
  - precision mode の解決
  - lock validation
  - merge executor 呼び出し
  - metrics 集計
  - lock release
- 状態所有
  - precision mode ごとの aggregate metrics は tracker が保持する。
  - request 固有の状態は session state に閉じ込める。
- 失敗モデル
  - `strict` で invalid token の場合は conflict 扱いで停止する。
  - executor の返却値が未知の場合は `conflicted` に正規化する。

### 3.5 Metrics 収集 CLI

- 責務
  - metrics source の収集計画作成
  - Prometheus と構造化ログの統合
  - suite 既定値の解決
  - JSON 出力と optional PushGateway 送信
- 状態所有
  - suite ごとの既定設定は `SUITES` で保持する。
  - 実行時解決は `MetricsCollectionPlan` が担う。
- 失敗モデル
  - source 未指定は即時に `MetricsCollectionError` を返す。
  - PushGateway 送信失敗は collection error として扱う。

### 3.6 agent-protocols Evidence ブリッジ

- 責務
  - `InferenceLogRecord` から `agent-protocols` の `Evidence` 契約へ写像する
  - ハッシュ、既定時刻、既定 environment を補完する
  - file sink へ JSON Lines として追記する
- 状態所有
  - logger は通常ログの書き込みと plugin 呼び出し責務のみを持つ
  - context extractor は logger 外部の入力契約を解決する
  - mapper は Evidence JSON の構築責務を持つ
  - writer は永続化責務のみを持つ
  - plugin config loader は import 文字列と options を `InferencePluginSpec` へ正規化する
- 失敗モデル
  - `extra.agent_protocol` が無い場合は no-op
  - `extra.agent_protocol` が不完全な場合は bridge 専用エラーで停止する
  - 通常ログ出力成功後に Evidence 側だけ失敗した場合は、呼び出し元へ失敗を返して再試行可能にする
  - import 文字列から plugin を構成できない場合は loader 専用エラーで停止する
  - YAML config を指定したが yaml loader が無い場合は config loader 専用エラーで停止する

## 4. 状態とデータフロー

### 4.1 Birdseye データフロー

1. 変更ファイルまたは `--targets` から seed を決める
2. focus resolver が `--radius` に応じて対象ノードを展開する
3. 既存 index / hot / caps の `generated_at` を観測する
4. serial allocator が次世代番号を決める
5. `index.json` と `hot.json` を同じ世代で更新する
6. 対象 capsule を同じ世代で更新する

### 4.2 AutoSave / Merge データフロー

1. AutoSave が request を受ける
2. rollout gate と checklist 完了条件を評価する
3. lock token と snapshot order を検証する
4. AutoSave commit 時に `autosave.snapshot.commit` を emit する
5. Merge は request から precision mode を解決する
6. Merge は strict 時に lock validity を必須化する
7. executor 実行後に `merge.pipeline.metrics` を emit する
8. lock release 可否を session が判定する

### 4.3 Metrics データフロー

1. CLI 引数と環境変数から収集計画を組み立てる
2. Prometheus と構造化ログから metrics source を読む
3. 正規化して `MetricsSnapshot` 相当の辞書へ統合する
4. 標準出力へ JSON を出す
5. `output_path` があれば `.ga/qa-metrics.json` へ書き出す
6. `pushgateway_url` があれば PushGateway へ送る

### 4.4 LLM 行動追跡データフロー

1. 呼び出し元が `StructuredLogger.inference()` へ prompt / response / model を渡す
2. `extra.agent_protocol` に TaskSeed / commit / actor などの追跡コンテキストを載せる
3. logger は通常の inference JSON 行を先に出力する
4. Evidence bridge が `InferenceLogRecord` を `Evidence` 契約へ変換する
5. sink が JSON Lines として証跡を永続化する

## 5. 拡張ポイント

### 5.1 optional config

- `governance/predictor.yaml`
  - 外部リポジトリが任意に提供する。
  - Cookbook 本体は未提供でも既定値で動く。

### 5.2 追加テレメトリ

- AutoSave / Merge / Metrics は payload 拡張を許容する。
- ただし既存キーの意味は変更しない。

### 5.3 Birdseye ノード追加

- 新規ノードは `index.json` の `nodes` と `caps/*.json` を同期追加する。
- `hot.json` へ入れるかは curated access の観点で判断する。

## 6. 設計上の不変条件

- `requirements` / `spec` / `design` / `CONTRACTS` は相互に矛盾しないこと。
- `index.json.generated_at` と `hot.json.generated_at` は同一更新サイクルであること。
- AutoSave の snapshot ID は project 単位で単調増加すること。
- Merge の `precision_mode` は `baseline` または `strict` に正規化されること。
- `.ga/qa-metrics.json` と `governance/predictor.yaml` は optional input として扱うこと。

## 7. 障害時の扱い

- Birdseye
  - 世代番号不整合や capsule 欠落時は再生成を要求する。
- AutoSave
  - rollout inactive は skip
  - invalid token は retryable error
  - snapshot order violation は non-retryable error
- Merge
  - strict で invalid token は conflict として扱う。
  - 未知 status は `conflicted` に正規化する。
- Metrics
  - source 未指定は即時エラー
  - PushGateway 送信失敗は収集失敗として通知する

## 8. 検証との対応

- 代表テスト
  - `tests/test_codemap_update.py`
  - `tests/autosave/test_project_lock_service.py`
  - `tests/merge/test_precision_mode_pipeline.py`
  - `tests/test_collect_metrics_cli.py`
  - `tests/perf/test_collect_metrics_autosave_merge.py`
- docs 対応
  - Birdseye 手順: `README.md` / `docs/BIRDSEYE.md` / `docs/birdseye/README.md` /
    `GUARDRAILS.md` / `RUNBOOK.md`
  - 受入基準: `EVALUATION.md`
  - 責務境界: `docs/interfaces.md`

## 9. 関連資料

- 要件: `docs/requirements.md`
- 仕様: `docs/spec.md`
- 受入基準: `EVALUATION.md`
- 実行手順: `RUNBOOK.md`
- 境界一覧: `docs/interfaces.md`
- 外部契約: `docs/CONTRACTS.md`
