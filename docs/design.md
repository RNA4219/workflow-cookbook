---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-04-09
next_review_due: 2026-05-09
---

# DESIGN

## 1. 設計方針

Workflow Cookbook は、単なる Markdown テンプレート集ではなく、
「運用ドキュメント」「最小読込用 Birdseye」「参照実装」「CI / Governance テンプレート」
を 1 つのリポジトリで同期させる設計を採る。

設計上の主眼は次の 3 点である。

1. 人間とエージェントが同じ入口から辿れること
2. 局所更新でコンテキスト負荷を抑えられること
3. docs と code の契約をテストで裏付けられること

## 2. 主要構成

### 2.1 ドキュメント層

- ルート文書
  - `README.md`: リポジトリ全体の初動入口
  - `RUNBOOK.md`: 実行手順と運用手順
  - `EVALUATION.md`: KPI と受入観点
  - `CHECKLISTS.md`: レビュー / リリース / 運用チェックリスト
  - `HUB.codex.md`: タスク分割と依存のハブ
  - `GUARDRAILS.md`: 行動制約と鮮度管理の基準
- `docs/`
  - `requirements.md`: repo 全体の要件
  - `spec.md`: 公開仕様
  - `design.md`: 設計説明
  - `CONTRACTS.md`: 外部拡張との feature detection 契約
  - `BIRDSEYE.md`: Birdseye の概念と更新導線

### 2.2 Birdseye 層

- `docs/birdseye/index.json`
  - ノード一覧、エッジ、capsule 参照を持つ軽量インデックス
- `docs/birdseye/caps/*.json`
  - 個別ノードごとの point read 用 capsule
- `docs/birdseye/hot.json`
  - 主要導線のホットリスト
- `tools/codemap/update.py`
  - Birdseye 資産の局所更新・全体更新を担う CLI
  - `--radius` により既定 ±2 hop の読込範囲を縮小・維持できる

### 2.3 参照実装層

- `tools/autosave/`
  - lock coordinator
  - rollout gate
  - snapshot monotonicity
  - autosave telemetry
- `tools/merge/`
  - precision mode pipeline
  - autosave lock coordination
  - merge telemetry / conflict handling
- `tools/perf/`
  - Prometheus と構造化ログの統合
  - `.ga/qa-metrics.json` への正規化出力

### 2.4 CI / Governance 層

- `.github/workflows/reusable/*.yml`
  - 派生リポジトリから再利用できる Python / Security CI
- `.github/workflows/*.yml`
  - phased rollout を含む Cookbook 自身の検証導線
- `governance/policy.yaml`
  - required jobs と自己改変境界
- `governance/predictor.yaml`
  - 外部拡張が任意に持ち込める feature detection 対象設定
  - 未提供でも Cookbook 側は既定値で動作する

## 3. 主要データ契約

- Birdseye
  - `generated_at`: 5 桁ゼロ埋め世代番号
  - `index.json`: ノード、エッジ、capsule 参照
  - `caps/*.json`: 要約、依存、リスク、関連テスト
- Metrics
  - `.ga/qa-metrics.json`: QA 系指標の既定集約先
- Governance
  - `policy.yaml`: required jobs とガード境界
  - `predictor.yaml`: 外部契約で任意提供される重み・閾値設定

## 4. 制御フロー

### 4.1 文書更新フロー

1. `requirements` で目的とスコープを定義する
2. `spec` で公開仕様を定義する
3. `design` で構成と制御フローを定義する
4. 実装・Birdseye・運用文書を同期する
5. `CHANGELOG.md` に差分を記録する

### 4.2 Birdseye 更新フロー

1. 変更ファイルまたは `--targets` から seed ノードを解決する
2. `--radius` に応じて対象ノードを近傍展開する
3. 対象 `caps/*.json` と必要に応じて `index.json` / `hot.json` を更新する
4. 関連文書と鮮度ルールの整合を確認する

### 4.3 参照実装検証フロー

1. AutoSave が lock / snapshot / rollout を検証する
2. Merge が precision mode と lock release を検証する
3. Metrics CLI が構造化ログと Prometheus 由来データを統合する
4. 関連テストで docs 契約との整合を担保する

## 5. リスクと緩和

- リスク: docs と code の乖離
  - 緩和: `requirements` / `spec` / `design` / `CONTRACTS` の同時更新と `CHANGELOG` 記録
- リスク: Birdseye の鮮度低下
  - 緩和: `codemap.update` と `GUARDRAILS.md` / `RUNBOOK.md` の同期
- リスク: 派生リポジトリでの再利用失敗
  - 緩和: reusable workflow と feature detection 契約を維持
- リスク: テスト基準の曖昧化
  - 緩和: 最低限の検証対象テストを `requirements` / `spec` に固定
