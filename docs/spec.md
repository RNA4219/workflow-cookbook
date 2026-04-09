---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-04-09
next_review_due: 2026-05-09
---

# SPEC

## 1. 概要

Workflow Cookbook は、QA / Governance-first の運用ドキュメント、Birdseye
資産、参照実装、CI / Governance テンプレートをまとめて提供する基盤リポジトリである。
本仕様は「このリポジトリが外部へどのようなふるまいを公開するか」を定義する。

## 2. 公開対象

- ドキュメントハブ
  - `README.md` は初動入口を提供する。
  - `HUB.codex.md` はタスク分割と依存導線の入口を提供する。
  - `RUNBOOK.md` は実行手順、`EVALUATION.md` は受入観点の入口として機能する。
- Birdseye / Codemap
  - `docs/birdseye/index.json` はノード、エッジ、capsule 参照を提供する。
  - `docs/birdseye/caps/*.json` は個別ノードの要約、依存、関連テスト、リスクを提供する。
  - `docs/birdseye/hot.json` は主要導線のホットリストを提供する。
  - `tools/codemap/update.py` は Birdseye 資産を更新する CLI を提供する。
- 参照実装
  - `tools/autosave/` はロック協調、単調増加、ロールアウトガードの参照実装を提供する。
  - `tools/merge/` は `baseline` / `strict` 精度モードとロック協調の参照実装を提供する。
  - `tools/perf/collect_metrics.py` は構造化ログと Prometheus 由来データを統合する CLI を提供する。
- CI / Governance テンプレート
  - `.github/workflows/reusable/*.yml` は派生リポジトリから `workflow_call` で再利用できる。
  - `governance/policy.yaml` は required jobs と自己改変境界の基準を提供する。
  - `docs/CONTRACTS.md` は feature detection で扱う外部拡張契約を定義する。

## 3. 公開インターフェース仕様

### 3.1 Birdseye / Codemap

- `tools/codemap/update.py` は次の主要引数を受け付けること。
  - `--targets`: 明示ターゲットによる更新対象の指定
  - `--emit`: `index` / `caps` / `index+caps` の出力制御
  - `--since`: 差分検出にもとづく対象抽出
  - `--radius`: 依存 hop 数の制御
- `--radius` の仕様は次のとおり。
  - 既定値は `2`
  - `0` は seed ノードのみ更新
  - `1` 以上は指定 hop 数まで近傍展開
  - 負数は CLI エラー
- `generated_at` は ISO 日時ではなく、5 桁ゼロ埋めの世代番号として扱うこと。

### 3.2 AutoSave

- AutoSave は少なくとも次を検証できること。
  - プロジェクトロックの取得・検証・解放
  - snapshot ID の単調増加
  - rollout gate による書き込み制御
- `autosave.snapshot.commit` 相当のテレメトリは、少なくとも snapshot ID、
  project ID、必要に応じて `latency_ms` / `lock_wait_ms` を含められること。

### 3.3 Merge

- Merge は少なくとも次を提供すること。
  - `baseline` / `strict` の精度モード
  - lock token の検証と release
  - `merged` / `conflicted` / `rolled_back` の結果表現
- Merge は AutoSave のロック協調モデルと矛盾しないこと。

### 3.4 Metrics / Governance

- `python -m tools.perf.collect_metrics --suite qa` は `.ga/qa-metrics.json` を既定出力先とする。
- `docs/CONTRACTS.md` に定義された `.ga/qa-metrics.json` と
  `governance/predictor.yaml` は feature detection で扱い、未提供でも Cookbook
  側が正常動作すること。
- CI phased rollout の段階は、`docs/ci_phased_rollout_requirements.md` と
  workflow 群の双方から追跡できること。

## 4. 互換性と運用ルール

- ドキュメント、Birdseye 生成物、参照実装、CI テンプレートは相互に矛盾しないこと。
- 公開インターフェースを変更する場合は、関連テストとドキュメントを同時に更新すること。
- 仕様変更は `CHANGELOG.md` の `[Unreleased]` に反映すること。

## 5. 最低限の受入観点

- 文書整合
  - `docs/requirements.md` / `docs/spec.md` / `docs/design.md` /
    `docs/CONTRACTS.md` が相互に矛盾しないこと。
- Birdseye 整合
  - `README.md`、`docs/BIRDSEYE.md`、`docs/birdseye/README.md`、
    `GUARDRAILS.md`、`RUNBOOK.md` の更新手順が一致していること。
- 最低限のテスト
  - `tests/test_codemap_update.py`
  - `tests/autosave/test_project_lock_service.py`
  - `tests/merge/test_precision_mode_pipeline.py`
  - `tests/test_collect_metrics_cli.py`
  - `tests/perf/test_collect_metrics_autosave_merge.py`
