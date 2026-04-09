---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-04-09
next_review_due: 2026-05-09
---

# Boundary Map

各機能の責務境界を共有するための一覧です。1機能につき1テーブルを追加し、
「提供するもの / 受け取るもの / 備考」を明記してください。

| 機能 | 提供するもの | 受け取るもの | 備考 |
|------|---------------|---------------|------|
| context_trimmer | `ContextTrimResult` JSON（`messages`、`omitted`, `token_budget` を含むトリミング後会話コンテキスト） | `ConversationLog` JSON（メッセージ列、`token_budget` 指定、`policy` 設定）、`context_trimmer` の操作ログ出力先 | 入出力スキーマ: `ContextTrimResult` / `ConversationLog`; 依存: structured logging sink。関連仕様: [docs/ROADMAP_AND_SPECS.md](./ROADMAP_AND_SPECS.md) の「Minimal Context Intake」。 |
| codemap_update_cli | `docs/birdseye/index.json`、`docs/birdseye/hot.json`、`docs/birdseye/caps/*.json` の更新済み生成物 | 既存 Birdseye 資産、`--targets` / `--emit` / `--since` / `--radius` 引数、差分元 git 状態 | 入出力仕様は [docs/spec.md](./spec.md) の「4.2 Birdseye / Codemap」を参照。`generated_at` は 5 桁ゼロ埋め世代番号で、`index.json` と `hot.json` は同じ更新サイクルへ揃える。 |
| autosave_project_lock | `AutoSaveResult`（`status`, `applied_snapshot_id`, `next_retry_at`）と `autosave.snapshot.commit` テレメトリ | `AutoSaveRequest`、`ProjectLockCoordinator`、feature flag state | 代表例外: `LockTokenInvalidError`, `SnapshotOrderViolation`。関連仕様: [docs/spec.md](./spec.md) の「4.3 AutoSave 参照実装」。 |
| merge_pipeline | `MergePipelineResult`（`status`, `precision_mode`, `resolved_snapshot_id`, `lock_released`）と `merge.pipeline.metrics` テレメトリ | `MergePipelineRequest`、`ProjectLockCoordinator`、feature flag state、merge executor | `strict` では valid な `lock_token` を要求する。関連仕様: [docs/spec.md](./spec.md) の「4.4 Merge 参照実装」。 |
| collect_metrics_cli | CLI 経由でエクスポートされる `MetricsSnapshot` JSON、`.ga/qa-metrics.json`、Prometheus PushGateway 互換メトリクス | `--metrics-url`、`--log-path`、`--suite qa`、`--pushgateway-url`、構造化ログ | どの入力ソースも無い場合は `MetricsCollectionError`。契約: [docs/CONTRACTS.md](./CONTRACTS.md) の `.ga/qa-metrics.json`。関連仕様: [docs/spec.md](./spec.md) の「4.5 Metrics 収集 CLI」。 |

追加ルール:

- 新規機能はPRでテーブル行を追加
- 廃止機能は行末に `(deprecated: YYYY-MM-DD)` を追記
- 相互依存がある場合は備考欄に関連機能を記載

このドキュメントにより、並列開発時の責務衝突を防ぎます。
