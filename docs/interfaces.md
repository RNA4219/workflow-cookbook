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
| ---- | ------------- | ------------- | ---- |
| context_trimmer | `ContextTrimResult` JSON（`messages`、`omitted`, `token_budget` を含むトリミング後会話コンテキスト） | `ConversationLog` JSON（メッセージ列、`token_budget` 指定、`policy` 設定）、`context_trimmer` の操作ログ出力先 | 入出力スキーマ: `ContextTrimResult` / `ConversationLog`; 依存: structured logging sink。関連仕様: [docs/ROADMAP_AND_SPECS.md](./ROADMAP_AND_SPECS.md) の「Minimal Context Intake」。 |
| codemap_update_cli | `docs/birdseye/index.json`、`docs/birdseye/hot.json`、`docs/birdseye/caps/*.json` の更新済み生成物 | 既存 Birdseye 資産、`--targets` / `--emit` / `--since` / `--radius` 引数、差分元 git 状態 | 入出力仕様は [docs/spec.md](./spec.md) の「4.2 Birdseye / Codemap」を参照。`generated_at` は 5 桁ゼロ埋め世代番号で、`index.json` と `hot.json` は同じ更新サイクルへ揃える。 |
| autosave_project_lock | `AutoSaveResult`（`status`, `applied_snapshot_id`, `next_retry_at`）と `autosave.snapshot.commit` テレメトリ | `AutoSaveRequest`、`ProjectLockCoordinator`、feature flag state | 代表例外: `LockTokenInvalidError`, `SnapshotOrderViolation`。関連仕様: [docs/spec.md](./spec.md) の「4.3 AutoSave 参照実装」。 |
| merge_pipeline | `MergePipelineResult`（`status`, `precision_mode`, `resolved_snapshot_id`, `lock_released`）と `merge.pipeline.metrics` テレメトリ | `MergePipelineRequest`、`ProjectLockCoordinator`、feature flag state、merge executor | `strict` では valid な `lock_token` を要求する。関連仕様: [docs/spec.md](./spec.md) の「4.4 Merge 参照実装」。 |
| agent_protocol_evidence_bridge | `agent-protocols` の `Evidence` schema に互換な JSON object、または JSON Lines 追記 | `InferenceLogRecord`、`extra.agent_protocol` コンテキスト、repo root、plugin / mapper / writer の組み合わせ | 通常ログは `StructuredLogger` が担当し、本境界は context 抽出・Evidence 変換・永続化だけを担当する。関連仕様: [docs/spec.md](./spec.md) の「4.5 `StructuredLogger` / agent-protocols Evidence 連携」。 |
| inference_plugin_loader | `InferenceLogPlugin` のインスタンス列 | `InferencePluginSpec[]`（`factory`, `options`, `enabled`） | `factory` は `module:attribute` 形式の import 文字列。plugin の生成と検証だけを担当し、logger 本体は loader 実装詳細を持たない。関連仕様: [docs/spec.md](./spec.md) の「4.5 `StructuredLogger` / agent-protocols Evidence 連携」。 |
| inference_plugin_config_loader | `InferencePluginSpec[]` | mapping または `.json` / `.yaml` / `.yml` の config file | top-level `inference_plugins` 配列、または配列直下を受け付ける。YAML は loader が存在する環境でのみ有効。shape は `schemas/inference-plugin-config.schema.json` で共有する。関連仕様: [docs/spec.md](./spec.md) の「4.5 `StructuredLogger` / agent-protocols Evidence 連携」。 |
| reflection_loop_orchestrator | `ReflectionSummary`、`SkillDraftRecord`、`RecallResponse`、`PeriodicNudge` の生成ルール | task / session / acceptance / evidence / docs reference | cookbook 自身は実状態を持たず、下流ソフトウェアへ提供する自己改善ループの契約だけを定義する。関連仕様: [docs/spec.md](./spec.md) の「4.6 自己改善ループ blueprint」。 |
| memory_curation_backend | 長期保持候補、短期保持候補、nudge 候補の分類結果 | `ReflectionSummary[]`、既存 memory state、review 状態 | memory store 自体は差し替え可能。関連設計: [docs/addenda/O_Adaptive_Improvement_Loop.md](./addenda/O_Adaptive_Improvement_Loop.md)。 |
| skill_evolution_pipeline | `SkillDraftRecord` の draft / review / approved / rejected 遷移 | 複雑 task の reflection、既存 skill、acceptance / evidence 参照 | review 未完了の draft を公開 skill として扱わない。関連設計: [docs/addenda/O_Adaptive_Improvement_Loop.md](./addenda/O_Adaptive_Improvement_Loop.md)。 |
| session_recall_resolver | `RecallResponse`（summary と根拠断片） | query、reflection index、acceptance、evidence、docs reference | raw transcript 全文ではなく summary と出典を返す。関連仕様: [docs/spec.md](./spec.md) の「4.6 自己改善ループ blueprint」。 |
| user_workspace_model_store | `UserModelSnapshot`、`WorkspaceModelSnapshot` | review 済み preference / constraint / output convention | 長期保持は review 済み情報のみ。関連設計: [docs/addenda/O_Adaptive_Improvement_Loop.md](./addenda/O_Adaptive_Improvement_Loop.md)。 |
| collect_metrics_cli | CLI 経由でエクスポートされる `MetricsSnapshot` JSON、`.ga/qa-metrics.json`、Prometheus PushGateway 互換メトリクス | `--metrics-url`、`--log-path`、`--suite qa`、`--pushgateway-url`、構造化ログ | どの入力ソースも無い場合は `MetricsCollectionError`。契約: [docs/CONTRACTS.md](./CONTRACTS.md) の `.ga/qa-metrics.json`。関連仕様: [docs/spec.md](./spec.md) の「4.7 Metrics 収集 CLI」。 |

追加ルール:

- 新規機能はPRでテーブル行を追加
- 廃止機能は行末に `(deprecated: YYYY-MM-DD)` を追記
- 相互依存がある場合は備考欄に関連機能を記載

このドキュメントにより、並列開発時の責務衝突を防ぎます。
