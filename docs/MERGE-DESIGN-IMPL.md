---
intent_id: INT-001
owner: docs-core
status: draft
last_reviewed_at: 2026-04-09
next_review_due: 2026-05-09
---

# Merge 実装設計

参照: [README.md](../README.md) / [HUB.codex.md](../HUB.codex.md) /
[docs/IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md) /
[docs/tasks/task-autosave-project-locks.md](tasks/task-autosave-project-locks.md) /
[docs/AUTOSAVE-DESIGN-IMPL.md](AUTOSAVE-DESIGN-IMPL.md)

## 精度モード

- `baseline`
  - `strict` 以外の候補を正規化した既定モード
  - lock token が未検証でも pipeline は継続できる
  - lock release は `lock_token` があれば許可できる
- `strict`
  - valid な `lock_token` を必須とする
  - token 無効時は `LockTokenInvalidError` を送出し、結果は conflict 系として扱う
- 解決ルール
  - `precision_mode_override` があればそれを優先
  - 無ければ `FlagState.merge_precision_mode()` を使う
  - `strict` 以外はすべて `baseline` に正規化する

## I/O 契約

### 入力

- `MergePipelineRequest`
  - `project_id: str`
  - `request_id: str`
  - `merged_snapshot: Mapping[str, object]`
  - `last_applied_snapshot_id: int`
  - `lock_token: str | None`
  - `autosave_lag_ms: float | None`
  - `latency_ms: float | None`
  - `lock_wait_ms: float | None`
  - `precision_mode_override: str | None`

### 出力

- `MergePipelineResult`
  - `status`
    - `merged`
    - `conflicted`
    - `rolled_back`
  - `precision_mode`
  - `resolved_snapshot_id: int | None`
  - `lock_released: bool`

### 実行結果の正規化

- executor の返却 `status` が上記 3 種以外なら `conflicted` に正規化する。
- strict で token 無効の場合は executor を呼ばずに失敗させる。

## ロック協調

- Merge は `ProjectLockCoordinator` に依存し、token validation と lock release
  を委譲する。
- `MergeSession.create_state()`
  - precision mode 解決
  - token 有無の確認
  - token 妥当性の事前評価
- `MergeSession.release_lock()`
  - `lock_token` があり、かつ次のいずれかで release する。
    - `lock_validated == True`
    - `precision_mode == "baseline"`
- strict モードで token invalid の場合は release を行わない。

## 実行フロー

1. request から `MergeSessionState` を組み立てる
2. strict かつ token invalid なら `LockTokenInvalidError`
3. `MergeOperation` を executor へ渡す
4. `status` を `merged/conflicted/rolled_back` に正規化する
5. lock release 可否を判定する
6. metrics を集計し `MergePipelineResult` を返す

## テレメトリ要件

- `merge.pipeline.metrics` を emit する。
- payload は少なくとも次を含む。
  - `precision_mode`
  - `status`
  - `merge.success.rate`
  - `merge.conflict.rate`
  - `merge.autosave.lag_ms`
  - `lock_validated`
  - `resolved_snapshot_id`
  - `latency_ms`（指定時のみ）
  - `lock_wait_ms`（指定時のみ）
- 同時に `StructuredLogger.inference()` へ次の metrics を渡す。
  - `merge.precision_mode`
  - `merge.success.rate`
  - `merge.conflict.rate`
  - `merge.autosave.lag_ms`
- `metrics_snapshot()` は mode 別 aggregate を
  `merge.success.rate|precision_mode=...` 形式で返す。

## 状態所有

- precision mode ごとの totals / successes / conflicts / lag は
  `MergeMetricsTracker` が保持する。
- request 固有の state は `MergeSessionState` に閉じ込める。
- merge 本体の業務ロジックは `MergeExecutor` に委譲し、pipeline は前後処理に集中する。

## 失敗時の扱い

- strict token invalid
  - `LockTokenInvalidError`
  - metrics 上は conflict 系で記録
- executor 不正 status
  - `conflicted` に正規化
- lock release 不可
  - `lock_released=False`
  - ただし pipeline 自体は結果を返す

## 実装上の意図

- 参照実装はバックオフ、UI 復帰、分散ロック期限管理などの本番論点をすべては持ち込まない。
- 中心となる責務は
  - precision mode の決定
  - strict token 必須化
  - lock release 条件
  - mode 別 metrics 集計
  に限定する。
- AutoSave との境界は `ProjectLockCoordinator` と telemetry 名で接続する。

---

- 逆リンク: [docs/tasks/task-autosave-project-locks.md](tasks/task-autosave-project-locks.md) /
  [docs/AUTOSAVE-DESIGN-IMPL.md](AUTOSAVE-DESIGN-IMPL.md) /
  [docs/IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md) /
  [README.md](../README.md) /
  [HUB.codex.md](../HUB.codex.md)
