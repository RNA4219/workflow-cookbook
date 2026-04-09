---
intent_id: INT-001
owner: docs-core
status: draft
last_reviewed_at: 2026-04-09
next_review_due: 2026-05-09
---

# AutoSave 実装設計

参照: [README.md](../README.md) / [HUB.codex.md](../HUB.codex.md) /
[docs/IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md) /
[docs/tasks/task-autosave-project-locks.md](tasks/task-autosave-project-locks.md) /
[docs/MERGE-DESIGN-IMPL.md](MERGE-DESIGN-IMPL.md)

## 不変条件

- `autosave.project_lock` が有効かつ checklist 完了済みの場合のみ、AutoSave の検証処理を実行する。
- 同一 `project_id` に対する `snapshot_id` は単調増加しなければならない。
- `lock_token` は空文字を許容せず、Merge 側 coordinator により検証可能でなければならない。
- commit 成功時は `applied_snapshot_id == request.snapshot_id` を返す。
- skip 時は `applied_snapshot_id` と `next_retry_at` を `None` にする。

## 例外設計

- 基底例外
  - `ProjectLockError`
- リトライ可能として扱う例外
  - `LockTokenInvalidError`
    - `lock_token` 欠落
    - coordinator による token 検証失敗
- リトライ不可として扱う例外
  - `SnapshotOrderViolation`
    - 既知の最終 snapshot ID 以下を再適用しようとした場合
- rollout checklist 未完了や flag 無効は例外ではなく `status="skipped"` で表現する。
- すべての失敗系分岐は監査ログへ `action` と `project_id`、`precision_mode`、
  `snapshot_id` を残す。

## I/O 契約

### 入力

- `AutoSaveRequest`
  - `project_id: str`
  - `snapshot_delta: Mapping[str, Any]`
  - `lock_token: str`
  - `snapshot_id: int`
  - `timestamp: datetime`
  - `precision_mode: str`
  - `latency_ms: float | None`
  - `lock_wait_ms: float | None`

### 出力

- `AutoSaveResult`
  - `status`
    - `ok`
    - `skipped`
  - `applied_snapshot_id: int | None`
  - `next_retry_at: datetime | None`

### 主要分岐

1. rollout inactive
   - `status="skipped"`
   - `applied_snapshot_id=None`
2. token invalid
   - `LockTokenInvalidError`
3. snapshot order violation
   - `SnapshotOrderViolation`
4. commit success
   - `status="ok"`
   - `applied_snapshot_id=request.snapshot_id`

## ロック協調

- AutoSave は lock lifecycle の実装本体を持たず、`ProjectLockCoordinator`
  に委譲する。
- coordinator には次の 2 操作を期待する。
  - `validate_token(project_id, token) -> bool`
  - `lock_release(project_id, token) -> None`
- AutoSave 側は commit 前に `validate_token` を呼び、token が無効なら
  `LockTokenInvalidError` を返す。
- lock release 自体は Merge pipeline 側が行い、AutoSave は token の妥当性と
  snapshot monotonicity の検証に集中する。

## セッション設計

- `ProjectLockService`
  - リクエスト受付とセッション起動を担う。
- `AutoSaveSnapshotSession`
  - rollout 判定
  - lock 検証
  - snapshot order 検証
  - commit 実行
- `AutoSaveSnapshotValidator`
  - coordinator を使った token 検証
  - project ごとの最終 snapshot ID 管理
  - commit テレメトリ送信

状態の持ち主は次のとおり。

- rollout flag
  - `FlagState`
- 最終 snapshot ID
  - `ProjectLockService._last_snapshot_id`
- telemetry sink
  - `TelemetryEmitter`

## テレメトリ要件

- 成功 commit 時には `autosave.snapshot.commit` を emit する。
- payload は少なくとも次を含む。
  - `project_id`
  - `snapshot_id`
  - `precision_mode`
  - `timestamp`
  - `latency_ms`（指定時のみ）
  - `lock_wait_ms`（指定時のみ）
- telemetry 送信失敗を握り潰す責務はこの実装には持たせず、sink 側の実装方針へ委譲する。
- 監査ログは `autosave_audit action=<...>` 形式の logger 出力で補完する。

## 実装上の意図

- CRDT、UI トースト、ローカルキャッシュ同期のようなプロダクト固有機能は、
  この参照実装の責務に含めない。
- 参照実装は「lock validation と snapshot monotonicity をどう検証するか」を
  最小単位で示すことを目的とする。
- Merge との協調点は token validation と `precision_mode` の共有、および
  telemetry 名の整合に限定する。

---

- 逆リンク: [docs/tasks/task-autosave-project-locks.md](tasks/task-autosave-project-locks.md) /
  [docs/MERGE-DESIGN-IMPL.md](MERGE-DESIGN-IMPL.md) /
  [docs/IMPLEMENTATION-PLAN.md](IMPLEMENTATION-PLAN.md) /
  [README.md](../README.md) /
  [HUB.codex.md](../HUB.codex.md)
