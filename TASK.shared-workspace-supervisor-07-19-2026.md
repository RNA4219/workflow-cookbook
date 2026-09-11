---
intent_id: TASK-SHARED-WORKSPACE-SUPERVISOR-20260719
owner: workflow-cookbook
status: completed
last_reviewed_at: 2026-07-19
next_review_due: 2026-08-19
---

# Task Seed: Shared Workspace Supervisor

2026-07-19作成時の完了Task。現在の適用条件・read共有・再利用手順は
[運用Runbook](docs/operations/shared-workspace-supervision-runbook.md) を参照する。
追跡用IDは [Task索引](docs/tasks/task-shared-workspace-supervisor-20260719.md) に対応付けた。

## Objective

複数のCodex task / sub-agentが同じWindows workspaceを使うとき、分析・read-only調査は安全に並列化し、shell・編集・評価などの可変I/Oは永続leaseで直列化する汎用coordinatorを提供する。

## Scope

- SQLite/WALによるprocess間coordinator
- canonical pathの親子overlapを考慮したshared read / exclusive write lease
- owner、task、job key、WIP key、TTL、heartbeat、release
- job key PRIMARY KEYによる重複実行防止とterminal result再利用
- append-only event ledger、status、hard-timeout read-only probe
- Codex supervisor用Runbook、JSON Schema、設定例、unit/CLI test

## Invariants

- state DBは既定でworkspace外のユーザーlocal領域へ置き、対象repoを汚さない。
- read lease同士だけを並列許可し、overlapするwrite/readを排他する。
- shellはread-only allowlistでない限りwrite扱いとする。
- lease tokenはSHA-256だけを永続化する。
- `succeeded` / `invalid` jobは同一job keyで再実行しない。
- probeは子processをhard timeoutし、incidentあたり1回の運用とする。
- 既存Agent_tools repoや外部processを強制停止する権限は持たない。

## Acceptance Criteria

- read/read、read/write、write/write、親子path、WIP競合がテストされる。
- heartbeat、token ownership、release、TTL失効、job completion/invalid/reuseがテストされる。
- 2 processの同時acquireでsingle writerが維持される。
- CLIのJSON出力、安定exit code、hard-timeout probeがテストされる。
- schema、Runbook、sample config、Acceptance recordが揃う。

## Rollback

新規 `tools/supervision/`、schema、tests、docs、sample、Task/Acceptanceだけを戻す。既存runtimeとユーザー差分には触れない。

## Current Result

- SQLite/WAL coordinator、CLI、JSON Schema、設定例、運用Runbookを実装した。
- Windows安全既定として同一ユーザー配下の可変I/Oを `workspace_io_global` / WIP=1で直列化した。
- Phase12 Top7固定Sample4のRunbookとSupervisor boardへ必須利用契約を接続した。
- unit/process-race/CLI lifecycle/probe/schemaの11テスト、Ruff、mypyがすべてpassした。
- 既存のユーザー差分とshipyard-cp runtimeは変更していない。
