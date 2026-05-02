# Completion Record

この文書は、完了済み作業の要約索引である。
詳細な検収証跡は `docs/acceptance/`、作業単位の背景と完了条件は `docs/tasks/`、
リリース単位の変更履歴は `CHANGELOG.md` / `docs/releases/` を正本にする。

RUNBOOK は日常運用と現在の判断に集中させ、完了済みの長い表や詳細証跡を
蓄積しない。

## 記録先の役割

| 記録先 | 役割 | 書く内容 |
|---|---|---|
| `docs/completion-record.md` | 完了事項の索引 | 日付、完了テーマ、正本リンク、短い判定 |
| `docs/tasks/*.md` | 作業単位の正本 | 背景、要求、完了条件、レビュー観点 |
| `docs/acceptance/*.md` | 検収証跡 | 実行コマンド、テスト結果、判定、参照資料 |
| `CHANGELOG.md` | リリース履歴 | ユーザー向け変更点 |
| `docs/releases/*.md` | リリース証跡 | release note、承認、rollback/rehearsal 証跡 |
| `RUNBOOK.md` | 運用入口 | 実行手順、現在の未解決事項、参照リンク |

## 追記テンプレート

```md
## YYYY-MM-DD <完了テーマ>

| 項目 | 状態 | 正本 |
|---|---|---|
| <作業名> | 完了 | <task / acceptance / release へのリンク> |

判定: go / hold / follow-up required
補足: <RUNBOOK に残す必要がない短い補足>
```

## 運用ルール

- 完了済みの詳細表は RUNBOOK に増やさず、本書へ索引として追記する。
- 本書だけで完了を主張しない。必ず `docs/tasks/` または
  `docs/acceptance/` などの正本へリンクする。
- RUNBOOK に完了済み項目を残す場合は、現在の運用判断に必要な最小限の
  1-2 行に留める。
- 大きな機能分割やドキュメント分割を終えた場合は、「何をどこへ分けたか」
  を本書に明示する。

## 2026-05-02 自己改善ループ schema 定義

| 項目 | 状態 | 正本 |
|---|---|---|
| ReflectionSummary schema | 完了 | `schemas/reflection-summary.schema.json` |
| SkillDraftRecord schema | 完了 | `schemas/skill-draft-record.schema.json` |
| RecallResponse schema | 完了 | `schemas/recall-response.schema.json` |
| UserModelSnapshot schema | 完了 | `schemas/user-model-snapshot.schema.json` |
| WorkspaceModelSnapshot schema | 完了 | `schemas/workspace-model-snapshot.schema.json` |
| PeriodicNudge schema | 完了 | `schemas/periodic-nudge.schema.json` |
| Sample config (ReflectionSummary) | 完了 | `examples/reflection-summary.sample.json` |
| Sample config (SkillDraftRecord) | 完了 | `examples/skill-draft-record.sample.json` |
| Schema validation test | 完了 | `tests/test_self_improvement_schemas.py` |

判定: go
補足: spec.md 4.6.2-4.6.6 の DTO 定義に基づく JSON Schema。21 tests passed。
