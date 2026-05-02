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
| Self-Improvement Loop Schemas | 完了 | [docs/acceptance/AC-20260502-01.md](acceptance/AC-20260502-01.md) |

判定: go
補足: spec.md 4.6.2-4.6.6 の DTO 定義に基づく JSON Schema。21 tests passed。schemas/, examples/, tests/ 詳細は acceptance レコード参照。

## 2026-05-02 自己改善ループ nudge checker 実装

| 項目 | 状態 | 正本 |
|---|---|---|
| stale reflection checker | 完了 | `tools/ci/check_stale_self_improvement.py` |
| stale skill draft checker | 完了 | `tools/ci/check_stale_self_improvement.py` |
| nudge checker test | 完了 | `tests/test_check_stale_self_improvement.py` |
| Session reflection | 完了 | `.workflow-cache/reflections/SESSION-20260502-002.json` |
| Skill draft (Task Seed auto-propagation) | draft | `.workflow-cache/skill-drafts/SKILL-DRAFT-002.json` |

判定: go
補足: 12 tests passed。O_Adaptive_Improvement_Loop.md 8節「次の実装候補」のnudge checker実装。

## 2026-05-03 Birdseye Freshness しきい値 365→90日移行

| 項目 | 状態 | 正本 |
|---|---|---|
| markdown.ymlしきい値変更 | 完了 | `.github/workflows/markdown.yml` L24-26 |
| RUNBOOK段階計画更新 | 完了 | `RUNBOOK.md` L112-114 |

判定: go
補足: 运用開始期間終了、90日しきい値移行完了。最終目標は30日。