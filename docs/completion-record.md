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
| stale reflection checker | 完了 | [docs/acceptance/AC-20260502-02.md](acceptance/AC-20260502-02.md) |
| stale skill draft checker | 完了 | `tools/ci/check_stale_self_improvement.py` |
| nudge checker test | 完了 | `tests/test_check_stale_self_improvement.py` |
| Session reflection | 完了 | `.workflow-cache/reflections/SESSION-20260502-002.json` |
| Skill draft (Task Seed auto-propagation) | draft | `.workflow-cache/skill-drafts/SKILL-DRAFT-002.json` |

判定: go
補足: 12 tests passed。O_Adaptive_Improvement_Loop.md 8節「次の実装候補」のnudge checker実装。

## 2026-05-03 Birdseye Freshness しきい値 365→90日移行

| 項目 | 状態 | 正本 |
|---|---|---|
| markdown.ymlしきい値変更 | 完了 | [docs/acceptance/AC-20260503-01.md](acceptance/AC-20260503-01.md) |
| RUNBOOK段階計画更新 | 完了 | `RUNBOOK.md` L112-114 |

判定: go
補足: 运用開始期間終了、90日しきい値移行完了。最終目標は30日。

## 2026-05-03 Task Seed完了propagation checker実装

| 項目 | 状態 | 正本 |
|---|---|---|
| propagation checker | 完了 | [docs/acceptance/AC-20260503-02.md](acceptance/AC-20260503-02.md) |
| propagation checker test | 完了 | `tests/test_check_task_completion_propagation.py` |

判定: go
補足: SKILL-DRAFT-002手順1実装。13 tests passed。done Task Seedのcompletion-record未反映をnudge検出。

## 2026-05-03 CI workflow RG-006追加 + sample config作成

| 項目 | 状態 | 正本 |
|---|---|---|
| RG-006 gate追加 | 完了 | [docs/acceptance/AC-20260503-03.md](acceptance/AC-20260503-03.md) |
| ci-config.md更新 | 完了 | `docs/ci-config.md` |
| UserModelSnapshot sample | 完了 | `examples/user-model-snapshot.sample.json` |
| WorkspaceModelSnapshot sample | 完了 | `examples/workspace-model-snapshot.sample.json` |
| RecallResponse sample | 完了 | `examples/recall-response.sample.json` |
| PeriodicNudge sample | 完了 | `examples/periodic-nudge.sample.json` |

判定: go
補足: SKILL-DRAFT-002手順3完了。sample config 4件作成。

## 2026-04-10 Autosave project locks

| 項目 | 状態 | 正本 |
|---|---|---|
| autosave project locks | 完了 | [docs/tasks/task-autosave-project-locks.md](tasks/task-autosave-project-locks.md) |

判定: go

## 2026-04-11 Gate hardening metrics birdseye

| 項目 | 状態 | 正本 |
|---|---|---|
| gate hardening metrics | 完了 | [docs/tasks/task-gate-hardening-metrics-birdseye.md](tasks/task-gate-hardening-metrics-birdseye.md) |

判定: go

## 2026-04-11 Gate hardening followup

| 項目 | 状態 | 正本 |
|---|---|---|
| gate hardening followup | 完了 | [docs/tasks/task-gate-hardening-followup.md](tasks/task-gate-hardening-followup.md) |

判定: go

## 2026-04-11 Gate hardening finalization

| 項目 | 状態 | 正本 |
|---|---|---|
| gate hardening finalization | 完了 | [docs/tasks/task-gate-hardening-finalization.md](tasks/task-gate-hardening-finalization.md) |

判定: go

## 2026-04-11 Metrics harvest operationalization

| 項目 | 状態 | 正本 |
|---|---|---|
| metrics harvest | 完了 | [docs/tasks/task-metrics-harvest-operationalization.md](tasks/task-metrics-harvest-operationalization.md) |

判定: go

## 2026-04-11 Birdseye freshness tightening

| 項目 | 状態 | 正本 |
|---|---|---|
| birdseye freshness | 完了 | [docs/tasks/task-birdseye-freshness-tightening.md](tasks/task-birdseye-freshness-tightening.md) |

判定: go

## 2026-04-11 Cross repo status review

| 項目 | 状態 | 正本 |
|---|---|---|
| cross repo status | 完了 | [docs/tasks/task-cross-repo-status-review.md](tasks/task-cross-repo-status-review.md) |

判定: go

## 2026-04-11 Acceptance normalization

| 項目 | 状態 | 正本 |
|---|---|---|
| acceptance normalization | 完了 | [docs/tasks/task-acceptance-normalization-ac-20260411-01.md](tasks/task-acceptance-normalization-ac-20260411-01.md) |

判定: go

## 2026-04-11 Product readiness implementation

| 項目 | 状態 | 正本 |
|---|---|---|
| product readiness | 完了 | [docs/tasks/task-product-readiness-implementation.md](tasks/task-product-readiness-implementation.md) |

判定: go

## 2026-04-11 Improvement backlog complete

| 項目 | 状態 | 正本 |
|---|---|---|
| improvement backlog | 完了 | [docs/tasks/task-improvement-backlog-complete.md](tasks/task-improvement-backlog-complete.md) |

判定: go

## 2026-04-15 Security priority response

| 項目 | 状態 | 正本 |
|---|---|---|
| security priority | 完了 | [docs/tasks/task-security-priority-response-20260415.md](tasks/task-security-priority-response-20260415.md) |

判定: go

## 2026-04-17 Enterprise supply chain hardening

| 項目 | 状態 | 正本 |
|---|---|---|
| supply chain hardening | 完了 | [docs/tasks/task-enterprise-supply-chain-hardening-20260417.md](tasks/task-enterprise-supply-chain-hardening-20260417.md) |

判定: go

## 2026-04-17 Enterprise release operations evidence

| 項目 | 状態 | 正本 |
|---|---|---|
| release operations | 完了 | [docs/tasks/task-enterprise-release-operations-evidence-20260417.md](tasks/task-enterprise-release-operations-evidence-20260417.md) |

判定: go

## 2026-04-17 Enterprise security governance hardening

| 項目 | 状態 | 正本 |
|---|---|---|
| security governance | 完了 | [docs/tasks/task-enterprise-security-governance-hardening-20260417.md](tasks/task-enterprise-security-governance-hardening-20260417.md) |

判定: go

## 2026-04-17 CI gate matrix alignment

| 項目 | 状態 | 正本 |
|---|---|---|
| CI gate matrix | 完了 | [docs/tasks/task-ci-gate-matrix-alignment-20260417.md](tasks/task-ci-gate-matrix-alignment-20260417.md) |

判定: go

## 2026-04-17 Branch protection enablement

| 項目 | 状態 | 正本 |
|---|---|---|
| branch protection | 完了 | [docs/tasks/task-branch-protection-enablement-20260417.md](tasks/task-branch-protection-enablement-20260417.md) |

判定: go

## 2026-04-17 Release evidence operational drill

| 項目 | 状態 | 正本 |
|---|---|---|
| release evidence drill | 完了 | [docs/tasks/task-release-evidence-operational-drill-20260417.md](tasks/task-release-evidence-operational-drill-20260417.md) |

判定: go

## 2026-04-17 Supply chain reproducibility followup

| 項目 | 状態 | 正本 |
|---|---|---|
| supply chain followup | 完了 | [docs/tasks/task-supply-chain-reproducibility-followup-20260417.md](tasks/task-supply-chain-reproducibility-followup-20260417.md) |

判定: go
