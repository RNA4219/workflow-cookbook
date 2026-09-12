# Completion Record

## 2026-09-12 フル準拠の一括コピー

固定コミットの全追跡ファイルを導入先へ設置するCLIと、コピー先だけで使える検査入口を追加。
既存AGENTSの保全、再実行、dry-run、hash照合、途中失敗の撤回を検証した。
新機能55件、全体1137件成功。wheelから567ファイルをコピーし、全Git blobを独立照合した。
任意の外部repo integration 3件はskip。コピー成功を導入先の運用準拠認定とは扱わない。
公開前レビューでAGENTSを追記方式へ修正。並行編集・不完全な書込を含む60ケースが成功。

- [Task 20260912-01](tasks/task-full-workflow-copy-20260912.md)
- [Acceptance AC-20260912-01](acceptance/AC-20260912-01.md)
- [一括コピーの仕様](contracts/full-workflow-copy.md)
- [Changelog](../CHANGELOG.md#unreleased)

## 2026-09-11 長い履歴と記憶更新の測定

128イベント・3条件の全504推論と独立監査を実施。解消済み質問が標準snapshotから欠ける制限を確認。
[Task](tasks/task-long-horizon-drift-20260911.md)と[技術検収](acceptance/AC-20260911-04.md)に結果と範囲を保存。
[Changelog](../CHANGELOG.md#unreleased)に測定完了と未解決事項を記録。

## 2026-09-11 原文計算と全件回帰

hash付き原文から全案へ同じ整数式を適用する計算API/CLIを追加。
新規67試験、全1071pytest成功、HATE/QEG Go。最終のモデル回帰は元の全192応答が正解。
初回の対照誤答も保存し、計算補助ありworkflowの限定回帰結果として記録する。

- [Task 20260911-03](tasks/task-source-calculations-20260911.md)
- [Acceptance AC-20260911-03](acceptance/AC-20260911-03.md)
- [原文計算の仕様](contracts/source-calculations.md)
- [Changelog](../CHANGELOG.md#unreleased)

## 2026-09-11 文脈の継続性

実装・自動受入完了。agent-taskstateの完全snapshotと必読全文を保持し、予算不足・状態変更・
根拠未解決では継続不可とする。全1004pytest成功、HATE→QEG Go。
モデルのドリフト率は未測定で、task全体はin_progressを維持する。

- [Task 20260911-02](tasks/task-context-continuity-20260911.md)
- [Acceptance AC-20260911-02](acceptance/AC-20260911-02.md)
- [文脈継続の仕様](contracts/task-context-continuity.md)
- [Changelog](../CHANGELOG.md#unreleased)

## 2026-09-11 HATE・QEGの実テスト接続

全976pytest成功、HATE正規化/export成功、QEGのvalidate/gate/record/outputs read成功。
判定Go。対象コードと固定ツールの前後hash一致を確認した。範囲はローカル自動テスト受入。

- [Task 20260911-01](tasks/task-hate-qeg-20260911.md)
- [Acceptance AC-20260911-01](acceptance/AC-20260911-01.md)
- [接続仕様](contracts/hate-qeg-test-gate.md)
- [Changelog](../CHANGELOG.md#unreleased)

task_id: 20260911-01

この文書は、完了済み作業の要約索引である。
詳細な検収証跡は `docs/acceptance/`、作業単位の背景と完了条件は `docs/tasks/`、
リリース単位の変更履歴は `CHANGELOG.md` / `docs/releases/` を正本にする。

RUNBOOK は日常運用と現在の判断に集中させ、完了済みの長い表や詳細証跡を
蓄積しない。

## 2026-09-10 評価・取得・観測・再開

4項目を仕様化して実装。131ケースを追加し、全889ケース、coverage 83.00%、型・lint・文書ゲートを確認。
実CLIでtaskstate再接続、memx cache更新、隔離wheelの4 CLIを検証した。実運用の性能差は未測定。

- [Task 20260910-02](tasks/task-workflow-evolution-20260910.md)
- [Acceptance AC-20260910-02](acceptance/AC-20260910-02.md)
- [利用手順](workflow-evolution.md)
- [Changelog 0082](../CHANGELOG.md#unreleased)

task_id: 20260910-02

## 2026-09-10 Workflow policy remediation

100ルールの実装・維持確認・条件外判断を完了。全pytest 675件、coverage 81.40%、配布物の5 CLIを検証した。
文書レビュー期限・旧caps・実運用観測の残件はAcceptanceへ記録した。

- [Task 20260910-01](tasks/task-policy-remediation-20260910.md)
- [Acceptance AC-20260910-01](acceptance/AC-20260910-01.md)
- [Changelog 0081](../CHANGELOG.md#unreleased)

task_id: 20260910-01

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

## 2026-07-14 Codemap discovery stability

| 項目 | 状態 | 正本 |
|---|---|---|
| configured discovery stability | 完了 | [docs/tasks/task-codemap-discovery-stability-20260714.md](tasks/task-codemap-discovery-stability-20260714.md) / [docs/acceptance/AC-20260714-01.md](acceptance/AC-20260714-01.md) |

判定: go
補足: Codemap / CLI 60 tests passed、`tools.codemap.update` coverage 91.27%。task_id: 20260714-01

## 2026-07-01 Operations automation extensions

| 項目 | 状態 | 正本 |
|---|---|---|
| docs review due automation | 完了 | [docs/acceptance/AC-20260701-01.md](acceptance/AC-20260701-01.md) |
| adoption tier enablement | 完了 | [docs/tasks/task-ops-automation-extensions-20260701.md](tasks/task-ops-automation-extensions-20260701.md) |
| plugin runtime timeout / trace | 完了 | [docs/acceptance/AC-20260701-01.md](acceptance/AC-20260701-01.md) |
| metrics regression checker | 完了 | [CHANGELOG.md#unreleased](../CHANGELOG.md#unreleased) |

判定: go
補足: Focused tests 39 passed、full pytest 561 passed。plugin trace Evidence は local JSON Lines まで対応。

## 2026-07-02 Readiness and downstream automation extensions

| 項目 | 状態 | 正本 |
|---|---|---|
| release readiness report | 完了 | [docs/acceptance/AC-20260702-01.md](acceptance/AC-20260702-01.md) |
| security posture diff | 完了 | [docs/tasks/task-readiness-downstream-automation-extensions-20260702.md](tasks/task-readiness-downstream-automation-extensions-20260702.md) |
| Birdseye remediation helper | 完了 | [docs/acceptance/AC-20260702-01.md](acceptance/AC-20260702-01.md) |
| CI Phase doctor | 完了 | [CHANGELOG.md#unreleased](../CHANGELOG.md#unreleased) |
| adaptive improvement ops | 完了 | [docs/tasks/task-readiness-downstream-automation-extensions-20260702.md](tasks/task-readiness-downstream-automation-extensions-20260702.md) |
| schema/sample/docs matrix | 完了 | [docs/acceptance/AC-20260702-01.md](acceptance/AC-20260702-01.md) |
| branch protection weekly audit | 完了 | [docs/acceptance/AC-20260702-01.md](acceptance/AC-20260702-01.md) |
| downstream onboarding doctor | 完了 | [docs/tasks/task-readiness-downstream-automation-extensions-20260702.md](tasks/task-readiness-downstream-automation-extensions-20260702.md) |

判定: go
補足: Focused tests 66 passed、full pytest 573 passed。

## 2026-05-03 v1.2.0 Release

| 項目 | 状態 | 正本 |
|---|---|---|
| INT-IMPROVEMENT-006 完全実装 | 完了 | [docs/releases/v1.2.0.md](releases/v1.2.0.md) |
| Version consistency checker | 完了 | [docs/acceptance/AC-20260503-05.md](acceptance/AC-20260503-05.md) |
| Stable CLI entrypoints | 完了 | [docs/acceptance/AC-20260503-06.md](acceptance/AC-20260503-06.md) |
| Docs gate escalation policy | 完了 | [docs/acceptance/AC-20260503-07.md](acceptance/AC-20260503-07.md) |
| Plugin capability catalog | 完了 | [docs/acceptance/AC-20260503-08.md](acceptance/AC-20260503-08.md) |
| Post-release validation | 完了 | [docs/tasks/task-release-v1.2.0-post-validation-20260503.md](tasks/task-release-v1.2.0-post-validation-20260503.md) / [docs/acceptance/AC-20260503-09.md](acceptance/AC-20260503-09.md) |

判定: go
補足: 549 tests passing、技術負債解消完了、Topics更新済み。task_id: 20260503-09

## 運用ルール

- 完了済みの詳細表は RUNBOOK に増やさず、本書へ索引として追記する。
- 本書だけで完了を主張しない。必ず `docs/tasks/` または
  `docs/acceptance/` などの正本へリンクする。
- RUNBOOK に完了済み項目を残す場合は、現在の運用判断に必要な最小限の
  1-2 行に留める。
- 大きな機能分割やドキュメント分割を終えた場合は、「何をどこへ分けたか」
  を本書に明示する。

## 2026-05-03 改善仕様拡充の検収

| 項目 | 状態 | 正本 |
|---|---|---|
| improvement spec acceptance | 完了 | [docs/tasks/task-improvement-spec-acceptance-20260503.md](tasks/task-improvement-spec-acceptance-20260503.md) / [docs/acceptance/AC-20260503-04.md](acceptance/AC-20260503-04.md) |
| next implementation prompt | planned | [docs/tasks/task-next-improvement-implementation-20260503.md](tasks/task-next-improvement-implementation-20260503.md) |

判定: go
補足: 文書仕様の検収結果は RUNBOOK に短く記録し、詳細は acceptance record に分離。

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

## 2026-07-11 OSS distribution and CI hardening

| Item | Status | Canonical record |
|---|---|---|
| fail-closed Python CI and 80% coverage | completed | [AC-20260711-01](acceptance/AC-20260711-01.md) |
| non-editable wheel CLI verification | completed | [task-oss-hardening-20260711.md](tasks/task-oss-hardening-20260711.md) |
| Actions SHA pinning and OSS hygiene | completed | [CHANGELOG.md](../CHANGELOG.md#unreleased) |
| Birdseye refresh | completed | [docs/birdseye/hot.json](birdseye/hot.json) |

Verdict: go

Documentation follow-up: [docs-review-audit-20260711.md](reports/docs-review-audit-20260711.md) records
the resolved mypy and recurring-review debt.

## 2026-09-12 導入検査の誤判定修正

- task_id: 20260912-02
- [docs/tasks/task-adoption-validation-20260912.md](tasks/task-adoption-validation-20260912.md) /
  [Acceptance](acceptance/AC-20260912-02.md) /
  [仕様](contracts/adoption-validation.md)
- Git往復、Tierの内容検査、テンプレート版unknownの扱いを修正した。
