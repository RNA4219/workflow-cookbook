---
task_id: 20260910-01
intent_id: INT-POLICY-20260910
owner: Codex
status: done
last_reviewed_at: 2026-09-10
next_review_due: 2026-10-10
---

# Workflow Policy Remediation

## Objective

棚卸し100ルール（必須12・条件付き42・推奨15・廃止31）を用途・実装・検証と整合させる。
100個の不具合という意味ではなく、維持確認と非該当判断も含む。

## Scope

- 共通方針・テンプレート、Birdseye、Task/Acceptance、指標・Gate、セキュリティ・release運用。
- 評価identityとcoordinatorは機能を持つAgent_tools版のみ。配置差を維持する。
- 実運用の効果観測、通知、release、commit/pushはこの改修の完了条件に含めない。

## Requirements

- 本文JSONの常時複製、固定100行／2ファイル上限を再導入しない。
- 実行結果の真実性、必要な承認、秘密情報、契約互換、採用済みcoverage 80%を維持する。
- Gateは主90日・補助180日・直近30日。機会数、正解ラベル、版、欠測を区別する。
- 指標の未計測／errorを実測0へ変換しない。意味保持率・圧縮率は校正が必要な補助指標として扱う。
- 独立readの共有と競合writeの排他を両立する。
- Birdseyeの生成世代と原文・要約のレビューを区別する。

## Work Packages

| パート | 件数 | 対象 |
| --- | ---: | --- |
| WP1 | 36 | 行動・検証・出力方針 |
| WP2 | 3 | 用途別評価identity |
| WP3 | 15 | Birdseyeと取得経路 |
| WP4 | 14 | Task・文書運用 |
| WP5 | 15 | 指標・観測・CI |
| WP6 | 14 | セキュリティ・release |
| WP7 | 3 | 共有workspace |
| 合計 | 100 | WP8で統合検証 |

## Evidence

- 詳細ID台帳: workspaceの `research/workflow-policy-remediation-20260910/progress.json`。
- 初期状態・既存変更hash: 同ディレクトリの `baseline.json`。
- パートごとの変更理由・検証: `WP1-report.json` ～ `WP7-report.json`。
- [Acceptance](../acceptance/AC-20260910-01.md)
- [Changelog](../../CHANGELOG.md#unreleased)

## Verification

100件改修の完了時点では全pytest 675件、行coverage 81.40%、Ruff、mypy、文書の必須ゲート、wheel/sdistと隔離CLI検証が成功。
結果と既存の警告・適用範囲は [Acceptance](../acceptance/AC-20260910-01.md) を参照する。

同日のテスト補強結果はAcceptanceの「追加検証」と、workspaceの
`research/workflow-policy-coverage-20260910/RESULTS.md` に記録した。
