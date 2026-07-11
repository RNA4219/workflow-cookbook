---
task_id: 20260417-06
intent_id: INT-SEC-011
owner: docs-core
status: done
last_reviewed_at: 2026-04-17
next_review_due: 2026-05-17
---

# Task Seed: Release Evidence Operational Drill

## 背景

- `RELEASE_APPROVAL_TEMPLATE.md`、`docs/releases/INDEX.md`、
  `RUNBOOK.md`、`Release_Checklist.md` は整備された。
- ただし、承認テンプレートと rollback 証跡導線を
  実際の release / rehearsal で 1 回以上回した証拠はまだ弱い。

## ゴール

- release approval、acceptance、rollback 証跡の導線を
  実運用または rehearsal で 1 回検証し、
  監査時に「使えるテンプレート」であることを示せるようにする。

## 実施対象

1. `docs/releases/RA-YYYYMMDD-XX.md` を 1 件作成する
2. Approval Type を `technical` / `security` / `risk_acceptance` のいずれかで分類する
3. `docs/acceptance/AC-*.md` と Release Approval Record を相互参照する
4. rollback 判定基準と証跡記録テーブルを rehearsal で 1 回埋める
5. `check_release_evidence.py` と `check_acceptance.py` が通ることを確認する

## 完了条件

- 承認記録が 1 件以上存在する
- release note / acceptance / approval record の参照がつながっている
- rollback 証跡が rehearsal でもよいので 1 件残る
- RUNBOOK と CHECKLISTS の該当項目を実行済みとして扱える

## 参照

- [RUNBOOK](../../RUNBOOK.md)
- [Release Checklist](../Release_Checklist.md)
- [Enterprise Readiness Assessment](../reports/enterprise-readiness-assessment-20260417.md)

## 完了証跡

- Release Approval: [RA-20260421-01](../releases/RA-20260421-01.md)
- Acceptance references: `AC-20260410-01`, `AC-20260410-02`
- Rollback drill: [RB-20260421-01](../releases/RB-20260421-01-monthly-drill.md)
- `check_release_evidence.py` と `check_acceptance.py` の検証経路を確認済み

## 例外理由

Acceptance record 不要: Operational Readiness Backlog の項目。
実施タイミングは次回 release または rehearsal 実施時。
完了時は `docs/releases/RA-*.md` と `CHECKLISTS.md` を更新する。
