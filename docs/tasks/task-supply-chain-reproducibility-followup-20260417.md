---
task_id: 20260417-07
intent_id: INT-SEC-012
owner: security
status: done
last_reviewed_at: 2026-04-17
next_review_due: 2026-05-17
---

# Task Seed: Supply Chain Reproducibility Follow-up

## 背景

- `requirements.txt`、SBOM、Dependabot、脆弱性 SLA まで導入され、
  supply chain governance は前進した。
- 一方で、dev dependencies の固定化、
  transitive dependencies の可視化、
  例外台帳の定期レビューは次の残課題である。

## ゴール

- dependency audit の再現性をもう一段上げ、
  supply chain 監査で説明しやすい状態にする。

## 実施対象

1. dev dependencies を固定するか、固定しない場合の方針を docs 化する
2. transitive dependencies を SBOM か補助ドキュメントで可視化する
3. dependency exception のレビュー周期を RUNBOOK / CHECKLISTS に反映する
4. 可能なら CI で例外レビュー期限を確認する導線を追加する

## 完了条件

- dev dependencies の扱いが説明できる
- transitive dependencies の見方が docs から追える
- 例外レビューの周期が RUNBOOK / CHECKLISTS に紐づく

## 参照

- [RUNBOOK](../../RUNBOOK.md)
- [Enterprise Readiness Checklist](../security/Enterprise_Readiness_Checklist.md)
- [Enterprise Readiness Assessment](../reports/enterprise-readiness-assessment-20260417.md)

## 例外理由

Acceptance record 不要: Operational Readiness Backlog の項目。
実施タイミングは依存更新方針見直し時または quarterly review。
完了時は `docs/security/Dependency_Governance.md` と `CHECKLISTS.md` を更新する。
