---
task_id: 20260417-01
intent_id: INT-SEC-003
owner: security
status: done
last_reviewed_at: 2026-04-17
next_review_due: 2026-05-17
completed_at: 2026-04-17
---

# Task Seed: Enterprise Supply Chain Hardening

## 背景

- `requirements.txt` と `pip-audit` により、
  dependency audit の再現性は改善した。
- ただし、lockfile、SBOM、脆弱性対応ルールの明文化が不足しており、
  供給網説明責任はまだ弱い。

## ゴール

- 依存関係の現在地、更新方法、監査方法、例外運用を
  第三者へ説明できる状態にする。

## 修正対象

1. lockfile または同等の再現性担保方式を決める
2. SBOM 生成方法を追加する
3. dependency update / vulnerability triage の運用方針を docs 化する
4. CI で最低限の supply chain 証跡を残せるようにする

## 完了条件

- lock / fixed input の方針が docs と CI で整合する
- SBOM 生成手順または自動化がある
- 脆弱性対応の担当・期限・例外運用が説明できる

## 参照

- [Enterprise Readiness Checklist](../security/Enterprise_Readiness_Checklist.md)
- [Enterprise Readiness Assessment](../reports/enterprise-readiness-assessment-20260417.md)
