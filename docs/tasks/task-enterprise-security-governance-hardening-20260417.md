---
task_id: 20260417-03
intent_id: INT-SEC-005
owner: security
status: done
last_reviewed_at: 2026-04-17
next_review_due: 2026-05-17
---

# Task Seed: Enterprise Security Governance Hardening

## 背景

- `collect_metrics.py` の SSRF 対策や
  `Bandit` の見直しにより、直近の弱点には対処できた。
- ただし、repo 横断の危険パターン棚卸し、
  `nosec` / skip の定期棚卸し、
  branch protection 実証などの governance 面は残っている。

## ゴール

- 個別修正で終わらず、
  危険パターンの横断管理と例外管理を継続できる状態にする。

## 実施内容

1. **Security Exception Registry 作成**: nosec / skip / suppress の台帳を定義
   - [Security_Exception_Registry.md](../security/Security_Exception_Registry.md)
2. **Security API Inventory 作成**: URL / subprocess / file access の棚卸し結果
   - [Security_API_Inventory.md](../security/Security_API_Inventory.md)
3. **Branch Protection Operation 作成**: branch protection 整合確認の運用導線
   - [Branch_Protection_Operation.md](../security/Branch_Protection_Operation.md)
4. **相互参照整備**: Enterprise Checklist / Security Review Checklist に新ドキュメント参照を追加

## 修正対象

1. URL / subprocess / file access の横断棚卸しを行う
2. `nosec` / skip / suppress の台帳を作る
3. branch protection と required checks の検証証跡を定期化する
4. 例外承認と見直し期限の運用を docs 化する

## 完了条件

- 主要な危険 API の棚卸し結果が残る
- suppress の理由と期限を追える
- branch protection と docs の整合を定期確認できる

## 参照

- [Security Review Checklist](../security/Security_Review_Checklist.md)
- [Enterprise Readiness Checklist](../security/Enterprise_Readiness_Checklist.md)
- [Enterprise Readiness Assessment](../reports/enterprise-readiness-assessment-20260417.md)
- [Security Exception Registry](../security/Security_Exception_Registry.md)
- [Security API Inventory](../security/Security_API_Inventory.md)
- [Branch Protection Operation](../security/Branch_Protection_Operation.md)
