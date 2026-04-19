---
task_id: 20260417-02
intent_id: INT-SEC-004
owner: docs-core
status: done
last_reviewed_at: 2026-04-19
next_review_due: 2026-05-19
completed_at: 2026-04-19
---

# Task Seed: Enterprise Release Operations Evidence

## 背景

- release checklist、runbook、release evidence checker は既に存在する。
- 一方で、rollback 実証、承認証跡、例外受容記録など、
  企業監査で問われやすい運用証跡はまだ弱い。

## ゴール

- release / approval / rollback を
  docs 参照だけでなく、証跡として残せる状態にする。

## 実施内容

1. **Rollback Evidence Record Template**: サンプル drill 記録を作成
   - [RB-20260419-01-sample-drill.md](../releases/RB-20260419-01-sample-drill.md)
2. **INDEX.md Rollback Events 更新**: drill 記録を追加、template usage 追記
3. **RUNBOOK.md rollback 手順**: 既存の rollback 手順と証跡記録様式が整備済み
4. **Release Approval Template**: Rollback History セクション既存

## 修正対象

1. release 承認記録のテンプレートを定義する ✓（既存）
2. rollback 実施手順に「確認項目」と「実施証跡」を追加する ✓（既存）
3. release evidence と acceptance record の相互参照を強化する ✓（既存）
4. 必要なら docs/releases 用の記録様式を補強する ✓（追加）

## 完了条件

- 誰が承認したかを残せる ✓
- rollback の実施・確認結果を記録できる ✓
- release note、acceptance、checker の整合が説明できる ✓

## 参照

- [Release Checklist](../Release_Checklist.md)
- [RUNBOOK](../../RUNBOOK.md)
- [Enterprise Readiness Assessment](../reports/enterprise-readiness-assessment-20260417.md)
