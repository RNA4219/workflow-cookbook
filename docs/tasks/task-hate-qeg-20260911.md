---
task_id: 20260911-01
intent_id: INT-WORKFLOW-HATE-QEG
owner: Codex
status: done
last_reviewed_at: 2026-09-11
next_review_due: 2026-12-11
---

# HATE・QEGによる自動テスト受入

依頼は「HATEとQEGを組み込んで実際にテストできてるか確認」。
[仕様](../contracts/hate-qeg-test-gate.md)に従い、再実行できるCLIと実行証跡を追加する。
既存の変更と過去のcompletion記録を保持する。

証跡保存先: workspaceの `research/workflow-hate-qeg-20260911/`。

完了条件: 実pytest→HATE正規化/export→QEG検証/判定/recordが追跡可能であり、
件数・coverage・判定・残る未評価範囲を記録していること。

[Acceptance AC-20260911-01](../acceptance/AC-20260911-01.md)で完了。
最終run-002は全976テスト成功、HATE→QEG全段成功、QEG Go、source前後一致。
