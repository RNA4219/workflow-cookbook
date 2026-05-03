---
task_id: 20260503-02
intent_id: INT-IMPROVEMENT-003
owner: RNA4219
status: completed
last_reviewed_at: 2026-05-03
next_review_due: 2026-06-03
---

# Task Seed: Task Seed Propagation Checker

## 背景

完了したTask Seedがcompletion-recordに未反映の場合をnudge検出。

## 完了条件

- `tools/ci/check_task_completion_propagation.py` 実装
- tests 13件以上パス

## Acceptance

- [AC-20260503-02](../acceptance/AC-20260503-02.md)

## 判定

完了