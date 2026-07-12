---
intent_id: TASK-SELF-IMPROVEMENT-GATE-EFFECTIVENESS-20260712
owner: workflow-cookbook
status: completed
last_reviewed_at: 2026-07-12
next_review_due: 2026-08-11
---

# Task Seed: Self-improvement Gate Effectiveness

## Objective

`self-improvement/v1`の契約、閾値、review-only Gate解析、repo責務正本を実装する。

## Invariants

- Gate削除、policy緩和、Skill公開を自動実行しない。
- hard-safety Gateをarchive候補にしない。
- 詳細repo責務は`governance/repo-responsibilities.yaml`だけを正本とする。

## Acceptance Criteria

- schema/sample/docs/Birdseye、golden shipyard export、閾値境界をテストする。
- `insufficient_data`をproposal対象外にし、actionを4値へ限定する。
- pytest全件と責務manifest CIを通す。
- 30日shadow運用はAcceptanceで追跡する。
