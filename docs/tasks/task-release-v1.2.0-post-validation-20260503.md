---
task_id: 20260503-09
intent_id: INT-RELEASE-001
owner: docs-core
status: done
last_reviewed_at: 2026-05-03
next_review_due: 2026-06-03
---

# Task Seed: v1.2.0 Post-Release Validation

## Objective

公開済み `v1.2.0` release について、GitHub Release、git tag、CHANGELOG、
release docs、version consistency、release evidence、主要 docs gate、full test の整合を検収する。

## Scope

- In:
  - `v1.2.0` git tag
  - GitHub Release `v1.2.0`
  - `CHANGELOG.md`
  - `docs/releases/v1.2.0.md`
  - release / version / acceptance / completion trace / CI gate matrix checker
  - full pytest
- Out:
  - 新規 release 作成
  - release note 本文の再発行

## Acceptance

- [docs/acceptance/AC-20260503-09.md](../acceptance/AC-20260503-09.md)

## Notes

- 初回 full pytest では `tests/test_check_version_consistency.py` のリリース前提コメントが残っており 1 件失敗した。
- `v1.2.0` tag が存在するリリース後状態に合わせて期待値を更新し、再実行で 549 tests passed を確認した。
