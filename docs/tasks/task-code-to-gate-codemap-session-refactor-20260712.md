---
task_id: 20260712-02
intent_id: INT-CODE-TO-GATE-REFINEMENT
owner: docs-core
status: completed
last_reviewed_at: 2026-07-12
next_review_due: 2026-08-11
acceptance_id: AC-20260712-02
---

# Task Seed: Codemap Session Responsibility Split

## Objective

Move generated-at serial allocation out of the Birdseye update session and remove its duplicate root-planning methods.

## Scope

- In: Codemap session, serial allocation, builder imports, and direct unit tests.
- Out: Birdseye JSON schema, update CLI options, and target-selection behavior.

## Requirements

- Keep `tools.codemap.update` exports for serial allocation and generated-at values.
- Make `BirdseyeRootBuilder` the sole implementation of focus, hot, and capsule planning.
- Remove the Code-to-gate LARGE_MODULE finding without adding a suppression.

## Affected Paths

- `tools/codemap/update/session.py`
- `tools/codemap/update/serial.py`
- `tools/codemap/update/capsule.py`
- `tests/test_codemap_update.py`

## Local Commands

```bash
uv run pytest tests/test_codemap_update.py -q
```

## Acceptance

- [AC-20260712-02](../acceptance/AC-20260712-02.md)
