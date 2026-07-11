---
task_id: 20260712-01
intent_id: INT-CODE-TO-GATE-REFINEMENT
owner: docs-core
status: active
last_reviewed_at: 2026-07-12
next_review_due: 2026-08-11
acceptance_id: AC-20260712-01
---

# Task Seed: Docs Review Checker Responsibility Split

## Objective

Split the docs review due checker without changing its CLI, report, JSON artifacts, or exit behavior.

## Scope

- In: `tools/ci/check_docs_review_due.py` and its new internal package.
- Out: review-date policy and generated artifact schemas.

## Requirements

- Preserve `python -m tools.ci.check_docs_review_due` and existing private compatibility imports.
- Separate review-status data, Markdown scanning, artifact rendering, and CLI orchestration.
- Remove the Code-to-gate LARGE_MODULE finding without adding a suppression.

## Affected Paths

- `tools/ci/check_docs_review_due.py`
- `tools/ci/docs_review_due/`
- `tests/test_check_docs_review_due.py`

## Local Commands

```bash
uv run pytest tests/test_check_docs_review_due.py -q
```

## Acceptance

- [AC-20260712-01](../acceptance/AC-20260712-01.md)
