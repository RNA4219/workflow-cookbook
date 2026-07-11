---
task_id: 20260712-03
intent_id: INT-CODE-TO-GATE-REFINEMENT
owner: docs-core
status: active
last_reviewed_at: 2026-07-12
next_review_due: 2026-08-11
acceptance_id: AC-20260712-03
---

# Task Seed: Context Pack Resolver Responsibility Split

## Objective

Split context-pack configuration, graph signals, and ranking while retaining resolver and package import compatibility.

## Scope

- In: resolver facade, config coercion, graph signal computation, ranking, and compression imports.
- Out: context-pack config schema, score semantics, candidate ordering, and token budget defaults.

## Requirements

- Keep `tools.context.pack` and `tools.context.pack.resolver` exports stable.
- Preserve candidate selection, PPR, score values, and section-token behavior.
- Remove the Code-to-gate LARGE_MODULE finding without adding a suppression.

## Affected Paths

- `tools/context/pack/config.py`
- `tools/context/pack/signals.py`
- `tools/context/pack/ranking.py`
- `tools/context/pack/resolver.py`
- `tests/test_context_pack.py`

## Local Commands

```bash
uv run pytest tests/test_context_pack.py -q
```

## Acceptance

- [AC-20260712-03](../acceptance/AC-20260712-03.md)
