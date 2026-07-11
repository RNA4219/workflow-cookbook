---
task_id: 20260712-04
intent_id: INT-CODE-TO-GATE-REFINEMENT
owner: docs-core
status: completed
last_reviewed_at: 2026-07-12
next_review_due: 2026-08-11
acceptance_id: AC-20260712-04
---

# Task Seed: Workflow Plugin Runtime Responsibility Split

## Objective

Separate runtime policy and trace data types plus Evidence serialization from plugin execution without changing runtime behavior.

## Scope

- In: runtime types, trace JSON/Evidence projection helpers, runtime compatibility delegates, and tests.
- Out: plugin config schema, capability method mapping, timeout/retry semantics, and Evidence schema.

## Requirements

- Preserve `tools.workflow_plugins.runtime` dataclass imports and method signatures.
- Preserve trace JSON, Evidence JSONL keys, ordering, timestamps, and hash inputs.
- Remove the Code-to-gate LARGE_MODULE finding without adding a suppression.

## Affected Paths

- `tools/workflow_plugins/runtime.py`
- `tools/workflow_plugins/runtime_types.py`
- `tools/workflow_plugins/runtime_evidence.py`
- `tests/test_workflow_plugin_runtime.py`

## Local Commands

```bash
uv run pytest tests/test_workflow_plugin_runtime.py -q
```

## Acceptance

- [AC-20260712-04](../acceptance/AC-20260712-04.md)
