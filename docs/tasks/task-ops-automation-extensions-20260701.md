---
task_id: 20260701-01
intent_id: INT-OPS-AUTOMATION-EXTENSIONS
owner: docs-core
status: completed
last_reviewed_at: 2026-07-01
next_review_due: 2026-08-01
acceptance_id: AC-20260701-01
---

# Task Seed: Operations Automation Extensions

## Objective

Implement the next set of workflow-cookbook automation extensions identified from
the requirements and expansion docs.

## Scope

- In:
  - Docs review due automation
  - Adoption tier assessment
  - Workflow plugin runtime timeout and trace diagnostics
  - Stable CLI entrypoint smoke coverage
  - Metrics regression detection
- Out:
  - GitHub branch protection changes
  - External SaaS production settings
  - Remote publication of plugin trace Evidence

## Requirements

- `check_docs_review_due.py` can emit owner grouping, PeriodicNudge JSON,
  Task Seed drafts, and review update plans without breaking existing `--check`.
- Downstream repositories can be assessed against Tier 0-3 adoption criteria.
- Tier 1/2 templates and an adoption guide exist.
- Workflow plugin runtime can return on timeout through thread isolation and
  export trace JSON / Evidence JSON Lines.
- `wfc-codemap-update` and `wfc-context-pack` have console script smoke tests.
- Metrics threshold checker can compare current metrics against a previous-good
  baseline and report regressions.

## Affected Paths

- `tools/ci/check_docs_review_due.py`
- `tools/ci/check_adoption_tier.py`
- `tools/ci/check_metrics_thresholds.py`
- `tools/workflow_plugins/runtime.py`
- `templates/`
- `docs/adoption-guide.md`
- `docs/adoption-tiers.md`
- `tests/`
- `README.md`
- `RUNBOOK.md`
- `CHANGELOG.md`

## Local Commands

```bash
uv run pytest tests/test_check_docs_review_due.py tests/test_check_adoption_tier.py tests/test_workflow_plugin_runtime.py tests/test_check_metrics_thresholds.py tests/test_cli_entrypoints.py -q
```

## Acceptance

- [AC-20260701-01](../acceptance/AC-20260701-01.md)

## Notes

- Remote publication of plugin trace Evidence remains a follow-up; this task adds
  local Evidence JSON Lines that can be attached to release or evidence records.
