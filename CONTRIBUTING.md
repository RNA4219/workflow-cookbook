---
intent_id: DOC-CONTRIBUTING
owner: RNA4219
status: active
last_reviewed_at: 2026-07-11
next_review_due: 2026-10-11
---

# Contributing

Thank you for improving Workflow Cookbook.

## Development Setup

1. Fork and clone the repository.
2. Install Python 3.11 or 3.12 and `uv`.
3. Run `uv sync --locked --extra dev`.
4. Create a focused branch from `main`.

## Change Workflow

- Define the objective and scope in a Task Seed under `docs/tasks/`.
- Add or update tests before changing behavior.
- Preserve existing `python tools/...` and console-script interfaces.
- Keep documentation, schema, samples, and runtime behavior synchronized.
- Record verification in `docs/acceptance/AC-YYYYMMDD-xx.md`.
- Add user-visible changes to the Unreleased section of `CHANGELOG.md`.

## Required Checks

Run the following before opening a pull request:

```sh
uv run ruff check .
uv run mypy .
uv run pytest -q --cov=tools --cov=security_headers --cov-fail-under=80
uv build
python tools/ci/check_ci_gate_matrix.py
python tools/ci/check_birdseye_freshness.py --check
```

Pull requests must describe intent, risk, rollback, Priority Score, and the linked
Acceptance Record. Small, independently reviewable changes are preferred.
