---
task_id: 20260711-01
intent_id: INT-OSS-HARDENING-20260711
owner: RNA4219
status: completed
last_reviewed_at: 2026-07-11
next_review_due: 2026-08-11
acceptance_id: AC-20260711-01
---

# Task Seed: OSS distribution and CI hardening

## Objective

Make workflow-cookbook reproducible, installable as a normal Python package, and
honest about CI, metrics, security, and release evidence.

## Scope

- Pin development tools and run Ruff, strict mypy, pytest coverage, and package
  build as fail-closed CI jobs.
- Remove synthetic metrics evidence from CI.
- Add OSS governance documents and remove tracked bytecode.
- Add repository-root and metrics-config controls to public CLIs.
- Pin GitHub Actions to immutable commit SHAs and reduce workflow permissions.
- Add a non-editable wheel smoke test for every console entrypoint.
- Record the migration debt and refresh Birdseye.

## Constraints

- Preserve direct `python tools/...` execution.
- Preserve the required GitHub check name `unit`.
- Keep combined `tools` and `security_headers` coverage at 80 percent.
- Do not publish to PyPI or change GitHub repository settings.

## Local Commands

```sh
uv sync --locked --extra dev
uv run ruff check .
uv run mypy .
uv run pytest -q --cov=tools --cov=security_headers --cov-fail-under=80
uv build
python tools/ci/smoke_wheel.py dist/*.whl
```

## Acceptance

- [AC-20260711-01](../acceptance/AC-20260711-01.md)
