---
task_id: 20260702-01
intent_id: INT-READINESS-DOWNSTREAM-AUTOMATION
owner: docs-core
status: completed
last_reviewed_at: 2026-07-02
next_review_due: 2026-08-02
acceptance_id: AC-20260702-01
---

# Task Seed: Readiness and Downstream Automation Extensions

## Objective

Implement the next wave of workflow-cookbook automation extensions for release
readiness, security diffing, Birdseye remediation, CI rollout diagnosis,
adaptive improvement utilities, schema/sample/docs coverage, branch protection
weekly audit, and downstream onboarding.

## Scope

- In:
  - Release / Acceptance / Evidence readiness report
  - Security posture snapshot and diff
  - Birdseye remediation suggestions
  - CI Phase doctor
  - Adaptive improvement utilities
  - Schema/sample/docs sync matrix
  - Branch protection weekly audit artifacts
  - Downstream onboarding doctor
- Out:
  - GitHub settings mutation
  - External SaaS production changes
  - Automatic publication of generated skill drafts

## Requirements

- Release readiness can combine acceptance, evidence, release records, security
  posture JSON, and metrics JSON.
- Security posture can export current state and compare against a baseline.
- Birdseye freshness failures include actionable remediation commands.
- CI rollout phase can be diagnosed from policy, workflows, and docs.
- Adaptive improvement operations can export reviewed memory, generate draft
  skill records, and build recall responses without storing unreviewed memory.
- Sample/docs sync can validate schema/sample/docs coverage as a matrix.
- Branch protection validation can generate weekly report, nudge, and Task Seed
  draft artifacts.
- Downstream onboarding can combine Adoption Tier with CI onboarding signals.
- Five-tool validation blockers are resolved or recorded in the same Task /
  Acceptance trace before release approval.
- Console script smoke tests work in isolated HATE real-repo environments where
  `pip` is not installed in the active Python environment.

## Affected Paths

- `tools/ci/generate_evidence_report.py`
- `tools/ci/check_security_posture.py`
- `tools/ci/check_birdseye_freshness.py`
- `tools/ci/check_ci_phase_doctor.py`
- `tools/ci/self_improvement_ops.py`
- `tools/ci/check_sample_docs_sync.py`
- `tools/ci/check_branch_protection.py`
- `tools/ci/check_downstream_onboarding.py`
- `tools/audit/verify_log_chain.py`
- `tests/`
- `README.md`
- `RUNBOOK.md`
- `CHANGELOG.md`

## Local Commands

```bash
uv run pytest tests/test_generate_evidence_report.py tests/test_check_security_posture.py tests/test_check_birdseye_freshness.py tests/test_check_ci_phase_doctor.py tests/test_self_improvement_ops.py tests/test_check_sample_docs_sync.py tests/test_check_branch_protection.py tests/test_check_downstream_onboarding.py -q
```

## Acceptance

- [AC-20260702-01](../acceptance/AC-20260702-01.md)
