---
intent_id: INT-ADOPTION-TIERS
owner: docs-core
status: active
last_reviewed_at: 2026-07-01
next_review_due: 2026-08-01
---

# Adoption Guide

This guide explains how to apply workflow-cookbook adoption tiers to a downstream
repository.

## 1. Assess Current Tier

```bash
python tools/ci/check_adoption_tier.py --repo /path/to/repo --json
```

Use `--min-tier` with `--check` when a repository must meet a required tier.

```bash
python tools/ci/check_adoption_tier.py --repo /path/to/repo --min-tier 2 --check
```

## 2. Add Missing Tier Documents

Tier 1 starts with navigation and scope:

```bash
cp templates/HUB.codex.md.template /path/to/repo/HUB.codex.md
cp templates/BLUEPRINT.md.template /path/to/repo/BLUEPRINT.md
```

Tier 2 adds operational validation:

```bash
cp templates/RUNBOOK.md.template /path/to/repo/RUNBOOK.md
cp templates/GUARDRAILS.md.template /path/to/repo/GUARDRAILS.md
cp templates/EVALUATION.md.template /path/to/repo/EVALUATION.md
```

After copying templates, update front matter fields such as `intent_id`, `owner`,
`last_reviewed_at`, and `next_review_due`.

## 3. Check Template Drift

If downstream documents keep `template_version`, compare them with the current
template set:

```bash
python tools/ci/check_adoption_tier.py --repo /path/to/repo --check-drift --json
```

`--check --check-drift` returns non-zero when a tracked template version differs.

## 4. Batch Assessment

Use a JSON list when assessing several repositories:

```json
[
  "C:/Users/ryo-n/Codex_dev/workflow-cookbook",
  {"repo": "C:/Users/ryo-n/Codex_dev/agent-taskstate"}
]
```

```bash
python tools/ci/check_adoption_tier.py --repo-list repos.json --json
```

## 5. Review Cadence

- Tier 0-1: review every 6 months.
- Tier 2: review every 3 months.
- Tier 3: review monthly or follow each document's `next_review_due`.
