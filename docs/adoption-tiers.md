---
intent_id: INT-ADOPTION-TIERS
owner: docs-core
status: active
last_reviewed_at: 2026-07-01
next_review_due: 2026-08-01
---

# Workflow Cookbook Adoption Tiers

workflow-cookbook を派生 repo に導入する際の段階基準。

## Tier Summary

| Tier | Name | Required Files | Purpose | Example Repos |
|------|------|---------------|---------|---------------|
| 0 | Minimal | `README.md` | Entry point only | scratch repos |
| 1 | Foundation | + `HUB.codex.md`, `BLUEPRINT.md` | AI navigation + scope definition | agent-state-gate |
| 2 | Operational | + `RUNBOOK.md`, `GUARDRAILS.md`, `EVALUATION.md` | Execution + policy + acceptance | manual-bb-test-harness, code-to-gate |
| 3 | Full | + `docs/acceptance/`, `docs/tasks/`, `docs/birdseye/` | Traceability + knowledge map | workflow-cookbook, shipyard-cp |

---

## Tier 0: Minimal

### Purpose
- 最小限の README のみ
- プロトタイプ、実験、scratch repo 向け

### Required Files
- `README.md`

### Advancement Criteria
- プロジェクトが継続利用される場合 → Tier 1 へ
- 複数エージェント/メンバーが関わる場合 → Tier 1 へ

---

## Tier 1: Foundation

### Purpose
- AI エージェントが repo を自律的に読み解ける
- プロジェクトの目的・範囲が明示される

### Required Files

| File | Role |
|------|------|
| `README.md` | Entry point, task classifier |
| `HUB.codex.md` | AI navigation hub, document routing |
| `BLUEPRINT.md` | Problem statement, scope, I/O contract, constraints |

### File Locations
- **Option A**: Root level (推奨)
  ```
  repo/
  ├── README.md
  ├── HUB.codex.md
  └── BLUEPRINT.md
  ```
- **Option B**: `workflow-cookbook/` directory
  ```
  repo/
  └── workflow-cookbook/
      ├── README.md
      ├── HUB.codex.md
      └── BLUEPRINT.md
  ```

### Advancement Criteria
- 継続的な運用が必要 → Tier 2 へ
- CI/CD 導入が必要 → Tier 2 へ
- 複数メンバーでの作業 → Tier 2 へ

---

## Tier 2: Operational

### Purpose
- 実行手順と品質基準が明示される
- 変更時の行動指針が定義される
- 検収基準が追跡可能

### Required Files

| File | Role |
|------|------|
| (Tier 1 files) | |
| `RUNBOOK.md` | Execution procedures, validation commands |
| `GUARDRAILS.md` | Principles, behavior guidelines, constraints |
| `EVALUATION.md` | Acceptance criteria, KPIs, test outline |

### Additional Recommended Files
- `CHANGELOG.md` - Change history (Keep a Changelog format)
- `CHECKLISTS.md` - Release checklists
- `SPEC.md` - Implementation specification notes

### Advancement Criteria
- 複数リリースが必要 → Tier 3 へ
- Task 追跡が必要 → Tier 3 へ
- ドキュメント間の依存関係を可視化したい → Tier 3 へ

---

## Tier 3: Full

### Purpose
- 完全な traceability (Task → Acceptance → Release)
- ドキュメント知識マップ (Birdseye)
- インシデント追跡

### Required Files & Directories

| Path | Role |
|------|------|
| (Tier 2 files) | |
| `docs/acceptance/` | Acceptance records (`AC-YYYYMMDD-xx.md`) |
| `docs/tasks/` | Task seeds (`TASK.*-MM-DD-YYYY.md`) |
| `docs/birdseye/index.json` | Knowledge map node graph |
| `docs/birdseye/hot.json` | Hot list (primary entry points) |
| `docs/birdseye/caps/*.json` | Capsule summaries per document |

### Optional Files
- `docs/IN-*.md` - Incident logs
- `docs/releases/` - Release approval records
- `governance/policy.yaml` - Self-modification bounds, SLOs

---

## Tier Assessment

`tools/ci/check_adoption_tier.py` で現在の tier を自動判定できる。

```bash
# Single repo assessment
python tools/ci/check_adoption_tier.py --repo /path/to/repo

# Multi-repo assessment
python tools/ci/check_adoption_tier.py --repo-list repos.json

# JSON output
python tools/ci/check_adoption_tier.py --repo . --json
```

### Expected Output

```
Repo: manual-bb-test-harness
Current Tier: 2 (Operational)

Tier 0 files: ✅ README.md
Tier 1 files: ✅ HUB.codex.md, ✅ BLUEPRINT.md
Tier 2 files: ✅ RUNBOOK.md, ✅ GUARDRAILS.md, ✅ EVALUATION.md
Tier 3 files: ⚠️  docs/acceptance/ (exists but empty)
             ❌ docs/tasks/
             ❌ docs/birdseye/

Recommendation: Advance to Tier 3 by adding docs/tasks/ and docs/birdseye/
```

---

## Template Usage

各 tier の雛形は `templates/` ディレクトリに配置。

```bash
# Tier 1 setup
cp templates/HUB.codex.md.template ./HUB.codex.md
cp templates/BLUEPRINT.md.template ./BLUEPRINT.md

# Tier 2 setup
cp templates/RUNBOOK.md.template ./RUNBOOK.md
cp templates/GUARDRAILS.md.template ./GUARDRAILS.md
cp templates/EVALUATION.md.template ./EVALUATION.md
```

テンプレートの front matter (`intent_id`, `owner`, `status` 等) は導入時に必ず更新する。

---

## Drift Prevention

### Template Version Tracking

各テンプレートには `template_version` field を含める。
派生 repo の文書が template より古い場合、drift 警告を発する。

```bash
# Check for template drift
python tools/ci/check_adoption_tier.py --repo . --check-drift
```

### Cross-Repo Sync

workflow-cookbook の template が更新された場合、派生 repo へ通知する。

1. `docs/UPSTREAM.md` に upstream repo を記録
2. 週次で `extract_upstream_changes.py` を実行
3. 差分があれば Task Seed を起票

---

## Adoption Examples

### Example 1: New QA Skill Repo (Tier 2)

```bash
# Initial setup
git init my-qa-skill
cd my-qa-skill

# Copy Tier 1 templates
cp ~/workflow-cookbook/templates/HUB.codex.md.template ./HUB.codex.md
cp ~/workflow-cookbook/templates/BLUEPRINT.md.template ./BLUEPRINT.md

# Copy Tier 2 templates
cp ~/workflow-cookbook/templates/RUNBOOK.md.template ./RUNBOOK.md
cp ~/workflow-cookbook/templates/GUARDRAILS.md.template ./GUARDRAILS.md
cp ~/workflow-cookbook/templates/EVALUATION.md.template ./EVALUATION.md

# Edit front matter
sed -i 's/your-handle/my-handle/' *.md
sed -i 's/2025-10-14/2026-05-04/' *.md

# Verify adoption
python ~/workflow-cookbook/tools/ci/check_adoption_tier.py --repo .
```

### Example 2: Existing Repo Upgrade (Tier 1 → 2)

```bash
# Check current tier
python ~/workflow-cookbook/tools/ci/check_adoption_tier.py --repo .
# Output: Current Tier: 1 (Foundation)

# Add Tier 2 files
cp ~/workflow-cookbook/templates/RUNBOOK.md.template ./RUNBOOK.md
cp ~/workflow-cookbook/templates/GUARDRAILS.md.template ./GUARDRAILS.md
cp ~/workflow-cookbook/templates/EVALUATION.md.template ./EVALUATION.md

# Customize
# ... edit files ...

# Verify
python ~/workflow-cookbook/tools/ci/check_adoption_tier.py --repo .
# Output: Current Tier: 2 (Operational) ✅
```

---

## Maintenance

### Tier Review Schedule

- **Tier 0-1**: 6ヶ月毎の review で十分
- **Tier 2**: 3ヶ月毎の review 推奨
- **Tier 3**: 毎月 review (front matter の `next_review_due` に従う)

### Tier Downgrade

プロジェクトが終了/縮小する場合、tier を下げてよい。
その場合は `CHANGELOG.md` に記録し、不要ファイルを削除する。

---

## References

- `docs/adoption-guide.md` - 詳細な導入手順
- `templates/` - 各 tier の雛形
- `tools/ci/check_adoption_tier.py` - Tier 判定ツール
- `BLUEPRINT.md` - workflow-cookbook 自体の設計方針
- `RUNBOOK.md` - 運用手順
