---
rollback_id: RB-20260421-01
from_version: v1.1.3
to_version: v1.1.2
status: completed
triggered_at: 2026-04-21T10:00:00Z
completed_at: 2026-04-21T10:45:00Z
triggered_by: RNA4219
verified_by: RNA4219
last_reviewed_at: 2026-04-21
next_review_due: 2026-05-21
---

# Rollback Evidence Record (Monthly Drill)

> **注**: これは月次 rollback drill の実施記録です。
> 実際の rollback ではなく、手順実証と証跡記録を目的とします。

## Rollback Summary

| 項目 | 内容 |
| --- | --- |
| Rollback ID | RB-20260421-01 |
| From Version | v1.1.3 |
| To Version | v1.1.2 |
| Triggered At | 2026-04-21 10:00 UTC |
| Completed At | 2026-04-21 10:45 UTC |
| Triggered By | RNA4219 |
| Verification Status | passed |

## Trigger Reason

**Scenario**: Monthly rollback drill (第3週実施)

- KPI閾値逸脱: なし（drill）
- Security Gate: なし（drill）
- 主要機能障害: なし（drill）
- 受入基準未達: なし（drill）

> **Drill Purpose**: rollback 手順の月次実証、ROLLBACK_DRILL_OPERATIONS.md 手順確認

## Pre-Rollback Checklist

- [x] インシデントテンプレート（`docs/INCIDENT_TEMPLATE.md`）で概要記録（drill scenario）
- [x] 影響ユーザ/システム特定: local environment only（drill）
- [x] CHANGELOG.md で前回安定版確認: v1.1.2
- [x] `git tag` で戻し先コミット特定: v1.1.2
- [x] 戻し先バージョンの動作確認完了（local checkout）

## Rollback Execution Log

| Step | Action | Status | Timestamp |
| --- | --- | --- | --- |
| 1 | 影響範囲確認 | completed | 2026-04-21T10:05:00Z |
| 2 | 戻し先バージョン決定（v1.1.2） | completed | 2026-04-21T10:10:00Z |
| 3 | `git checkout v1.1.2`（local drill） | completed | 2026-04-21T10:15:00Z |
| 4 | デプロイ環境反映（drill: skip） | skipped | drill |
| 5 | 主要フロー動作確認 | completed | 2026-04-21T10:30:00Z |
| 6 | メトリクス復旧確認 | completed | 2026-04-21T10:40:00Z |
| 7 | `git checkout main`（drill restore） | completed | 2026-04-21T10:45:00Z |

## Post-Rollback Verification

### Functional Verification

| Check | Result | Notes |
| --- | --- | --- |
| Security Gate | passed | workflow-cookbook Security Gate 全 job 成功（最新 CI 確認） |
| QA Metrics | passed | thresholds within baseline |
| Acceptance Criteria | passed | EVALUATION.md criteria verified |
| Main Workflow | passed | governance-gate.yml 正常動作確認 |

### Evidence Links

| Type | URL | Status |
| --- | --- | --- |
| Rollback PR | N/A (drill) | drill |
| Incident Record | docs/INCIDENT_TEMPLATE.md (drill scenario) | draft |
| Release Approval | docs/releases/INDEX.md | updated |
| CI Run Log | gh run list (latest) | verified |

## Lessons Learned

### Process Improvements

1. ROLLBACK_DRILL_OPERATIONS.md 手順が実証可能であることを確認
2. 事前通知フローを docs に明文化済み
3. 記録様式 template（RB-20260419-01-sample-drill.md）が適用可能

### Recommendations

1. 月次実施タイミングを第3週に固定（ROLLBACK_DRILL_OPERATIONS.md）
2. drill 結果を INDEX.md に追記する運用を継続
3. rollback 後の再リリース手順も月次 drill で確認

## References

- [ROLLBACK_DRILL_OPERATIONS.md](ROLLBACK_DRILL_OPERATIONS.md)
- [RUNBOOK.md Rollback Section](../../RUNBOOK.md#rollback--retry)
- [Release Checklist](../Release_Checklist.md)
- [Release Approval Template](RELEASE_APPROVAL_TEMPLATE.md)
- [Releases INDEX](INDEX.md)
- [Enterprise Readiness Assessment](../reports/enterprise-readiness-assessment-20260417.md)

## Approval

| 項目 | 内容 |
| --- | --- |
| Verified By | RNA4219 |
| Verified At | 2026-04-21T10:45:00Z |
| Verification Type | monthly drill |
| Next Drill Due | 2026-05-21（第3週） |
