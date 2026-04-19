---
rollback_id: RB-20260419-01
from_version: v1.3.0
to_version: v1.2.0
status: completed
triggered_at: 2026-04-19T10:30:00Z
completed_at: 2026-04-19T11:15:00Z
triggered_by: RNA4219
verified_by: RNA4219
last_reviewed_at: 2026-04-19
next_review_due: 2026-06-19
---

# Rollback Evidence Record (Sample Drill)

> **注**: これは rollback drill のサンプル記録です。
> 実際の rollback ではこの形式で証跡を残します。

## Rollback Summary

| 項目 | 内容 |
| --- | --- |
| Rollback ID | RB-20260419-01 |
| From Version | v1.3.0 |
| To Version | v1.2.0 |
| Triggered At | 2026-04-19 10:30 UTC |
| Completed At | 2026-04-19 11:15 UTC |
| Triggered By | RNA4219 |
| Verification Status | passed |

## Trigger Reason

**Scenario**: Sample rollback drill to verify rollback process

- KPI閾値逸脱: なし（drill）
- Security Gate: なし（drill）
- 主要機能障害: なし（drill）
- 受入基準未達: なし（drill）

> **Drill Purpose**: rollback 手順と証跡記録の実証確認

## Pre-Rollback Checklist

- [x] インシデントテンプレート（`docs/INCIDENT_TEMPLATE.md`）で概要記録
- [x] 影響ユーザ/システム特定
- [x] CHANGELOG.md で前回安定版確認: v1.2.0
- [x] `git tag` で戻し先コミット特定: v1.2.0
- [x] 戻し先バージョンの動作確認完了

## Rollback Execution Log

| Step | Action | Status | Timestamp |
| --- | --- | --- | --- |
| 1 | 影響範囲確認 | completed | 2026-04-19T10:35:00Z |
| 2 | 戻し先バージョン決定 | completed | 2026-04-19T10:40:00Z |
| 3 | `git checkout v1.2.0` | completed | 2026-04-19T10:45:00Z |
| 4 | デプロイ環境反映（drill: skip） | skipped | drill |
| 5 | 主要フロー動作確認 | completed | 2026-04-19T11:00:00Z |
| 6 | メトリクス復旧確認 | completed | 2026-04-19T11:10:00Z |
| 7 | インシデントサマリ更新 | completed | 2026-04-19T11:15:00Z |

## Post-Rollback Verification

### Functional Verification

| Check | Result | Notes |
| --- | --- | --- |
| Security Gate | passed | workflow-cookbook Security Gate 全 job 成功 |
| QA Metrics | passed | thresholds within baseline |
| Acceptance Criteria | passed | EVALUATION.md criteria verified |
| Main Workflow | passed | governance-gate.yml 正常動作 |

### Evidence Links

| Type | URL | Status |
| --- | --- | --- |
| Rollback PR | N/A (drill) | drill |
| Incident Record | docs/INCIDENT_TEMPLATE.md (sample) | draft |
| Release Approval | docs/releases/RA-YYYYMMDD-XX (sample) | updated |
| CI Run Log | N/A (drill) | drill |

## Lessons Learned

### Process Improvements

1. rollback 手順の実証を RUNBOOK.md で明文化済み
2. 証跡記録様式を docs/releases/RELEASE_APPROVAL_TEMPLATE.md で定義済み
3. INDEX.md に Rollback Events セクション追加済み

### Recommendations

1. 定期的な rollback drill（月次または release 前）
2. rollback 時の通知導線の明文化
3. rollback 後の再リリース手順の標準化

## References

- [RUNBOOK.md Rollback Section](../../RUNBOOK.md#rollback--retry)
- [Release Checklist](../Release_Checklist.md)
- [Release Approval Template](RELEASE_APPROVAL_TEMPLATE.md)
- [Releases INDEX](INDEX.md)
- [Enterprise Readiness Assessment](../reports/enterprise-readiness-assessment-20260417.md)

## Approval

| 項目 | 内容 |
| --- | --- |
| Verified By | RNA4219 |
| Verified At | 2026-04-19T11:15:00Z |
| Verification Type | drill |
| Next Drill Due | 2026-06-19 |