---
release_approval_id: RA-YYYYMMDD-XX
release_version: vX.Y.Z
intent_id: INT-XXX
status: draft  # draft|approved|deployed|rolled_back
owner: your-handle
approved_at: null
approved_by: null
last_reviewed_at: 2026-04-17
next_review_due: 2026-05-17
---

# Release Approval Record Template

> **使用タイミング**: リリース準備開始時にこのテンプレートをコピーし、
> `docs/releases/RA-YYYYMMDD-XX.md` として作成する。
> **使用者**: Release Manager または担当者

## Approval Summary

| 項目 | 内容 |
| --- | --- |
| Release Version | vX.Y.Z |
| Release Type | <!-- major or minor or patch or hotfix --> |
| Approved By | <!-- 承認者名/ハンドル --> |
| Approved At | <!-- YYYY-MM-DD HH:MM UTC --> |
| Approval Type | <!-- technical or security or risk_acceptance --> |
| Evidence Links | <!-- PR URL, QA結果, Security Gate --> |

## Approval Checklist

- [ ] Security Gate 全項目成功（`.github/workflows/security.yml`）
- [ ] QAメトリクス閾値達成（`check_metrics_thresholds.py --check`）
- [ ] CHANGELOG 通番整合確認
- [ ] 配布物 LICENSE/NOTICE 同梱確認
- [ ] Security Review Checklist レビューフェーズ完了
- [ ] 受入基準（`EVALUATION.md#acceptance-criteria`）達成
- [ ] Acceptance Record（`docs/acceptance/AC-*.md`）verified
- [ ] Release Checklist 全項目完了

## Risk Assessment

| 項目 | 内容 |
| --- | --- |
| Risk Level | <!-- low or medium or high --> |
| Risk Description | <!-- リスク内容 --> |
| Mitigation Measures | <!-- 緩和措置 --> |
| Rollback Plan | <!-- RUNBOOK.md 参照 --> |

## Evidence

### Pre-Release

| 種別 | URL / Path | Status |
| --- | --- | --- |
| QA Evidence | <!-- URL --> | <!-- pass/fail --> |
| Security Gate Log | <!-- URL --> | <!-- pass/fail --> |
| Metrics Baseline | <!-- .ga/qa-metrics.json --> | <!-- pass/fail --> |
| Acceptance Record | <!-- docs/acceptance/AC-*.md --> | <!-- verified/pending --> |

### Post-Release

| 種別 | URL / Path | Status |
| --- | --- | --- |
| Deployment Log | <!-- URL --> | <!-- success/failure --> |
| Verification Result | <!-- 確認結果 --> | <!-- pass/fail --> |

## References

- Intent: <!-- INT-XXX -->
- PR: <!-- #XXX -->
- CHANGELOG: <!-- [vX.Y.Z](../../CHANGELOG.md) -->
- Release Note: <!-- [docs/releases/vX.Y.Z.md](vX.Y.Z.md) -->
- Acceptance: <!-- [docs/acceptance/AC-YYYYMMDD-XX.md](../acceptance/AC-*.md) -->

## Rollback History

<!-- ロールバック発生時のみ記録 -->

| 項目 | 内容 |
| --- | --- |
| Rolled Back At | <!-- YYYY-MM-DD HH:MM UTC --> |
| From Version | <!-- vX.Y.Z --> |
| To Version | <!-- vX.Y.W --> |
| Reason | <!-- 原因 --> |
| Evidence URL | <!-- PR/インシデント URL --> |
| Verified At | <!-- 確認日時 --> |

---

## 使用フロー

1. **リリース準備開始**: テンプレートをコピー → RA-YYYYMMDD-XX.md 作成
2. **Approval Checklist 完了**: 各項目を確認しチェック
3. **承認完了**: approved_at / approved_by 記入、status → approved
4. **リリース実行**: status → deployed
5. **ロールバック発生**: Rollback History 記録、status → rolled_back
