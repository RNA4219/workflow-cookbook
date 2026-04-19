---
intent_id: DOC-OPS-001
owner: ops-core
status: active
last_reviewed_at: 2026-04-17
next_review_due: 2026-05-17
---

# Releases Index

## Release Notes

| Version | Date | Summary | File |
| --- | --- | --- | --- |
| v1.2.0 | 2026-04-10 | Improvement Backlog completion | [v1.2.0.md](v1.2.0.md) |
| v1.1.3 | 2026-04-11 | Product readiness polish | [v1.1.3.md](v1.1.3.md) |
| v1.1.2 | 2026-04-09 | Security Gate improvements | [v1.1.2.md](v1.1.2.md) |
| v1.1.1 | 2026-04-09 | Markdown lint fixes | [v1.1.1.md](v1.1.1.md) |
| v1.1.0 | 2026-04-09 | Evidence / Codemap improvements | [v1.1.0.md](v1.1.0.md) |
| v1.0.0 | 2025-10-16 | Stable Template API | [v1.0.0.md](v1.0.0.md) |
| v0.1.0 | 2025-10-13 | Initial release | [v0.1.0.md](v0.1.0.md) |

## Release Approval Mapping

> Release Approval Record（RA-XXX）と Release Note（vX.Y.Z）の対応表

| Approval ID | Version | Status | Approved At | Approved By | File |
| --- | --- | --- | --- | --- | --- |
| <!-- RA-YYYYMMDD-XX --> | <!-- vX.Y.Z --> | <!-- draft or approved or deployed --> | <!-- YYYY-MM-DD --> | <!-- handle --> | <!-- [RA-XXX.md](RA-XXX.md) --> |

## Acceptance Mapping

> Release と Acceptance Record の相互参照

| Version | Acceptance IDs | Status |
| --- | --- | --- |
| v1.2.0 | AC-20260410-01, AC-20260410-02 | verified |
| v1.1.3 | AC-20260411-01〜08 | unreleased |

> 詳細は [docs/acceptance/INDEX.md](../acceptance/INDEX.md) を参照

## Rollback Events

> ロールバック履歴の概要

| Rolled Back At | From Version | To Version | Reason | Evidence |
| --- | --- | --- | --- | --- |
| 2026-04-19 | v1.3.0 (drill) | v1.2.0 | Rollback drill | [RB-20260419-01](RB-20260419-01-sample-drill.md) |

### Rollback Template Usage

- **ファイル**: [RB-20260419-01-sample-drill.md](RB-20260419-01-sample-drill.md)（サンプル）
- **作成手順**:
  1. Rollback 実施時に `RB-YYYYMMDD-XX.md` を作成
  2. Pre-Rollback Checklist 完了
  3. Rollback Execution Log 記録
  4. Post-Rollback Verification 実施
  5. INDEX.md の Rollback Events に追加

---

## Template Usage

### Release Approval Template

- **ファイル**: [RELEASE_APPROVAL_TEMPLATE.md](RELEASE_APPROVAL_TEMPLATE.md)
- **使用タイミング**: リリース準備開始時
- **使用者**: Release Manager
- **作成手順**:
  1. テンプレートをコピー → `RA-YYYYMMDD-XX.md` 作成
  2. Front Matter 設定（release_version, intent_id）
  3. Approval Checklist 完了
  4. 承認完了時に approved_at / approved_by 記入
  5. INDEX.md に追加

### Release Note Template

- 既存の `docs/releases/vX.Y.Z.md` はリリースノート（変更内容記録）
- Release Approval Record（RA-XXX）は承認プロセスの証跡
- 両者を相互参照させる

---

## Verification Commands

```bash
# Release evidence check
python tools/ci/check_release_evidence.py --check

# Acceptance check
python tools/ci/check_acceptance.py --check

# Security posture check
python tools/ci/check_security_posture.py --check --github-repo owner/name
```
