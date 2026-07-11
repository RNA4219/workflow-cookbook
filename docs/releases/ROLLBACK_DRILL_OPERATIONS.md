---
intent_id: INT-OPS-001
owner: ops-core
status: active
last_reviewed_at: 2026-07-11
next_review_due: 2026-08-10
---

# Rollback Drill Operations Guide

このドキュメントは rollback drill の月次運用手順を定める。

## 実施タイミング

- **月次**: 毎月第3週の平日
- **Release 前**: major/minor release の 1週前
- **例外**: security incident 発生時は実施免除（実 rollback で代替）

## 担当

| 役割 | 担当 | 責務 |
| --- | --- | --- |
| Drill Owner | RNA4219 | drill 計画、実施、記録 |
| Verification Owner | RNA4219 | 事後確認、証跡レビュー |
| Notification Owner | RNA4219 | 事前通知、完了通知 |

## 通知フロー

### 事前通知（実施3日前）

- 通知先: repo owner、関連担当者
- 内容:
  - 実施日時
  - drill 範囲（version、環境）
  - 影響想定
- 形式: GitHub issue comment または Slack

### 完了通知（実施当日）

- 通知先: 同上
- 内容:
  - drill 結果（成功/失敗）
  - 証跡ファイル URL
  - lessons learned（改善点）
- 形式: GitHub issue comment + INDEX.md 更新

## 実施手順

### Pre-Drill Checklist

1. `docs/INCIDENT_TEMPLATE.md` で drill scenario 記録
2. 影響範囲確認（environment、dependencies）
3. 戻し先 version 確認（`git tag`、CHANGELOG.md）
4. 戻し先 version の動作確認（local test）

### Drill Execution

1. rollback 手順実行（RUNBOOK.md 参照）
2. 実施ログ記録（timestamp、command、result）
3. 環境反映確認（drill は local または staging）

### Post-Drill Verification

1. Security Gate 成功確認
2. QA metrics 閾値確認
3. acceptance criteria 確認
4. evidence links 確認

## 記録様式

### 記録ファイル

- `docs/releases/RB-YYYYMMDD-XX.md`
- template: `RB-20260419-01-sample-drill.md` 参照

### 必須項目

- rollback_id
- from_version / to_version
- triggered_at / completed_at
- triggered_by / verified_by
- trigger_reason（drill 明記）
- pre-rollback checklist
- execution log
- post-rollback verification
- lessons learned

### INDEX.md 更新

- `docs/releases/INDEX.md` の Rollback Events に追記

## Lessons Learned 整理

### 改善点分類

- **Process**: 手順の不備、記録様式の改善
- **Tool**: CI/ツールの不整合、自動化余地
- **Docs**: docs の不備、相互参照の強化

### Follow-up

- 改善点は `docs/tasks/` に Task Seed 作成
- critical は即時対応、minor は backlog

## 年次レビュー

- **タイミング**: 毎年 Q1
- **内容**:
  - drill 実施回数
  - success rate
  - lessons learned 集計
  - 手順改訂要否

## 参照

- [RUNBOOK.md Rollback Section](../../RUNBOOK.md#rollback--retry)
- [Rollback Drill Sample](RB-20260419-01-sample-drill.md)
- [Release Checklist](../Release_Checklist.md)
- [Enterprise Readiness Assessment](../reports/enterprise-readiness-assessment-20260417.md)
