---
intent_id: INT-SEC-013
owner: security
status: active
last_reviewed_at: 2026-07-11
next_review_due: 2026-08-10
---

# Ruleset Migration Analysis

本ドキュメントは classic branch protection から GitHub Rulesets への移行可否を分析する。

## 現状

- **Classic Branch Protection**: `main` branch に適用済み
- **Rulesets**: 未使用（repo-level、org-level ともに空）

### Classic Branch Protection 設定（2026-04-19）

| 設定項目 | 現状 |
| --- | --- |
| required_status_checks | governance, unit, Allowlist Guard, Semgrep, Bandit, Gitleaks, Dependency Audit & SBOM（7 checks） |
| strict status checks | 有効 |
| required_pull_request_reviews | 1 approval, dismiss stale reviews 有効 |
| required_conversation_resolution | 有効 |
| enforce_admins | 無効 |
| allow_force_pushes | 無効 |
| allow_deletions | 無効 |

## Rulesets vs Classic Branch Protection

### Rulesets の利点

| 特徴 | Classic | Ruleset |
| --- | --- | --- |
| **Scope** | branch 単位 | branch pattern（wildcard 可）、repo 単位、org 単位 |
| **Bypass actors** | enforce_admins で on/off | actor 単位で granular に bypass 設定 |
| **Merge queue** | 未対応 | 対応（merge 順序制御） |
| **Deployment 要求** | 未対応 | 対応（deploy 成功後 merge 可） |
| **組織標準適用** | repo ごと設定 | org-level ruleset で一括適用可 |

### Rulesets の制限

| 特徴 | Classic | Ruleset |
| --- | --- | --- |
| **API endpoint** | `/branches/{branch}/protection` | `/rulesets` |
| **既存ツール互換性** | 高 | docs 更新必要（check_branch_protection.py 等） |
| **個人 repo** | 自由設定 | org-level は Enterprise plan 必要 |
| **GitHub Pages** | 一部制限あり | 同様 |

### 現状との整合性

- **required checks**: Ruleset で同等設定可（`required_status_checks` rule）
- **PR reviews**: Ruleset で同等設定可（`pull_request` rule）
- **conversation resolution**: Ruleset で同等設定可
- **force push / deletion**: Ruleset で同等設定可（`deletion` rule）

## 移行判断基準

### 移行メリットがある条件

1. **org 標準適用**: 複数 repo で共通設定を一元管理したい
2. **wildcard branch**: `release/*`, `feature/*` 等 pattern で保護したい
3. **granular bypass**: admin 以外の特定 actor に bypass 設定したい
4. **merge queue**: merge 順序制御が必要

### 移行しない理由（現状維持）

1. **個人 repo / small team**: org-level ruleset は Enterprise plan 必要
2. **ツール互換性**: `check_branch_protection.py` 等 docs 更新コスト
3. **設定数**: 1 branch protection で十分
4. **リスク**: 移行中の設定不整合

## 推奨判断

### 現状（2026-04-19）

- **判定**: **Classic Branch Protection 維持**
- **理由**:
  1. 本 repo は個人管理、org-level ruleset のメリット享受不可
  2. 1 branch（main）のみ保護、wildcard 不要
  3. 既存ツール（check_branch_protection.py）との整合確認済み
  4. 移行コスト > メリット

### 将来移行条件

以下の条件で移行検討:

1. **org 標準**: 組織で複数 repo の branch protection を一元管理する必要発生
2. **Enterprise plan**: org が Enterprise plan に upgrade
3. **multi-branch**: `release/*`, `hotfix/*` 等 wildcard 保護が必要
4. **ツール更新**: `check_branch_protection.py` が Ruleset API 対応完了

## 移行手順（将来実施時）

### Phase 1: 設定確認

1. classic branch protection JSON export
2. Ruleset 用 JSON schema に変換
3. docs との整合確認

### Phase 2: 移行実施

1. repo-level ruleset 作成
2. classic branch protection 削除
3. CI で ruleset 設定確認

### Phase 3: docs 更新

1. `docs/security/Branch_Protection_Operation.md` 更新
2. `check_branch_protection.py` Ruleset API 対応
3. `docs/ci-config.md` 更新

### Phase 4: 検証

1. PR 作成で ruleset 動作確認
2. bypass actor 設定確認
3. CI gate 結果確認

## 関連資料

- [Branch Protection Operation](./Branch_Protection_Operation.md)
- [Enterprise Readiness Checklist](./Enterprise_Readiness_Checklist.md)
- [GitHub Rulesets Documentation](https://docs.github.com/rest/repos/rules)
- [GitHub Branch Protection Documentation](https://docs.github.com/rest/repos/branches)
