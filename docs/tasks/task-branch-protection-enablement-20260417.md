---
task_id: 20260417-05
intent_id: INT-SEC-010
owner: security
status: done
last_reviewed_at: 2026-04-17
next_review_due: 2026-05-17
---

# Task Seed: Branch Protection Enablement

## 背景

- `check_ci_gate_matrix.py` の mapping は workflow 実態に合わせて修正済み。
- 当初は GitHub API で
  `gh api repos/RNA4219/workflow-cookbook/branches/main/protection`
  を取得すると `404 Branch not protected` が返っていた。
- 2026-04-17 に classic branch protection を live 設定し、
  required checks と PR review 条件の適用を確認した。

## ゴール

- `governance/policy.yaml` の論理 gate ID に対応する concrete checks を、
  GitHub 側でも required checks として強制できる状態にする。

## 実施対象

1. `main` に classic branch protection または ruleset を設定する
2. required checks を以下に揃える
   - `governance`
   - `unit`
   - `Allowlist Guard`
   - `Semgrep`
   - `Bandit`
   - `Gitleaks`
   - `Dependency Audit & SBOM`
3. `check_branch_protection.py --protection-json <json>` が通ることを確認する
4. docs と設定差分があれば `docs/ci-config.md` / `Branch_Protection_Operation.md` を同期する

## 完了条件

- `gh api repos/RNA4219/workflow-cookbook/branches/main/protection` もしくは
  ruleset export で required checks が確認できる
- `check_branch_protection.py` が live export に対して通る
- docs の対応表と GitHub 側設定が一致する

## 実施結果

- 2026-04-17 に `main` へ classic branch protection を設定
- required checks を以下の 7 本に統一
  - `governance`
  - `unit`
  - `Allowlist Guard`
  - `Semgrep`
  - `Bandit`
  - `Gitleaks`
  - `Dependency Audit & SBOM`
- 追加設定
  - `strict: true`
  - PR review 必須: 1 approvals
  - stale review dismiss: 有効
  - conversation resolution: 必須
  - force push / deletion: 無効
- `py -3 tools/ci/check_branch_protection.py --protection-json branch-protection.json`
  で live export に対して整合確認済み

## 参照

- [ci-config](../ci-config.md)
- [Branch Protection Operation](../security/Branch_Protection_Operation.md)
- [Enterprise Readiness Assessment](../reports/enterprise-readiness-assessment-20260417.md)
