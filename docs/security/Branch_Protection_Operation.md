---
intent_id: INT-SEC-008
owner: security
status: active
last_reviewed_at: 2026-04-17
next_review_due: 2026-05-17
---

# Branch Protection Operation

本ドキュメントは branch protection と
`docs/ci-config.md` / `governance/policy.yaml` の整合を
定期確認する運用導線を明文化する。

## 整合確認の目的

- GitHub 上の required check と docs の記述が一致している
- policy.yaml の論理 gate ID と実 check 名の対応が追える
- 例外や skip が branch protection で意図せず反映されない

## 整合確認手順

### 1. Branch Protection JSON 取得

```sh
gh api repos/RNA4219/workflow-cookbook/branches/main/protection > branch-protection.json
```

### 2. policy.yaml との照合

```sh
python tools/ci/check_branch_protection.py --protection-json branch-protection.json
```

- `governance/policy.yaml` の `ci.required_jobs` を読み込み
- `LOGICAL_TO_REPO_CHECK` mapping で concrete check 名へ変換
- branch protection JSON と照合

### 3. docs/ci-config.md との照合

```sh
python tools/ci/check_ci_gate_matrix.py
```

- workflow / docs / policy.yaml の整合を確認

### 4. 定期実行

月次または release 前に実行:

- branch protection JSON export
- check_branch_protection.py
- check_ci_gate_matrix.py
- 結果を docs または CI log に保存

## 2026-04-17 時点の live 設定

`main` には classic branch protection を適用済み。

- required checks
  - `governance`
  - `unit`
  - `Allowlist Guard`
  - `Semgrep`
  - `Bandit`
  - `Gitleaks`
  - `Dependency Audit & SBOM`
- strict status checks: 有効
- required pull request reviews: 1 approvals
- dismiss stale reviews: 有効
- required conversation resolution: 有効
- force pushes: 無効
- deletions: 無効

確認コマンド:

```sh
gh api repos/RNA4219/workflow-cookbook/branches/main/protection > branch-protection.json
python tools/ci/check_branch_protection.py --protection-json branch-protection.json
```

期待結果:

```text
Branch protection matches governance/policy.yaml logical gate IDs.
```

## 現行 required jobs 対応表

| 論理 gate ID | workflow | job/check 名 | branch protection での表示 |
| --- | --- | --- | --- |
| `governance-gate` | governance-gate.yml | `governance` | Governance Gate / governance |
| `python-ci` | test.yml | `unit` | test / unit |
| `security-ci` | security.yml | `Allowlist Guard`, `Semgrep`, `Bandit`, `Gitleaks`, `Dependency Audit & SBOM` | Security Gate / 各 job check |

**注**: `python-ci` は論理名。
downstream repo では caller 側 job 名を使ってよいが、
本 repo では `.github/workflows/test.yml` の `unit` を concrete check として扱う。

### 2026-04-17 の不整合と対応

`check_ci_gate_matrix.py` 実行で以下の不整合を発見:

1. **security.yml job 名**: 実際は `allowlist_guard`, `semgrep`, `bandit`, `gitleaks`, `dep_audit`
   - `security-ci` という単一 job は存在しないため、
     論理 gate ID から複数 concrete check へ対応づける必要があった

2. **tests.yml 不在**: `python-ci` の mapping で `.github/workflows/tests.yml` を参照していたが、
   実際は `.github/workflows/test.yml`（単数形）で `unit` を使うのが正しい

**対応**: `check_branch_protection.py` / `check_ci_gate_matrix.py` /
`docs/ci-config.md` を更新し、
`task-ci-gate-matrix-alignment-20260417.md` として記録した。

## 例外・変更の手順

### required check 追加

1. workflow / job を追加
2. `governance/policy.yaml` の `ci.required_jobs` へ論理 ID を追記
3. `docs/ci-config.md` の対応表を更新
4. branch protection で check を required に設定
5. `check_branch_protection.py` / `check_ci_gate_matrix.py` で検証

### required check 削除

1. policy.yaml から論理 ID を削除
2. docs/ci-config.md の対応表を更新
3. branch protection で check を optional に変更
4. 検証スクリプトで確認

### workflow / job 名変更

1. docs/ci-config.md の対応表を先に更新
2. workflow / job 名を変更
3. branch protection の check 名を更新
4. 検証スクリプトで確認

**禁止**: docs と branch protection のどちらかだけを変更する運用

## CI での自動検証（将来実装）

- workflow で branch protection JSON を定期 export
- check_branch_protection.py を実行
- 結果を artifact または comment で通知

## 関連資料

- [ci-config](../ci-config.md)
- [ci_phased_rollout_requirements](../ci_phased_rollout_requirements.md)
- [Security Exception Registry](./Security_Exception_Registry.md)
- [Enterprise Readiness Checklist](./Enterprise_Readiness_Checklist.md)
