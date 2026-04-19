---
report_id: REP-20260417-ENTERPRISE
owner: security
status: active
last_reviewed_at: 2026-04-19
next_review_due: 2026-05-19
---

# Enterprise Readiness Assessment (2026-04-19 Update)

## 結論

`workflow-cookbook` は、セキュリティ修正、CI 改善、supply chain docs、
branch protection live enforcement、rollback drill 記録により
企業利用で説明可能な品質・統制・証跡の基盤が整った。

現時点の総合判定は `B+` とする。

- 強み:
  危険実装への即応、supply chain docs 改善、branch protection live enforcement、
  rollback drill 記録、Runbook / Acceptance / Release docs の土台
- 弱み:
  定期演習の実証ログ蓄積、ruleset 移行の検討

## セクション別採点（2026-04-19 Update）

| セクション | 評価 | 根拠 |
| --- | --- | --- |
| Secure Coding | B | `collect_metrics.py` に URL 検証と危険スキーム拒否。失敗系テスト追加済み。repo 横断棚卸しは未確認。 |
| CI Security Gates | B+ | branch protection live enforcement 確認済み。required checks 7本が docs と整合。例外台帳あり。 |
| Supply Chain / Dependency Governance | B+ | SBOM 生成、Dependabot、脆弱性 SLA、例外台帳テンプレート追加済み。dev dependencies 固定化完了。lockfile 方針 docs 化済み。 |
| Release / Change Management | B+ | rollback drill 記録サンプル追加。承認テンプレート、INDEX 相互参照、rollback 手順 docs 化済み。 |
| Ops / Incident Readiness | B | incident template、sample incident、docs freshness CI あり。演習・定期レビュー証拠は弱い。 |
| Documentation / Auditability | A- | Task / Acceptance / Runbook / Release docs 相互参照整備。Enterprise checklist、例外台帳、API inventory、branch protection docs 揃った。 |

## 総合コメント

この repo は、「危険な実装を安全側へ直せる」「docs と CI を同期できる」
「rollback 手順を drill で実証できる」という意味で企業利用の基盤が整った。

次の強化余地:

1. 定期 rollback drill の月次実施と証跡蓄積
2. classic branch protection から ruleset 移行の判断
3. transitive dependencies の可視化

## 推奨アクション（2026-04-19）

1. rollback drill を月次または release 前に実施し、証跡蓄積
2. ~~dev dependencies 固定方針を docs 化~~ ✓ 完了（pyproject.toml + Dependency_Governance.md）
3. ruleset 移行可否を org 標準と整合判断

## 根拠メモ

### 1. Secure Coding

確認できたこと:

- `tools/perf/collect_metrics.py` で URL 検証が入り、
  `file://`、loopback、private network、link-local などが拒否される
- `tests/test_url_validation_security.py` に失敗系テストが追加されている

不足:

- URL / subprocess / file access の同種パターンを
  repo 横断で棚卸しした証跡は見当たらない

### 2. CI Security Gates

確認できたこと:

- `.github/workflows/security.yml` が存在する
- `Bandit` の `B310` は除外されていない
- `Dependabot` 設定ファイルがある
- `check_security_posture.py` / `check_release_evidence.py` が存在する
- `check_ci_gate_matrix.py` と `check_branch_protection.py` の mapping 不整合は修正された

不足:

- `nosec` / skip 例外の定期棚卸し台帳はない
- GitHub API で `main` branch protection を取得でき、
  classic branch protection に required checks が設定済み
- required checks は `governance`、`unit`、
  `Allowlist Guard`、`Semgrep`、`Bandit`、`Gitleaks`、
  `Dependency Audit & SBOM` の 7 本で live enforcement されている
- ruleset は未使用であり、将来的に org 標準へ寄せる余地はある

### 3. Supply Chain / Dependency Governance

確認できたこと:

- `requirements.txt` が固定入力として使われる
- `pyproject.toml` dev依存が `==` で固定（pytest, pytest-cov, bandit, pip-audit）
- SBOM 生成スクリプトと CI artifact upload が追加された
- `Dependabot` による pip 依存監視が入った
- 脆弱性 SLA と例外台帳テンプレートが docs 化された
- dev dependencies 固定方針が `Dependency_Governance.md` に記載

不足:

- transitive dependencies の可視化は今後の強化余地がある

### 4. Release / Change Management

確認できたこと:

- `Release_Checklist.md` がある
- `RUNBOOK.md` で release evidence / acceptance / CI gate の導線がある
- `docs/releases/` に履歴がある
- `RELEASE_APPROVAL_TEMPLATE.md` と `docs/releases/INDEX.md` が追加された
- acceptance / release evidence checker 側の関連付けが強化された

不足:

- rollback が docs 参照に留まり、
  実証ログや drill 記録までは確認できない
- 新しい承認テンプレートを用いた実運用ログはまだ少ない

### 5. Ops / Incident Readiness

確認できたこと:

- `RUNBOOK.md` に運用確認手順がある
- `docs/INCIDENT_TEMPLATE.md` と sample incident がある
- docs freshness / review due を確認する CI がある

不足:

- 定期演習の記録が見えにくい
- 監視項目はあるが、閾値逸脱時の責任分界や応答時間目標は弱い

## 詳細分析

この repo は、
「危険な実装を安全側へ直せる」
「docs と CI をある程度同期できる」
という意味では健全である。

一方で、企業利用で重視されるのは
「一度直した」ことではなく、
「次も同じ品質で回る」「例外が説明できる」「監査時に追える」
ことである。

その観点では、
次の 2 系統を優先して埋めるべきである。

1. Release / rollback / approval の実運用証跡の蓄積
2. branch protection の ruleset 移行可否と例外照合自動化の整理
3. transitive dependency 可視化（将来検討）

## 推奨アクション

1. release / rollback / approval の証跡テンプレートを実運用で 1 回以上回す
2. ~~dev dependencies 固定と transitive dependency 可視化の方針を追加する~~ ✓ dev固定完了
3. classic branch protection から ruleset へ移すかを決め、運用標準を一本化する

## 関連 Task Seed

- [task-enterprise-supply-chain-hardening-20260417](../tasks/task-enterprise-supply-chain-hardening-20260417.md)
- [task-enterprise-release-operations-evidence-20260417](../tasks/task-enterprise-release-operations-evidence-20260417.md)
- [task-enterprise-security-governance-hardening-20260417](../tasks/task-enterprise-security-governance-hardening-20260417.md)
- [task-ci-gate-matrix-alignment-20260417](../tasks/task-ci-gate-matrix-alignment-20260417.md)
- [task-branch-protection-enablement-20260417](../tasks/task-branch-protection-enablement-20260417.md)
