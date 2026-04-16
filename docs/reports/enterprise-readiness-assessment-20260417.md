---
report_id: REP-20260417-ENTERPRISE
owner: security
status: active
last_reviewed_at: 2026-04-17
next_review_due: 2026-05-17
---

# Enterprise Readiness Assessment (2026-04-17)

## 結論

`workflow-cookbook` は、
セキュリティ修正と CI 改善により
「危険な穴が明確に残っている状態」からは前進した。
ただし、現時点では
「上場企業が継続利用するソフトとして十分に説明可能な品質水準」
と断言するには証拠が不足している。

現時点の総合判定は `B` とする。

- 強み:
  危険実装への即応、supply chain docs の改善、
  Runbook / Acceptance / Release docs の土台
- 弱み:
  branch ruleset 不在、
  例外管理と rollback 運用の実証不足

## セクション別採点

| セクション | 評価 | 根拠 |
| --- | --- | --- |
| Secure Coding | B | `collect_metrics.py` に URL 検証と危険スキーム拒否が入った。失敗系テストも追加済み。一方で、同種パターンの repo 横断棚卸しは未確認。 |
| CI Security Gates | B | `Bandit` / dependency audit / CI gate mapping docs 改善に加え、`main` に classic branch protection を設定し、required checks の live enforcement を確認した。ruleset への移行や例外台帳の自動照合は今後の強化余地。 |
| Supply Chain / Dependency Governance | B- | `requirements.txt` の固定入力、CycloneDX SBOM 生成、Dependabot、脆弱性 SLA、例外台帳テンプレートが追加され、C+ から前進した。 |
| Release / Change Management | B | `CHECKLISTS.md`、`Release_Checklist.md`、`RUNBOOK.md`、`docs/releases/` に加え、承認記録テンプレートと rollback 証跡導線が追加され、B- から前進した。 |
| Ops / Incident Readiness | B | `RUNBOOK.md`、incident template、sample incident、docs freshness 系チェックがある。日常運用の入口は揃っているが、演習・定期レビューの証拠は弱い。 |
| Documentation / Auditability | B+ | Task / Acceptance / Runbook / Release docs の導線に加え、Enterprise 判定チェックリスト、承認テンプレート、例外台帳、API inventory、branch protection operation docs が揃った。 |

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
- SBOM 生成スクリプトと CI artifact upload が追加された
- `Dependabot` による pip 依存監視が入った
- 脆弱性 SLA と例外台帳テンプレートが docs 化された

不足:
- lockfile がない
- dev dependencies の固定化はまだ弱い
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

## 総合コメント

この repo は、
「危険な実装を安全側へ直せる」
「docs と CI をある程度同期できる」
という意味では健全である。

一方で、企業利用で重視されるのは
「一度直した」ことではなく、
「次も同じ品質で回る」「例外が説明できる」「監査時に追える」
ことである。

その観点では、
次の 3 系統を優先して埋めるべきである。

1. Release / rollback / approval の実運用証跡の蓄積
2. Supply chain の再現性強化（dev 依存固定、transitive 可視化）
3. branch protection の ruleset 移行可否と例外照合自動化の整理

## 推奨アクション

1. release / rollback / approval の証跡テンプレートを実運用で 1 回以上回す
2. dev dependencies 固定と transitive dependency 可視化の方針を追加する
3. classic branch protection から ruleset へ移すかを決め、運用標準を一本化する

## 関連 Task Seed

- [task-enterprise-supply-chain-hardening-20260417](../tasks/task-enterprise-supply-chain-hardening-20260417.md)
- [task-enterprise-release-operations-evidence-20260417](../tasks/task-enterprise-release-operations-evidence-20260417.md)
- [task-enterprise-security-governance-hardening-20260417](../tasks/task-enterprise-security-governance-hardening-20260417.md)
- [task-ci-gate-matrix-alignment-20260417](../tasks/task-ci-gate-matrix-alignment-20260417.md)
- [task-branch-protection-enablement-20260417](../tasks/task-branch-protection-enablement-20260417.md)
