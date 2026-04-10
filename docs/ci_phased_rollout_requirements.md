---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-04-09
next_review_due: 2026-05-09
---

# CI 段階的導入 要件定義

## 1. 目的

- Workflow Cookbook 本体と派生リポジトリに対して、CI を一度に最大化するのではなく、
  Phase 0 から Phase 3 まで段階的に強化する。
- `governance/policy.yaml`、GitHub Actions workflow、Runbook、Checklist の
  記述差異をなくし、どのジョブが gate でどのジョブが拡張枠かを明確にする。
- 「存在しない想定ジョブ」を書くのではなく、実在する workflow と reusable workflow を
  起点に導入判断できる状態を作る。

## 2. スコープ

- 対象
  - 本 repo の `.github/workflows/*.yml`
  - `.github/workflows/reusable/*.yml`
  - `governance/policy.yaml`
  - `docs/ci-config.md`
  - `RUNBOOK.md` / `CHECKLISTS.md` / `EVALUATION.md`
- 非対象
  - デプロイ自動化
  - 本番環境の runtime health check
  - 外部 SaaS 固有のビルド設定

## 3. 設計原則

- Phase 定義は workflow 名と job 名に直接対応づける。
- gate 判定の正本は `governance/policy.yaml` の `ci.required_jobs` とする。
- `ci.required_jobs` は論理 gate ID とし、GitHub 上の実 check 名への対応は
  `docs/ci-config.md` の対応表で管理する。
- branch protection の必須チェック名は、GitHub 上で表示される job 名と一致させる。
- docs-only 変更時の軽量化は許容するが、skip 条件は workflow 内へ明示する。

## 4. 現行 CI 資産

### 4.1 Gate 系

- `governance-gate.yml`
  - job 名: `governance`
  - 役割: code change に対する governance rule と INT metrics 集約
- `reusable/python-ci.yml`
  - job 名: `ci`
  - 表示名: `Python CI`
  - 役割: Ruff と Pytest による Python 基本ゲート
- `security.yml`
  - job 名: `security-ci`
  - 役割: reusable security gate を本 repo 側で呼び出す Phase 3 正本入口
- `security-ci.yml`
  - job 名: `security-ci`
  - 役割: reusable security gate の互換入口

### 4.2 補助系

- `tests.yml`
  - 役割: Phase 1 の最小 Python テストゲート
- `test.yml`
  - 役割: Phase 2 の lint / typecheck / unit / build / e2e 集約ゲート
- `codeql.yml`
  - 役割: 高コストな静的解析
- `links.yml` / `markdown.yml`
  - 役割: ドキュメント品質の補助チェック

## 5. フェーズ別要件

### Phase 0

- 目的
  - ドキュメント中心の repo 立ち上げ時に、最低限の安全策だけを有効にする。
- 必須
  - `governance-gate.yml`
  - `links.yml`
  - `markdown.yml`
- 任意
  - `tests.yml`
- 昇格条件
  - `README.md` と `docs/ci-config.md` に採用 workflow を記載済み
  - `CHECKLISTS.md` の日次確認項目に CI 結果確認が存在する

### Phase 1

- 目的
  - Python ベースの最小回帰と governance gate を常時有効化する。
- 必須
  - `governance-gate`
  - `python-ci`
- 代表実装
  - reusable: `.github/workflows/reusable/python-ci.yml`
  - repo 内補助構成: `.github/workflows/tests.yml`
- 任意
  - `test.yml`
  - `security.yml`
- 昇格条件
  - `governance/policy.yaml` の `ci.required_jobs` に `governance-gate` と
    `python-ci` が登録されている
  - `python-ci` の論理名と GitHub 上の check 名対応が `docs/ci-config.md` に記載されている
  - Python の最低限回帰テスト群が安定して通る

### Phase 2

- 目的
  - lint / typecheck / unit / build / e2e をまとめた開発者向け集約ゲートを有効にする。
- 必須
  - Phase 1 の必須ジョブ
  - `test.yml` の導入有無を repo ごとに判断し、導入時は branch protection と整合させる
- 代表実装
  - `.github/workflows/test.yml`
- 任意
  - `codeql.yml`
  - `security.yml`
- 昇格条件
  - `RUNBOOK.md` に CI 実行手順と失敗時確認手順がある
  - `CHANGELOG.md` に CI 拡張が記録されている
  - 実行時間の増加が受容可能である

### Phase 3

- 目的
  - セキュリティ系 workflow を gate に昇格し、GA 相当の統制にする。
- 必須
  - `governance-gate`
  - `python-ci`
  - `security-ci`
- 代表実装
  - `.github/workflows/security.yml`
  - `.github/workflows/reusable/security-ci.yml`
- 任意
  - `.github/workflows/security-ci.yml`
  - `codeql.yml`
  - 高コストな追加監査ジョブ
- 昇格条件
  - `governance/policy.yaml` の `ci.required_jobs` と branch protection が一致
  - `security-ci` の論理名が `.github/workflows/security.yml` の job `security-ci`
    に対応づいている
  - `EVALUATION.md` の受入基準と矛盾しない
  - セキュリティ gate の失敗時運用が `RUNBOOK.md` から辿れる

## 6. `policy.yaml` との対応

- `governance/policy.yaml` の現行 required jobs は次の 3 つである。
  - `governance-gate`
  - `security-ci`
  - `python-ci`
- 本 repo では、これらを Phase 1 から Phase 3 へ順に有効化する論理 gate ID の正本として扱う。
- branch protection や運用確認で使う実 check 名は `docs/ci-config.md` の対応表に従う。
- workflow 名、job 名、branch protection 表示名がずれる場合は、
  まず対応表を更新し、そのうえで必要なら policy の論理名見直しを行う。

## 7. フェーズ移行プロセス

1. Phase 変更候補を `CHANGELOG.md` の `[Unreleased]` へ記録する。
2. `.github/workflows/` と `governance/policy.yaml` の対応を確認する。
3. `docs/ci-config.md`、`RUNBOOK.md`、`CHECKLISTS.md`、必要なら
   `EVALUATION.md` を更新する。
4. branch protection 上の必須チェック名を見直す。
5. 移行後の失敗パターンと rollback 手順を確認する。

## 8. ロールバック方針

- CI 導入により開発速度や安定性へ大きな悪影響が出た場合は、直前 Phase の required jobs へ戻す。
- rollback 時は次を同時に更新する。
  - `governance/policy.yaml`
  - `docs/ci-config.md`
  - `CHANGELOG.md`
- docs と branch protection のどちらかだけを戻す運用は禁止する。

## 9. トレーサビリティ

- Phase と workflow の対応は、本書と `.github/workflows/*` の Phase コメントで追跡する。
- gate の正本は `governance/policy.yaml` に置く。
- branch protection の export は `tools/ci/check_branch_protection.py` で検証できる状態を保つ。
- 受入観点は `EVALUATION.md`、運用観点は `RUNBOOK.md`、日次確認は `CHECKLISTS.md` で補完する。

## 10. 今後の具体化候補

### 実装済み

- branch protection の exported 設定との差分検証
  - `tools/ci/check_branch_protection.py` で実装済み (v1.1.0)
  - policy.yaml と branch protection の整合を CI で検証可能

### 未実装 / backlog 移管

- codeql.yml を Phase 3 の必須へ昇格するかどうかの判断基準整備
  - 現状: Phase 3 任意 (高コストな静的解析)
  - backlog: `docs/addenda/P_Expansion_Candidates.md` へ移管検討
