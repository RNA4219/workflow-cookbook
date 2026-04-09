---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-04-09
next_review_due: 2026-05-09
---

# CI 設定ガイド

## 現行の基準点

- gate 名の正本は `governance/policy.yaml` の `ci.required_jobs` とする。
- ここでの `required_jobs` は GitHub の生の check 名ではなく、論理 gate ID として扱う。
- 現行の required jobs は次の 3 つ。
  - `governance-gate`
  - `security-ci`
  - `python-ci`
- Phase ごとの考え方は [docs/ci_phased_rollout_requirements.md](./ci_phased_rollout_requirements.md)
  を参照する。

## 論理 gate ID と実 check 名

| `required_jobs` の値 | この repo での正本 workflow | この repo で確認する job / check | 備考 |
| --- | --- | --- | --- |
| `governance-gate` | `.github/workflows/governance-gate.yml` | job `governance` | GitHub 上では workflow 名 `Governance Gate` 配下の `governance` job を基準に確認する。 |
| `python-ci` | `.github/workflows/reusable/python-ci.yml` | downstream では caller 側で付ける job 名を正本とする。本 repo では補助的に `.github/workflows/tests.yml` の `pytest` と reusable の `Python CI` を参照する。 | `python-ci` は論理名であり、本 repo では reusable workflow 自体が主契約。 |
| `security-ci` | `.github/workflows/security.yml` | job `security-ci` | `.github/workflows/security-ci.yml` は互換入口として残すが、branch protection で優先して参照する正本は `security.yml` 側とする。 |

## Branch Protection 検証

- GitHub API から branch protection JSON を取得できる環境では、
  `python tools/ci/check_branch_protection.py --protection-json <json>` を実行して、
  `governance/policy.yaml` の論理 gate ID と実 check 名対応を検証する。
- repo 内の `policy.yaml` / workflow / `docs/ci-config.md` の整合は、
  `python tools/ci/check_ci_gate_matrix.py` で検証する。
- 例:

```sh
gh api repos/RNA4219/workflow-cookbook/branches/main/protection > branch-protection.json
python tools/ci/check_branch_protection.py --protection-json branch-protection.json
python tools/ci/check_ci_gate_matrix.py
```

- このスクリプトは `governance-gate -> governance`、
  `python-ci -> pytest`、`security-ci -> security-ci` を本 repo の期待値として照合する。
- downstream repo で異なる check 名を採用する場合は、対応表と運用文書を合わせて更新する。

## 自動キャンセル設定

すべての GitHub Actions ワークフローには、
同一ブランチ/PR 上で最新の実行だけを保持するための `concurrency` ブロックを追加しています。

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true
```

- `group` はワークフロー名と PR 番号（またはブランチ名）の組み合わせで定義します。
  これにより PR と push のどちらでも古い実行をまとめてキャンセルできます。
- `cancel-in-progress: true` により、新しい Run が開始されたタイミングで進行中の古い Run を自動的に停止します。

## Workflow 対応表

| Workflow | 主用途 | Phase |
| --- | --- | --- |
| `.github/workflows/governance-gate.yml` | governance rule と INT metrics 集約 | Phase 0-3 |
| `.github/workflows/reusable/python-ci.yml` | 派生リポ向け Python baseline gate | Phase 1-2 |
| `.github/workflows/tests.yml` | 本 repo の最小 Python テスト | Phase 1 補助 |
| `.github/workflows/test.yml` | lint / typecheck / unit / build / e2e 集約 | Phase 2 |
| `.github/workflows/security.yml` | 本 repo の Phase 3 正本セキュリティ gate 入口 | Phase 3 |
| `.github/workflows/security-ci.yml` | `security.yml` へ寄せるための互換入口 | Phase 3 互換 |
| `.github/workflows/codeql.yml` | 高コストな静的解析 | Phase 3 optional |
| `.github/workflows/links.yml` / `.github/workflows/markdown.yml` | docs 品質チェック | Phase 0-3 |

CI の個別構成は `.github/workflows/` ディレクトリ内の各 YAML を参照してください。
branch protection で required check を設定するときは、この表ではなく
「論理 gate ID と実 check 名」の表を正本に使います。
