---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-07-11
next_review_due: 2026-08-11
---

# CI 設定ガイド

## 現行の基準点

- gate 名の正本は `governance/policy.yaml` の `ci.required_jobs` とする。
- ここでの `required_jobs` は GitHub の生の check 名ではなく、論理 gate ID として扱う。
- 現行の required jobs は次の 4 つ。
  - `governance-gate`
  - `security-ci`
  - `python-ci`
  - `docs-gate`
- Phase ごとの考え方は [docs/ci_phased_rollout_requirements.md](./ci_phased_rollout_requirements.md)
  を参照する。

## 論理 gate ID と実 check 名

| `required_jobs` の値 | この repo での正本 workflow | この repo で確認する job / check | 備考 |
| --- | --- | --- | --- |
| `governance-gate` | `.github/workflows/governance-gate.yml` | job `governance` | GitHub 上では workflow 名 `Governance Gate` 配下の `governance` job を基準に確認する。 |
| `python-ci` | `.github/workflows/test.yml` | job `unit` | `python-ci` は論理名。downstream では caller 側 job 名を使ってよいが、本 repo では `test.yml` の `unit` job を concrete check として扱う。 |
| `security-ci` | `.github/workflows/security.yml` | `Allowlist Guard`, `Semgrep`, `Bandit`, `Gitleaks`, `Dependency Audit & SBOM` | `security-ci` は論理名。branch protection では `security.yml` の複数 job を concrete checks として扱う。 |
| `docs-gate` | `.github/workflows/markdown.yml` | job `docs-gate` | RG-002〜RG-005 の docs governance checker を集約。内部 steps は front matter, acceptance, birdseye, runbook slimming (RG-003), completion trace (RG-004), agent-tools-hub boundary (RG-005)。 |
| `metrics-contract-smoke` | `.github/workflows/markdown.yml` | job `metrics-contract-smoke` | Sample data validates only the metrics contract; it is not production evidence. |

## Python CI contract

The `.github/workflows/test.yml` workflow is fail-closed:

- `lint` runs pinned Ruff.
- `typecheck` runs strict mypy. Narrow legacy exceptions are enumerated in
  `pyproject.toml` and tracked in `TECH_DEBT.md`.
- `unit` is the stable required check name and enforces combined coverage for
  `tools` and `security_headers` at 80 percent.
- `python-312` verifies compatibility with the second supported runtime.
- `build` creates wheel and sdist artifacts, installs the wheel non-editably in
  a clean virtual environment, and runs `--help` for all console entrypoints.

Development dependencies are synchronized with
`uv sync --locked --extra dev`. CI must not use unpinned linter or type-checker
installs.

## Docs Gate 内部 checker 対応

| Gate ID | Checker | Status | 備考 |
| --- | --- | --- | --- |
| RG-002 | `check_birdseye_freshness.py` | warning → fail | `--max-verified-age-days` で鮮度判定。 |
| RG-003 | `check_runbook_slimming.py` | warning | 完了済み表肥大化を検出。 |
| RG-004 | `check_completion_trace.py` | warning (default) | `--require-acceptance-for-done` で error 昇格可。 |
| RG-005 | `check_agent_tools_hub_boundary.py` | warning | routing table 複製を検出。 |
| RG-006 | `check_task_completion_propagation.py` | warning | done Task Seedのcompletion-record未反映をnudge。 |
| RG-007 | `check_version_consistency.py` | warning | pyproject.toml / README badge / CHANGELOG / git tag / docs/releases 整合確認。 |

## Docs Gate Escalation Policy

checker ごとの stage は `governance/policy.yaml` の `ci.checker_stages` で定義。

### Stage 定義

| Stage | 挙動 | 目的 |
| --- | --- | --- |
| `observe` | 検出のみ、CI pass | 新規 checker の導入期間。実績収集。 |
| `warn` | stderr に警告出力、CI pass | 修正促進。担当者に通知。 |
| `enforce` | stderr にエラー出力、CI fail | 必須対応。merge block。 |

### 昇格条件

| 昇格パス | 条件 |
| --- | --- |
| `observe` → `warn` | 30日経過 + 検出率 < 5% (安定動作確認) |
| `warn` → `enforce` | 30日経過 + 検出率 < 1% (ほぼ解消) + 担当者同意 |

### Rollback 条件

| Rollback | 条件 |
| --- | --- |
| `enforce` → `warn` | 検出率 > 10% (大量誤検出) + 48h以内 |
| `warn` → `observe` | 検出率 > 20% (根本原因未解決) + 担当者判断 |

### 現行 Stage 状態

| Gate ID | Stage | 昇格履歴 |
| --- | --- | --- |
| RG-002 | enforce | 365日(warn) → 90日(enforce) 2026-05-03 |
| RG-003 | warn | 2026-05-02導入、observe期間終了 |
| RG-004 | warn | 2026-05-02導入、--require-acceptance-for-done で error 昇格可 |
| RG-005 | warn | 2026-05-02導入 |
| RG-006 | warn | 2026-05-03導入 |
| RG-007 | warn | 2026-05-03導入 (新規) |

### Stage 変更手順

1. `governance/policy.yaml` の `checker_stages` を更新
2. `docs/ci-config.md` の「現行 Stage 状態」を更新
3. 対応 workflow で `--check` フラグ調整 (enforce の場合は必須)
4. PR 作成、reviewer に stage 変更理由を明記

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
  `python-ci -> unit`、`security-ci -> Allowlist Guard / Semgrep / Bandit / Gitleaks / Dependency Audit & SBOM`
  を本 repo の期待値として照合する。
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
| `.github/workflows/test.yml` | lint / typecheck / unit / build / e2e 集約 | Phase 2 |
| `.github/workflows/security.yml` | 本 repo の Phase 3 正本セキュリティ gate 入口 | Phase 3 |
| `.github/workflows/security-ci.yml` | `security.yml` へ寄せるための互換入口 | Phase 3 互換 |
| `.github/workflows/codeql.yml` | 高コストな静的解析 | Phase 3 optional |
| `.github/workflows/links.yml` / `.github/workflows/markdown.yml` | docs 品質チェック | Phase 0-3 |

CI の個別構成は `.github/workflows/` ディレクトリ内の各 YAML を参照してください。
branch protection で required check を設定するときは、この表ではなく
「論理 gate ID と実 check 名」の表を正本に使います。
