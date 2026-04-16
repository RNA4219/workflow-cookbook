---
task_id: 20260417-04
intent_id: INT-SEC-009
owner: docs-core
status: done
last_reviewed_at: 2026-04-17
next_review_due: 2026-05-17
---

# Task Seed: CI Gate Matrix Alignment

## 背景

- `check_ci_gate_matrix.py` 実行により、
  `python-ci` と `security-ci` の concrete check mapping が
  workflow 実態とずれていることが分かった。
- `python-ci` は存在しない `tests.yml` と `pytest` を参照していた。
- `security-ci` は単一 job `security-ci` 前提だったが、
  実際の `security.yml` は複数 job で構成されている。

## 実施内容

1. `python-ci` の workflow mapping を `.github/workflows/test.yml` に修正した
2. `python-ci` の concrete check を `unit` に修正した
3. `security-ci` を複数 concrete check
   (`Allowlist Guard`, `Semgrep`, `Bandit`, `Gitleaks`, `Dependency Audit & SBOM`)
   として扱うように修正した
4. `docs/ci-config.md` を実態に合わせて更新した
5. checker tests を更新した

## 完了条件

- `py -3 tools/ci/check_ci_gate_matrix.py` が通る
- `py -3 -m pytest tests/test_check_branch_protection.py tests/test_check_ci_gate_matrix.py` が通る
- `docs/ci-config.md` の対応表が workflow 実態と一致する

## 参照

- [Branch Protection Operation](../security/Branch_Protection_Operation.md)
- [ci-config](../ci-config.md)
