---
task_id: 20260503-05
intent_id: INT-IMPROVEMENT-006
owner: docs-core
status: completed
last_reviewed_at: 2026-05-03
next_review_due: 2026-06-03
---

# Task Seed: Next Improvement Implementation

## 指示文

`workflow-cookbook` の `BLUEPRINT.md`、`docs/requirements.md`、`docs/spec.md` に追加された
先行改善案を、既存の互換性を壊さず小さな実装単位へ落としてください。

優先順位は次の通りです。

1. version consistency checker
   - README badge、`pyproject.toml`、`CHANGELOG.md`、`docs/releases/*.md`、git tag の整合を確認する。
   - 既存 `tools/ci/check_release_evidence.py` を拡張するか、小さな checker を追加する。
2. stable CLI entrypoint
   - 既存 `python tools/...` 入口を維持したまま、package entrypoint の薄い wrapper を検討する。
   - 追加する場合は既存 script と同等結果を返す smoke test を入れる。
3. docs gate escalation policy
   - `docs/ci-config.md` または `governance/policy.yaml` で checker ごとの stage を追えるようにする。
   - `observe` / `warn` / `enforce` の昇格条件と rollback 条件を明記する。
4. plugin capability catalog
   - `tools/workflow_plugins/README.md`、schema、runtime、sample config の capability 定義を照合できる形にする。
5. large module split policy
   - `TECH_DEBT_REGISTER.md` の分割計画に沿って、互換 wrapper を残す前提で最小分割から進める。

## 制約

- 既存 `python tools/...` 入口を削除しない。
- Agent_tools 全体の repo routing は `agent-tools-hub` の責務とし、`workflow-cookbook` へ複製しない。
- 実装する場合は、最小 failing test を先に置き、関連 docs と `CHANGELOG.md` を同時に更新する。
- 完了時は acceptance record を追加し、RUNBOOK には短い結果サマリだけを残す。

## 検証候補

- `uv run python tools/ci/check_front_matter.py --check`
- `uv run python tools/ci/check_acceptance.py --check`
- `uv run python tools/ci/check_ci_gate_matrix.py`
- 追加 checker の unit test
