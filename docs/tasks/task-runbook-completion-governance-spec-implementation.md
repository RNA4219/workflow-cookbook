---
task_id: 20260502-01
intent_id: INT-001
owner: docs-core
status: planned
last_reviewed_at: 2026-05-02
next_review_due: 2026-06-02
---

# Task Seed: RUNBOOK Completion Governance Spec Implementation

## 背景

`docs/requirements.md` と `docs/spec.md` に、次の 3 つの仕様が追加された。

1. RUNBOOK slimming
2. Task / Acceptance / Completion trace
3. Agent-tools-hub boundary

このタスクは、仕様を実装へ落とすためのエージェント向け指示文書である。
ただし、repo 選定や Agent_tools 全体のルーティングは `agent-tools-hub` の責務であり、
`workflow-cookbook` 側では重複実装しない。

## ゴール

- RUNBOOK が完了済み詳細表で肥大化する変更を検出できるようにする。
- `docs/tasks/*.md`、`docs/acceptance/*.md`、`docs/completion-record.md` の
  完了トレースを確認できるようにする。
- `workflow-cookbook/HUB.codex.md` が `agent-tools-hub` の横断 routing table を
  複製していないことを確認できるようにする。

## 非ゴール

- `agent-tools-hub` の repo map、Skill routing、Agent_tools 全体の入口整理を
  `workflow-cookbook` 側へ移植しない。
- 他 repo の詳細な使い分け表を `workflow-cookbook/HUB.codex.md` に追加しない。
- 本タスクでは仕様の実装のみを扱い、完了済み履歴の棚卸しや過去 acceptance の再判定は行わない。

## 修正対象候補

実装時は、既存ツールを優先して拡張する。

1. RUNBOOK slimming check
   - 候補: `tools/ci/` 配下に新規 checker を追加する、または既存 docs checker へ統合する。
   - 入力: `RUNBOOK.md`, `docs/completion-record.md`, `docs/tasks/*.md`, `docs/acceptance/*.md`
   - 出力: warning / failure summary

2. Task / Acceptance / Completion trace
   - 候補: `tools/ci/check_task_acceptance_sync.py` の拡張。
   - `status: done` の task が acceptance record または例外理由を持つか確認する。
   - `docs/completion-record.md` の各項目が task / acceptance / release / changelog の
     いずれかへリンクしているか確認する。

3. Agent-tools-hub boundary check
   - 候補: docs lint として `HUB.codex.md` を確認する。
   - `workflow-cookbook/HUB.codex.md` が Agent_tools 全体の repo routing table を
     複製していないことを検出する。
   - 横断 repo 選定が必要な記述では `agent-tools-hub` 参照へ誘導する。

4. Docs
   - `RUNBOOK.md`
   - `docs/completion-record.md`
   - `docs/requirements.md`
   - `docs/spec.md`
   - 必要なら `README.md` または `HUB.codex.md`

## 実装順

1. 既存 checker の有無を確認する。
   - `tools/ci/check_task_acceptance_sync.py`
   - docs review / markdown / acceptance 系 checker
2. 仕様に対する最小 failing test を追加する。
3. 既存 checker を拡張できる場合は拡張する。
4. 拡張が不自然な場合のみ、小さな専用 checker を追加する。
5. RUNBOOK / completion record / HUB の文書を、checker の期待に合わせて最小更新する。
6. `CHANGELOG.md` の `[Unreleased]` に実装差分を追記する。

## TDD / 検証

### RUNBOOK slimming

- RUNBOOK に完了済み詳細表が追加された fixture で warning になること。
- RUNBOOK に `docs/completion-record.md` への短い参照だけがある場合は pass すること。
- RUNBOOK の incident / rollback / release 実行手順は false positive にしないこと。

### Task / Acceptance / Completion trace

- `status: done` の task に acceptance 参照がある場合は pass すること。
- `status: done` の task に acceptance 参照も例外理由もない場合は warning または failure になること。
- completion record の項目に正本リンクがない場合は failure になること。
- completion record が task / acceptance / release / changelog のいずれかへリンクしている場合は pass すること。

### Agent-tools-hub boundary

- `workflow-cookbook/HUB.codex.md` が Agent_tools 全体の repo routing を複製する fixture で warning になること。
- `workflow-cookbook` 内の Birdseye / Task Seed / Acceptance / CI / Evidence 手順は pass すること。
- 他 repo 連携の説明が plugin config / Evidence / Acceptance / Task state の接続点に限定されている場合は pass すること。

## 完了条件

- RUNBOOK slimming、Task/Acceptance/Completion trace、agent-tools-hub boundary の
  3 仕様に対応する checker または既存 checker 拡張がある。
- 追加/更新した checker の unit test がある。
- `docs/requirements.md` と `docs/spec.md` の仕様と実装が矛盾していない。
- `RUNBOOK.md` が完了済み詳細表を持たず、必要な場合は `docs/completion-record.md` へリンクしている。
- `workflow-cookbook/HUB.codex.md` が `agent-tools-hub` の横断 routing table を複製していない。
- `CHANGELOG.md` の `[Unreleased]` に実装内容が記録されている。

## レビュー観点

- `agent-tools-hub` と `workflow-cookbook` の責務が混ざっていないか。
- RUNBOOK を細くする目的なのに、RUNBOOK へ新しい長文セクションを追加していないか。
- completion record が「完了の正本」になっておらず、task / acceptance / release / changelog へ辿れるか。
- warning と failure の基準が既存 CI 運用に対して強すぎないか。
- 小さな文書変更で過剰に fail しないか。

## 参照

- [docs/requirements.md](../requirements.md)
- [docs/spec.md](../spec.md)
- [docs/completion-record.md](../completion-record.md)
- [RUNBOOK.md](../../RUNBOOK.md)
- [HUB.codex.md](../../HUB.codex.md)
- [tools/ci/check_task_acceptance_sync.py](../../tools/ci/check_task_acceptance_sync.py)
