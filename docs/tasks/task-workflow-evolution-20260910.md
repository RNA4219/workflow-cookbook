---
task_id: 20260910-02
intent_id: INT-WORKFLOW-EVOLUTION
owner: Codex
status: done
last_reviewed_at: 2026-09-10
next_review_due: 2026-12-10
---

# ワークフローの評価・取得・観測・復旧

## Objective

固定課題で改善を比較し、必要な文書を予算内で取得し、実行成否を追跡し、中断後に安全に再開できる基盤を接続する。

## Scope

Agent_tools版の既存manifest、Birdseye、workflow plugin、Evidence、coordinatorとagent-taskstateを利用する。
実装そのもののfixture試験と実運用効果の評価を区別する。

## 実装順序

1. [評価比較の仕様](../contracts/workflow-benchmark.md): 凍結入力、対応のある比較、欠測と失敗の区別。
2. [段階開示](../contracts/progressive-context.md): 予算・索引鮮度に基づく取得。
3. [trace](../contracts/workflow-run-observation.md): task/runと最終検収結果の接続。
4. [checkpoint](../contracts/workflow-checkpoint.md): 成果物照合と再開。

利用者向け入口: [利用手順](../workflow-evolution.md)。

## Evidence

workspaceの `research/workflow-evolution-20260910/HANDOFF.md` に進捗・実行結果を記録する。

[Acceptance AC-20260910-02](../acceptance/AC-20260910-02.md) で技術検証を完了。
4仕様と実装、131ケース追加、全889ケース成功。実運用での性能比較は別途行う。
