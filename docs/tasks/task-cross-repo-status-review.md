---
task_id: 20260411-06
intent_id: INT-001
owner: docs-core
status: planned
last_reviewed_at: 2026-04-11
next_review_due: 2026-05-11
---

# Task Seed: Cross-Repo Status Review

## 背景

- `workflow-cookbook` 側の gate hardening と backlog 実装は一通り完了した。
- 次の自然な段階は、
  `Agent_tools` 配下の周辺 repo に同種の運用品質差分が残っていないかを確認すること。
- 特に
  `agent-taskstate`、
  `memx-resolver`、
  必要なら `Agent_tools` ルート docs は、
  workflow / acceptance / README / runbook の整合を改めて点検する価値が高い。

## ゴール

- 他 repo に横展開すべき運用品質改善を洗い出す。
- `workflow-cookbook` で整えた gate / acceptance / docs 同期の観点を再利用する。
- repo ごとに「今すぐ直すもの」と「候補として積むもの」を分離する。

## 修正対象

1. review 対象の棚卸し
   - `Agent_tools` ルート
   - `agent-taskstate`
   - `memx-resolver`
   の README / HUB / RUNBOOK / acceptance / CI を確認する。

2. 観点の横展開
   - acceptance が実装状態より強い完了表現になっていないか
   - smoke / sample / real metrics の責務が混線していないか
   - stale / freshness / review due の運用手順が docs にあるか
   - CI / policy / branch protection 相当の記述差分がないか

3. 結果の文書化
   - repo ごとに findings と次タスクを残す。
   - すぐ直す項目は task seed 化し、
     中長期候補は backlog / expansion 候補へ振り分ける。

## TDD / 検証

1. review
   - repo ごとに review finding が 0 件か、
     あるいは task seed へ起票済みであることを確認する。

2. docs
   - 参照先が古くなっていないことを確認する。

## 完了条件

- `Agent_tools` / `agent-taskstate` / `memx-resolver` の現状レビューが完了する
- 指摘事項が task seed または backlog へ整理される
- `workflow-cookbook` で整備した観点の再利用方法が残る

## 検収観点

- repo ごとに review 観点が統一されているか
- ただの感想ではなく、次の着手単位まで落ちているか
- `workflow-cookbook` での学びが他 repo に再利用されているか

## 参照

- [C:/Users/ryo-n/Codex_dev/Agent_tools/README.md](C:/Users/ryo-n/Codex_dev/Agent_tools/README.md)
- [C:/Users/ryo-n/Codex_dev/Agent_tools/HUB.codex.md](C:/Users/ryo-n/Codex_dev/Agent_tools/HUB.codex.md)
- [C:/Users/ryo-n/Codex_dev/agent-taskstate/README.md](C:/Users/ryo-n/Codex_dev/agent-taskstate/README.md)
- [C:/Users/ryo-n/Codex_dev/memx-resolver/README.md](C:/Users/ryo-n/Codex_dev/memx-resolver/README.md)
