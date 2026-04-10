---
task_id: 20260411-03
intent_id: INT-001
owner: docs-core
status: done
last_reviewed_at: 2026-04-11
next_review_due: 2026-05-11
---

# Task Seed: Gate Hardening Finalization

## 背景

- 直近の `Metrics / Birdseye Gate Hardening` で、実装と docs の大枠は揃った。
- ただし検収レビューで次の 2 点が残った。
  - `tools/ci/check_birdseye_freshness.py` は `--max-verified-age-days` を受けても stale を warning 扱いに留め、CI failure にならない。
  - `docs/acceptance/AC-20260411-01.md` は `approved` だが、Evidence が sample 実行中心で、実 metrics gate の完了を強く言い切れていない。
- このままだと、task seed / acceptance / checklist の完了表現と、実際の gate の効き方がずれる。
- 注記: RG-001（実 metrics gate の `.ga/qa-metrics.json` 未生成時の扱い）は本タスクのスコープ外。
  `task-gate-hardening-followup.md` で対応。

## ゴール

- Birdseye freshness の stale を CI で fail できるようにする。
- acceptance record の表現を、実装状態と一致させる。
- task seed / checklist / acceptance の3点を同じ完了基準へそろえる。

## 修正対象

1. Birdseye freshness gate の failure 化
   - `tools/ci/check_birdseye_freshness.py` で、`--max-verified-age-days` を超えた `last_verified_at` を非ゼロ終了に結びつく failure として扱う。
   - 既存の JSON summary は維持してよいが、warning と failure の意味をはっきり分ける。
   - `markdown.yml` からの呼び出しが、stale を見逃さない状態にする。

2. acceptance record の再同期
   - `docs/acceptance/AC-20260411-01.md` の scope / evidence / verification result を見直し、
     「未完了のものを approved にしていない」表現へ合わせる。
   - 実 metrics gate がまだ sample ベースなら、その限界を明記する。
   - 逆に実 metrics gate を実装済みにするなら、その証跡を明示する。

3. docs 間の整合
   - `docs/tasks/task-gate-hardening-metrics-birdseye.md`
   - `docs/addenda/Q_Remaining_Gap_Checklist.md`
   - `CHECKLISTS.md`
   を、最終的な挙動に合わせて更新する。

## TDD / 検証

1. Birdseye
   - stale な `last_verified_at` を与えたときに exit code が 1 になるテストを追加する。
   - stale でないケースは従来どおり通ることを確認する。

2. acceptance
   - `approved` にする条件が、Evidence と矛盾していないことを確認する。
   - もし未完了が残るなら、`approved` を外して `reviewing` / `draft` 相当に戻す。

3. docs
   - `markdownlint-cli2` を通す。
   - `CHECKLISTS.md` と task seed の記述が一致することを確認する。

## 完了条件

- stale Birdseye が CI で fail する
- acceptance record が実装状態と一致する
- task seed / checklist / acceptance の記述差分がなくなる
- 追加したテストまたは workflow 検証が通る

## 検収観点

- `--max-verified-age-days` が warning だけで終わっていないか
- `approved` acceptance が実態以上の完了表現になっていないか
- docs の完了表現が実装と同じ基準になっているか

## 参照

- [tools/ci/check_birdseye_freshness.py](../../tools/ci/check_birdseye_freshness.py)
- [docs/acceptance/AC-20260411-01.md](../acceptance/AC-20260411-01.md)
- [docs/tasks/task-gate-hardening-metrics-birdseye.md](./task-gate-hardening-metrics-birdseye.md)
- [docs/addenda/Q_Remaining_Gap_Checklist.md](../addenda/Q_Remaining_Gap_Checklist.md)
- [CHECKLISTS.md](../../CHECKLISTS.md)
