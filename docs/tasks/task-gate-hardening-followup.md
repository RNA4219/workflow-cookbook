---
task_id: 20260411-02
intent_id: INT-001
owner: docs-core
status: planned
last_reviewed_at: 2026-04-11
next_review_due: 2026-05-11
---

# Task Seed: Gate Hardening Follow-up

## 背景

- `Metrics / Birdseye Gate Hardening` の実装後レビューで、
  P1 / P2 の未解消点が追加で見つかった。
- 現状の
  [`.github/workflows/markdown.yml`](../../.github/workflows/markdown.yml)
  は、`.ga/qa-metrics.json` が無い場合に
  実 metrics gate を skip して成功扱いにしている。
- 現状の
  [tools/ci/check_birdseye_freshness.py](../../tools/ci/check_birdseye_freshness.py)
  は、`--max-verified-age-days` を付けても stale を warning 扱いに留め、
  CI gate として fail しない。
- その一方で
  [docs/acceptance/AC-20260411-01.md](../acceptance/AC-20260411-01.md)
  は `approved` として閉じられており、
  実装状態より強い完了表現になっている。

## ゴール

- 実 metrics gate を「ファイルがあれば確認」ではなく、
  必要条件を満たす本番 gate として成立させる。
- Birdseye freshness check が stale を CI failure として返す。
- acceptance / checklist / task seed の完了表現を、実装状態と一致させる。

## 修正対象

1. 実 metrics gate の必須化
   - `.ga/qa-metrics.json` を CI 内で確実に生成する step を追加するか、
     対象 branch / workflow では未生成を failure にする。
   - sample JSON は疎通確認用として残してよいが、
     実 metrics gate の代替にしない。

2. Birdseye stale の failure 化
   - `tools/ci/check_birdseye_freshness.py` で
     `--max-verified-age-days` 超過を warning ではなく
     非ゼロ終了に結びつく failure として扱う。
   - 既存の JSON summary でも failure と warning の意味が分かれるようにする。

3. docs / acceptance の再同期
   - [docs/acceptance/AC-20260411-01.md](../acceptance/AC-20260411-01.md)
   - [docs/addenda/Q_Remaining_Gap_Checklist.md](../addenda/Q_Remaining_Gap_Checklist.md)
   - [docs/tasks/task-gate-hardening-metrics-birdseye.md](task-gate-hardening-metrics-birdseye.md)
   - [CHECKLISTS.md](../../CHECKLISTS.md)
   を、実際の gate 成立条件に合わせて更新する。

## TDD / 検証

1. metrics
   - `.ga/qa-metrics.json` が無い場合に fail する条件、
     または生成 step により file が確実に作られる条件を
     workflow / test で検証する。
   - sample JSON に依存しなくても実 metrics gate が成立することを確認する。

2. Birdseye
   - stale な `last_verified_at` を与えたとき、
     checker が非ゼロ終了することをテストで確認する。
   - stale ではない場合は従来どおり pass することを確認する。

3. docs
   - acceptance record に
     「未完了のまま approved しない」ことを確認する。
   - checklist のチェック済み項目が実装状態と一致することを確認する。

## 完了条件

- `.ga/qa-metrics.json` 未生成のまま実 metrics gate が素通りしない
- stale Birdseye が CI で fail する
- acceptance / checklist / task seed の表現が実装状態と一致する
- 追加テストまたは workflow 検証が通る

## 検収観点

- 実 metrics gate が sample のみで代替されていないか
- `--max-verified-age-days` が warning だけで終わっていないか
- approved 済み acceptance が実態以上の完了表現になっていないか
- follow-up 完了後に docs 間の記述差分が残っていないか

## 参照

- [`.github/workflows/markdown.yml`](../../.github/workflows/markdown.yml)
- [tools/ci/check_birdseye_freshness.py](../../tools/ci/check_birdseye_freshness.py)
- [docs/acceptance/AC-20260411-01.md](../acceptance/AC-20260411-01.md)
- [docs/addenda/Q_Remaining_Gap_Checklist.md](../addenda/Q_Remaining_Gap_Checklist.md)
