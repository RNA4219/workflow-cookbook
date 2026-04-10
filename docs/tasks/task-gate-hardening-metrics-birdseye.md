---
task_id: 20260411-01
intent_id: INT-001
owner: docs-core
status: done
last_reviewed_at: 2026-04-11
next_review_due: 2026-05-11
---

# Task Seed: Metrics / Birdseye Gate Hardening

## 背景

- 直近の改修で `tools/ci/check_metrics_thresholds.py` と
  `tools/ci/check_birdseye_freshness.py` を追加したが、
  検収レビューでは「置いただけ」で終わっている懸念が残った。
- 現在の CI は
  [`.github/workflows/markdown.yml`](../../.github/workflows/markdown.yml)
  から checker を呼んでいるものの、
  metrics は固定 sample のみ、
  Birdseye は freshness window なしで実行している。
- そのため、運用上の stale や KPI 逸脱を実メトリクスベースで
  十分に止められていない。

## ゴール

- metrics checker を「固定 sample の疎通確認」ではなく、
  実際に収集された QA metrics に対する gate として機能させる。
- Birdseye checker を構造整合だけでなく、
  `last_verified_at` を使った freshness 判定まで含めて CI へ載せる。
- README / RUNBOOK / CHECKLISTS / backlog docs の説明が、
  実際の CI 挙動と一致する状態にする。

## 修正対象

1. metrics gate の実運用化
   - `.ga/qa-metrics.json` を生成する workflow か既存の Python CI /
     governance 系 workflow に
     `python tools/ci/check_metrics_thresholds.py --check` を組み込む。
   - 固定 sample は「設定例」「疎通確認」として残してよいが、
     本番 gate は実 metrics に対して実行する。

2. Birdseye freshness gate の強化
   - CI で
     `python tools/ci/check_birdseye_freshness.py --check --max-verified-age-days <N>`
     を実行し、
     `last_verified_at` の stale も検知できるようにする。
   - `<N>` は運用上妥当な日数を docs と合わせて固定する。

3. docs と CI の同期
   - [README.md](../../README.md)
   - [RUNBOOK.md](../../RUNBOOK.md)
   - [CHECKLISTS.md](../../CHECKLISTS.md)
   - [docs/BIRDSEYE.md](../BIRDSEYE.md)
   - [docs/addenda/N_Improvement_Backlog.md](../addenda/N_Improvement_Backlog.md)
   - [docs/addenda/P_Expansion_Candidates.md](../addenda/P_Expansion_Candidates.md)
   を、最終的な CI 実装に合わせて再確認・更新する。

## TDD / 検証

1. metrics
   - 実 metrics file を生成する経路に対して、
     threshold checker が fail / pass する統合テストまたは workflow 内検証を追加する。
   - sample JSON を使うテストは残しつつ、
     実収集フロー経由の検証を別に持つ。

2. Birdseye
   - `--max-verified-age-days` を付けた CLI 実行の想定値を追加検証する。
   - stale な `last_verified_at` が CI で期待どおり検知されることを確認する。

3. docs
   - `markdownlint-cli2` で関連 docs を確認する。
   - `RUNBOOK` と `CHECKLISTS` の記述が workflow 実装と一致することを確認する。

## 完了条件

- metrics checker が実 metrics に対して CI で実行される
- Birdseye checker が freshness window 付きで CI で実行される
- docs と workflow の説明差分が解消される
- 追加したテストまたは workflow 検証が通る

## 検収観点

- `sample を通しているだけ` になっていないか
- stale Birdseye が CI で検知されるか
- `README` / `RUNBOOK` / `CHECKLISTS` の記述が現実の CI と一致しているか
- 例外的に skip する条件があるなら docs に明記されているか

## 参照

- [tools/ci/check_metrics_thresholds.py](../../tools/ci/check_metrics_thresholds.py)
- [tools/ci/check_birdseye_freshness.py](../../tools/ci/check_birdseye_freshness.py)
- [governance/metrics_thresholds.yaml](../../governance/metrics_thresholds.yaml)
- [examples/qa-metrics.sample.json](../../examples/qa-metrics.sample.json)
- [`.github/workflows/markdown.yml`](../../.github/workflows/markdown.yml)
