---
task_id: 20260411-04
intent_id: INT-001
owner: docs-core
status: done
last_reviewed_at: 2026-04-11
next_review_due: 2026-05-11
---

# Task Seed: Metrics Harvest Operationalization

## 背景

- 現在の `markdown-quality` workflow では、
  [`.ga/qa-metrics.json`](../../.ga/qa-metrics.json) を
  smoke baseline として生成し、
  metrics gate 自体の存在確認と threshold CLI の疎通確認を行っている。
- これは立ち上げ期の安全策としては有効だが、
  実測 metrics に基づく継続運用とはまだ分離されている。
- 今後は
  [`.github/workflows/reusable/metrics-harvest.yml`](../../.github/workflows/reusable/metrics-harvest.yml)
  と
  [tools/perf/collect_metrics.py](../../tools/perf/collect_metrics.py)
  を正本経路として、
  定期的な metrics 収集と gate の運用を始めたい。

## ゴール

- smoke baseline と実測 metrics を docs / workflow 上で明確に分離する。
- 実測 metrics を定期収集する workflow を運用へ載せる。
- metrics gate の「疎通確認」と「実運用」を別の責務として追えるようにする。

## 修正対象

1. metrics 収集経路の整理
   - `metrics-harvest.yml` を、定期実行または手動実行で
     `.ga/qa-metrics.json` を更新できる運用へ整える。
   - `collect_metrics.py` の入力前提と出力契約を docs に再掲する。

2. workflow 責務の分離
   - `markdown-quality` は smoke / baseline / schema sanity check の位置付けに固定する。
   - 実測 metrics による監視・定期更新は別 workflow へ分離する。

3. docs 同期
   - [README.md](../../README.md)
   - [RUNBOOK.md](../../RUNBOOK.md)
   - [CHECKLISTS.md](../../CHECKLISTS.md)
   - [docs/CONTRACTS.md](../CONTRACTS.md)
   - [docs/addenda/P_Expansion_Candidates.md](../addenda/P_Expansion_Candidates.md)
   に、smoke baseline と実測 metrics の役割差を明記する。

## TDD / 検証

1. metrics
   - 実測 metrics 更新 workflow が `.ga/qa-metrics.json` を出力することを確認する。
   - `check_metrics_thresholds.py` がその出力を読めることを確認する。

2. docs
   - baseline と production metrics を混同しない記述になっていることを確認する。
   - markdownlint と関連 checker を通す。

## 完了条件

- 実測 metrics を更新する定期または手動 workflow が docs 付きで整備される
- smoke baseline と production metrics の責務が docs 上で分離される
- metrics gate の運用手順が `RUNBOOK.md` と `CHECKLISTS.md` から辿れる

## 検収観点

- baseline 生成が実測 metrics の代替として誤読されないか
- `.ga/qa-metrics.json` の正本更新経路が明確か
- metrics 閾値チェックが実運用フローに接続されているか

## 参照

- [`.github/workflows/markdown.yml`](../../.github/workflows/markdown.yml)
- [`.github/workflows/reusable/metrics-harvest.yml`](../../.github/workflows/reusable/metrics-harvest.yml)
- [tools/perf/collect_metrics.py](../../tools/perf/collect_metrics.py)
- [tools/ci/check_metrics_thresholds.py](../../tools/ci/check_metrics_thresholds.py)
