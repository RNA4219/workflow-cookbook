---
task_id: 20260411-05
intent_id: INT-001
owner: docs-core
status: planned
last_reviewed_at: 2026-04-11
next_review_due: 2026-05-11
---

# Task Seed: Birdseye Freshness Tightening

## 背景

- 現在の Birdseye freshness gate は
  `--max-verified-age-days 365` で運用開始できる状態になっている。
- stale を failure にできるようになったため、
  次の課題は「安全にしきい値を短縮し、運用品質を上げる」こと。
- ただし、いきなり 30 日へ寄せると運用負荷が読めないため、
  まず 90 日、その後 30 日への段階短縮を検討したい。

## ゴール

- Birdseye freshness のしきい値を段階的に引き締める。
- stale 修復フローを `RUNBOOK` / `CHECKLISTS` に明記する。
- stale failure を運用負荷ではなく習慣化された点検へ落とし込む。

## 修正対象

1. freshness しきい値の見直し
   - `markdown-quality` の `--max-verified-age-days` を
     365 → 90 → 30 の順で見直す計画を docs 化する。
   - 現時点で即変更するか、段階計画だけ先に置くかを明確にする。

2. remediation 導線
   - stale failure 時に実行すべき標準コマンドを
     `RUNBOOK.md` と `CHECKLISTS.md` に明記する。
   - `docs/BIRDSEYE.md` と `tools/codemap/README.md` の手順も必要なら同期する。

3. docs / acceptance 同期
   - freshness gate が「ある」だけでなく
     「どの程度の stale を許容するか」まで追えるようにする。

## TDD / 検証

1. Birdseye
   - しきい値内の `last_verified_at` は通り、
     しきい値外は fail することを確認する。
   - 代表値 365 / 90 / 30 のうち、採用値に対するテストまたは検証ログを残す。

2. docs
   - stale failure 後の標準復旧手順が docs から辿れることを確認する。

## 完了条件

- freshness しきい値の次段計画が docs に明記される
- stale remediation 手順が `RUNBOOK.md` / `CHECKLISTS.md` に反映される
- Birdseye 運用手順と CI 設定の記述差分がなくなる

## 検収観点

- stale failure の次の一手が docs から迷わず辿れるか
- しきい値の変更理由と運用方針が明記されているか
- Birdseye docs と workflow の値が一致しているか

## 参照

- [tools/ci/check_birdseye_freshness.py](../../tools/ci/check_birdseye_freshness.py)
- [docs/BIRDSEYE.md](../BIRDSEYE.md)
- [tools/codemap/README.md](../../tools/codemap/README.md)
- [RUNBOOK.md](../../RUNBOOK.md)
- [CHECKLISTS.md](../../CHECKLISTS.md)
