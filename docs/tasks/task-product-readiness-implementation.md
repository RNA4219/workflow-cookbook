---
task_id: 20260411-08
intent_id: INT-001
owner: docs-core
status: done
last_reviewed_at: 2026-04-11
next_review_due: 2026-05-11
---

# Task Seed: Product Readiness Implementation

## 背景

- `workflow-cookbook` は、Birdseye、Task Seed、acceptance、CI、security、
  release evidence、cross-repo plugin まで一通りそろい、
  「個人用の便利 repo」を超えた成熟度に達している。
- 一方で、プロダクト級と呼ぶには、
  機能の有無だけでなく、
  onboarding、定常運用、証跡、再利用性、他 repo への展開まで含めた
  仕上げが必要である。
- その判定基準として
  [R_Product_Readiness_Checklist.md](../addenda/R_Product_Readiness_Checklist.md)
  を追加した。
- 次は、この checklist を実際の実装・運用改善へ落とし込み、
  少なくとも `P0` を満たした状態で
  「プロダクト級」と言えるようにしたい。

## ゴール

- `R_Product_Readiness_Checklist.md` の `P0` をすべて満たす。
- 可能な範囲で `P1` も前進させる。
- 個別改善をバラバラに進めず、
  `README` / `RUNBOOK` / `CHECKLISTS` / acceptance / CI / metrics の
  つながりとして仕上げる。

## 優先実装順

1. P0 の成立条件を満たす
   - README / RUNBOOK / CHECKLISTS / requirements/spec/design の役割分担確認
   - acceptance record の表現正規化
   - Birdseye / codemap の docs と CI の整合確認
   - required gate / policy / release evidence / security posture の再確認

2. P1 のうち効果が高いものを進める
   - smoke baseline と production metrics の責務分離
   - metrics 収集と確認導線の整備
   - Birdseye stale 復旧手順の docs 化
   - acceptance index / release / evidence の導線強化

3. P2 は task seed 化まででよい
   - cross-repo 横展開
   - optional capability の整理
   - quality summary / stale docs review の定期化

## 修正対象

1. Docs / Onboarding
   - [README.md](../../README.md)
   - [RUNBOOK.md](../../RUNBOOK.md)
   - [CHECKLISTS.md](../../CHECKLISTS.md)
   - [HUB.codex.md](../../HUB.codex.md)
   - [docs/requirements.md](../requirements.md)
   - [docs/spec.md](../spec.md)
   - [docs/design.md](../design.md)

2. Acceptance / Release
   - [docs/acceptance/INDEX.md](../acceptance/INDEX.md)
   - `docs/acceptance/AC-*.md`
   - [CHANGELOG.md](../../CHANGELOG.md)
   - `docs/releases/*.md`

3. CI / Governance / Metrics
   - [`.github/workflows/markdown.yml`](../../.github/workflows/markdown.yml)
   - [docs/ci-config.md](../ci-config.md)
   - [governance/policy.yaml](../../governance/policy.yaml)
   - [tools/ci/check_metrics_thresholds.py](../../tools/ci/check_metrics_thresholds.py)
   - [tools/ci/check_birdseye_freshness.py](../../tools/ci/check_birdseye_freshness.py)

4. 関連する既存 task seed
   - [task-metrics-harvest-operationalization.md](./task-metrics-harvest-operationalization.md)
   - [task-birdseye-freshness-tightening.md](./task-birdseye-freshness-tightening.md)
   - [task-cross-repo-status-review.md](./task-cross-repo-status-review.md)
   - [task-acceptance-normalization-ac-20260411-01.md](./task-acceptance-normalization-ac-20260411-01.md)

## TDD / 検証

1. CI
   - required gate が green であることを確認する。
   - `docs/ci-config.md` と workflow の役割差分がないことを確認する。

2. Acceptance / Release
   - acceptance record が scope を越えた完了表現になっていないことを確認する。
   - release evidence と changelog / release docs の関係が追えることを確認する。

3. Metrics / Birdseye
   - `.ga/qa-metrics.json` の位置づけが smoke baseline か実測値か、
     docs と workflow で一致していることを確認する。
   - Birdseye freshness のしきい値と stale 復旧手順が docs から辿れることを確認する。

4. Docs
   - `markdownlint-cli2` を通す。
   - README から最初の検証コマンドまで迷わず辿れることを確認する。

## 完了条件

- `R_Product_Readiness_Checklist.md` の `P0` がすべて満たされる
- `P1` の主要項目について、少なくとも実装済みか task seed 化済みになっている
- docs / acceptance / CI / metrics の記述差分が整理される
- プロダクト級と表現する根拠を docs から説明できる

## 検収観点

- README だけで初見利用者が入口を理解できるか
- acceptance / release / security / metrics の証跡が相互参照できるか
- smoke baseline と production 運用を混同しないか
- 「個人用の便利 repo」ではなく
  「他者へ渡せる運用基盤」として説明できるか

## 参照

- [R_Product_Readiness_Checklist.md](../addenda/R_Product_Readiness_Checklist.md)
- [README.md](../../README.md)
- [RUNBOOK.md](../../RUNBOOK.md)
- [CHECKLISTS.md](../../CHECKLISTS.md)
- [docs/acceptance/INDEX.md](../acceptance/INDEX.md)
- [docs/addenda/P_Expansion_Candidates.md](../addenda/P_Expansion_Candidates.md)
