---
intent_id: INT-001
owner: docs-core
status: active
last_reviewed_at: 2026-04-11
next_review_due: 2026-05-11
---

# R. Product Readiness Checklist

## 1. 目的

- `workflow-cookbook` を「個人用に便利な repo」から
  「他者へ安心して渡せるプロダクト級の運用基盤」へ引き上げるための
  判定基準を明文化する。
- 単なる機能有無ではなく、
  docs、CI、acceptance、security、運用導線、再利用性まで含めて確認する。

## 2. 判定の考え方

- `CHECKLISTS.md` は日常運用と release 作業のチェックを扱う。
- 本書は「プロダクトとして成立したと言えるか」を確認するための
  上位チェックリストとして扱う。
- すべてを一度に完了させる必要はないが、
  少なくとも `P0` は満たしてから
  「プロダクト級」と表現する。

## 3. Readiness Levels

### 3.1 P0: 成立条件

- [x] README だけで導入目的、主要機能、最初のコマンドが分かる
- [x] `RUNBOOK.md` と `CHECKLISTS.md` から日常運用と復旧手順が辿れる
- [x] `docs/requirements.md` / `docs/spec.md` / `docs/design.md` の役割分担が崩れていない
- [x] `docs/acceptance/` の records が、実装状態より強い完了表現になっていない
- [x] Birdseye / codemap の更新手順が docs と CI で一致している
- [x] CI の required gate が green で、`governance/policy.yaml` と整合している
- [x] security / release / acceptance の証跡が release 判定から辿れる
- [x] Python 系の基準 coverage 80% を継続的に満たしている
- [x] ライセンス、公開メタデータ、README の記述が一致している

### 3.2 P1: 実用条件

- [ ] smoke 用 baseline と production 用データの責務が docs / workflow で分離されている
- [ ] metrics の収集、閾値判定、確認手順が `RUNBOOK.md` から辿れる
- [ ] Birdseye freshness failure 後の復旧手順が docs 化されている
- [ ] acceptance index を見れば task / acceptance / release の関係が追える
- [ ] release evidence checker と security posture checker が release 導線に組み込まれている
- [ ] README の多言語版が最低限同じ機能集合を説明している
- [ ] 新規 contributor が `README.md` と `RUNBOOK.md` だけで最初の検証を再現できる

### 3.3 P2: 量産条件

- [ ] `workflow-cookbook` の運用観点を `agent-taskstate` / `memx-resolver` へ横展開できる
- [ ] plugin host / Evidence / docs resolve の使い分けが明確に説明されている
- [ ] downstream software 向け optional capability が必須フローを邪魔しない
- [ ] release / acceptance / evidence を横断した品質サマリを生成できる
- [ ] front matter の review 期限や stale docs の棚卸しが定期運用されている

## 4. 分野別チェック

## 4.1 Docs / Onboarding

- [x] README に Quick Start がある
- [x] README に repo の役割と非対象が書かれている
- [x] 主要 docs への入口が 1 回のスクロールで見つかる
- [x] README / RUNBOOK / CHECKLISTS / HUB の説明が矛盾していない
- [x] Windows 利用者向けの `python` 実行注意が明記されている

## 4.2 CI / Quality

- [x] `markdown-quality`
- [x] `python-tests`
- [x] `test`
- [x] `cross-repo-integration`
- [x] `release-evidence`
- [x] `Security Gate`
- [x] `CodeQL`

上記が green であり、役割が `docs/ci-config.md` と一致していること。

## 4.3 Acceptance / Release

- [x] acceptance record は scope を越えた完了表現になっていない
- [x] release ごとの主要変更が `CHANGELOG.md` と `docs/releases/` で追える
- [x] acceptance index から最新の承認記録へすぐ辿れる
- [x] review finding を task seed へ落とす流れが機能している

## 4.4 Security / Governance

- [x] security checklist と workflow の関係が docs から分かる
- [x] branch protection / required jobs / docs の差分が管理されている
- [x] release 判定前に security posture を確認する手順がある
- [x] forbidden paths や自己改変境界が docs で追える

## 4.5 Observability / Metrics

- [x] `.ga/qa-metrics.json` の位置づけが明記されている
- [x] baseline と実測値を混同しない
- [x] threshold check の意味が docs に書かれている
- [x] Birdseye freshness のしきい値と運用方針が説明されている

## 5. 現時点の自己評価ガイド

- `P0` がすべて満たされていれば、
  `workflow-cookbook` は「プロダクト級の基盤」と表現してよい。
- `P1` まで満たせば、
  「他者へ渡しても運用しやすいプロダクト」と言いやすい。
- `P2` は拡張性と量産性の段階であり、
  ここが未完でも `P0` / `P1` を満たしていれば
  実用上のプロダクト価値は十分にある。

## 6. 次の優先候補

1. metrics-harvest の実運用化
2. Birdseye freshness しきい値の段階短縮
3. `agent-taskstate` / `memx-resolver` への横展開レビュー
4. release / acceptance / evidence の横断サマリ生成

## 7. 参照

- [README.md](../../README.md)
- [RUNBOOK.md](../../RUNBOOK.md)
- [CHECKLISTS.md](../../CHECKLISTS.md)
- [docs/requirements.md](../requirements.md)
- [docs/acceptance/INDEX.md](../acceptance/INDEX.md)
- [docs/addenda/P_Expansion_Candidates.md](P_Expansion_Candidates.md)
