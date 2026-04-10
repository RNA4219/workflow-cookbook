---
intent_id: INT-001
owner: docs-core
status: active
last_reviewed_at: 2026-04-11
next_review_due: 2026-05-11
---

# Q. Remaining Gap Checklist

## 1. 目的

- `workflow-cookbook` の次回仕上げに向けて、
  「未実装そのもの」よりも
  「docs / CI / task 状態のズレ」を優先的に潰すための確認表を置く。
- 実装者、レビュー担当、リリース担当が
  同じ観点で残課題を確認できるようにする。

## 2. 現在確認できている主なギャップ

| ID | 優先度 | 領域 | 内容 | 根拠 |
| :-- | :-- | :-- | :-- | :-- |
| RG-001 | P1 | CI | 実 metrics gate が `.ga/qa-metrics.json` 未生成時に skip 成功となる | `.github/workflows/markdown.yml`, `docs/tasks/task-gate-hardening-followup.md` |
| RG-002 | P1 | CI | Birdseye freshness checker が stale を warning のみで返し、CI failure にならない | `tools/ci/check_birdseye_freshness.py`, `docs/tasks/task-gate-hardening-followup.md` |
| RG-003 | P2 | Docs / Acceptance | acceptance record が未完了事項を含んだまま `approved` 扱いになっている | `docs/acceptance/AC-20260411-01.md`, `docs/tasks/task-gate-hardening-followup.md` |

## 2.1 完了済みギャップ

| ID | 完了日 | 内容 |
| :-- | :-- | :-- |
| RG-001 | 2026-04-11 | 実 metrics gate 未生成時 failure 化 |
| RG-002 | 2026-04-11 | Birdseye stale を failure 化 |
| RG-003 | 2026-04-11 | acceptance record を実装状態へ再同期 |

## 3. 実装優先チェック

### 3.1 P1

- [x] RG-001: `.ga/qa-metrics.json` 未生成のまま実 metrics gate が素通りしないようにする
- [x] RG-002: Birdseye stale を warning ではなく failure として扱う

### 3.2 P2

- [x] RG-003: acceptance / checklist / task seed の完了表現を実装状態へ合わせる

## 4. 分野別チェックリスト

## 4.1 CI / Gate

- [x] 実 metrics file を入力にした threshold check が CI で動く
- [x] sample JSON は疎通確認用と明記され、本番 gate と混同しない
- [x] Birdseye freshness check に `--max-verified-age-days` が付いている
- [x] stale な `last_verified_at` を CI で fail できる
- [x] `governance/policy.yaml`、workflow、`docs/ci-config.md` の記述が一致する

## 4.2 Docs / Task State

- [x] `done` task の背景に未実装リストが残っていない
- [x] backlog 完了済み項目と task seed の説明が矛盾していない
- [x] `CHANGELOG.md` の Known limitations が現状と一致する
- [x] README / RUNBOOK / CHECKLISTS のコマンド例が実際の実行環境と合う
- [x] acceptance record が実装状態より強い完了表現になっていない

## 4.3 Release Readiness

- [x] release evidence checker の前提ファイルが docs に明記されている
- [x] acceptance / release / changelog の参照関係が追える
- [x] security / metrics / Birdseye の gate が「説明だけ」で終わっていない

## 4.4 Local Developer Experience

- [x] Windows の既定 `python` 解決で Store stub に当たる場合の回避策が書かれている
- [x] `py -3` と `uv run` の使い分けが docs で統一されている
- [x] ローカル検証手順と GitHub Actions 上の手順差分が説明されている

## 5. すすめ方

1. まず P1 の RG-001 から RG-004 を片づける
2. 次に P2 の docs / local ops のズレを縮める
3. 修正後に `CHECKLISTS.md` と `RUNBOOK.md` を同期する
4. 最後に acceptance または task seed へ検収結果を反映する

## 6. 関連文書

- [Task Seed: Gate Hardening Follow-up](../tasks/task-gate-hardening-followup.md)
- [Task Seed: Metrics / Birdseye Gate Hardening](../tasks/task-gate-hardening-metrics-birdseye.md)
- [N. Improvement Backlog](N_Improvement_Backlog.md)
- [CI 段階的導入 要件定義](../ci_phased_rollout_requirements.md)
- [Checklists](../../CHECKLISTS.md)
