# N. Improvement Backlog

## 1. 目的

`workflow-cookbook` とその周辺 repo に対して、今後の改善候補を
優先度・対象 repo・狙いごとに整理する。

この文書は「未実装の欠陥一覧」ではなく、現状を前提に
次の品質向上や運用自動化の打ち手を集約する backlog として扱う。

## 2. 優先度の見方

- `P1`
  - 近い将来に運用品質へ直接効く。
  - CI / 検収 / リリース事故の予防効果が高い。
- `P2`
  - 効果は大きいが、直近の gate を壊すものではない。
  - 自動化や観測性を一段押し上げる。
- `P3`
  - すぐ困らないが、repo が大きくなるほど効いてくる。
  - 索引化、棚卸し、運用負債の抑制に効く。

## 3. 改善候補

| ID | 優先度 | 領域 | 対象 repo | 目的 | 具体案 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| IB-001 | P1 | Security | workflow-cookbook | Security 通知を後追い対応にしない | GitHub の Dependabot / vulnerability alerts を恒常運用へ寄せ、`docs/security/` と release 手順へ反映する |
| IB-002 | P1 | Release | workflow-cookbook | リリース証跡の抜け漏れを防ぐ | `CHANGELOG.md`、`docs/releases/`、tag、GitHub Release の整合を検証する CLI を追加する |
| IB-003 | P1 | Acceptance | workflow-cookbook, agent-taskstate | task と検収の不一致をなくす | `done` task と `approved` acceptance の双方向整合チェックを追加する |
| IB-004 | P1 | Integration CI | workflow-cookbook, agent-taskstate, memx-resolver | cross-repo 連携破壊を早期検知する | 3 repo の sample config をまとめて確認する統合 workflow を追加する |
| IB-005 | P1 | Metrics | workflow-cookbook | 数値を取るだけでなく判定まで自動化する | `collect_metrics.py` の出力に閾値判定 CLI を追加し、warn / fail を返せるようにする |
| IB-006 | P1 | Birdseye | workflow-cookbook | Birdseye の stale 放置を減らす | `generated_at` と `mtime` に加え、一定期間未更新時の freshness check を追加する |
| IB-007 | P2 | Docs Resolve | workflow-cookbook, memx-resolver | 検収前に必要資料を読み漏らさない | `docs.resolve` / `docs.ack` / `docs.stale_check` の結果を PR gate からも参照できるようにする |
| IB-008 | P2 | Acceptance Index | workflow-cookbook, agent-taskstate | acceptance record の棚卸しを楽にする | `approved` / `rejected` / `draft` の集計と task 対応表を index に出力する |
| IB-009 | P2 | Sample Sync | workflow-cookbook | sample と docs の乖離を防ぐ | `examples/` と参照 docs の対応表を作り、差分チェックを CLI 化する |
| IB-010 | P2 | Upstream | workflow-cookbook | upstream / fork の手動棚卸しを減らす | `UPSTREAM.md` と `UPSTREAM_WEEKLY_LOG.md` の更新差分を抽出する補助 CLI を追加する |
| IB-011 | P2 | Plugin Host | workflow-cookbook | plugin 追加時の壊れやすさを下げる | capability ごとの timeout / error policy / tracing hook を plugin runtime に追加する |
| IB-012 | P2 | TaskState | workflow-cookbook, agent-taskstate | task 正本を強くする | Task Seed / Acceptance / Evidence の対応関係を state export できるようにする |
| IB-013 | P2 | Evidence | workflow-cookbook, agent-protocols | 検収証跡を再利用しやすくする | acceptance result と Evidence record の対応を簡易レポート化する |
| IB-014 | P3 | Docs Hub | workflow-cookbook | docs が増えても迷いにくくする | `Task / Acceptance / Skill / Plugin / CI` の用途別ナビを 1 枚にまとめる |
| IB-015 | P3 | Addenda / ADR | workflow-cookbook | 設計資料の散逸を抑える | `docs/addenda/` と `docs/ADR/` の索引を自動生成し、更新時にリンク切れを検査する |
| IB-016 | P3 | Security Docs | workflow-cookbook | セキュリティ文書の陳腐化を防ぐ | `docs/security/` の最終更新日と release 記録を突き合わせるチェックを追加する |
| IB-017 | P3 | Release Notes | workflow-cookbook | 過去の出荷履歴を検索しやすくする | `docs/releases/` の一覧と主要変更点サマリを自動生成する |
| IB-018 | P3 | Knowledge Reuse | workflow-cookbook, memx-resolver | 過去の判断根拠を再利用しやすくする | release / acceptance / incident を `memx-resolver` から横断参照する導線を整備する |

## 4. repo 連携で解きやすい項目

### 4.1 `agent-taskstate` と相性が良い

- IB-003
- IB-008
- IB-012
- IB-013

使いどころ:

- task と acceptance の対応を state として持つ
- acceptance index の集計元を markdown scan だけに依存させない
- Task Seed / Acceptance / Evidence の関係を後から追えるようにする

### 4.2 `memx-resolver` と相性が良い

- IB-006
- IB-007
- IB-010
- IB-018

使いどころ:

- 読むべき docs の解決
- 既読 ack と stale 判定
- release / incident / acceptance の根拠文書を再参照する導線

### 4.3 `agent-protocols` と相性が良い

- IB-013

使いどころ:

- LLM 行動追跡と acceptance / release 証跡を結びつける
- 監査向けの最小レポートを生成する

## 5. 実施順のおすすめ

1. IB-001 Security 恒常対策
2. IB-002 Release 証跡チェック
3. IB-003 Task / Acceptance 双方向整合
4. IB-004 3 repo 統合 CI
5. IB-005 Metrics 判定自動化
6. IB-006 Birdseye freshness check

## 6. 完了済み

- IB-001
  - Dependabot / vulnerability alerts / secret scanning の恒常対策を
    security posture checker と docs へ反映済み
- IB-002
  - `CHANGELOG.md`、`docs/releases/`、git tag、GitHub release を照合する
    release evidence checker を追加済み
- IB-004
  - `workflow-cookbook` / `agent-taskstate` / `memx-resolver` を横断する
    cross-repo integration workflow を追加済み

## 7. 更新ルール

- backlog へ追加するときは ID を採番する
- 実装した項目は `CHANGELOG.md` に記録する
- 実装後は本書から削除せず、完了済みセクションへ移すか、
  後継タスクへリンクする
- cross-repo で実装する場合は対象 repo を必ず更新する

## 8. 参照先

- [README.md](../../README.md)
- [RUNBOOK.md](../../RUNBOOK.md)
- [EVALUATION.md](../../EVALUATION.md)
- [docs/ROADMAP_AND_SPECS.md](../ROADMAP_AND_SPECS.md)
- [docs/addenda/J_Test_Engineering.md](J_Test_Engineering.md)
