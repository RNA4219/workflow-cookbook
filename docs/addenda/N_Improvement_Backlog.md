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

全項目完了済み。詳細は「6. 完了済み」セクション参照。

| ID | 状態 | 完了日 |
| :-- | :-- | :-- |
| IB-001 | ✓ 完了 | 2026-04-17 |
| IB-002 | ✓ 完了 | 2026-04-17 |
| IB-003 | ✓ 完了 | 2026-04-10 |
| IB-004 | ✓ 完了 | 2026-04-17 |
| IB-005 | ✓ 完了 | 2026-04-11 |
| IB-006 | ✓ 完了 | 2026-04-11 |
| IB-007 | ✓ 完了 | 2026-04-17 |
| IB-008 | ✓ 完了 | 2026-04-10 |
| IB-009 | ✓ 完了 | 2026-04-10 |
| IB-010 | ✓ 完了 | 2026-04-10 |
| IB-011 | ✓ 完了 | 2026-04-17 |
| IB-012 | ✓ 完了 | 2026-04-10 |
| IB-013 | ✓ 完了 | 2026-04-10 |
| IB-014 | ✓ 完了 | 2026-04-10 |
| IB-015 | ✓ 完了 | 2026-04-10 |
| IB-016 | ✓ 完了 | 2026-04-10 |
| IB-017 | ✓ 完了 | 2026-04-10 |
| IB-018 | ✓ 完了 | 2026-04-10 |

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

全項目完了済み (2026-05-03確認)。新規改善案は BLUEPRINT.md / docs/requirements.md / docs/spec.md の「先行改善案」セクション参照。

## 6. 完了済み

- IB-001
  - Dependabot / vulnerability alerts / secret scanning の恒常対策を
    security posture checker と docs へ反映済み
- IB-002
  - `CHANGELOG.md`、`docs/releases/`、git tag、GitHub release を照合する
    release evidence checker を追加済み
- IB-003
  - Task/Acceptance 双方向整合チェッカー `check_task_acceptance_bidirectional.py` 追加
- IB-004
  - `workflow-cookbook` / `agent-taskstate` / `memx-resolver` を横断する
    cross-repo integration workflow を追加済み
- IB-005
  - `tools/ci/check_metrics_thresholds.py` と
    `governance/metrics_thresholds.yaml` を追加し、
    `.ga/qa-metrics.json` の KPI 閾値判定を warn / fail で自動化済み
- IB-006
  - `tools/ci/check_birdseye_freshness.py` を追加し、
    `generated_at`・`mtime`・caps 参照・`last_verified_at` の
    freshness check を実行可能にした
- IB-007
  - `docs-resolve-pr-gate.yml` workflow 追加、PR comment へ docs resolve 結果投稿
- IB-008
  - Acceptance Index 生成ツールを拡張し、
    status summary / records テーブル形式を出力
- IB-009
  - `check_sample_docs_sync.py` 追加、sample と docs の同期チェック
- IB-010
  - `extract_upstream_changes.py` 追加、UPSTREAM 差分抽出
- IB-011
  - Plugin runtime に timeout / retry / error policy と tracing 機能を追加
- IB-012
  - `export_task_state.py` 追加、TaskState JSON export 機能
- IB-013
  - Evidence / Acceptance 連携レポート生成ツール `generate_evidence_report.py` 追加
- IB-014
  - `generate_navigation_hub.py` 追加、用途別 docs hub 自動生成
- IB-015
  - ADR / addenda 索引自動生成ツール `generate_docs_index.py` 追加
- IB-016
  - `check_security_docs_freshness.py` 追加、security docs 更新チェック
- IB-017
  - `generate_release_notes.py` 追加、Release Notes 自動生成
- IB-018
  - `knowledge_reuse.py` 追加、release/acceptance/incident 横断参照 CLI

### Known Limitations 完了

- 0005
  - SLO バッジ自動生成ツール `generate_slo_badges.py` を追加し、
    `policy.yaml` の SLO 値を README へ自動反映

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
