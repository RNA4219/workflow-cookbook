# P. Expansion Candidates

## 1. 目的

`workflow-cookbook` の次段改善を、実装 backlog より広い視点で整理する。

本書は「今すぐ着手するタスク」だけでなく、
将来的に価値が高い拡張や運用高度化の候補を集約する。
直近実装候補は [N_Improvement_Backlog.md](N_Improvement_Backlog.md) を正本とし、
本書は中長期の設計メモとして扱う。

## 2. 対象領域

- docs / acceptance / release / security の運用高度化
- Birdseye / metrics / evidence の観測強化
- cross-repo plugin host の拡張
- downstream software 向け blueprint の具体化
- `agent-taskstate` / `memx-resolver` / `agent-protocols` と
  相性のよい運用拡張の整理

## 3. 候補一覧

### 3.1 Operations / Governance

- release / acceptance / evidence の横断レポート
  - 1 回の release がどの acceptance record と evidence に支えられているかを
    1 枚で可視化する
- security posture の差分サマリ
  - 前回成功時からの security settings 変更を比較できるようにする
- branch protection export の定期監査
  - `check_branch_protection.py` の結果を週次レポート化する
- docs 更新期限の自動棚卸し
  - front matter の `next_review_due` が近い docs を一覧化する

### 3.2 Quality / Observability

- metrics regression checker
  - 閾値内でも、前回成功値から悪化した指標を warn する
- acceptance / task / release の品質サマリ生成
  - coverage、検収、security、release evidence の結果をまとめる
- Birdseye stale remediation helper
  - stale 検知後に次の標準コマンドを提案する wrapper を追加する
- docs / sample / schema の同期チェック
  - `examples/`、schema、README / Runbook の記述ずれを検出する

### 3.3 Cross-Repo Plugins

- plugin timeout / retry / isolation policy
  - capability ごとに timeout と失敗隔離ルールを持てるようにする
- plugin tracing / structured diagnostics
  - host と plugin の呼び出し履歴を Evidence へ流せるようにする
- plugin capability catalog
  - どの repo がどの capability を提供しているかを自動列挙する
- downstream config doctor
  - plugin config と repo layout から導入可否を診断する CLI を追加する

### 3.4 Adaptive Improvement Loop

- reflection summary schema
  - release 後 reflection の最小 DTO を定義する
- curated memory snapshot
  - reviewed な判断だけを downstream software へ渡す export 形式を定義する
- optional skill draft generator
  - reflection 結果から下書き skill を提案するが、自動適用はしない
- workspace model review flow
  - user/workspace model を reviewed 状態でのみ公開できる運用を定義する

### 3.5 Docs / Navigation

- 用途別 docs hub
  - `Task / Acceptance / Security / Plugin / CI / Birdseye` の入口を
    1 枚に束ねる
- ADR / addenda / releases の索引自動生成
  - docs 増加時も見失いにくくする
- multilingual docs policy
  - 日本語 / 英語 / 中国語でどこまで同期するかをルール化する
- skill / README / About の同期チェック
  - 公開メタデータと docs のズレを継続検出する

## 4. repo 連携で特に効く項目

### 4.1 `agent-taskstate`

- acceptance / release / evidence の関係グラフ
- state export からの progress dashboard
- task 完了条件の厳格化

### 4.2 `memx-resolver`

- docs resolve / ack / stale の週次監査
- upstream / incident / release の横断参照
- reviewed docs set の選定補助

### 4.3 `agent-protocols`

- evidence timeline と release report の接続
- plugin 実行 tracing の記録先
- reflection / recall の監査用証跡

## 5. 進め方の目安

1. まず [N_Improvement_Backlog.md](N_Improvement_Backlog.md) の `P1` / `P2`
   を実装する
2. その後、本書の項目から
   `運用自動化` → `品質観測` → `cross-repo diagnostics` の順に広げる
3. downstream software へ出す機能は、必須機能ではなく
   optional capability として定義する

## 6. 更新ルール

- 実装段階へ入った項目は [N_Improvement_Backlog.md](N_Improvement_Backlog.md) へ移す
- 仕様化した項目は `requirements/spec/design/interfaces` にリンクを追加する
- 実装完了後は `CHANGELOG.md` に記録し、本書には後継項目だけを残す

## 7. 参照先

- [README.md](../../README.md)
- [RUNBOOK.md](../../RUNBOOK.md)
- [CHECKLISTS.md](../../CHECKLISTS.md)
- [docs/ROADMAP_AND_SPECS.md](../ROADMAP_AND_SPECS.md)
- [N_Improvement_Backlog.md](N_Improvement_Backlog.md)
- [O_Adaptive_Improvement_Loop.md](O_Adaptive_Improvement_Loop.md)
