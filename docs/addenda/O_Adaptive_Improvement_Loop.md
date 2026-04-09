---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-04-10
next_review_due: 2026-05-10
---

# O. Adaptive Improvement Loop

## 1. 位置付け

本書は、`workflow-cookbook` が下流ソフトウェアへ提供する
「自己改善オペレーション」機能群の補助設計である。
ここで扱うのは `hermes-agent` との接続や互換実装ではなく、
そこから着想を得た repo 非依存の運用パターンを
`workflow-cookbook` の独自機能として再定義したものである。

## 2. 目的

下流ソフトウェアが次を実現できるようにする。

- セッションごとの作業結果を要約して次回へ持ち越す
- 複雑な作業後に skill 改善候補を生成する
- skill 実行中の学びを差分として記録する
- 過去セッションから関連文脈を検索して再利用する
- ユーザーやワークスペースごとの傾向を安全に再利用する
- 上記を review / acceptance / evidence と矛盾なく運用する

## 2.1 有効化タイミング

- 本機能は任意機能であり、全ての下流ソフトウェアに必須ではない。
- 想定する主な有効化タイミングはリリース後運用である。
- 実装中、作成途中、仕様が固まっていない段階では無効化できることを前提とする。
- 未有効化時でも、通常の task / acceptance / release / security フローを妨げない。

## 3. 基本原則

### 3.1 独立性

- `workflow-cookbook` は `hermes-agent` を dependency としない。
- 同一機能名や内部データ構造を前提にしない。
- 参考にするのは運用アイデアのみであり、実装契約は本 repo が定義する。

### 3.2 外向き実装

- 実際の学習ループ本体は、`workflow-cookbook` 自身ではなく
  下流ソフトウェアが実装・利用する。
- `workflow-cookbook` は次を提供する。
  - 要件
  - 仕様
  - 境界定義
  - review / acceptance / evidence との接続方針

### 3.3 repo 非依存

- 特定 repo 固有のファイル名や DB を必須にしない。
- 最低限必要な入力は task / session / evidence / skill / docs reference に絞る。
- plugin 形式で差し替え可能にする。

### 3.4 安全性

- 自己改善候補は即時自動反映しない。
- 少なくとも draft / review / approved の段階を持つ。
- acceptance と evidence の参照先を残し、監査可能にする。

### 3.5 非侵襲性

- 作成途中の change set に割り込んで mandatory action を増やさない。
- reflection / recall / nudge / skill draft は optional operation として扱う。
- 無効化時は no-op で終了し、主要作業フローに影響を与えない。

## 4. 機能モジュール

### 4.1 Session Reflection

- セッション終了時に次を抽出する。
  - 目的
  - 実施した変更
  - 学んだこと
  - 未解決事項
  - 次回の推奨アクション
- 出力は repo 非依存の `ReflectionSummary` とする。

### 4.2 Memory Curation

- reflection を長期保持候補と短期保持候補へ分類する。
- 長期保持は skill 改善、運用ルール、ユーザー嗜好、危険パターンを中心とする。
- 短期保持は直近作業の継続文脈として扱う。

### 4.3 Periodic Nudges

- 長期間未整理の reflection や stale な docs / acceptance / release evidence を
  定期的に見直すための nudge を生成する。
- nudge は自動実行ではなく、次回セッションで参照できる提案とする。
- nudge はリリース前の未完了作業に対する blocking signal として扱わない。

### 4.4 Skill Evolution

- 複雑な task 完了後、skill 化候補を draft として生成できる。
- 既存 skill 実行中に得た改善点は skill patch candidate として記録できる。
- draft / patch candidate は review を通過してから公開対象へ昇格する。

### 4.5 Cross-Session Recall

- 過去セッションから関連 reflection / acceptance / evidence / docs 参照を
  検索し、要約して返す。
- recall は raw transcript 全展開ではなく、要約結果と出典リンクを返す。

### 4.6 User / Workspace Model

- ユーザーごとの好み、承認閾値、レビュー傾向、出力形式の嗜好を
  `UserModelSnapshot` として保持できる。
- workspace ごとの制約、repo の注意点、頻出 runbook を
  `WorkspaceModelSnapshot` として保持できる。
- いずれも明示的に review された情報のみを長期保持対象とする。

## 5. 他 repo への適用方針

### 5.1 最低限必要な入力

- task ID または session ID
- reflection 対象の evidence / acceptance / docs reference
- skill registry への書き込み口
- memory store または search backend

### 5.2 差し替え対象

- memory store
- search / summarization backend
- skill registry
- scheduler / nudge generator
- user model store

### 5.3 差し替え不要の共通契約

- reflection summary の最低フィールド
- skill draft の状態遷移
- recall response の最低フィールド
- evidence / acceptance とのリンク方法

## 6. workflow-cookbook で担保すること

- docs による契約定義
- plugin host / interface への差し込み口
- acceptance / evidence / release / security と矛盾しない運用パターン
- 下流 repo が review 付きで導入できるようにする task 化の導線
- 任意機能として段階導入・段階無効化できる前提

## 7. 導入時の禁止事項

- `workflow-cookbook` 自身へ学習ループ本体をベタ実装しない
- 単一 agent 実装の内部 API を正本としない
- user model や long-term memory を無審査で永続化しない
- skill 自動生成結果を review 無しで公開しない

## 8. 次の実装候補

- `ReflectionSummary` / `SkillDraftRecord` / `RecallResponse` の schema 化
- self-improvement plugin capability の追加
- acceptance / evidence と skill draft の関連付け checker
- stale reflection に対する nudge checker
