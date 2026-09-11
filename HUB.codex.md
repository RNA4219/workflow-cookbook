---
intent_id: INT-001
owner: RNA4219
status: active   # draft|active|deprecated
last_reviewed_at: 2026-07-11
next_review_due: 2026-08-11
---

# Agent Tool Policy — Dual Stack

## Runtimes

- ネイティブの関数呼び出しツールを利用できる環境。
- ネイティブツールはなく、外部オーケストレータが本文の JSON を実行する環境。

## Rules

1. ネイティブツールが使える場合は、実際に登録された名前と引数で関数呼び出しを行う。
   呼び出し内容を本文の JSON に複製する必要はない。
2. ネイティブツールが使えず、対応する外部オーケストレータがある場合に限り、
   その契約に沿った `tool_request` JSON を本文に一度だけ出力する。例：

   ```tool_request
   {"name":"web.search","arguments":{"q":"..."}}
   ```

3. 同じ依頼を関数呼び出しと JSON 封筒の両方から実行しない。
   外部オーケストレータの実行結果が届くまでは、実行済みとして扱わない。
4. どちらの実行経路も使えない場合は、未実行であることと必要な手順を説明する。
   ツールの実行結果を捏造しない。
5. プラットフォーム固有マクロはそのまま残し、展開しない。
6. 既定の記述言語は日本語とし、コード識別子は必要に応じて原表記を使う。

## Logical Tool Names

以下は論理名の例。実行時は利用環境に登録されたツール名・スキーマ、または外部オーケストレータの契約に従う。

- web.search{q, recency?, domains?}
- web.open{url}
- drive.search{query, owner?, modified_after?}
- gmail.search{query, max_results?}
- calendar.search{time_min?, time_max?, query?}

## Output Contract

外部runnerやAPIが書式を要求する場合は、その出力契約に従う。
通常の説明は、成果・変更範囲・実際の検証・未完事項が分かる形式と長さで記述する。
`plan`/`patch`/`tests`/`commands`/`notes` は整理例であり、全応答の必須見出しではない。

## HUB.codex.md

リポジトリ内の仕様・運用MDを集約し、エージェントがタスクを自動分割できるようにするハブ定義。
`BLUEPRINT.md` など既存ファイルに加えて、オーケストレーション専用のMD（例: `orchestration/*.md`）も取り込む。

## 1. 目的

- リポジトリ配下の計画資料から作業ユニットを抽出し、優先度順に配列
- オーケストレーションMD（ワークフロー全体の段取り記載）を検出し、必要な子タスクへ展開
- 生成されたタスクリストを `TASK.*-MM-DD-YYYY` 形式の Task Seed へマッピング

## 2. 入力ファイル分類

- **Blueprint** (`BLUEPRINT.md`): 要件・制約・背景。優先順: 高。
  備考: 最上位方針。
- **Runbook** (`RUNBOOK.md`): 実行手順・コマンド。優先順: 中。
  備考: 具体的操作。
- **Guardrails** (`GUARDRAILS.md`): ガードレール/行動指針。優先順: 高。
  備考: 全メンバー必読。
- **Incident Logs** (`docs/IN-*.md`): インシデント記録（影響・再発防止など）。優先順: 高。
  備考: 再発防止策とフォローアップ抽出。
- **Evaluation** (`EVALUATION.md`): 受け入れ基準・品質指標。優先順: 中。
  備考: 検収条件。
- **Evaluation Identity Contract** (`docs/contracts/evaluation-identity-contract.md`): 比較評価の
  実装・入力・runner・用途別profile・測定集合・結果artifactを結ぶ契約。優先順: 高。
  備考: 評価系Task Seedは開始前にmanifest preflightを通し、agentは実行前に必読。
- **Checklist** (`CHECKLISTS.md`): リリース/レビュー確認項目。優先順: 低。
  備考: 後工程。
- **Orchestration** (`orchestration/*.md`): ワークフロー構成・依存関係。優先順: 可変。
  備考: 最優先のブロッカーを提示。
- **Birdseye Map** (`docs/birdseye/index.json` など): 依存トポロジと役割を把握。優先順: 高。
  備考: `plan` 出力にノードID/役割を埋め込む基準面。
- **Task Seeds** (`TASK.*-MM-DD-YYYY`): 既存タスクドラフト。優先順: 高。
  備考: 未着手タスクの候補。
- **Evidence** (`agent-protocols` 連携): LLM 行動追跡と証跡記録。優先順: 中。
  備考: `RUNBOOK.md#Observability` と `docs/CONTRACTS.md` を参照。

補完資料一覧:

- `README.md`: リポジトリ概要と参照リンク
- `CHANGELOG.md`: 完了タスクと履歴の記録
- `.github/PULL_REQUEST_TEMPLATE.md`: PR 作成時のチェック項目（Intent/リスク/Canary連携）
- `.github/ISSUE_TEMPLATE/bug.yml`: Intent ID と自動ゲート確認を必須化した不具合報告フォーム
- `governance/policy.yaml`: QA が管理する自己改変境界・カナリア中止条件・SLO
- `governance/prioritization.yaml`: 設計更新の優先度スコア計算ルール
- `docs/IN-*.md`: インシデントログ本体。Blueprint/Evaluation との相互リンクを維持し、再発防止策の同期を確認
- `docs/INCIDENT_TEMPLATE.md`: 検知/影響/5Whys/再発防止/タイムラインのインシデント雛形
- `docs/TASKS.md`: Task Seed 運用ガイドとテンプレートの要点
- `docs/ADR/README.md`: 最新 ADR 索引と更新手順
- `CODEOWNERS`: `/governance/**` とインシデント雛形を QA 管轄とする宣言
- `LICENSE`: OSS としての配布条件（MIT）
- `.github/release-drafter.yml`: リリースノート自動整形のテンプレート
- `.github/workflows/release-drafter.yml`: Release Drafter の CI 設定
- `docs/UPSTREAM.md`: Workflow Cookbook 派生リポからの知見取り込み手順と評価基準
- `docs/UPSTREAM_WEEKLY_LOG.md`: Upstream 差分確認の週次ログテンプレート
- `docs/addenda/A_Glossary.md`: 用語定義を参照するための補足資料
- `docs/addenda/D_Context_Trimming.md`: コンテキストトリミング指標・検証フローの詳細ガイド
- `docs/addenda/G_Security_Privacy.md`: SAC 原則に準拠したキー管理・ログマスキング等の運用ディテールを参照
- `datasets/README.md`: データセット取得履歴とハッシュを管理。データ保持レビュー時は本表で収集状況を確認

更新日: 2025-10-24

## 3. 自動タスク分割フロー

1. **スキャン**: ルートと `orchestration/` 配下を再帰探索し、Markdown front matter
   (`---`) を含むファイルを優先取得。
2. **必要な根拠の取得**: Birdseyeの形式・対象・登録・鮮度が作業に合う場合、indexと必要なcapsを使う。
   質問と返却量に応じて0/1/2hopを選び、未登録・破損・不適合なら通常検索で原文へ到達する。
   根拠のパスを記録し、存在しないノードIDや未読の内容を補わない。
3. **ノード生成**: 各ファイルから `##` レベルの節をノード化し、`Priority`
   `Dependencies` などのキーワードを抽出。
4. **依存解決**: Orchestrationノードに含まれる依存パスを解析し、該当セクションを子ノードとして連結。
5. **インシデント抽出**: `docs/IN-*.md` のインシデントセクションを走査。
   再発防止やテスト強化の箇条書きを Task Seed 候補としてタグ付け。
6. **粒度調整**: ノード内の ToDo / 箇条書きを、依存関係と検収可能性でまとめる。
   人間の作業計画で `<= 0.5d` を目安にできるが、時間換算や細分化を一律に強制しない。
7. **テンプレート投影**: 各作業ユニットを `TASK.*-MM-DD-YYYY` 形式の Task Seed
   (`Objective` `Requirements` `Commands`) へ変換し、欠損フィールドは元資料の該当行を引用。
   比較評価・実験・復元・昇格・提出を含む場合は、評価identity manifestへのリンクと
   preflight/postrun commandを必ず含める。
8. **出力整形**: 優先度、依存、担当の有無でソートし、GitHub Issue もしくは
   PR下書きとしてJSON/YAMLに整形。
9. **タスク化**: タスクは独立性が保てる粒度まで分割し、責務の重複(コンフリクト)を避ける。
   レビューと並行変更に応じて短いブランチと適切な同期方法を選ぶ。共有履歴の書き換えを一律には要求しない。
   リスクがある、タスクが重なっている場合は**Task Seeds** (`TASK.*-MM-DD-YYYY`)に記載を行うこと。

## 4. ノード抽出ルール

- Front matter内の `priority`, `owner`, `deadline` を最優先で採用
- 節タイトルに `[Blocker]` を含む場合は依存解決フェーズで最上位へ昇格
- 箇条書きのうち `[]` or `[ ]` 形式はチェックリスト扱い、`- [ ]` はタスク分解対象。詳細ステータスは後述`Task Status & Blockers`参照
- コードブロックはコマンドサンプルとして `Commands` セクションに集約

- **Task Status & Blockers**

```yaml
許容ステータス（Allowed）
- `[]` or `[ ]` or `- [ ]`：未着手・未割り振り
- planned：バックログ。着手順待ち
- active：受付済/優先キュー入り（担当/期日が付いた状態）
- in_progress：着手中
- reviewing：見直し中（レビュー/ふりかえり/承認待ち）
- blocked：ブロック中（外的依存で進められない）
- done：完了

遷移例（標準）
planned → active → in_progress → reviewing → done
ブロック例（例外）
in_progress → blocked → in_progress（解除後に戻す）
```

## 5. 出力例（擬似）

```yaml
- task_id: 20240401-01
  source: orchestration/api-rollout.md#Phase1
  objective: API Gateway ルーティング切替の段階実行
  scope:
    in: [infra/aws/apigw]
    out: [legacy/cli]
  requirements:
    behavior:
      - Blue/Green 切替時にダウンタイム0
    constraints:
      - 既存API破壊禁止
  commands:
    - terraform plan -target=module.api_gateway
  dependencies:
    - 20240331-ops-01
```

## 6. 運用メモ

- Orchestration MD には `## Phase` `## Stage` 等の段階名を揃える
- タスク自動生成ツールはドライランでJSON出力を確認後にIssue化
- 生成後は `CHANGELOG.md` へ反映済みタスクを移すことで履歴が追える
- Birdseye 鮮度: `docs/birdseye/index.json.generated_at` は 5 桁ゼロ埋めの世代番号として扱い、関連差分に対して未更新または Birdseye 資源間で不整合なら再収集を要求。
  該当 Capsule も同時更新。
- `codemap.update` は Birdseye 再生成時のみ実行。対象・読込量に応じて `--radius 0/1/2` を選ぶ。
  生成世代だけでは原文・要約を確認済みとせず、source_sha256とreviewの一致を確認する。
  ネイティブツールが使える場合は関数呼び出しを行い、使えず外部オーケストレータがある場合のみ
  `tool_request` を出力する。同じ依頼を両方の経路へ送らない。
