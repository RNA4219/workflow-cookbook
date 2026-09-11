---
intent_id: INT-FULL-WORKFLOW-COPY
owner: workflow-cookbook
status: active
last_reviewed_at: 2026-09-12
next_review_due: 2026-12-12
---

# フル準拠の参照一式を一度でコピーする

## 要求と適用範囲

ユーザー要求は「1発でWorkflow-cookbookのフル準拠をコピーする機能」。
一度のCLI実行で、指定したコミットの全追跡ファイルと、導入先エージェントの参照入口を設置する。
ルールの短い再要約を正本にせず、HUB、Guardrails、Task Seed、Acceptance、Birdseye、CI、
Evidence、HATE/QEG接続、taskstateの契約と実装を原文のまま参照できるようにする。

「フル」はコピー元コミットの収録範囲を表す。コピーだけで導入先の実装・運用・品質ゲートの
合格を保証しない。コピー元のTask、Acceptance、評価結果は上流の資料であり、導入先の実績ではない。
外部ツール本体、認証情報、未追跡ファイル、作業中差分、Git履歴はコピー対象外。
導入先のCI設定、外部サービス、ブランチ保護設定、リポジトリ固有の承認条件は自動変更しない。

## 入出力

```sh
python -m tools.adoption --repo /path/to/target
python -m tools.adoption --repo /path/to/target --dry-run
python -m tools.adoption --repo /path/to/target --check
```

コピー元は既定でカレントディレクトリ。`--source`でcheckout、`--ref`でcommit/tag/branchを指定できる。
Gitが解決したcommit IDとtree IDを固定し、作業ツリーの変更を取り込まない。
wheel版は`wfc-copy --source /path/to/workflow-cookbook --repo /path/to/target`を使う。
コピー時はPython 3.11以上とGitが必要。検査はコピー元checkoutやGitへの接続なしでも実行できる。

- `workflow-cookbook/upstream/`: 全追跡ファイル。相対参照と実行ファイルの属性を保持する。
- `workflow-cookbook/ADOPTION.md`: 導入先への適用方法、原文入口、残る設定・検証の説明。
- `workflow-cookbook/manifest.json`: format version、commit/tree、全ファイルのSHA-256とmode。
- `workflow-cookbook/verify.py`: 導入時のCLIコード。コピー元なしで単独検査でき、hashも記録する。
- `AGENTS.md`: 既存のバイト列を残し、管理対象の参照ブロックを末尾へ追加する。

stdoutはJSON。exit 0はコピー完了・同一内容の再実行・dry-run成功・整合性検査成功、
exit 1は不正入力、衝突、破損、I/O失敗。全出力で運用準拠は`not_evaluated`とする。

## 保全と整合性

- コピー元と導入先の包含関係を拒否し、自己コピーを防ぐ。
- シンボリックリンク、junction、submodule、通常ファイル以外のコピー元要素は拒否する。
- 新規領域は一時領域に準備してから配置する。既存の同名領域や管理ブロックには上書きしない。
- 準備中の失敗では一時領域を撤回する。公開後に失敗した場合はコピー領域とAGENTSを保持してエラーにする。
- 既存AGENTSはO_APPENDで追記し、全体置換や元の内容への書き戻しは行わない。
  新規AGENTSは排他的に作成する。追記前後の内容・実体の変化、短い書込は要対応として返す。
- 追記は全プロセス間のトランザクションではない。競合時には利用者の変更と追記を残し、
  自動削除・再試行をせず、現物とmanifestを確認してから管理ブロックの整理や再導入を行う。
- 再実行では元のcommit/tree、全管理ファイル、AGENTS参照ブロックが一致する場合だけ成功する。
- ファイルの欠落、改変、追加、不正manifest、不完全な管理ブロックは失敗とし、黙って修復しない。
- 別版への切替は既存コピーと衝突として返す。自動更新で独自差分を消す機能は今回に含めない。
- ハッシュはローカルコピーの整合性を検査するもので、署名による配布元認証ではない。

## 導入先の指示

上位指示、ユーザーが許可した作業、導入先固有の指示を確認し、共有契約を適用する。
上流のowner、数値目標、環境パス、承認条件を導入先へ無条件に移植しない。
適用できない条件や未接続の外部ツールは根拠付きで記録し、実行済みとしない。
目的・制約・進捗・決定・未解決事項を維持し、解決済みの履歴は必要時に参照する。
全履歴を毎回復元する義務、固定の行数・ファイル数上限、本文JSONへのツール呼び出し複製義務を追加しない。

## 検証

[Task](../tasks/task-full-workflow-copy-20260912.md)と
[Acceptance](../acceptance/AC-20260912-01.md)に実行結果を記録する。
空の導入先、既存AGENTSの保持、再実行、dry-run、改変・追加・欠落、途中失敗、
不正パス、別版衝突、Gitの実CLI、wheelからのCLI実行を検査する。
これはコピー機能の実装検証であり、モデル性能・長期ドリフト率の比較実験ではない。
