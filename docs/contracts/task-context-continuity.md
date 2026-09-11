---
intent_id: INT-CONTEXT-CONTINUITY
owner: Codex
status: active
last_reviewed_at: 2026-09-11
next_review_due: 2026-12-11
---

# taskstateを使った文脈の継続

## 目的と優先順位

入力の削減率を目標にしない。目的・制約・完了条件・決定・未解決事項・根拠を保持し、
途中再開でも作業の方向がずれない入力を組み立てる。tokenや時間は、この条件を満たした後の補助指標とする。
2026-09-11のDGX pilotは全文対照20/20に対し段階取得8/20、負の対照不成立だった。
6000 bytesの一律圧縮を運用上の推奨値として扱わない。

## 保存と再構成

内部状態の正本は既存agent-taskstate。task.goalとstateのconstraints、done_when、current_step、
current_summary、context_policy、refsを保存し、決定と未解決事項は既存decision/questionへ記録する。
ユーザーが目的や制約を変更したときは、その変更を正本へ反映する。更新を無視して古い方針を固定する機能ではない。
state更新は既存revision CASを使い、実行中stepの再開は既存checkpoint/reconcile契約に従う。

`TaskstateCLI.build_context`で既存`context build --reason recovery --rebuild-level L2`を呼び、
完全snapshotを追記保存する。`tools.context.taskstate.assemble`はそのsnapshot全体をJSONで保持する。
goal、state、決定、質問、run、解決済み情報を再要約・件数制限・文字数切捨てしない。
proposed/accepted等のstatusも保持し、提案を決定済みに変更しない。
bundle ID、generator version、source refs、diagnosticsとsnapshot hashを結果へ接続する。

ここでの完全snapshotは、agent-taskstateの標準bundleに含まれる内容全体を指す。
DB内の全履歴を復元する意味ではない。標準bundleはopen質問を対象とし、answered質問は含めない。
[128イベントの測定](../evidence/long-horizon-drift-20260911/RESULTS.md)では、
保存状態は正しくても解消済み質問IDが復元回答から欠落した。
解消済み質問・回答本文・根拠を必要とする用途は、この入口だけで情報が揃うとは判断しない。
過去のrejected決定も標準一覧から除外される。
解決済みの質問・回答・根拠や過去の決定は、必要時に履歴の正本へ問い合わせる。
全履歴を毎回の標準bundleやプロンプトへ復元することは要求しない。
この省略自体を欠陥や長期ドリフト防止の失敗とは判定しない。

## 必読原文と予算

`state.context_policy.workflow_context.required_documents`へ次を登録する。

```json
{
  "required_documents": [
    {"path": "docs/spec.md", "sha256": "sha256:<原文の64桁hash>"}
  ]
}
```

登録は利用者・呼出元の責務。質問に必要な原文集合を自動で完全判定する仕組みではない。
pathはrepo内の実ファイル、hashは登録した版との一致を要求する。
protected snapshotと全必読原文を先に確保し、残りの予算で関連資料を取得する。
hash不一致、必読不存在、予算不足を黙って除外しない。
予算は毎回の呼出しで指定し、必要最小bytesを返す。予算不足ではcontextを空にし、処理継続不可とする。
呼出元は予算を増やすか、正本側で合意されたtask分割・参照整理を行う。自動削減はしない。

## 状態照合と結果

呼出元が`task_id`と`expected_revision`を指定する。
最初のbundleのtask/state IDとrevision、必須field、完全snapshotを検証する。
資料取得後に別のbundleを再構成し、snapshotのhashが一致することを確認する。
state以外のgoal・decision・questionの変更も、snapshotが変われば検出する。
これは観測した前後の一致であり、最後の照合後の変更をロックする分散トランザクションではない。

- `ready`: 完全snapshotと登録必読原文を保持し、前後snapshot・原文hashが一致。
- `insufficient_budget`: 必須内容の必要bytesが予算を超えた。
- `state_changed`: 取得中にsnapshotが変わった。
- `needs_evidence`: resolverの欠落・unsupported・partial、または必読も検索結果もない。
- `invalid`: 不正入力、別task、古いrevision、改変必読、CLI障害。

`ready`以外はcontextをモデルへ渡さず、原因を解消してから再構成する。
`ready`は構造的な保持条件であり、モデルの意味的な正しさやドリフト率ゼロの保証ではない。
stateだけで完結する処理は明示的な`state_only`を使い、資料不足を隠す目的で切り替えない。
bundleはDBに追記されるが、task/state/decision/questionをこの入口から更新しない。
外部trackerへの書き込みやモデル呼び出しも行わない。

## CLIと検証

```sh
python -m tools.context.taskstate --state-client state-client.json --task-id TASK-ID --expected-revision 1 --repo-root . --query "次の作業" --budget-bytes 80000
```

state-clientはcheckpointと同じcommand/cwd/db/timeout形式。
80000は使用例であり固定推奨値ではない。stdoutはJSON、exit 0=ready、2=要対応、1=invalid。
返却bytesとモデルtokenは区別する。

必須状態の無損失保持、任意資料による予算圧迫、境界bytes、資料改変、異なるtask/revision、
decision変更、未解決根拠、CLI失敗を試験する。実agent-taskstate CLIでDB再接続後の復元も確認する。
モデルの長時間ドリフト率を評価するときは、別manifestで複数ターン・中断復元・ユーザー方針変更を凍結する。
今回の実装試験をその実モデル効果と呼ばない。
