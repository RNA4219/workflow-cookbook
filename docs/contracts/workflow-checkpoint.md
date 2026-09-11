---
intent_id: INT-WORKFLOW-EVOLUTION
owner: Codex
status: active
last_reviewed_at: 2026-09-10
next_review_due: 2026-12-10
---

# 永続checkpointと再開 v1

## 要求と保存先

既存agent-taskstateのstate get/patchとrevision CASを利用する。
context_policy.workflow_checkpointへ保存し、他のcontext_policy、制約、typed_ref、task statusを保持する。
別のtask状態DBや状態schemaを追加しない。taskと初期stateの作成は既存CLIで事前に行う。

planはschema_version=1.0、task_id、run_id、policy_version（実装版も識別する）、steps（順序付き一意文字列）、
inputs（path/sha256配列）を持つ。pathはworkspace相対の実ファイル、hashはsha256:形式。
入力の追加・並び・方針・実装の変更はplan hashを変える。再開時は保存済みhashと全入力を照合する。
job keyは既存deterministic_job_keyへplan hashを含めて生成する。同じplanとworkspaceに同じkeyを割り当てる。

## 状態遷移と成果物

startで全stepをpendingとしてCAS保存する。同じplanの再startは現状を返し、初期化し直さない。
beginは最初の未完了stepだけをrunningへCAS保存し、その成功後に呼出元が処理を実行する。
completeはrunningのstepだけをcompletedへ更新し、1件以上の確定済み成果物のpath/hashを照合して保存する。
同じ成果物によるcomplete再送は重複実行せず受理する。異なる成果物による書換えは拒否する。
ファイル出力は呼出元が一時ファイルからrename等で確定してからcompleteする。

statusは保存状態・入力・全完了成果物を照合する。完了済みは再実行せず、pendingをremainingとして返す。
runningがあればneeds_reconciliationとし、次のbeginを拒否する。
中断がbegin後・処理実行前でも、結果不明の外部操作と区別できないため自動再試行しない。
reconcileには利用者が検証した記録ファイルのpath/hashと、completedまたはpendingの明示判断を要求する。
completed判断には成果物も必要。pending判断は再試行許可を意味し、判断記録をcheckpointへ保持する。
採点や外部サービス照会の正しさは記録作成者の責務。自動で外部操作を実行しない。

## leaseと中断の境界

変更には既存WorkspaceCoordinatorから取得したworkspace全体のwrite leaseを必須とする。
task/job/workspace、lease ID、fenceと有効性を照合し、heartbeatでtokenを検証する。秘密tokenはcheckpointへ保存しない。
期限切れ、別task/job、read leaseでは保存できない。並行state更新はexpected_revision不一致で止まり、勝手に上書き・再試行しない。
長時間処理中のheartbeatは呼出元の責務。coordinatorとtaskstateは別DBなので原子的な分散commitや外部副作用のexactly-onceは保証しない。
lease確認後の停止・CAS応答喪失ではstateを再取得し照合する。協調しないwriterに対するファイルロックを提供するものではない。

finalizeは全step完了と成果物照合の後で既存coordinator.finishを呼ぶ。taskの最終検収合格とは別のjob完了である。
checkpoint保存後・finish前で止まった場合も、再取得したleaseでfinalizeできる。
terminal jobの再利用はstatus=completedの照合結果が必要。invalid jobや改変成果物を成功へ昇格させない。

## API・CLI・検証

tools.workflow_plugins.checkpointのTaskstateCLIとWorkflowCheckpointを提供する。
TaskstateCLIは信頼するCLIのargv配列・cwd・DBを設定し、shellを介さず実行する。JSON envelopeの失敗とtimeoutを伝える。
Python CLIの標準出力をUTF-8に固定し、Windowsの日本語状態もJSONとして受け取る。
大きいpatchは一時JSON fileで渡し、CLI応答後に削除する。CLI timeout時は成功不明として再取得を促す。

`python -m tools.workflow_plugins.checkpoint --help` にAPI相当の入口を用意する。
plan、state-client設定、workspace、coordinator-rootを明示する。変更時のみlease grantのJSON fileを渡す。
statusのneeds_reconciliationはexit 2、契約違反・競合はexit 1、正常はexit 0。

DB再接続、途中停止、完了再送、入力/成果物変更、revision競合、lease失効、reconcile、
保存後finish前の復旧をfixtureで検証する。外部副作用を起こす試験や性能評価は含めない。
