---
intent_id: INT-WORKFLOW-EVOLUTION
owner: Codex
status: active
last_reviewed_at: 2026-09-10
next_review_due: 2026-12-10
---

# 評価・取得・観測・再開の利用手順

実装対象はAgent_tools版workflow-cookbook。repo rootで実行する。
この機能を追加したこと自体は、モデル成功率や速度の改善を意味しない。

| 順序 | 仕様 | 実装入口 |
| --- | --- | --- |
| 1 | [固定課題の比較](contracts/workflow-benchmark.md) | `python -m tools.evaluation` |
| 2 | [予算付き取得](contracts/progressive-context.md) | `python -m tools.context.progressive` |
| 3 | [task/runの観測](contracts/workflow-run-observation.md) | `python -m tools.workflow_plugins.run_report` |
| 4 | [checkpoint](contracts/workflow-checkpoint.md) | `python -m tools.workflow_plugins.checkpoint` |

## 1. 比較条件を凍結する

[非game manifestテンプレート](../templates/evaluation-measurement-manifest.template.json) をコピーし、
単一仮説・変更差分・負の対照、実装snapshot、runner、datasetを設定する。
benchmark固有の設定とrecord形式は仕様を参照する。source/runnerは単一ファイルまたはarchiveで固定する。

```sh
python -m tools.evaluation schedule --manifest frozen.json --dataset cases.json
python -m tools.evaluation compare --manifest frozen.json --dataset cases.json --observations observations.json --output-dir comparison-001
```

scheduleの順で利用者のrunnerが実行・採点し、観測を保存する。compareはモデルを起動しない。
比較ごとに1項目だけを変えれば、その概念を外した場合との差も検証できる。
train課題は集計せず、testの反復は課題内の変動として扱う。
欠測usageはnull、測定失敗はerror/timeout/crashにする。生成したreportのdecisionを確認する。

## 2. 文書をbyte予算内で取得する

目的・制約を継続して保持する作業では、先に[taskstateからの再構成](contracts/task-context-continuity.md)を使う。
以下の単独取得は補助資料の入口。削減率を目標にせず、必読の不足量に応じて予算を確保する。

```sh
python -m tools.context.progressive --repo-root . --query "checkpoint 再開" --budget-bytes 12000 --max-hops 2 --scope docs
```

`--required` は全文確保する資料、`--target-documents` は通常選択の目標件数。
必読が予算を超える場合は不足量を確認して予算を調整する。
`--plugin-config examples/workflow_plugins.cross_repo.sample.json --task-id 20260910-02` で既存memxの必読選択を接続できる。
このsampleは隣接repoの配置とplugin依存の導入を前提とする。
利用環境で `python -m pip install -e ../agent-taskstate -e ../memx-resolver` を実行する。
必要なpluginだけを設定してもよい。

context_bytesは実際の返却UTF-8量、source_io_bytesはretriever内部の読込量。
plugin内部のI/Oとモデルtoken数は別途測定する。raw contextをそのまま本文JSONへ複製する必要はない。

### taskstateで作業の方向を保持する

task.goal、stateのconstraints/done_when/current_step/current_summary、decision/questionを正本へ保存する。
`state.context_policy.workflow_context.required_documents`には必読原文のpath/sha256を登録する。
登録・更新は既存state patchのexpected_revisionを使い、他のcontext_policyを保持する。

```sh
python -m tools.context.taskstate --state-client state-client.json --task-id TASK-ID --expected-revision 1 --repo-root . --query "検証を続ける" --budget-bytes 80000
```

state-clientの形式は下記checkpointの例と共通。出力のready_for_model=trueを確認してからcontextを使う。
完全snapshotと必読全文は削らない。予算不足ならrequired_bytesを確認し、予算または合意済みの作業範囲を見直す。
`state_changed`は最新状態の再読込、`needs_evidence`は根拠の解決が必要。
stateだけで完結する作業に限り`--state-only`を使う。6000 bytes等の一律圧縮へ戻さない。
この入口は既存DBへbundleを追記するが、task状態の判断やユーザー方針を勝手に更新しない。

## 3. 同じrunへtraceと採点を接続する

```python
import time
from pathlib import Path
from tools.context.progressive import retrieve
from tools.workflow_plugins.runtime import RunContext, WorkflowPluginRuntime

started_at = time.time()
runtime = WorkflowPluginRuntime.from_config(
    "examples/workflow_plugins.cross_repo.sample.json",
    run_context=RunContext("20260910-02", "run-001"),
)
context = retrieve(
    Path("."), "checkpoint 再開", 12000,
    runtime=runtime, task_id="20260910-02",
)
runtime.write_traces_json("traces.json")
finished_at = time.time()
```

この始終を含むoutcome.jsonを採点者が作る。検収対象の作業も同じ時間区間へ含める。
acceptedは採点結果、未使用・未計測のinput_tokens/output_tokens/costはnullにする。
task_id/run_idはtraceと一致させ、acceptance_idで検収記録へ接続する。

```sh
python -m tools.workflow_plugins.run_report --traces traces.json --outcome outcome.json
```

正式Evidenceが必要なら `--evidence-context evidence-context.json` を追加する。
内容は `task_seed_id`（TS数値ID）、`base_commit`、`head_commit`、`actor`。
出力bundleのlinksがspanとEvidenceと検収を接続する。tool成功だけでacceptedをtrueにしない。

## 4. checkpointを保存して再開する

agent-taskstate CLIが利用可能な環境で、専用のtaskと初期stateを既存CLIで作る。
state-client.jsonは次の形式。各pathは実環境の絶対pathへ置き換える。

```json
{
  "command": ["agent-taskstate"],
  "cwd": "/absolute/workspace",
  "db": "/absolute/state/tasks.sqlite3",
  "timeout": 30
}
```

plan.jsonはtask_id/run_id/policy_version/steps/inputsを仕様どおり設定する。
policy_versionには利用する実装版も含め、inputsには仕様・設定など再開判断に必要な実ファイルのhashを入れる。
同じworkspace、coordinator保存先、plan、taskstate DBを再開時にも使用する。

```python
from pathlib import Path
from tools.evaluation.workflow import read_object
from tools.supervision.workspace_coordinator import WorkspaceCoordinator
from tools.workflow_plugins.checkpoint import TaskstateCLI, WorkflowCheckpoint

store = TaskstateCLI(**read_object(Path("state-client.json")))
coordinator = WorkspaceCoordinator(".", state_root=".workflow-cache/coordinator")
checkpoint = WorkflowCheckpoint(store, coordinator, read_object(Path("plan.json")))
checkpoint.grant = coordinator.acquire(
    mode="write", owner="runner", task_id=checkpoint.task_id,
    job_key=checkpoint.job_key,
)
if not checkpoint.grant["acquired"]:
    # terminal結果を使う場合もcheckpoint.status()で全成果物を照合する。
    raise RuntimeError(checkpoint.grant["reason"])
status = checkpoint.start()
if status["status"] == "needs_reconciliation":
    raise RuntimeError("結果不明のstepを確認してreconcileする")
```

処理直前に `checkpoint.transition("begin", step_id)` を呼ぶ。
成果物を確定した後、path/sha256配列を渡して `transition("complete", step_id, artifacts=[...])` を呼ぶ。
次回はstatus.remainingの順に進める。全件完了後にfinalizeする。job完了はtask検収とは別の記録になる。
長い処理ではcoordinator.heartbeatを継続する。lease tokenを共有文書・ログへ保存しない。

CLIにはstart/status/begin/complete/reconcile/finalizeがある。変更にはcoordinator.acquireの返却JSONを
`--grant` で渡す。grantをファイルへ保存する場合は利用者だけが読める場所に置き、終了後に除去する。
`--artifacts` はartifacts配列を持つJSON、`--record` は判断記録ファイルのpath/sha256を持つJSON。
reconcileのpendingは再試行許可、completedは成果物確認済みの明示判断である。

## 原文に基づく計算を追加する

数値の判断には[原文計算API](contracts/source-calculations.md)を利用できる。
呼出元が原文全文・hash・式・値の場所を指定すると、全案の計算結果と原文位置を返す。

```python
from tools.calculation.sources import derive, render

receipt = derive(documents, recipe)
model_context = restored_context + render(receipt)
```

documentsとrecipeは契約の形式で構成する。元の必読原文とsnapshotは維持し、追加後の予算を確認する。
receiptも実行証跡へ保存する。モデルの採用案や最終回答を正解表で書き換える処理ではない。

## 検証の範囲

[長い履歴の実測結果](evidence/long-horizon-drift-20260911/RESULTS.md)では、
目的・制約など7項目は全文履歴・確定保存・モデル更新の各96回答で保持した。
一方、標準snapshotに含まれない解消済み質問IDはtaskstateの2条件で全192回答から欠落した。
モデル更新192回の保存状態は正しかった。長期無ドリフトや性能優位を保証する結果ではない。
詳細な制限と次の改修対象は[診断](evidence/long-horizon-drift-20260911/DIAGNOSTIC.md)を参照する。

通常のunit fixtureは単独checkoutで動く。隣接agent-taskstate/memx-resolverがある環境では、
実CLIのDB再接続とmemx cacheの更新試験も実行する。隣接repoがない環境はその2件をskipする。

```sh
python -m pytest -q tests/evaluation tests/test_progressive_context.py tests/test_workflow_run_report.py tests/test_workflow_checkpoint.py
```

実モデル・実利用taskでの比較、Windows以外の動作、分散トランザクションはこのfixture検証では証明しない。
対応する [Task](tasks/task-workflow-evolution-20260910.md) と [Acceptance](acceptance/AC-20260910-02.md) を参照する。

## HATE・QEGで実行証跡を検証する

[接続仕様](contracts/hate-qeg-test-gate.md)のローカル受入runnerは、全pytestを新しく実行し、
HATEの正規化・exportとQEGのvalidate・gate・record・出力hash検証まで処理する。
HATE 0.3とbuild済みQEG 0.4のcheckout、pytest/covを含むPythonが必要。

```powershell
python -m tools.ci.hate_qeg `
  --repo . `
  --python C:/path/to/test-python.exe `
  --hate-root C:/path/to/harness-auto-test-evidence `
  --hate-python C:/path/to/harness-auto-test-evidence/.venv/Scripts/python.exe `
  --qeg-root C:/path/to/quality-evidence-graph `
  --node C:/path/to/node.exe `
  --output C:/path/to/new-evidence-directory
```

出力先は各repo外の新規directoryにする。`result.json`が件数・coverage・QEG判定、
`receipt.json`が実行情報、`source-snapshot.json`が未commit変更を含む対象hash、
`quality-evidence-record.json`がQEG正本。コマンド別のlogと終了codeも保存される。
終了0はこの自動テスト受入の成功、2は非Goまたは検証失敗、1は入力や実行基盤の問題。
既存CIへの導入時はHATE/QEGのruntimeを用意して同じCLIを呼ぶ。
手動BB、モデル性能、本番deploy、リリース承認はこの判定に含めない。
