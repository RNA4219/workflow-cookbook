---
intent_id: INT-WORKFLOW-EVOLUTION
owner: Codex
status: active
last_reviewed_at: 2026-09-10
next_review_due: 2026-12-10
---

# task/runのtraceと検収結果 v1

## 要求

tool呼び出しが成功したことと、taskが検収を満たしたことを区別する。
既存PluginTraceをtask_id/run_id/invocation_id/span_idで接続する。
通常runtimeは従来どおり動作し、RunContextを指定したrunだけ確実にtraceを収集する。

## 収集

RunContext(task_id, run_id)をWorkflowPluginRuntimeへ渡す。
1回のinvokeに1つのinvocation_id、各retryに一意span_idを振り、attemptは1起算。
contextはインスタンスごとに固定し、他runへ混ぜない。raw入力・出力本文は追加しない。
既存のtrace_enabled設定も維持する。context未指定の旧traceは従来形式のまま利用できる。

## 検収入力と集計

`python -m tools.workflow_plugins.run_report --traces <JSON array> --outcome <JSON>`。

outcomeはschema_version=1.0、task_id、run_id、acceptance_id、accepted（bool）、
started_at/finished_at（epoch秒、有限、終了>=開始）、input_tokens/output_tokens/cost（非負またはnull）、
cost_currency（costありなら非空文字列）を持つ。採点者が作成する検収結果であり、traceから合格を推測しない。
ファイルSHA-256をレポートに保存し、判断の正しさは採点者の責務として残す。

各traceの識別子・時刻・attemptの連続性・重複を検証し、別task/runや未終了traceを拒否する。
task時間はoutcomeの始終、tool時間は各spanの合計であり、並列時には同じ値にならない。
capability別の回数・失敗・timeout・合計時間、invocation数、retry数、最終acceptedを出す。
欠測usageはnull。採点不合格は正常な観測結果として残し、入力エラーと混同しない。

## Evidenceの接続

任意の `--evidence-context <JSON>` はtask_seed_id、base_commit、head_commit、actorを持つ。
既存Evidence projectorを使用し、正式なEV数値ID・TS数値IDと必須フィールドを維持する。
外部Evidence schemaへ独自fieldを足さない。
bundleのlinksでEvidence ID、span ID、run ID、acceptance IDを接続する。
既存Evidence exporterの呼び出しと既定ID形式は互換維持し、新bundleの正式IDは明示的に選択する。
Evidenceのtool成否を最終検収のacceptedへ昇格させない。

## 受入条件

retry後成功・tool成功/task不合格・別run混入・重複span・attempt欠落・時刻不正・未計測usageを検証する。
正式Evidence schemaとの整合をfixtureで照合する。既存logger/inference plugin APIを変更しない。
