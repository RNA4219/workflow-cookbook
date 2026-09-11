---
intent_id: INT-WORKFLOW-EVOLUTION
owner: Codex
status: active
last_reviewed_at: 2026-09-10
next_review_due: 2026-12-10
---

# 固定課題によるワークフロー比較 v1

## 要求と境界

評価条件の違い・欠測・失敗を成功率や高速化に混ぜない。
既存[evaluation identity](evaluation-identity-contract.md) 1.1の非game profileを利用する。
外部モデルの起動・採点・課金はrunner側の責務。本ツールは実行順を作り、保存済み観測と成果物を検証・集計する。
任意コマンドを実行しない。モデル未使用はnot_usedとして記録する。

## 入力

- 凍結manifest: 既存checkerのpreflightを満たす。primary_metricはsuccess_rate。
- identity.legacy/candidate.source_pathは実装snapshotの単一file（archiveも可）。
  source_sha256と実ファイルを照合する。input_sha256は両方datasetのSHA-256と一致。
- evaluation.configuration.benchmark: repeats（正整数）、seed（非負整数）、reading_budget_bytes（正整数）、
  output_budget_tokens（正整数またはnull）、runner_path（runner実装snapshot file）。
  runner_pathはidentity.runner.sha256と照合する。全pathはmanifest所在ディレクトリ基準。
- dataset: schema_version=1.0、cases配列。case_idは一意、splitはtrain/test、
  inputとoracleはobject、negative_controlはbool。test課題のみ比較し、最低1つの負の対照を要求。
- observations: schema_version=1.0、manifest_sha256（凍結manifestのfile hash）、records配列。
  recordは一意なrun_id、case_id、repeat（0起算）、variant（legacy/candidate）、
  status（completed/error/timeout/crash）、success（completedのみbool、それ以外null）、
  wall_time_ms（有限・非負）、reading_bytes（非負整数）、
  input_tokens/output_tokens/interventions（非負整数または未計測null）、
  artifact_path/artifact_sha256を持つ。artifactはobservations所在ディレクトリ基準。

## 実行と比較

`python -m tools.evaluation schedule --manifest ... --dataset ...` は対ごとに条件順を乱択した計画JSONをstdoutへ返す。
同じseedで同じ順を作る。モデル出力の同一性をseedだけでは保証しない。

`python -m tools.evaluation compare --manifest ... --dataset ... --observations ... --output-dir ...`
は全課題×反復×2条件の完全な対応を要求する。
未知課題、train混入、重複、欠落、版hash不一致、不正数値、成果物不一致を拒否する。
reading予算超過、またはoutput予算が指定されているのに未計測・超過なら拒否する。
負の対照は両条件が同じsuccessでなければ負の対照不成立とする。

成功率はtest課題の全試行を分母とし、error/timeoutも失敗として含める。
負の対照は通常課題の性能集計から除き、別に成否を報告する。
時間・token・介入は条件別に分子・件数・平均・中央値・p95を返し、欠測はnullのまま件数を残す。
同じcase/repeatで対を作り、successと時間のcandidate-minus-legacyを返す。
反復を独立課題と誤認せず、課題ごとに平均した差の分布を併記する。信頼区間や有意差はこの版では主張しない。

## 出力と失敗

新規output directoryにreport.jsonとcompleted-manifest.jsonを一括確定する。
既存directoryは上書きしない。元manifest・dataset・observationsは変更しない。
reportのSHA-256・全record数・測定障害数・candidate成功率をcompleted-manifestへ記録し、
既存postrun検査を実行する。障害や負の対照不成立ではreportを保持し、decision=measurement_errorまたはinvalid_control。
障害なしでもdecision=owner_review_requiredであり、自動昇格・性能改善判定をしない。
exit 0=比較可能、1=不正入力、2=測定障害/負の対照不成立。例外のtracebackを通常のCLI出力へ混ぜない。

## 受入条件

- 条件の順序、test/train分離、予算、対応数、hashをテストする。
- 改善・悪化・全失敗・未計測・負の対照差をそれぞれ検証する。
- エラーを除外した成功率や欠測を0にしたコストを出さない。
- CLI出力と既存preflight/postrunの連携をfixtureで確認する。
- 実運用効果の測定とは区別して記録する。
