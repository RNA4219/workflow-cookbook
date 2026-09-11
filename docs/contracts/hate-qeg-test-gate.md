---
intent_id: INT-WORKFLOW-HATE-QEG
owner: Codex
status: active
last_reviewed_at: 2026-09-11
next_review_due: 2026-12-11
---

# HATEからQEGへの実テスト接続

## 対象と責務

workflow-cookbookの現在の作業ツリーでpytestを実行し、JUnitとcoverageをHATEの公開CLIで正規化する。
HATEのexportを保存し、QEG 0.4のnative_graph契約へ投影してvalidate、gate、recordを実行する。
HATE/v1 bundleはQEG 0.2のnative graphとは形が異なるため、直接同一schemaとして扱わない。
接続処理は `tools.ci.hate_qeg`、判定はQEGが所有する。

## 実行契約

- 出力先は対象repo外の新規directory。再実行で以前の証跡を上書きしない。
- HATEのsource/schema、QEGの実行dist/schemaとNode依存を出力先のtoolchainへコピーする。
  コピー前後のhashを照合し、以降はコピーだけを使う。元repoで別作業が進んでも実行途中に切り替えない。
  元repoのrevision/dirtyと実際のコピーのhashを残す。Python/Node実行ファイルは指定した既存runtimeを使う。
- 対象repoのHEADに加え、Git管理対象と未追跡・非ignoreファイルのSHA-256一覧を固定する。
  未commit変更を含む一覧のhashをbuildIdへ結び、実行前後の一致を確認する。
- pytestは全testsを実行し、JUnit XML、Cobertura XML、coverage JSON、終了codeと開始・完了時刻を保存する。
  Windowsでは子processにもPYTHONUTF8=1を継承する。実行依存は事前に導入する。
- HATEはci-context.jsonのgeneric-ciとして記録する。GitHub Actions実行を装わない。
  `p0a`、`export qeg`のstdout/stderr、終了code、正規化原本を保存する。
- JUnitの全caseとHATEのidentity・statusを一対一照合する。空集合、重複、欠落、対象run/revision不一致は停止する。
  failed/errorをfail、skippedをskippedとしてQEGへ渡す。未認識statusをpassにしない。
- pytest自体の終了codeと全体coverage閾値80%も独立した実行結果としてQEGへ渡す。
  個別テストが全passでもcoverage失敗やpytest異常終了を隠さない。
- native実行原本にはHATEから得たcase identity、status、run、完了時刻と対象buildを保持する。
  元JUnit/HATE/bundle/coverage/実行receiptもrequired input artifactとしてhash検証させる。
  実行mode=realはpytestを実際に起動した意味で、各テスト内のstubやmockを実サービス利用と主張しない。
- QEGはstandard profile、requireExecutedTests=true、期限24時間の技術テスト受入policyを使用する。
  requirement→obligation→placement→test→execution→原本を辿れるようにする。
  policyHashは今回のpolicy実体から計算し、waiverとrelease approvalは付与しない。
- QEGのvalidate/gate/record/outputs readの結果と各コマンド終了codeを残す。
  validateの期待値は「全テスト成功・coverage閾値充足・証跡完全ならGo」という本仕様から実行前に保存する。
  実際のGate出力を期待値へコピーしない。
  判定がgoでも検証command失敗・source変更・HATE部分出力があれば接続処理の成功としない。

## 受入

実際の全pytestをHATE→QEGまで通す。別の合成入力による接続試験で、失敗case、skip、欠落、hash不一致を
goにできないことを確認する。合成試験の結果を製品機能の成功件数に加算しない。
件数とcoverageは今回の原本から集計し、直近4機能のcoverageと全体coverageを分ける。

今回の判定範囲はローカルの自動テスト受入。RanD、Code-to-gate、手動BB、モデル実運用の性能比較、
本番deploy、リリース承認はこの実行では評価しない。未実行stageを実行済みにしない。
