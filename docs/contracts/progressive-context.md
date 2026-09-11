---
intent_id: INT-WORKFLOW-EVOLUTION
owner: Codex
status: active
last_reviewed_at: 2026-09-10
next_review_due: 2026-12-10
---

# 予算付き段階取得 v1

## 要求

原文への到達と実際の返却量を両立する。既存context packの推定token予算は変更せず、
新しい取得入口ではUTF-8 bytesを正確に数える。bytesをモデルtoken数とは呼ばない。

## APIとCLI

`tools.context.progressive.retrieve` と `python -m tools.context.progressive` を提供する。
入力はrepo_root、query、budget_bytes、max_hops（0/1/2）、target_documents、scope_paths、required_paths。
CLIは `--repo-root --query --budget-bytes` が必須、`--scope` と `--required` は繰り返せる。
scope既定はrepo内Markdown。生成Birdseye・依存環境・Git管理内部は通常検索対象から除く。
repo外へ解決されるpathは使わない。

## 選択順

1. 必読原文を先に確保する。pluginを指定した場合は既存docs.resolveのrequiredを合流する。
2. file-map形式のBirdseyeから、質問に語句一致するcapsuleを選ぶ。
   原文hashとreviewのsummary hashを既存source_freshness契約で確認し、不明・古いcapsは根拠に使わない。
3. 件数と予算に余地があれば、選んだcapsのdeps_outを最大max_hopsまで展開する。
4. 足りないときはscope内原文の語句検索へ戻る。索引欠落・旧形式・壊れた索引は通常検索を妨げない。
5. 原文を確保し、重複を除き、path付きcontextを返す。関連度は語句一致のヒューリスティックで、回答の正解判定ではない。

context全体（path見出しを含む）のUTF-8 bytesを予算内にする。大きい通常文書は一致箇所付近の抜粋にし、
excerpt=trueと行番号を付ける。必読は全文を保持し、必読だけで予算を超えた場合はcontextを空にして
status=insufficient_budget、必要bytesを返す。読めていないものへackしない。
空結果はnot_found、予算や目標件数で途中停止した場合はpartial、それ以外はselected。
selectedは質問への回答が十分という判定ではない。

## 観測と互換性

source I/O bytes、返却context bytes、選択経路、hop、原文SHA-256、警告、所要時間を返す。
source_io_scope=retriever_onlyであり、plugin内部のI/Oはこのbyte数へ含めない。
pluginありでは既存memx-resolverのsignature-aware cacheに委譲し、hostへ別cacheを作らない。
plugin errorsは明示して失敗させ、required文書を黙って捨てない。pluginのread receiptは変更しない。
CLIはJSONのみをstdoutへ出し、通常処理0、入力・plugin障害1、必読予算不足2。

## 受入条件

indexの未存在・古さ・不正形状、新文書未登録、依存cycle、0/1/2hop、重複、必読不足、
日本語のbyte境界、repo外path、予算ぴったり、plugin cache経路をfixtureで検証する。
通常全文検索との実運用の優劣は別の凍結評価で判断する。
