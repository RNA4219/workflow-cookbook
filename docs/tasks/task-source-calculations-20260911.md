---
task_id: 20260911-03
intent_id: INT-SOURCE-CALCULATIONS
owner: Codex
status: done
---

# 元の全課題を全問正解にする

ユーザー依頼: 「怪しいなあ。全て正解になるまで粘ってくださる？」。
前回の完全正答は全文履歴11/20、agent-taskstate16/20。
誤りは計算値であり、採点を変えず原因へ対処する。

[仕様](../contracts/source-calculations.md)を実装し、実際のモデル入力へ接続する。
汎用整数計算・原文への根拠付け・全行保持を単体試験し、全体pytestとHATE/QEGを実行する。
旧結果を保存したまま、同じ12課題・2反復・4段階・2条件を新manifestで実行する。
全192応答を同じ判定で採点し、全件正解・計測障害なし・対照一致を目標とする。
全問正解でも長期ドリフトや任意の実作業での全問正解を保証したとは扱わない。

新しい実験の作業記録はworkspaceの `research/workflow-dgx-calculation-eval-20260911/TASK.md`。
同ディレクトリに凍結manifest・結果・失敗記録を保持する。

## 実施結果

全1071pytestとHATE/QEG Go、追加コードの行/分岐coverage100%。
最終attempt-002の全192応答が正解。attempt-001の対照誤答と全原本を保存した。
[AC-20260911-03](../acceptance/AC-20260911-03.md)に根拠と範囲を記録。
