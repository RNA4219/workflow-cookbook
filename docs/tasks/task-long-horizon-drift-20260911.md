---
task_id: 20260911-04
intent_id: INT-LONG-HORIZON-DRIFT
owner: Codex
status: done
---

# 長い履歴と記憶更新のドリフト評価

ユーザー依頼: 「長期ドリフト防止検証も実施しましょう」。
前回4段階の回帰では計算誤りを解消したが、長い履歴とモデルによる記憶更新は未評価だった。

新規6シナリオ・128イベント・16更新batch・8回答点・2反復で、全文履歴と
agent-taskstateの確定更新／モデル抽出更新を比較する。負の対照を含め504推論を予定する。
目的・制約・決定・未解決事項・許可の保持と明示変更への追従を、計算や形式の誤りから分けて集計する。
品質成績の良否を問わず、失敗原本と実測条件を保存して報告する。

[公開用の測定記録](../evidence/long-horizon-drift-20260911/README.md)と
[固定仕様](../evidence/long-horizon-drift-20260911/PROTOCOL.md)を参照する。
全原本の正本はworkspaceの `research/workflow-long-horizon-drift-20260911/` に保持する。
2つの比較manifestでpreflight/postrunを実施し、独立auditを記録する。
暦日で数か月の運用や実務全般の無ドリフトを保証したとは扱わない。

## 実施結果

全504要求・独立監査・postrun・31fixture検証を完了。
解消済み質問を復元入力に含めない制限と、項目別の成績を記録した。
[AC-20260911-04](../acceptance/AC-20260911-04.md)を参照。
