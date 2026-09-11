---
task_id: 20260911-02
intent_id: INT-CONTEXT-CONTINUITY
owner: Codex
status: in_progress
---

# 文脈の削減より継続性を優先する

ユーザーの指摘: 「減らしすぎじゃない？ドリフトさせないのが大事」「agenttaskstate使うのもええ」。

- [仕様](../contracts/task-context-continuity.md)を先に固定する。
- agent-taskstateの完全snapshotと必読原文を保持する組立API/CLIを追加する。
- budget不足・状態変更・根拠欠落時は継続可能として返さない。
- 既存checkpoint/CAS、agent-taskstate DB/schemaを再利用する。
- 実CLIのDB再接続と単体回帰を実行し、HATE→QEGで自動テスト証跡を確認する。
- 過去のDGX pilot原本とinvalid_control判定は変更しない。

この作業の検証は実装・復元の受入であり、モデル性能の比較実験を実行したとは記録しない。

## 実施結果

実装と自動受入は完了。[AC-20260911-02](../acceptance/AC-20260911-02.md)を参照。
全1004テスト、HATE/QEG各CLI成功、QEG Go。新規モジュールの行coverageは91.28%。
今回のtaskを既存agent-taskstate CLIへ登録し、DB再接続から必須文脈を復元した。
実モデルの複数ターンドリフト率は未評価のためin_progressとし、未解決事項をDBにも残す。
過去の日本語検索・capsule順位の問題は本変更で解決済みとせず、別の取得改善として保持する。

## 後続の測定

[128イベントの測定](../evidence/long-horizon-drift-20260911/README.md)でモデル更新と復元を評価した。
目的・制約等7項目は保持したが、解決済み質問IDが標準bundleから欠ける制限が残る。
上の「未評価」は実装受入時点の記録であり、長期ドリフト防止の達成は引き続き未完了。
