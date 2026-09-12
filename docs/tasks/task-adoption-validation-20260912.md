---
task_id: 20260912-02
intent_id: INT-ADOPTION-VALIDATION
owner: workflow-cookbook
status: done
---

# コピーと導入検査の誤判定を修正する

[検査仕様](../contracts/adoption-validation.md)に従い、Git checkout後の整合性、
Tierの内容検査、テンプレート版不明の扱いを修正する。
[受入記録](../acceptance/AC-20260912-02.md)に回帰テストと配布CLIの結果を残す。
モデル性能や長期ドリフト率の比較実験は行わない。

- Git設定と既存AGENTSを保全し、元のSHA-256を保持する。
- 正当なテキストのLF/CRLF変換と内容変更を区別する。
- 空・種類違い・破損・比較不能を合格にしない。
- コピー更新機能と外部接続doctorは今回の3件の修正後に扱う別の拡張候補。
