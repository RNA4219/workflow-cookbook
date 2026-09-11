---
task_id: 20260912-01
intent_id: INT-FULL-WORKFLOW-COPY
owner: workflow-cookbook
status: done
---

# フル準拠の参照一式を一括コピーする

- 要求: 1回の操作でWorkflow Cookbookのフル準拠をコピーする。
- [仕様](../contracts/full-workflow-copy.md)に従い、固定コミットの全ファイルと導入先の参照入口を設置する。
- 既存ファイルを保全し、再実行と整合性検査を提供する。
- [受入記録](../acceptance/AC-20260912-01.md)へCLI、配布境界、異常系の実結果を記載する。
- 単体fixtureとコピーの実CLI検証を実施する。モデル比較評価は実施しない。

## 完了結果

`wfc-copy`、元checkoutからのmodule実行、コピー先の単独検査を実装した。
新機能55件と全体1137件が成功し、wheelからの実コピーと全567 blobの独立照合を確認した。
外部repo前提の任意integration 3件はskip。詳細は
[AC-20260912-01](../acceptance/AC-20260912-01.md)を参照する。
