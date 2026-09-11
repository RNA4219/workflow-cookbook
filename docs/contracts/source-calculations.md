---
intent_id: INT-SOURCE-CALCULATIONS
owner: Codex
status: active
last_reviewed_at: 2026-09-11
next_review_due: 2026-12-11
---

# 原文からの厳密な整数計算

## 要求

文脈を保存しても、モデルが単価を取り違えたり整数商を切り上げたりする誤りは残る。
`tools.calculation.sources` は、hashで指定した原文の値と呼出元の式を使い、
選択肢の全行について厳密に整数計算する。特定課題の正解表、採用済み案、モデル回答は入力にしない。
計算結果と原文位置をモデルへ渡し、モデル自身が現在の目的・決定に合う行を選んで回答する。
回答を採点oracleで補正したり、失敗例だけに答えを埋め込んだりしない。

## 入力と公開API

`derive(documents, recipe)` のdocumentsはpathをkeyとする
`{"content": "原文全文", "sha256": "sha256:..."}` のmapping。
関数はファイル・ネットワークへアクセスせず、渡された原文と期待hashを照合する。
recipeのschema_versionは `1.0`。

```json
{
  "schema_version": "1.0",
  "expression": "(budget-reserve)//unit_cost",
  "scalars": {
    "budget": {"path": "limits.md", "key": "budget"},
    "reserve": {"path": "limits.md", "key": "reserve"}
  },
  "table": {
    "path": "catalog.md",
    "row_key": "plan",
    "bindings": {"unit_cost": "unit_cost"}
  }
}
```

scalarは一意な `key=整数` 行、tableは一意に特定できる単純なMarkdown tableを扱う。
全行を計算し、採用案の自動選択は行わない。曖昧な表、重複行key、列不足、非整数、
hash不一致はエラーにする。quoted/escaped pipe等の複雑なMarkdown表は対象外。
小数を含む金額などは呼出元が同一単位の整数へ正規化した原文を用意する。

`tools.calculation.exact.evaluate(expression, bindings)` は整数の `+ - * // %`、
括弧、単項符号、名前を扱う。`//` は負数でも数学的なfloorで、浮動小数点を使わない。
関数呼出し、属性参照、通常除算、累乗、暗黙の数値変換は受け付けない。
0除算、未束縛・未使用の変数、非整数を明示エラーにする。
式512文字・AST64node・深さ16・整数256bit・表512行を上限とし、超過時に切り捨てない。

## 証跡とモデル入力

結果は式、全行の入力値・計算結果・演算trace、原文path/hash/行番号を含む。
`render(receipt)` は全行と元のpathを表示する。raw receiptは別途保存する。
元のsnapshotや必読原文を置換せず、補助計算として追加する。
文脈全体の予算は呼出元が再確認し、不足なら不足として扱う。
元の仕様・入力値の正しさやモデルの行選択・計算結果の転記まで保証するものではない。

CLIは `python -m tools.calculation.sources --input request.json`。
stdoutはJSON、終了0=計算完了、1=不正入力。失敗時に数値を捏造しない。

## 検証

- 整数商の境界、負数、0、余り、単価変更、行順序変更、異なる名前の表で検証する。
- 各入力と結果の原文位置、hash、全行保持、エラー条件を確認する。
- ランダムな数値で商・余りの恒等式を確認し、既知課題の数値へ固定していないことを検証する。
- 実モデル評価は別manifestを生成前に凍結する。元課題・全反復・全段階・同一入力の対照を維持する。
- 計算補助ありのworkflowの成績と、補助なしのモデル単体の成績を区別する。
