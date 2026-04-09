# J. Test Engineering

## 1. 目的

実装工程で単体テスト・結合テスト・カバレッジ確認を標準化し、
「動いたから終わり」ではなく「検証可能な品質工程を通した」状態を
既定にする。

## 2. 標準

- 単体テスト
  - 変更した関数・クラス・分岐を直接検証する。
  - 正常系だけでなく、代表的な異常系と境界値を含める。
- 結合テスト
  - モジュール境界、CLI、設定読み込み、永続化、外部契約の接点を
    代表シナリオで検証する。
- カバレッジ
  - 標準の下限は 80% とする。
  - Python では `pytest --cov=. --cov-report=term-missing --cov-fail-under=80`
    を既定コマンドとする。

## 3. 実装工程

1. 先に `Tests` セクションへ Unit / Integration の対象を書く。
2. 実装と同時に単体テストを追加する。
3. I/O 契約や CLI を持つ変更は結合テストを 1 本以上追加する。
4. ゲート実行時に coverage を確認する。
5. 検収記録へ実行コマンド、結果、未達時の理由を残す。

## 4. 未達時の扱い

- coverage が 80% 未満の場合は、そのままマージしない。
- 例外を認める場合は次を必須とする。
  - `docs/acceptance/AC-*.md` に理由を記録
  - PR 本文に対象範囲とフォローアップを記載
  - 次回タスクで解消する Task Seed を起票

## 5. 参照先

- [EVALUATION.md](../../EVALUATION.md)
- [RUNBOOK.md](../../RUNBOOK.md)
- [CHECKLISTS.md](../../CHECKLISTS.md)
- [TASK.codex.md](../../TASK.codex.md)
