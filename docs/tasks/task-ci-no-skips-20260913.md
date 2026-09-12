---
task_id: 20260913-01
intent_id: INT-CI-INTEGRATION-NO-SKIPS
owner: workflow-cookbook
status: in_progress
---

# CIの依存準備不足によるskipを解消する

[CI設定](../ci-config.md) / [受入](../acceptance/AC-20260913-01.md)

## Scope

公開commitを固定したmemx/taskstate/HATE/QEGをLinuxのPython 3.11/3.12で準備する。
必須テストのskipを失敗にし、JUnitとskip理由を保存する。
WindowsのPOSIX対象外を明示し、Linuxで同じ権限テストを必須実行する。
過去のmanual-bb Gate・実行証跡、依存repo自体、ブランチ保護設定は変更しない。
本作業はCIと機能fixtureの検証であり、比較評価実験やモデル性能測定は行わない。

## Verification

skip制御の独立pytest回帰、公開依存版での連携、全体pytest、Ruff/mypyと文書ゲートを実行する。
GitHub Actions上の3構成のJUnitでskip 0と対象ケースの実行を確認する。
