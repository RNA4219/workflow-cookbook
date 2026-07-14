---
task_id: 20260714-01
intent_id: INT-001
owner: RNA4219
status: done
last_reviewed_at: 2026-07-14
next_review_due: 2026-08-14
---

# Task Seed: Codemap discovery stability

## Objective

`codemap.config.json` に基づくノード発見を非破壊かつ衝突安全にし、
CIで追跡可能な検収証跡を残す。

## Scope

- In:
  - acceptance Markdown / JSONの設定駆動発見
  - `.lakda/**` の除外
  - 未変更capsuleのbytes・世代番号保持
  - capsuleパス衝突の拒否
  - Codemap / CLI回帰テスト
- Out:
  - capsule命名規則の破壊的変更
  - 既存のREADMEおよびexport-ingest作業ツリー変更

## Requirements

- Behavior:
  - includeに一致しexcludeに一致しないファイルだけをindexへ追加する
  - 未変更capsuleを再書込みしない
  - 複数ノードが同一capsuleパスへ変換される場合は明示的に失敗する
  - no-op caps更新は既存の5桁世代番号を報告する
- Constraints:
  - 既存CLIとBirdseye JSONの後方互換を維持する
  - Python変更のcoverageを80%以上にする

## Affected Paths

- `tools/codemap/update/capsule.py`
- `tools/codemap/update/session.py`
- `tests/test_codemap_update.py`
- `docs/acceptance/AC-20260714-01.md`

## Verification

- Codemap / CLI回帰: 60 passed
- `tools.codemap.update` coverage: 91.27%
- Python構文検査: passed
- Acceptance形式検査: passed

## Acceptance

- [AC-20260714-01](../acceptance/AC-20260714-01.md)
- [EVALUATION.md#Acceptance Criteria](../../EVALUATION.md#acceptance-criteria)

## Notes

- capsule命名規則は維持し、衝突時は黙って上書きせずエラーを返す。
- フォローアップ: なし
