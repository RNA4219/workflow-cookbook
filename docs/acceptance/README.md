---
intent_id: INT-001
owner: your-handle
status: active
last_reviewed_at: 2026-04-10
next_review_due: 2026-05-10
---

# Acceptance Records

`docs/acceptance/` は変更ごとの検収記録を残す場所です。

## 使い方

1. [ACCEPTANCE_TEMPLATE.md](ACCEPTANCE_TEMPLATE.md) を複製する
2. `AC-YYYYMMDD-xx.md` 形式で保存する
3. front matter と各見出しを埋める
4. PR 本文の `Acceptance Record` からこのファイルへリンクする
5. 必要なら `python tools/ci/generate_acceptance_index.py --plugin-config examples/workflow_plugins.cross_repo.sample.json`
   で一覧を再生成する

## 命名規則

- `AC-YYYYMMDD-xx.md`
- 例: `AC-20260410-01.md`

## 必須項目

- front matter
  - `acceptance_id`
  - `task_id`
  - `intent_id`
  - `owner`
  - `status`
  - `reviewed_at`
  - `reviewed_by`
- 本文見出し
  - `## Scope`
  - `## Acceptance Criteria`
  - `## Evidence`
  - `## Verification Result`

## 検証

```sh
python tools/ci/check_acceptance.py --check
```

## 一覧

- plugin 連携で生成する一覧:
  [INDEX.md](INDEX.md)
