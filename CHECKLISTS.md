---
intent_id: INT-001
owner: your-handle
status: active   # draft|active|deprecated
last_reviewed_at: 2025-10-14
next_review_due: 2025-11-14
---

# Checklists

## Daily

- 入力到着の確認
- 失敗通知の有無
- 主要メトリクス閾値

## Release

- [docs/Release_Checklist.md](docs/Release_Checklist.md) を参照して全体手順を確認
- 変更点の要約
- リリースノート（`CHANGELOG.md` など）へ必要最小の項目を追記
- 未反映の `TASK.*` が残っていないか確認し、成果を `[Unreleased](CHANGELOG.md#unreleased)` へ通番付きで転記済みかチェック
- 受け入れ基準に対するエビデンス
- 影響範囲の再確認
- PR に `type:*` および `semver:*` ラベルを付与済み
- [Security Review Checklist](docs/security/Security_Review_Checklist.md) に沿って準備→実装→レビューの各フェーズを完了し、リリース判定と証跡を残す
- 配布物へ `LICENSE` / `NOTICE` を同梱済み

## Hygiene

- 命名・ディレクトリ整備
- ドキュメント差分反映
- フォーク差分記録の最新化（[`docs/FORK_NOTES.md`](docs/FORK_NOTES.md) をリリース前レビューと突合）
