---
intent_id: INT-001
owner: your-handle
status: active   # draft|active|deprecated
last_reviewed_at: 2025-10-14
next_review_due: 2025-11-14
---

# Checklists

## Development

- `TASK.*` を起票・更新し、[docs/TASKS.md](docs/TASKS.md) の運用ルールに沿ってスコープとフォローアップを同期
- 着手前に [docs/ROADMAP_AND_SPECS.md](docs/ROADMAP_AND_SPECS.md) と [`GUARDRAILS.md`](GUARDRAILS.md) を読み合わせ、最小差分と既存ガードレールへ整合
- テストを先行させ、[docs/ROADMAP_AND_SPECS.md](docs/ROADMAP_AND_SPECS.md) で定義された TDD フロー（`EVALUATION.md#test-outline` 参照）を完了
- 例外や設定変更は [docs/security/Security_Review_Checklist.md](docs/security/Security_Review_Checklist.md) の該当フェーズで可否を確認
- Runbook 連携が必要な作業は [`RUNBOOK.md`](RUNBOOK.md) へ手順差分を反映し、参照リンクを Task Seed に追記

## Pull Request / Review

- 失敗させたテストが緑化する最小コミット単位を維持し、差分を可視化
- `CHANGELOG.md` の `[Unreleased](CHANGELOG.md#unreleased)` に Task Seed 番号付きで成果を追記
- PR 説明欄から [docs/TASKS.md](docs/TASKS.md)・[docs/ROADMAP_AND_SPECS.md](docs/ROADMAP_AND_SPECS.md)
  ・[`RUNBOOK.md`](RUNBOOK.md) 等の参照先へ遷移できるようリンクを付す
- レビュー観点は [docs/security/Security_Review_Checklist.md](docs/security/Security_Review_Checklist.md) の「レビュー」節と
  [`GUARDRAILS.md`](GUARDRAILS.md) を再確認
- ラベル運用・テンプレ遵守は `HUB.codex.md` と `TASK.codex.md` のタスク分割フローに合わせる

## Ops / Incident

- インシデント初動は [`RUNBOOK.md`](RUNBOOK.md) の該当セクションを実行し、必要な通知経路を確保
- セキュリティ対応は [docs/security/Security_Review_Checklist.md](docs/security/Security_Review_Checklist.md) のインシデント項目を完了
- 復旧後の再発防止策を `TASK.*` と [docs/TASKS.md](docs/TASKS.md) の手順に沿って起票し、Runbook へ差分を反映
- 主要メトリクスの変動は [docs/ROADMAP_AND_SPECS.md](docs/ROADMAP_AND_SPECS.md) のモニタリング要件に沿って記録

## Daily

- 入力到着の確認
- 失敗通知の有無
- 主要メトリクス閾値

## Release

- 実装・レビューの完了条件は「Development」「Pull Request / Review」を満たしていることを前提に進行
- [docs/Release_Checklist.md](docs/Release_Checklist.md) を参照して全体手順を確認
- 変更点の要約
- リリースノート（`CHANGELOG.md` など）へ必要最小の項目を追記
- 未反映の `TASK.*` が残っていないか確認し、成果を `[Unreleased](CHANGELOG.md#unreleased)` へ通番付きで転記済みかチェック
- [docs/UPSTREAM.md](docs/UPSTREAM.md) を参照し、対象派生リポの取り込み評価が最新であるか
  確認する
- [docs/UPSTREAM_WEEKLY_LOG.md](docs/UPSTREAM_WEEKLY_LOG.md) に未反映の検証ログ・
  フォローアップが残っていないか確認する
- 新規 ADR を含むリリースでは [docs/ADR/README.md](docs/ADR/README.md) の索引更新を完了し、レビューフローで確認する
- 受け入れ基準に対するエビデンス
- 影響範囲の再確認
- PR に `type:*` および `semver:*` ラベルを付与済み
- [Security Review Checklist](docs/security/Security_Review_Checklist.md) に沿って準備→実装→レビューの各フェーズを完了し、リリース判定と証跡を残す
- 配布物へ `LICENSE` / `NOTICE` を同梱済み

## Hygiene

- 命名・ディレクトリ整備
- ドキュメント差分反映
- フォーク差分記録の最新化（[`docs/FORK_NOTES.md`](docs/FORK_NOTES.md) をリリース前レビューと突合）
