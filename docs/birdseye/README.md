# Birdseye データセット運用ガイド

Birdseye は、Workflow Cookbook の知識マップを 3 層（Index / Capsules / Hot）で提供します。
Guardrails の「Bootstrap → Index → Caps」という読込順序を支えるデータセットとして、各層の成果物と鮮度管理ルールをこのディレクトリで管理します。

## ディレクトリ構成

- `index.json`
  - 役割: Birdseye ノード一覧と隣接関係（Edges）の基盤データ
  - Guardrails 連携: `plan` や `notes` で ±hop を抽出する際の一次ソース
- `caps/`
  - 役割: 各ノードのカプセル要約（`<path>.json` を 1 ノード 1 ファイルで保持）
  - Guardrails 連携: `deps_out`・公開 API・リスク情報を提供
- `hot.json`
  - 役割: 主要ノードのホットリスト
  - Guardrails 連携: 「頻出入口ホットリスト」を満たし、即時参照を補助
- `README.md`
  - 役割: データセット運用手順
  - Guardrails 連携: Birdseye の生成・検証フローを共有
- `../BIRDSEYE.md`
  - 役割: フォールバック用の人間向け導線
  - Guardrails 連携: 最終手段として Edges / Hot / 更新手順を提示

## `codemap.update` 実行手順

Birdseye の生成・更新は `tools/codemap/update.py` を介して行います。
Guardrails が要求する「自動再生成＋鮮度確認」を満たすため、以下のコマンドを基準としてください。

```bash
python -m tools.codemap.update \
  --targets docs/birdseye/index.json,docs/birdseye/hot.json \
  --emit index+caps
```

```bash
python -m tools.codemap.update \
  --since \
  --radius 1 \
  --emit caps
```

1. `--targets` にはindex/hot/capsのBirdseye資源を列挙します。原文パスを直接渡さないでください。
   原文の変更から対象を導く場合は `--since <ref>` を使います。
2. `--emit` で出力対象を指定します。現在は `index+caps` が標準です。
   `--radius` を省略した場合は既定で ±2 hop を探索し、`0` を指定すると seed ノード自身だけを更新します。
3. `docs/birdseye/index.json` と `docs/birdseye/hot.json` を同一ターゲットで指定すると、両データセットの鮮度が揃います。
   出力後は `index.json.generated_at` / `hot.json.generated_at` が 5 桁ゼロ埋めの世代番号として同じ更新サイクルへ進んでいるか確認し、
   必要に応じてホットリスト項目の `last_verified_at` が対象ノードの最新確認日を反映しているか点検します。
4. 差分をレビューし、`docs/BIRDSEYE.md` のフォールバック情報と矛盾がないことをチェックしてからコミットします。

> 手動編集が必要な場合でも、Birdseye スキーマ（`id`・`role`・`caps`・`edges` など）とパス命名規則（`/` を `.` に置換）を崩さないでください。

## ホットリストと鮮度管理

- `docs/birdseye/hot.json` は `index.json` の再生成時に自動同期され、`refresh_command` と `index_snapshot`
  で更新履歴を追跡します。必要に応じて各ホットリスト項目に記録された `last_verified_at` を確認し、
  対象ノードの確認日が反映されているかを点検します。
- ホットリストの構成は `README.md`・`GUARDRAILS.md`・`HUB.codex.md` など主要導線を中心に選定し、Workflow Cookbook 本流のホットリスト基準を参考に `edges` を明示しています。
- `docs/BIRDSEYE.md` ではホットリストの概要と Edges を人間が参照できるよう再掲しているため、変更時は両ドキュメントの整合性を確認します。

## Guardrails との整合

- Birdseyeが質問に合う場合に活用し、未登録・不適合・鮮度不明の場合は通常検索で原文へ戻ります。
- source_sha256は原文の観測hashです。再生成はsummary/reviewを確認済みにしません。
  原文と要約を確認した後だけreviewのsource_sha256・summary_sha256・reviewed_atを記録します。
- `tools.codemap.source_freshness.summary_digest` でrole/summary/API/依存/リスク/試験をhash化します。
  確認記録を整備した範囲ではcheckerの `--require-reviewed` を利用できます。
- 旧capsのhash・review欠落はwarningです。段階的に移行し、日付だけで未確認を隠さないでください。
- フォールバック時には `docs/BIRDSEYE.md` の Edges / Hot / 更新手順を参照し、必要に応じて本 README の `codemap.update` 手順に合流してください。
- Birdseye を更新した場合は、関連するチェックリストや運用ドキュメント（`CHECKLISTS.md`・`RUNBOOK.md` など）にも鮮度情報を反映し、リポジトリ全体の整合を保ちます。
