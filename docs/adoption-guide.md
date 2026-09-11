---
intent_id: INT-ADOPTION-TIERS
owner: docs-core
status: active
last_reviewed_at: 2026-07-01
next_review_due: 2026-08-01
---

# Adoption Guide

This guide explains how to apply workflow-cookbook adoption tiers to a downstream
repository.

## 一度で全体をコピーする

全ルールを参照できる形で導入する場合は、Workflow Cookbookのcheckoutから実行します。
導入先ディレクトリは作成済みで、コピー元と包含関係のない場所を指定します。

```sh
python -m tools.adoption --repo /path/to/target
```

Windows例（checkoutをカレントディレクトリにして実行）:

```powershell
uv run wfc-copy --repo "C:\work\my-project"
```

wheel版はコピー元checkoutを指定します。

```sh
wfc-copy --source /path/to/workflow-cookbook --repo /path/to/target --ref HEAD
```

- `workflow-cookbook/upstream/`へ指定コミットの全追跡ファイルをコピーします。
- `workflow-cookbook/ADOPTION.md`で原文への入口と導入先への適用方法を案内します。
- 既存`AGENTS.md`を保持して管理ブロックを追加します。既存のREADMEやCI設定は保全します。
- `manifest.json`にcommit/treeとファイルhashを記録します。
- 未コミット差分、未追跡ファイル、Git履歴、外部ツール本体や認証は含めません。

```sh
# 書き込み前の確認
python -m tools.adoption --repo /path/to/target --dry-run

# コピー元やGitに接続せず欠落・改変・管理ブロックを検査
python -m tools.adoption --repo /path/to/target --check

# 導入先ルートだけで検査（元checkout・Git・追加依存なし）
python -B workflow-cookbook/verify.py --repo . --check
```

同じ版を再実行すると`unchanged`、別版や独自変更との衝突はエラーになります。
自動上書きはありません。更新するときは既存の独自差分を確認し、別領域へのコピーと比較します。
AGENTSへの接続は追記で行い、並行編集を全体置換で失わないようにします。
公開後に競合・I/Oエラーを検出した場合は、コピー領域とAGENTSを残してexit 1を返します。
現物とmanifestを確認して管理ブロックを整理するか、別領域で再導入してください。
stdoutはJSON、exit 0は成功、1は要対応です。

この機能の「フル」は固定版の収録範囲です。通常のTier判定は導入先自身の文書構造を調べる別の検査です。
コピーした上流のTask/Acceptance/Birdseyeを、導入先の実績やコード索引として数えないでください。
導入先の仕様、Task/Acceptance、CI、HATE/QEG等の接続と実際の受入試験は、対象作業で整えます。
コピー成功時も運用準拠は`not_evaluated`です。
[仕様](contracts/full-workflow-copy.md)に保全と検証の境界を記載しています。

## 1. Assess Current Tier

```bash
python tools/ci/check_adoption_tier.py --repo /path/to/repo --json
```

Use `--min-tier` with `--check` when a repository must meet a required tier.

```bash
python tools/ci/check_adoption_tier.py --repo /path/to/repo --min-tier 2 --check
```

## 2. Add Missing Tier Documents

Tier 1 starts with navigation and scope:

```bash
cp templates/HUB.codex.md.template /path/to/repo/HUB.codex.md
cp templates/BLUEPRINT.md.template /path/to/repo/BLUEPRINT.md
```

Tier 2 adds operational validation:

```bash
cp templates/RUNBOOK.md.template /path/to/repo/RUNBOOK.md
cp templates/GUARDRAILS.md.template /path/to/repo/GUARDRAILS.md
cp templates/EVALUATION.md.template /path/to/repo/EVALUATION.md
```

After copying templates, update front matter fields such as `intent_id`, `owner`,
`last_reviewed_at`, and `next_review_due`. Set the adopted scope and `template_version`.
既存の独自差分を保全し、必要なファイル・機能だけを導入してください。

## 3. Check Template Drift

If downstream documents keep `template_version`, compare them with the current
template set:

```bash
python tools/ci/check_adoption_tier.py --repo /path/to/repo --check-drift --json
```

`--check --check-drift` returns non-zero when a tracked template version differs.

## 4. Batch Assessment

Use a JSON list when assessing several repositories:

```json
[
  "../workflow-cookbook",
  {"repo": "../agent-taskstate"}
]
```

```bash
python tools/ci/check_adoption_tier.py --repo-list repos.json --json
```

## 5. Review Cadence

継続性・変更頻度・重要度・確認機会で責任者と周期を決めます。
Tier 0-1の6か月、Tier 2の3か月、Tier 3の毎月は初期の目安です。
採用済みの期限は内容確認後に更新し、Gate効果の90/180/30日観測窓を一括適用しません。
