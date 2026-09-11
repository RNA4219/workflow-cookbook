---
task_id: YYYYMMDD-xx
intent_id: INT-001
owner: RNA4219
status: active   # draft|active|deprecated
last_reviewed_at: 2026-07-11
next_review_due: 2026-08-11
---

# Task Seed Template

## メタデータ

```yaml
task_id: YYYYMMDD-xx
repo: https://github.com/owner/repo
base_branch: main
work_branch: feat/short-slug
priority: P1|P2|P3
langs: [auto]   # auto | python | typescript | go | rust | etc.
```

## Objective

{{一文で目的}}

## Scope

- In: {{対象(ディレクトリ/機能/CLI)を箇条書き}}
- Out: {{非対象(触らない領域)を箇条書き}}

## Requirements

- Behavior:
  - {{期待挙動1}}
  - {{期待挙動2}}
- I/O Contract:
  - Input: {{型/例}}
  - Output: {{型/例}}
- Constraints:
  - 既存API破壊なし / 不要な依存追加なし
  - 変更範囲に必要な Lint/Type/Test を実行し、失敗・未実行の理由を記録する
  - 本repoの Python 系ゲートは coverage 80% 以上。導入先は採用済みの基準・測定対象に従う
- Acceptance Criteria:
  - {{検収条件1}}
  - {{検収条件2}}

### Evaluation Identity（比較評価・実験・復元・昇格・提出を含む場合は必須）

- Manifest: `docs/evaluation-manifests/EV-YYYYMMDD-01.json`
- Before run:
  `uv run python tools/ci/check_evaluation_identity_manifest.py --manifest <path> --stage preflight --check`
- After run:
  `uv run python tools/ci/check_evaluation_identity_manifest.py --manifest <path> --stage postrun --check`
- 共通identity・profile・測定単位・データ集合・model/policy版・結果artifact hashを記録する。
  gameのみdeck/native runtime/opponent set/G50等を要求する。非gameはmeasurementテンプレートを使う。
- postrunが通るまでregistry更新・提出・外部公開を行わない。read-only診断・文書lint・checkerのfixture試験は適用外。

## Affected Paths

- {{glob例: backend/src/**, frontend/src/hooks/**, tools/*.sh}}

## Local Commands（存在するものだけ実行）

```bash
## Python
ruff check . && black --check . && mypy --strict . && pytest --cov=. --cov-report=term-missing --cov-fail-under=80 -q

## TypeScript/Node
pnpm lint && pnpm typecheck && pnpm test
npm run lint && npm run typecheck && npm test

## Go
go vet ./... && go test ./...

## Rust
cargo fmt --check && cargo clippy -- -D warnings && cargo test

## Fallback（Makefile に ci target がある場合のみ）
make ci
```

## Deliverables

- PR: タイトル/要約/影響/ロールバックに加え、本文へ `Intent: INT-xxx` と `## EVALUATION` アンカーを明記
  - `Acceptance Record: docs/acceptance/AC-YYYYMMDD-xx.md` を追記
  - 必要なら `Priority Score: <number>` を追記
- Artifacts: 変更パッチ、テスト、必要ならREADME/CHANGELOG差分
  - 検収記録: `docs/acceptance/AC-YYYYMMDD-xx.md`

---

## Plan

### Steps

1) 現状把握（対象ファイル列挙、既存テストとI/O確認）
2) 小さな差分で仕様を満たす実装
3) sample::fail の再現手順/前提/境界値を洗い出し、必要な工程を増補
4) 必要な検証を追加/更新（挙動が明確ならテスト先行。既存試験で足りる場合は再利用）
5) コマンド群でゲート通過
6) ドキュメント最小更新（必要なら）

## Patch

***Provide a unified diff. Include full paths. New files must be complete.***

## Tests

### Outline

- Unit:
  - {{case-1: 入力→出力の最小例}}
  - {{case-2: エッジ/エラー例}}
- Integration:
  - {{代表シナリオ1つ}}
- Coverage:
  - {{対象モジュールと採用済みの基準。本repoは80%。対象外なら理由}}

## Commands

### Run gates

- （上の "Local Commands" から存在し変更に必要なコマンドを選び、実結果と未実行理由を記録する）

## Notes

### Rationale

- {{設計判断を1～2行}}

### Risks

- {{既知の制約/互換性リスク}}

### Follow-ups

- {{後続タスクあれば}}
