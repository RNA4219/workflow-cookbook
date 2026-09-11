---
intent_id: INT-SEC-004
owner: security
status: active
last_reviewed_at: 2026-07-11
next_review_due: 2026-08-10
---

# Dependency Governance Policy

このドキュメントは、`workflow-cookbook` の
依存関係管理・更新・監査・例外運用の方針を定める。

## 1. 依存関係の現在地

本プロジェクトの依存関係は以下で管理:

| ファイル | 用途 | 固定方式 |
| --- | --- | --- |
| `uv.lock` | 開発・検証環境の解決結果 | 推移的依存・解決条件を固定 |
| `requirements.txt` | 現行CI監査入力 | `==` 固定バージョン |
| `pyproject.toml` | package宣言、dev/optional依存 | runtimeは許容範囲、devは固定 |

**現状の依存一覧**:

- PyYAML 6.0.3 (本番依存)
- pytest, pytest-cov, bandit, pip-audit (dev依存)

## 2. Lockfile 方針

開発・検証環境は `uv.lock` を正本として `uv sync --locked --extra dev` で構築する。
直接依存の `==` 指定だけを推移的依存を含む完全なlockと見なさない。

- 配布packageのruntime許容範囲と、開発環境の解決結果を区別する。
- `requirements.txt` を参照する既存監査CIを維持し、宣言・lock・監査入力の整合を確認する。
- バージョン更新はDependabot PRまたは手動レビューで、宣言とlockを同じ変更に含める。
- 利用先では実際のパッケージ管理・配布方式に合わせたlockを採用し、根拠なく完全再現を保証しない。

## 3. 依存更新方法

### 自動更新

- **Dependabot**: GitHub Actions + pip 依存を週次監視
- PR 生成 → CI検証 → マージ承認

### 手動更新

1. 対象の依存宣言を更新し、`uv lock` の差分をレビューする。必要な `requirements.txt` 監査入力も同期する
2. `uv sync --locked --extra dev` で解決結果を確認し、`pip-audit -r requirements.txt` と配布環境に必要な監査を行う
3. CI で Bandit/Semgrep/pip-audit が通ることを確認
4. PR 作成・レビュー・マージ

### dev依存更新（Dependabot）

- Dependabot が `pyproject.toml` の dev依存を週次監視
- バージョン更新PR作成 → CI検証 → マージ承認
- 脆弱性発見時は critical/high を優先対応

## 4. 脆弱性監査方法

### CI での監査

- `security.yml` で `pip-audit -r requirements.txt` を実行
- 脆弱性発見時は CI失敗 → PRブロック

### 手動監査

```bash
pip-audit -r requirements.txt
```

## 5. 脆弱性対応 SLA

以下は本repoで採用した期限であり、Gateの90/180/30日観測窓によって延長しない。利用先の運用では責任者と合意した契約を明記する。

| 重大度 | 対応期限 | 対応内容 |
| --- | --- | --- |
| Critical | 24時間以内 | 即時パッチまたは依存削除 |
| High | 7日以内 | バージョン更新または例外申請 |
| Medium | 30日以内 | バージョン更新計画 |
| Low | 次リリース | 通常更新サイクル |

## 6. 例外運用

### 例外受容条件

- 代替ライブラリが存在しない
- 脆弱性が実exploit不可能（条件付き）
- 修正版が未リリース

### 例外承認プロセス

1. `docs/security/dependency_exceptions.md` に例外理由を記録
2. 検出ID・対象版／実コード・到達条件・緩和策・責任者・影響範囲を明記
3. 期限付き承認（最大90日）
4. 定期レビューで再評価

## 7. SBOM 生成

- CI で `.ga/sbom.json` を生成（CycloneDX JSON形式）
- リリース時にSBOMを対象版・配布物hashと対応付けて保存する。採用した生成器の収集範囲を記録し、未収集の推移的依存まで網羅したと見なさない

```bash
python -m tools.security.generate_sbom --output .ga/sbom.json
```

## 8. 資産可視化

- SBOM: `.ga/sbom.json`
- 監査ログ: CI artifacts
- 例外台帳: `docs/security/dependency_exceptions.md`

## 関連資料

- [Enterprise Readiness Checklist](./Enterprise_Readiness_Checklist.md)
- [security.yml](../../.github/workflows/security.yml)
- [requirements.txt](../../requirements.txt)
