---
intent_id: INT-SEC-004
owner: security
status: active
last_reviewed_at: 2026-04-19
next_review_due: 2026-05-17
---

# Dependency Governance Policy

このドキュメントは、`workflow-cookbook` の
依存関係管理・更新・監査・例外運用の方針を定める。

## 1. 依存関係の現在地

本プロジェクトの依存関係は以下で管理:

| ファイル | 用途 | 固定方式 |
| --- | --- | --- |
| `requirements.txt` | 本番依存 | `==` 固定バージョン |
| `pyproject.toml` | dev/optional依存 | `==` 固定バージョン |

**現状の依存一覧**:

- PyYAML 6.0.3 (本番依存)
- pytest, pytest-cov, bandit, pip-audit (dev依存)

## 2. Lockfile 方針

`requirements.txt` を **lockfile 相当**として扱う。

- 全ての本番依存は `==` で完全固定
- dev依存も `==` で完全固定（CI再現性確保）
- バージョン更新は Dependabot PR または手動レビューで実施

**理由**: pip-tools/pip-compile は現状の依存数に対して過剰。
`requirements.txt` + `pyproject.toml` 固定 + pip-audit監査 で
enterprise 相当の再現性を担保。dev依存固定によりCI環境の安定性を確保。

## 3. 依存更新方法

### 自動更新

- **Dependabot**: GitHub Actions + pip 依存を週次監視
- PR 生成 → CI検証 → マージ承認

### 手動更新

1. `requirements.txt` または `pyproject.toml` のバージョンを更新
2. `pip-audit -r requirements.txt` で脆弱性確認
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
2. リスク評価と影響範囲を明記
3. 期限付き承認（最大90日）
4. 定期レビューで再評価

## 7. SBOM 生成

- CI で `.ga/sbom.json` を生成（CycloneDX JSON形式）
- リリース時に SBOM を artefact として保存

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