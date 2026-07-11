---
intent_id: INT-SEC-005
owner: security
status: active
last_reviewed_at: 2026-07-11
next_review_due: 2026-08-10
---

# Dependency Exceptions Registry

このドキュメントは、脆弱性監査で見つかった依存例外の受容理由を記録する。

## 例外一覧

現在記録された例外はありません。

## 例外記録フォーマット

新しい例外を追加する場合は、以下の形式で記録:

```markdown
### EXC-YYYY-NNN: [パッケージ名] [バージョン]

- **脆弱性**: CVE-XXXX-XXXXX
- **重大度**: Critical/High/Medium/Low
- **受容理由**: [理由]
- **影響範囲**: [本プロジェクトでの利用状況]
- **期限**: YYYY-MM-DD (最大90日)
- **承認者**: [GitHub username]
- **再評価日**: YYYY-MM-DD
```

## 例外受容条件

以下の条件を満たす場合のみ例外を受容:

1. 代替ライブラリが存在しない
2. 脆弱性が実exploit不可能（条件付き）
3. 修正版が未リリース

## 定期レビュー

- 毎月 `next_review_due` で指定した日付に再評価
- 期限切れ例外は自動的に削除または更新が必要

## 関連資料

- [Dependency Governance Policy](./Dependency_Governance.md)
- [Enterprise Readiness Checklist](./Enterprise_Readiness_Checklist.md)
