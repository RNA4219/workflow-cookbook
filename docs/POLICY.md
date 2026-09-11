---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-07-11
next_review_due: 2026-08-10
---

# INT Policy

## 1. INT フォーマット

- 形式: `INT-<カテゴリ>-<番号>` または `INT-<番号>` を基本形とし、大文字英数字とハイフンのみを使用する。
- 正規表現: `INT-[0-9A-Z]+(?:-[0-9A-Z]+)*`
- governance検証が適用されるPR本文・テンプレートでは `Intent: INT-xxx` を必須とする。

## 2. 許可カテゴリ

- `FEATURE`: 新規機能追加や大規模改善。
- `FIX`: バグ修正や不具合是正。
- `CHORE`: メンテナンス・リファクタリング・CI 調整などの内部作業。
- `DOCS`: ドキュメント整備や情報更新。
- `OPS`: 運用変更や手順更新。
- カテゴリはいずれも大文字で記載し、必要に応じて `INT-FEATURE-123` のようにサフィックスを連結する。

## 3. 正規表現

- Intent 検証: `Intent\s*[：:]\s*INT-[0-9A-Z]+(?:-[0-9A-Z]+)*`
- [`Priority Score`](addenda/A_Glossary.md#priority-score): `Priority\s*Score\s*:\s*[0-9]+`
- INT Logs の日付行: `^\s*-\s*[0-9]{4}-[0-9]{2}-[0-9]{2}:`
- すべてのパターンは `tools/ci/check_governance_gate.py` の検証ロジックと同期させ、変更時は双方を同時更新する。

## 4. 運用ルール

1. 採用したgovernance workflowの適用条件に従い、PRのIntent Metadataを記録する。
   本repoはcode変更でIntent・EVALUATION・Priority Scoreを検証する。docs-only等のskip条件はworkflowを正本とする。
   skipは文書検査・内容確認・必要な承認の免除ではない。導入先へ同じメタデータを一律強制しない。
2. `INT Logs` セクションでは Intent の承認・変更履歴を時系列で記録し、日付・概要・関係者を箇条書きで残す。
3. Intent 番号は `governance/policy.yaml` の禁止パスに抵触しない作業のみ紐づけ、逸脱する場合は事前に承認を得る。
4. [`Priority Score`](addenda/A_Glossary.md#priority-score) は候補の優先順位比較に使い、算定根拠と `governance/prioritization.yaml` の計算版を示す。
   採用済みゲートが要求する場合はその形式で記録するが、点数だけで着手・完了・品質を判断しない。
5. テンプレートや検証ロジックを変更する場合は、本ドキュメントを更新し、関連する CI テスト（`test_pr_template_contains_required_sections`）を緑の状態で維持する。
