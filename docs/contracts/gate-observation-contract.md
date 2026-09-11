---
intent_id: INT-GATE-OBSERVATION
owner: RNA4219
status: active
last_reviewed_at: 2026-09-10
next_review_due: 2026-12-10
---

# Gate観測契約

Gateの効果やstageを判断するため、主90日・補助180日・直近30日を別々に集計する。
検出率と誤検知率を区別し、日数やproxy値だけで自動昇格・降格しない。

## 入力

`tools/ci/check_gate_observations.py --observations <bundle.json> --as-of <ISO日時>` を使う。
as-of省略時は現在UTC。実運用の観測bundleを指定し、sampleや単体fixtureを実績に流用しない。

bundleの必須項目:

- schema_version: 1.0、gate_id、観測開始時点observation_started_at（timezone付き）。
- versions: model/runner/policyの3つの版。利用しないモデルは理由付きnot_usedを記録する。
- minimum_opportunities: 責任者が測定設計に応じて合意した正の整数。
- records: 各適用機会のid・timestamp・versions・outcome・override。

outcomeはtrue_positive / false_positive / true_negative / false_negative / error。
正当な違反検出はtrue_positive。検出が誤りだった機会だけfalse_positiveとする。
overrideは担当者が判定を上書きしたかを示すbool。正誤判定とは別に記録する。
未評価の正誤を推測せず、確認できるまで集計対象を整備する。
機会IDは重複させず、版なし・不明なoutcome・観測範囲外の時刻は入力エラーとする。

## 集計と判断

| 項目 | 定義 |
|---|---|
| eligible_opportunities | 指定版に一致し、対象窓内にある適用機会 |
| detection_rate | (TP + FP) / 正誤分類済み機会 |
| false_positive_rate | FP / (FP + TN)。実際に違反がない機会に対する誤検知 |
| false_negative_rate | FN / (TP + FN)。実際に違反がある機会に対する見逃し |
| overrides / error | 上書き数と計測障害数。正当な検出と分ける |
| excluded_version_mismatch | 版が違うため除外した機会数 |

窓は(as-ofから日数を引いた時点, as-of]。90/180/30日は重複し、件数を合算しない。
窓全体の観測期間や最低機会数が足りなければinsufficient_data、障害があればmeasurement_error。
分母ゼロの率はnullであり、実測0と異なる。ready_for_reviewはデータがレビュー可能という意味。
責任者が正誤ラベル・未捕捉機会・運用負荷・修正可能性・緊急性を確認してstage変更を判断する。

このCLIはローカル集計のみで、CI設定・保護設定・通知・定期実行・stageを変更しない。
観測器の単体試験の成功と、実運用での長期効果を区別する。
入力集合の完全性や正誤ラベルの妥当性は収集側とレビュー側で検証し、件数だけで保証しない。
