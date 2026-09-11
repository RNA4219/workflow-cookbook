---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-07-11
next_review_due: 2026-08-10
---

# Contracts (Cookbook ↔ External)

Cookbook は独立して動作します。外部拡張（例: 子リポジトリ）は、以下の契約を任意に満たすことで、機能を拡張できます。

## Artifacts

- `.ga/qa-metrics.json`: CI メトリクスの任意拡張。`python -m tools.perf.collect_metrics --suite qa`
  で生成され、存在すれば Metrics Harvest が自動で取り込みます。

## 測定契約

- Context Trimmerの `run.statistics` は `statistics_schema: "1.1"` と `semantic_status` を記録する。
  未計測・エラー時は `semantic_retention` を省略し、数値0を補わない。
  旧ログの数値はlegacy値として読めるが、実測済み状態を遡って保証しない。
- [Gate観測契約](contracts/gate-observation-contract.md): 90日を主観測、180日を補助、30日を直近確認とし、件数・正解ラベル・版・欠測を区別する。既存のCIゲート設定を自動変更しない。

## Config

- `governance/predictor.yaml`: 予測ガバナンス用の重みや閾値。存在しない場合は既定値で実行されます。

```yaml
paths: {"src/providers/**": 5, "src/core_ext/**": 4, "docs/**": 1, "tests/**": 2}
size: {small: 0, medium: 2, large: 4, xlarge: 6}
retry_history_weight: 3
fail_history_weight: 5
threshold_warn: 7
threshold_block: 12
```

## Conventions

- すべて feature detection（存在検出）で扱われ、未提供でも Cookbook 側は正常に動作します。
