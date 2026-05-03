---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-05-03
next_review_due: 2026-06-03
---

# Technical Debt Register

code-to-gate 分析で検出された技術的債務の記録と対応計画。

## 検出日: 2026-05-03 (更新)

## 1. LARGE_MODULE - モジュール肥大化

### 1.1 tools/perf/collect_metrics/types.py → rules.py + helpers.py + extractor.py (分割済み: 2026-05-03)

**分割後**:

| Module | 行数 | 内容 |
|---|---|---|
| collect_metrics/rules.py | 231 | Rule dataclasses, Protocols |
| collect_metrics/helpers.py | 277 | derive_average, coerce functions, precision mode helpers |
| collect_metrics/extractor.py | 145 | MetricDefinition, MetricDefinitionRegistry, MetricExtractor |

**判定**: 完了 - tests 20件 全てパス

### 1.2 tools/ci/check_governance_gate.py → governance_gate/ package (分割済み: 2026-05-03)

**分割後**:

| Module | 行数 | 内容 |
|---|---|---|
| governance_gate/resolver.py | 246 | PRBodyResolver, CategoryHintResolver, resolution helpers |
| governance_gate/rules.py | 194 | ValidationRule classes, patterns, constants |
| governance_gate/validator.py | 126 | ValidationContext, ValidationOutcome, PRBodyValidator |
| governance_gate/cli.py | 64 | parse_arguments, main |
| governance_gate/`__init__.py` | 83 | Package exports |
| governance_gate/`__main__.py` | 7 | python -m entry point |

**判定**: 完了 - tests 33件 全てパス

### 1.4 関数数過多モジュール

**判定**: 対応不要 - 全ファイル500行以下

| Module | Functions | 行数 | 状態 |
|---|---|---|---|
| `tools/protocols/evidence_bridge.py` | 35 | 446 | 500行以下、分割不要 |
| `tools/security/allowlist_guard.py` | 22 | 308 | 500行以下、分割不要 |
| `tools/perf/context_trimmer.py` | 21 | 309 | 500行以下、分割不要 |
| `tools/autosave/project_lock_service.py` | 22 | 291 | 500行以下、分割不要 |

## 2. DOCSTRING欠損 - 解消済み

### 2.1 tools/protocols/evidence_bridge.py (完了: 2026-05-03)

**判定**: 完了 - 全クラス/関数にdocstring追加

### 2.2 tools/security/allowlist_guard.py (完了: 2026-05-03)

**判定**: 完了 - 全クラス/関数にdocstring追加

## 3. UNSAFE_DELETE - 妥当性確認済み

### 3.1 tools/audit/purge_logs.py

**判定**: Safe (正当な audit tool)

- `older_than_days > 0` 検証あり
- `cutoff_timestamp` 明示的期限判定
- audit log 用ツールとして設計

**対応**: 抑制設定 `.ctg-suppressions.yaml` で `status: reviewed-safe` 記録

### 3.2 tools/workflow_plugins/runtime.py

**判定**: False Positive

- `_traces.clear()` は trace ログクリア、実データ削除ではない

**対応**: 抑制設定で false positive 記録

## 4. 抑制・除外設定

### 除外対象

- `.venv/` - サードパーティライブラリ
- `node_modules/`
- `__pycache__/`
- `docs/` - ドキュメント (コードではない)

### 抑制ファイル

`.ctg-suppressions.yaml` で管理。

## 5. 定期再評価

次回 code-to-gate 実行: 2026-06-03 (月次)

```bash
code-to-gate analyze . --config ctg.config.yaml --emit all --out .qh
code-to-gate readiness . --policy governance/policy.yaml --from .qh --out .qh
```
