---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-07-12
next_review_due: 2026-08-11
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

**対応**: 抑制設定 `.ctg/suppressions.yaml` で `status: reviewed-safe` 記録

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

`.ctg/suppressions.yaml` で管理。

## 5. 定期再評価

2026-07-11 再評価では、strict mypy、Ruff、full pytest、coverage gateの通過後に、
Code-to-gate改善候補として4モジュールを未解決台帳へ登録した。

2026-07-12 解消確認では、Code-to-gate v1.5.0 の run
`ctg-202607111755-local` が unsuppressed finding 0、suppressed 5、
readiness failed condition 0 を記録した。

| Module | 解消内容 | 検証 |
| --- | --- | --- |
| `tools/ci/check_docs_review_due.py` | data / scan / artifacts / CLI packageへ分割 | CLI互換・focused test 4 passed |
| `tools/codemap/update/session.py` | serial分離、RootBuilder重複メソッド削除 | focused test 47 passed |
| `tools/context/pack/resolver.py` | config / signals / rankingへ分割、facade化 | focused test 16 passed |
| `tools/workflow_plugins/runtime.py` | types / Evidence projection分離、delegate化 | focused test 8 passed |

**判定**: 上記4件は解消済み。新規抑制・期限付き例外は追加していない。

残存する抑制は `.ctg/suppressions.yaml` の既存5件のみであり、対象4モジュールは
抑制対象ではない。次回の定期再評価は既存抑制期限に合わせて 2026-08-10 とする。

```bash
node C:\\Users\\ryo-n\\Codex_dev\\code-to-gate\\dist\\cli.js analyze . --emit all --out <artifact-dir> --cache force
node C:\\Users\\ryo-n\\Codex_dev\\code-to-gate\\dist\\cli.js readiness . --policy governance/policy.yaml --from <artifact-dir> --out <artifact-dir>
```
