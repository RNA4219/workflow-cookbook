---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-05-02
next_review_due: 2026-06-02
---

# Technical Debt Register

code-to-gate 分析で検出された技術的債務の記録と対応計画。

## 検出日: 2026-05-02

## 1. LARGE_MODULE - モジュール肥大化

### 1.1 tools/perf/collect_metrics.py (1334 lines)

**現状**: メトリクス収集の統合ハブ。CLI、Prometheus 取得、ログ解析、集計、出力すべてを含む。

**分割計画**:
| 新モジュール | 内容 | 行数見積 |
|---|---|---|
| `collect_metrics/cli.py` | CLI 入口、argparse、main | ~100 |
| `collect_metrics/prometheus.py` | Prometheus API 取得、metric fetch | ~300 |
| `collect_metrics/logs.py` | StructuredLogger ログ解析 | ~250 |
| `collect_metrics/aggregation.py` | 集計ロジック、prefix normalization | ~400 |
| `collect_metrics/output.py` | JSON 出力、PushGateway 送信 | ~150 |

**優先度**: Medium (Q2)
**依存**: 既存テスト `tests/test_collect_metrics_cli.py` は CLI入口のみなので分割影響小

### 1.2 tools/codemap/update.py (908 lines)

**現状**: Birdseye 更新の統合入口。CLI、graph 操作、capsule 生成、hot.json 管理。

**分割計画**:
| 新モジュール | 内容 | 行数見積 |
|---|---|---|
| `codemap/update/cli.py` | CLI 入口、argparse、main | ~80 |
| `codemap/update/graph.py` | index.json 操作、node/edge 管理 | ~300 |
| `codemap/update/capsule.py` | capsule 生成、要約、依存解析 | ~350 |
| `codemap/update/hot.py` | hot.json 管理、refresh_command | ~100 |

**優先度**: Medium (Q2)
**依存**: `tests/test_codemap_update.py` は CLI入口テスト

### 1.3 tools/context/pack.py (860 lines)

**現状**: context packing ツール。pack、resolve、compression。

**分割計画**:
| 新モジュール | 内容 | 行数見積 |
|---|---|---|
| `pack/cli.py` | CLI 入口 | ~80 |
| `pack/resolver.py` | docs resolve、依存解析 | ~350 |
| `pack/compression.py` | compression、trimming | ~300 |

**優先度**: Low (Q3)

### 1.4 関数数過多モジュール

| Module | Functions | 対応 |
|---|---|---|
| `tools/protocols/evidence_bridge.py` | 35 | Evidence 生成複雑性を submodule 化検討 |
| `tools/security/allowlist_guard.py` | 22 | allowlist 検証ロジックを分離 |
| `tools/perf/context_trimmer.py` | 21 | trimming ロジックを別 module 化 |
| `tools/autosave/project_lock_service.py` | 22 | lock service を分離 |

## 2. UNSAFE_DELETE - 妥当性確認済み

### 2.1 tools/audit/purge_logs.py

**判定**: Safe (正当な audit tool)
- `older_than_days > 0` 検証あり
- `cutoff_timestamp` 明示的期限判定
- audit log 用ツールとして設計

**対応**: 抑制設定 `.ctg-suppressions.yaml` で `status: reviewed-safe` 記録

### 2.2 tools/workflow_plugins/runtime.py

**判定**: False Positive
- `_traces.clear()` は trace ログクリア、実データ削除ではない

**対応**: 抑制設定で false positive 記録

## 3. 抑制・除外設定

### 除外対象

- `.venv/` - サードパーティライブラリ
- `node_modules/`
- `__pycache__/`
- `docs/` - ドキュメント (コードではない)

### 抑制ファイル

`.ctg-suppressions.yaml` で管理。

## 4. 定期再評価

次回 code-to-gate 実行: 2026-06-02 (月次)

```bash
code-to-gate analyze . --config ctg.config.yaml --emit all --out .qh
code-to-gate readiness . --policy governance/policy.yaml --from .qh --out .qh
```