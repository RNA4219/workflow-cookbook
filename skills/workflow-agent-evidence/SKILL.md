---
name: workflow-agent-evidence
description: workflow-cookbookのLLM活動ログをagent-protocols Evidence契約へ接続するSkill。StructuredLogger、evidence_bridge、plugin_loader/config、sample、schema、testsの整合性を保つ場合に使用。
---

# Workflow Agent Evidence

workflow-cookbookのLLM行動追跡をagent-protocols Evidence契約へ接続する。

## 概要

| 役割 | 内容 |
|------|------|
| **Logging** | StructuredLogger による LLM 活動記録 |
| **Bridge** | Evidence 契約への変換・接続 |
| **Plugin** | config/loader による拡張点 |
| **Schema** | inference-plugin-config.schema.json 整合 |

## Quick Start

```sh
# 1. core files 確認
cat tools/perf/structured_logger.py
cat tools/protocols/evidence_bridge.py

# 2. tests 実行
uv run pytest tests/test_structured_logger.py tests/test_agent_protocol_evidence.py -q

# 3. plugin config 検証
python tools/workflow_plugins/validate_workflow_plugin_config.py --config examples/inference_plugins.agent_protocol.sample.json
```

## Integration Points

### Core Files

| File | Role |
|------|------|
| `tools/perf/structured_logger.py` | Generic LLM activity logger |
| `tools/protocols/evidence_bridge.py` | Evidence contract mapping |
| `tools/protocols/plugin_loader.py` | Plugin discovery |
| `tools/protocols/plugin_config.py` | Config parsing |
| `tools/protocols/README.md` | Protocol docs |

### Samples

| File | Role |
|------|------|
| `examples/inference_plugins.agent_protocol.sample.json` | Plugin config sample |
| `examples/agent_protocol_evidence_consumer.sample.py` | Consumer sample |
| `schemas/inference-plugin-config.schema.json` | Config schema |

## Standard Flow

### 1. 統合点確認

優先ファイル:
- `tools/perf/structured_logger.py`
- `tools/protocols/evidence_bridge.py`
- `tools/protocols/plugin_loader.py`
- `tools/protocols/plugin_config.py`

詳細: `references/workflow-cookbook.md`

### 2. Architecture維持

- `StructuredLogger` を generic に保つ
- 契約固有の mapping を `tools/protocols/` 内に配置
- plugin boundary を `handle_inference(record)` に固定
- config/sample/schema/docs/tests を同時更新

### 3. Evidence契約維持

必須フィールドを削除・変更しない:
- `taskSeedId`, `baseCommit`, `headCommit`
- `inputHash`, `outputHash`, `diffHash`
- `model`, `tools`, `environment`
- `startTime`, `endTime`, `actor`
- `policyVerdict`, `mergeResult`, `staleStatus`

詳細: `references/agent-protocols.md`

### 4. Docs/Test同期

実装変更時に更新:
- `docs/requirements.md`, `docs/spec.md`, `docs/design.md`
- `RUNBOOK.md`, `CHANGELOG.md`

### 5. Validate

```sh
# core tests
uv run pytest tests/test_structured_logger.py tests/test_agent_protocol_evidence.py tests/test_plugin_loader.py tests/test_plugin_config.py -q

# coverage gate
uv run pytest tests/ --cov=. --cov-fail-under=80 -q
```

## References

| File | Content |
|------|---------|
| `references/workflow-cookbook.md` | Integration paths, validation commands |
| `references/agent-protocols.md` | Required Evidence fields, review points |

## Agents

- `agents/claude.yaml` — Claude metadata
- `agents/openai.yaml` — OpenAI metadata