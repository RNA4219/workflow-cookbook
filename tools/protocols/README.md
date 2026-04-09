# Protocol Plugins

`tools/protocols/` は、`StructuredLogger` へ外部契約連携を差し込むための
plugin 群と loader をまとめるディレクトリです。

現在の主用途は、`agent-protocols` の `Evidence` 契約へ
LLM 行動追跡ログをミラーすることです。

## 役割

- `evidence_bridge.py`
  - `InferenceLogRecord` を `Evidence` JSON へ変換する
  - context extractor / environment resolver / mapper / writer / plugin を提供する
- `plugin_loader.py`
  - `module:attribute` 形式の import 文字列から plugin factory を解決する
  - `InferencePluginSpec` から plugin を生成する
- `plugin_config.py`
  - mapping または `.json` / `.yaml` / `.yml` から `InferencePluginSpec` を読む
  - top-level `inference_plugins` 配列、または配列直下を受け付ける

## 基本 API

plugin は `handle_inference(record)` を実装した任意オブジェクトです。

```python
from tools.perf.structured_logger import StructuredLogger


class MyPlugin:
    def handle_inference(self, record) -> None:
        print(record.inference_id)


logger = StructuredLogger(
    name="workflow.metrics",
    path="metrics.log",
    plugins=[MyPlugin()],
)
```

## import 文字列から組み立てる

`plugin_loader.py` は `module:attribute` 形式の factory を受け付けます。

```python
from tools.perf.structured_logger import StructuredLogger
from tools.protocols.plugin_loader import InferencePluginSpec

logger = StructuredLogger.from_plugin_specs(
    name="workflow.metrics",
    path="metrics.log",
    plugin_specs=[
        InferencePluginSpec(
            factory="tools.protocols.evidence_bridge:create_agent_protocol_evidence_plugin",
            options={
                "path": "evidence.jsonl",
                "repo_root": "C:/Users/ryo-n/Codex_dev/workflow-cookbook",
            },
        )
    ],
)
```

## config から組み立てる

`StructuredLogger.from_plugin_config(...)` は mapping または config file を受け付けます。

```python
from tools.perf.structured_logger import StructuredLogger

logger = StructuredLogger.from_plugin_config(
    name="workflow.metrics",
    path="metrics.log",
    plugin_config="examples/inference_plugins.agent_protocol.sample.json",
)
```

参照サンプル:

- [`examples/inference_plugins.agent_protocol.sample.json`](../../examples/inference_plugins.agent_protocol.sample.json)
- [`examples/agent_protocol_evidence_consumer.sample.py`](../../examples/agent_protocol_evidence_consumer.sample.py)
- [`schemas/inference-plugin-config.schema.json`](../../schemas/inference-plugin-config.schema.json)

## Evidence plugin

`agent-protocols` 向けの既定 factory は
`tools.protocols.evidence_bridge:create_agent_protocol_evidence_plugin` です。

この plugin は `extra.agent_protocol` を見て動作します。
最低限必要なのは次のキーです。

- `evidence_id`
- `task_seed_id`
- `base_commit`
- `head_commit`
- `actor`

`extra.agent_protocol` が無い場合は no-op で、通常ログだけを書きます。

## 設計方針

- `StructuredLogger` は plugin 呼び出しだけを担当する
- 外部契約固有の知識は `tools/protocols/` 側へ閉じ込める
- plugin は差し替え可能で、logger 本体を変更しない
- config loader は宣言的な導入を補助するが、未使用でも既存コードはそのまま動く
- config file の shape は schema と sample を同期して管理する
