# Workflow Plugins

`tools/workflow_plugins/` は、`workflow-cookbook` から sibling repo の機能を
疎結合に読み込むための host です。

## 目的

- `agent-taskstate` を Task Seed / Acceptance / Evidence の状態 backend として使う
- `memx-resolver` を docs resolve / ack / stale-check backend として使う
- `workflow-cookbook` 本体は plugin capability と config だけを知る

## 設定

workflow plugin config は JSON または YAML で、`workflow_plugins` 配列を持ちます。

```json
{
  "workflow_plugins": [
    {
      "factory": "agent_taskstate_workflow_plugin.plugin:create_plugin",
      "python_paths": ["../agent-taskstate"]
    }
  ]
}
```

## capabilities

| Capability ID | Method Name | Result Type | Description |
| --- | --- | --- | --- |
| `task_state.sync` | `sync_task_acceptance` | `TaskAcceptanceSyncReport` | Task/Acceptance state sync |
| `acceptance.index` | `build_acceptance_index` | `AcceptanceIndexResult` | Acceptance index markdown |
| `docs.resolve` | `resolve_docs` | `DocsResolveResult` | Required/recommended docs |
| `docs.ack` | `ack_docs` | `DocsAckResult` | Docs read acknowledgment |
| `docs.stale_check` | `stale_check` | `DocsStaleResult` | Stale docs check |

capability と method の対応は `interfaces.py` の `CAPABILITY_METHOD_NAMES` で定義され、
`schemas/plugin-capability-catalog.schema.json` / `examples/plugin-capability-catalog.sample.json`
で照合可能。

## 検証

config の shape だけを見る:

```sh
python tools/workflow_plugins/validate_workflow_plugin_config.py --plugin-config examples/workflow_plugins.cross_repo.sample.json
```

import / instantiate まで確認する:

```sh
python tools/workflow_plugins/validate_workflow_plugin_config.py --plugin-config examples/workflow_plugins.cross_repo.sample.json --instantiate --emit-json
```

## 実装メモ

- host 側は `interfaces.py` の Protocol / coercion helper を正本にする
- `runtime.py` の `invoke_first` / `invoke_all` で capability dispatch を共通化している
- `agent-taskstate` plugin は markdown scan を `store.py` に切り出している
- `agent-taskstate` plugin は result dataclass と acceptance renderer を分離している
- `memx-resolver` plugin は docs selection policy、receipt store、resolve cache store を分離している
- docs resolve cache は host ではなく `memx-resolver` plugin 側で signature-aware に保持する

## 代表コマンド

```sh
python tools/ci/check_task_acceptance_sync.py --plugin-config examples/workflow_plugins.cross_repo.sample.json

# done task に acceptance を必須化したい場合
python tools/ci/check_task_acceptance_sync.py --plugin-config examples/workflow_plugins.cross_repo.sample.json --require-acceptance-for-done
python tools/ci/generate_acceptance_index.py --plugin-config examples/workflow_plugins.cross_repo.sample.json
python tools/context/workflow_docs.py --plugin-config examples/workflow_plugins.cross_repo.sample.json resolve --task-id 20260410-01
```
