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

- `task_state.sync`
- `acceptance.index`
- `docs.resolve`
- `docs.ack`
- `docs.stale_check`

capability ごとに必要な method 名も固定されています。

- `task_state.sync` -> `sync_task_acceptance`
- `acceptance.index` -> `build_acceptance_index`
- `docs.resolve` -> `resolve_docs`
- `docs.ack` -> `ack_docs`
- `docs.stale_check` -> `stale_check`

host は plugin 読み込み時にこの対応を検証します。

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
python tools/ci/generate_acceptance_index.py --plugin-config examples/workflow_plugins.cross_repo.sample.json
python tools/context/workflow_docs.py --plugin-config examples/workflow_plugins.cross_repo.sample.json resolve --task-id 20260410-01
```
