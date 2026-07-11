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

## Cross-repo integration

The integration workflow checks out `agent-taskstate` and `memx-resolver` as sibling
repositories and installs their declared runtime dependencies before instantiation:

```sh
python -m pip install -e ../agent-taskstate -e ../memx-resolver
```

This keeps the sample plugin config aligned with the dependencies declared by the
external plugin repositories.

## Runtime policy / tracing

`WorkflowPluginRuntime` は capability ごとに `PluginPolicy` を指定できる。

| Field | Default | Description |
| --- | --- | --- |
| `timeout_seconds` | `30.0` | plugin 呼び出しが戻るまでの上限秒数。`0` 以下で無効。 |
| `retry_count` | `0` | 失敗後の再試行回数。 |
| `retry_delay_seconds` | `1.0` | 再試行間隔。 |
| `continue_on_error` | `false` | 失敗時に例外ではなく `None` 結果として継続する。 |
| `trace_enabled` | `false` | 呼び出し trace を runtime に蓄積する。 |
| `isolation_mode` | `thread` | timeout を実時間で返すための thread isolation。`inline` も指定可能。 |

trace は `runtime.trace_payload()` で JSON 化でき、`runtime.write_traces_json(path)`
でファイルへ書き出せる。timeout や retry の診断証跡を release / evidence の
補助資料として残す用途を想定する。

`agent-protocols` 互換の Evidence JSON Lines が必要な場合は
`runtime.write_trace_evidence_jsonl(...)` を使う。`task_seed_id`、`base_commit`、
`head_commit`、`actor` を渡すことで、各 trace を `kind: Evidence` の record として
出力できる。

## 実装メモ

- host 側は `interfaces.py` の Protocol / coercion helper を正本にする
- `runtime.py` の `invoke_first` / `invoke_all` で capability dispatch を共通化している
- timeout を使う運用では `PluginPolicy.isolation_mode="thread"` を既定とし、
  長時間 plugin から runtime が戻れるようにする
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
