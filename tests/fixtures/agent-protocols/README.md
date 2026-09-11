# Evidence schema fixture

2026-09-10に隣接agent-protocols/schemasから複製した契約fixture。
外部schemaに独自fieldを追加していない。更新時は元schemaとhashを確認する。

| ファイル | SHA-256 |
| --- | --- |
| Evidence.schema.json | ea35f5262d63444e01fc4aeeaa62dd8fbec3a507f3ea6d7968da127a14b8609f |
| common.schema.json | fccfc91e3c1acf0c567bcbb09b5bc1cccb460ac91b11c8c8b5f5c12cd7f48f21 |

テストはDraft 2020-12で検証し、common参照をローカルRegistryへ登録する。
schemaの取得元はagent-protocolsのREADMEに記載された正式契約。ネットワーク取得をテストの前提にしない。
