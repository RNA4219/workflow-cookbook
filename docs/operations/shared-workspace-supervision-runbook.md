# Shared Workspace Supervision Runbook

## 目的

複数workerが共有workspaceを使う場合に、可変状態の競合を防ぐ。Supervisorは依存関係・対象path・共有資源を特定し、独立readをbatch化する。競合write、Git index、lock、生成物等の更新はleaseとAuditで管理する。

実装は `tools/supervision/workspace_coordinator.py`。
状態は既定で `%USERPROFILE%/.workflow-cookbook/supervision/coordinator.sqlite3` に置かれ、対象repoを汚さない。

## 強制する運用

1. **独立分析・read-only検査**は同じ段階でbatch実行できる。依存する読み取りは直列にする。
2. **filesystem read**は共有可変状態との競合を確認する。coordinator参加workerは `--mode read` を使い、既定ではglobal WIPを指定しない。
3. **編集・評価出力・hash-manifest更新**はwrite leaseで保護する。shellはコマンドの実際の副作用で分類する。
4. 既定scopeは `.`。独立したpathでもindex・lock・cache・生成物を共有するwriteは同じscopeまたはWIPで排他する。
5. 評価runner等の共有資源にだけWIP=1を設定する。完了前Auditは依存する成果物・後続writeを対象にし、無関係なreadまで止めない。
6. `succeeded` は過去の完了記録。現入力・版・コマンドidentityと成果物hashを確認してから再利用する。`invalid` は原因を修正しidentityを更新するまで再実行しない。
7. fence tokenが古いworkerの出力は採用しない。期限切れleaseはheartbeatで復活できない。

## 1. 決定的job key

timestampを入れず、scope、role、stage、workspace、source revision、input hashes、command identity、output rootから生成する。

```powershell
uv run python tools/supervision/workspace_coordinator.py job-key `
  --workspace C:\work\repo `
  --scope evaluation --role Evaluator --stage G200 `
  --source-revision abc123 --input-hash sha256:deck --input-hash sha256:runner `
  --command-identity fixed-sample4-v1 --output-root artifacts/run-1
```

## 2. lease取得

既定scopeはworkspace全体で、重複writeは排他される。CLI/APIとも `--wip-key` は省略可能で、重複readは共有できる。
環境で実際にsandbox競合が確認された場合や共有runnerには、理由と解除条件を記録してglobal WIPを指定する。

```powershell
uv run python tools/supervision/workspace_coordinator.py acquire `
  --workspace C:\work\repo --mode write --owner supervisor `
  --task-id TASK-001 --job-key sha256:<digest> `
  --wip-key workspace_io_global --ttl-seconds 300
```

返却された `lease_id`、`lease_token`、`fence_token` を実行記録へ固定する。`lease_token`はstdoutに一度だけ現れ、DBにはhashだけが保存される。

readを共有する場合は `--mode read` とし、`--wip-key` を省略する。上の例の `workspace_io_global` を明示すると、同じstate DBを使うworkspace間でも直列になる。CLIの旧既定が必要な運用はこの引数を明示する。

## 3. heartbeat

60秒以内を目安に更新する。

```powershell
uv run python tools/supervision/workspace_coordinator.py heartbeat `
  --workspace C:\work\repo --lease-id <id> --lease-token <token> --ttl-seconds 300
```

期限切れ後は失敗する。新しいleaseを取得し、古いfence tokenの成果物を隔離する。

## 4. 完了・無効・解放

成功時:

```powershell
uv run python tools/supervision/workspace_coordinator.py finish `
  --workspace C:\work\repo --lease-id <id> --lease-token <token> `
  --job-key sha256:<digest> --outcome succeeded --result-ref artifacts/result.json
```

契約違反・用途別の必須件数不足・不正identityは `--outcome invalid`。0試合の条件はgame評価に適用する。
取り下げ／再試行可能な一時障害は理由と処理済み状態を記録して `release` する。期限切れを再取得する前も部分適用・冪等性・残予算を確認する。

`succeeded` の返却値は `reuse_requires_validation: true` を含む。coordinatorはresult参照先の内容を自動検証しない。
Supervisorは入力hash（未commit差分、model/runner/policy版を含む）とcommand identityを再計算し、同じjob keyであることを確認する。
さらにresult manifestに記録した成果物hash・実在・受入結果を照合する。欠損／不一致なら再利用せず記録を保全し、復旧条件を固定した新しいidentityで作業する。

## 5. 状態とAudit

```powershell
uv run python tools/supervision/workspace_coordinator.py status `
  --workspace C:\work\repo --events 50
```

SQLiteの`events`はappend-only。lease tokenはevent・statusへ出さない。

## 6. bounded I/O probe

I/O不調時は1 incidentにつき1回だけ実行する。

```powershell
uv run python tools/supervision/workspace_coordinator.py probe `
  --workspace C:\work\repo --timeout-ms 5000
```

probeは子processで実行し、timeout時に親が終了させる。失敗後に反復せず、部分適用を確認して停止する。

## Supervisorチェックリスト

- 子へ渡す役割が分析専用か、I/Oを伴うか分類した。
- coordinator参加workerのI/O前に対象scope、job key、read/write leaseを固定した。
- 共有状態・runner等に必要なWIPだけを設定した。
- heartbeatとfence tokenを監視した。
- 後続writeが依存する成果物をcompletion/Auditで確認した。
- 完了jobのidentityと成果物hashを照合してから再利用した。
- invalid identityを同一generationで再実行していない。

## 制約

これは協調的排他である。coordinatorを使わない外部processをOSレベルで停止しない。したがってSupervisorの指示、Task/Runbook、各agentの起動promptへ利用必須を明記する。
