---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-07-11
next_review_due: 2026-08-11
---

# SPEC

## 1. 目的

Workflow Cookbook は、QA / Governance-first の運用ドキュメント、Birdseye
資産、参照実装、CI / Governance テンプレートを一体で提供する基盤リポジトリである。
本仕様書は、このリポジトリが外部へ公開するふるまい、入出力、運用上の互換条件を定義する。

## 2. 適用範囲

- 対象
  - ルートドキュメント群
  - `docs/birdseye/` の生成物
  - `tools/codemap/update.py`
  - `tools/autosave/` / `tools/merge/` / `tools/perf/`
  - 下流ソフトウェア向けの自己改善ループ契約
  - `agent-protocols` の `Evidence` 契約へ接続する追跡ブリッジ
  - `.github/workflows/` と `governance/`
- 非対象
  - 本番ホスティング実体
  - 外部 SaaS の本番設定
  - Cookbook 外部リポジトリ固有の追加要件

## 3. 想定利用者

- メンテナ
  - repo の文書、生成物、参照実装を保守する。
- 派生リポジトリ導入者
  - reusable workflow、policy、運用文書を再利用する。
- AI エージェント
  - Birdseye とハブ文書を最小読込の起点として利用する。
- QA / Ops / Security 担当
  - 受入基準、KPI、セキュリティ導線、CI 段階導入を評価する。

## 4. 機能仕様

### 4.1 ドキュメントハブ

- ルート文書の責務は次のとおり固定する。
  - `README.md`: 初動入口
  - `HUB.codex.md`: タスク分割入口
  - `RUNBOOK.md`: 実行手順入口
  - `EVALUATION.md`: 受入基準入口
  - `CHECKLISTS.md`: リリースと衛生チェック入口
  - `GUARDRAILS.md`: 行動制約と鮮度管理入口
- `docs/requirements.md` / `docs/spec.md` / `docs/design.md` /
  `docs/CONTRACTS.md` は互いに矛盾してはならない。
- 仕様変更時は `CHANGELOG.md` の `[Unreleased]` に差分を記録する。

#### 4.1.1 RUNBOOK slimming

- `RUNBOOK.md` は次の情報を保持する。
  - 実行手順
  - 現在の運用判断
  - 未解決事項
  - 参照リンク
  - incident / rollback / release などの実行時手順
- `RUNBOOK.md` は次の情報を原則として保持しない。
  - 完了済みタスクの長い一覧
  - 検収結果の詳細表
  - リリース履歴の詳細
  - `docs/tasks/*.md` や `docs/acceptance/*.md` と同じ内容の複製
- RUNBOOK に完了済み項目を書く場合は、現在の運用判断に必要な短い参照に限定する。
- 完了済み詳細を記録する場合は `docs/completion-record.md` へ索引として記載し、
  詳細は task / acceptance / release / changelog の正本へリンクする。

#### 4.1.2 RUNBOOK slimming check

機械チェックは、少なくとも次の条件を検出対象にする。

| check | 条件 | 期待動作 |
|---|---|---|
| Completed table growth | RUNBOOK に完了済み項目の表が一定行数以上追加される | warning |
| Missing canonical link | RUNBOOK の完了済み記述に task / acceptance / completion record へのリンクがない | warning |
| Duplicate completion detail | RUNBOOK と completion record / acceptance に同じ詳細表が重複する | warning |
| Current-ops exception | 現在の運用判断に必要な短い完了参照である | pass |

この check は `tools/ci/check_runbook_slimming.py` を実装入口とする。

### 4.2 Birdseye / Codemap

#### 4.2.1 `docs/birdseye/index.json`

- 最低限次のトップレベルキーを保持する。
  - `generated_at`
  - `nodes`
  - `edges`
- `generated_at`
  - 5 桁ゼロ埋めの世代番号であること。
  - 例: `00025`
- `nodes`
  - キーはノード ID であること。
  - 値は少なくとも次を持つこと。
    - `role`
    - `caps`
    - `mtime`
- `edges`
  - 2 要素配列の配列であること。
  - 各要素は `["from", "to"]` の形式で依存関係を表すこと。

#### 4.2.2 `docs/birdseye/hot.json`

- 最低限次のトップレベルキーを保持する。
  - `generated_at`
  - `index_snapshot`
  - `refresh_command`
  - `curation_notes`
  - `nodes`
- `generated_at`
  - `index.json` と同じ更新サイクルの 5 桁ゼロ埋め世代番号であること。
- `nodes[*]`
  - 少なくとも次を持つこと。
    - `id`
    - `role`
    - `reason`
    - `caps`
    - `edges`
    - `last_verified_at`

#### 4.2.3 `docs/birdseye/caps/*.json`

- カプセルは point read 用の最小要約として振る舞う。
- 最低限次のキーを持つこと。
  - `id`
  - `role`
  - `summary`
  - `deps_in`
  - `deps_out`
  - `risks`
  - `tests`
  - `generated_at`
- 必要に応じて `public_api` を持てること。

#### 4.2.4 `codemap.update` CLI

- エントリポイントは `python -m tools.codemap.update` とする。
- 主要引数は次のとおり。
  - `--targets`
    - 明示ターゲットによる更新対象指定
  - `--emit`
    - `index` / `caps` / `index+caps`
  - `--since`
    - `git diff --name-only <ref>...HEAD` ベースの対象抽出
  - `--radius`
    - 依存 hop 数制御
- `--radius` の仕様は次のとおり。
  - 既定値は `2`
  - `0` は seed ノードのみ更新
  - `1` 以上は指定 hop 数まで近傍展開
  - 負数は CLI エラー
- ルートターゲットに `docs/birdseye/`、`index.json`、`hot.json`、`caps/`
  を含めた場合は、全カプセルを探索起点として扱うこと。
- `index.json` を更新する場合、`hot.json` も同じ更新サイクルへ揃えること。

### 4.3 AutoSave 参照実装

#### 4.3.1 入力

- `AutoSaveRequest` は少なくとも次のフィールドを持つ。
  - `project_id`
  - `snapshot_delta`
  - `lock_token`
  - `snapshot_id`
  - `timestamp`
  - `precision_mode`
  - `latency_ms`（任意）
  - `lock_wait_ms`（任意）

#### 4.3.2 出力

- `AutoSaveResult` は少なくとも次のフィールドを持つ。
  - `status`
  - `applied_snapshot_id`
  - `next_retry_at`
- `status` は少なくとも次を返せること。
  - `ok`
  - `skipped`

#### 4.3.3 検証ルール

- ロックトークンの検証を行うこと。
- snapshot ID の単調増加を検証すること。
- rollout gate と checklist 完了条件を評価できること。

#### 4.3.4 例外とテレメトリ

- 主な例外は次のとおり。
  - `LockTokenInvalidError`
  - `SnapshotOrderViolation`
- commit 時には `autosave.snapshot.commit` を emit できること。
- payload は少なくとも次を含められること。
  - `project_id`
  - `snapshot_id`
  - `precision_mode`
  - `latency_ms`（任意）
  - `lock_wait_ms`（任意）

### 4.4 Merge 参照実装

#### 4.4.1 入力

- `MergePipelineRequest` は少なくとも次のフィールドを持つ。
  - `project_id`
  - `request_id`
  - `merged_snapshot`
  - `last_applied_snapshot_id`
  - `lock_token`
  - `autosave_lag_ms`（任意）
  - `latency_ms`（任意）
  - `lock_wait_ms`（任意）
  - `precision_mode_override`（任意）

#### 4.4.2 出力

- `MergePipelineResult` は少なくとも次のフィールドを持つ。
  - `status`
  - `precision_mode`
  - `resolved_snapshot_id`
  - `lock_released`
- `status` は次のいずれかであること。
  - `merged`
  - `conflicted`
  - `rolled_back`

#### 4.4.3 精度モード

- `precision_mode` は `baseline` または `strict` とする。
- `strict` では valid な `lock_token` を必須とする。
- `baseline` では lock release を許可できること。

#### 4.4.4 テレメトリ

- Merge は `merge.pipeline.metrics` を emit できること。
- payload は少なくとも次を含められること。
  - `precision_mode`
  - `status`
  - `merge.success.rate`
  - `merge.conflict.rate`
  - `merge.autosave.lag_ms`
  - `lock_validated`
  - `resolved_snapshot_id`
  - `latency_ms`（任意）
  - `lock_wait_ms`（任意）

### 4.5 `StructuredLogger` / agent-protocols Evidence 連携

#### 4.5.1 連携対象

- `StructuredLogger` は通常の JSON Lines ログ出力に加え、任意の `evidence_sink`
  互換引数に加え、任意個の `plugins` を受け取れること。
- 各 plugin は `handle_inference(record)` を実装し、logger 本体は plugin の中身を
  知らずに推論レコードを引き渡せること。
- plugin は import 文字列 `module:attribute` と options の組み合わせから
  生成できること。
- `StructuredLogger.from_plugin_specs(...)` は plugin spec 配列から logger を
  組み立てられること。
- `StructuredLogger.from_plugin_config(...)` は mapping または config file path
  から plugin spec を解決して logger を組み立てられること。
- config root は top-level の `inference_plugins` 配列、または配列直下のどちらかを
  受け付けること。
- file config は少なくとも `.json` を受け付け、`.yaml` / `.yml` は yaml loader
  が利用可能な環境で受け付けること。
- config file の shape は `schemas/inference-plugin-config.schema.json` で共有し、
  sample config と同期すること。
- Evidence 連携先の契約は
  `../agent-protocols/schemas/Evidence.schema.json`
  を正本とすること。
- 追跡対象は LLM の推論 1 回ごとの行動証跡とし、`InferenceLogRecord` から
  `Evidence` へ 1:1 で変換すること。

#### 4.5.2 発火条件

- `StructuredLogger.inference()` は既存どおり常に通常ログを 1 行出力すること。
- `extra.agent_protocol` が無い場合、Evidence の生成は行わず通常ログだけで完了すること。
- `extra.agent_protocol` がある場合、Evidence sink は schema 互換の JSON を生成すること。
- `extra.agent_protocol` があるにもかかわらず必須フィールドが欠落する場合は、
  変換専用の例外を返して不正な Evidence を出力しないこと。

#### 4.5.3 `extra.agent_protocol` の最小契約

- 最低限次のキーを受け付けること。
  - `evidence_id`
  - `task_seed_id`
  - `base_commit`
  - `head_commit`
  - `actor`
- 次のキーは任意入力として受け付けること。
  - `start_time`
  - `model_version`
  - `parameters`
  - `parameters_hash`
  - `tools`
  - `policy_verdict`
  - `stale_status`
  - `merge_result`
  - `diff`
  - `diff_hash`
  - `approvals_snapshot`
  - `environment`

#### 4.5.4 Evidence 生成ルール

- 共通フィールドは次の固定値または導出値を使うこと。
  - `schemaVersion`: `1.0.0`
  - `kind`: `Evidence`
  - `state`: `Published`
  - `version`: `1`
- 時刻は次のように解決すること。
  - `createdAt` / `updatedAt` / `endTime`: `InferenceLogRecord.timestamp`
  - `startTime`: `extra.agent_protocol.start_time` があればそれ、無ければ `timestamp`
- ハッシュは `sha256:<hex>` 形式で正規化入力から算出すること。
  - `inputHash`: `prompt`
  - `outputHash`: `response`
  - `diffHash`: `diff_hash` があればそれ、無ければ `diff`
  - `model.parametersHash`: `parameters_hash` があればそれ、無ければ `parameters`
- `model` は次のように解決すること。
  - `name`: `InferenceLogRecord.model`
  - `version`: `model_version` があればそれ、無ければ `unknown`
  - `parametersHash`: 上記導出値
- `tools` は `extra.agent_protocol.tools` があればそれを使い、
  無ければ `["StructuredLogger"]` を使うこと。
- `environment` は次の既定値を持てること。
  - `os`: 実行環境の OS 名
  - `runtime`: 実行中 Python ランタイム
  - `containerImageDigest`: `uncontainerized`
  - `lockfileHash`: repo root の既知 lockfile から導出したハッシュ。
    lockfile が無い場合は sentinel 値のハッシュを使うこと。
- `staleStatus` は指定が無ければ次を使うこと。
  - `classification`: `fresh`
  - `evaluatedAt`: `InferenceLogRecord.timestamp`
- `mergeResult` は指定が無ければ `{"status": "not_applicable"}` を使うこと。
- `policyVerdict` は指定が無ければ `manual_review_required` を使うこと。

#### 4.5.5 出力

- Evidence sink は 1 Evidence につき 1 JSON object を生成すること。
- file writer plugin は UTF-8 の JSON Lines として末尾追記できること。
- Evidence 出力は通常ログの内容を変更してはならないこと。

### 4.6 自己改善ループ blueprint

#### 4.6.1 適用方針

- 本機能は `workflow-cookbook` 自身へ `hermes-agent` を組み込むものではない。
- `workflow-cookbook` は、下流ソフトウェアが独自実装できる
  自己改善ループの契約を外向きに提供する。
- 本機能は任意であり、未導入の下流ソフトウェアに必須ではない。
- 本機能は原則としてリリース後運用で有効化する。
- 開発中や作成途中の変更に対して、本機能の利用を必須条件としない。
- `workflow-cookbook` は次を正本として扱う。
  - reflection summary
  - skill draft
  - recall response
  - user / workspace model snapshot

#### 4.6.2 ReflectionSummary

- `ReflectionSummary` は少なくとも次を持つこと。
  - `session_id`
  - `task_id` または `intent_id`
  - `objective`
  - `changes`
  - `lessons`
  - `open_questions`
  - `next_actions`
  - `sources`
- `sources` は acceptance / evidence / docs reference のいずれかを保持できること。

#### 4.6.3 SkillDraftRecord

- `SkillDraftRecord` は少なくとも次を持つこと。
  - `draft_id`
  - `source_session_id`
  - `title`
  - `problem`
  - `proposed_steps`
  - `review_state`
  - `linked_acceptance_ids`
  - `linked_evidence_ids`
- `review_state` は少なくとも次を扱えること。
  - `draft`
  - `review`
  - `approved`
  - `rejected`
- `approved` 以外の draft は公開 skill として扱わないこと。

#### 4.6.4 RecallResponse

- `RecallResponse` は少なくとも次を持つこと。
  - `query`
  - `summary`
  - `hits`
  - `stale`
- `hits[*]` は少なくとも次を持つこと。
  - `source_type`
  - `source_id`
  - `excerpt`
  - `reason`
- recall は raw transcript 全文ではなく、
  summary と根拠断片に正規化して返すこと。

#### 4.6.5 UserModelSnapshot / WorkspaceModelSnapshot

- `UserModelSnapshot` は少なくとも次を持つこと。
  - `user_id`
  - `preferences`
  - `approval_style`
  - `output_conventions`
  - `reviewed_at`
- `WorkspaceModelSnapshot` は少なくとも次を持つこと。
  - `workspace_id`
  - `constraints`
  - `preferred_docs`
  - `reviewed_at`
- 長期保持される snapshot は review 済みであること。

#### 4.6.6 Periodic Nudges

- nudge は少なくとも次を持つこと。
  - `nudge_id`
  - `reason`
  - `target_kind`
  - `target_ref`
  - `suggested_action`
  - `created_at`
- nudge は自動変更ではなく、次回セッションへの提案として扱うこと。
- nudge はリリース前の未完了作業へ割り込んで必須フロー化しないこと。

#### 4.6.7 差し替え可能性

- 次の要素は下流ソフトウェアで差し替え可能であること。
  - memory store
  - search backend
  - summarizer
  - skill registry
  - scheduler
- 上記差し替えにかかわらず、`ReflectionSummary`、
  `SkillDraftRecord`、`RecallResponse` の最低フィールドは維持すること。

### 4.7 Metrics 収集 CLI

#### 4.7.1 入力ソース

- 収集元は次のいずれか、または両方を受け付けること。
  - `--metrics-url`
  - `--log-path`
- どちらも未指定の場合は `MetricsCollectionError` を返すこと。
- 既定エラーメッセージは
  `No metrics input configured: provide --metrics-url or --log-path`
  を用いること。

#### 4.7.2 suite

- `--suite qa` を提供すること。
- `qa` suite の既定出力先は `.ga/qa-metrics.json` とすること。

#### 4.7.3 出力

- 結果は標準出力へ JSON として出力すること。
- `output_path` がある場合はファイルにも書き出すこと。
- `--pushgateway-url` 指定時は PushGateway へ PUT 送信できること。

### 4.8 CI / Governance テンプレート

- `.github/workflows/reusable/*.yml` は `workflow_call` により派生リポジトリから再利用できること。
- `governance/policy.yaml` は少なくとも次の責務を持つこと。
  - 論理 gate ID としての `required_jobs` の基準
  - `forbidden_paths` の基準
- 論理 gate ID と GitHub 上の実 check 名の対応は `docs/ci-config.md` で管理すること。
- `docs/ci_phased_rollout_requirements.md` と workflow 群は、Phase 0〜3
  の段階導入方針を追跡できること。
- Python CI は単体テスト・結合テストの実行と coverage 下限 80% の確認を
  標準で行えること。

### 4.9 Security baseline

- `.github/dependabot.yml` は GitHub Actions 依存更新を週次で監視すること。
- `.github/workflows/security.yml` は security posture 確認と reusable security CI を連結すること。
- security posture 確認では少なくとも次を検証できること。
  - `docs/security/SAC.md`
  - `docs/security/Security_Review_Checklist.md`
  - vulnerability alerts
  - Dependabot security updates
  - secret scanning
  - push protection
- security posture の検証 CLI は GitHub token がある場合に remote repository settings を確認できること。

### 4.10 外部契約

- `docs/CONTRACTS.md` の契約は feature detection で扱うこと。
- 少なくとも次を optional な外部入力として扱えること。
  - `.ga/qa-metrics.json`
  - `governance/predictor.yaml`
- これらが未提供でも Cookbook 側は正常動作しなければならない。

### 4.11 Task / Acceptance / Completion trace

#### 4.11.1 正本の役割

| artifact | 正本として扱う内容 |
|---|---|
| `docs/tasks/*.md` | 作業の背景、目的、要求、完了条件、レビュー観点 |
| `docs/acceptance/*.md` | 実行コマンド、テスト結果、検収判定、参照資料 |
| `docs/completion-record.md` | 完了事項の要約索引と正本リンク |
| `CHANGELOG.md` | ユーザー向け変更履歴 |
| `docs/releases/*.md` | リリース証跡、承認、rollback/rehearsal 記録 |

#### 4.11.2 Trace rule

- `status: done` の Task Seed は、原則として対応する acceptance record を参照する。
- acceptance record が不要な小変更では、Task Seed または completion record に
  例外理由を短く記載する。
- completion record は task / acceptance / release / changelog のいずれかへリンクし、
  単独の正本として完了を主張しない。
- completion record の 1 項目は次の最小情報を持つ。
  - 日付
  - 完了テーマ
  - 状態
  - 正本リンク
  - 判定 (`go` / `hold` / `follow-up required`)

#### 4.11.3 Sync check

同期チェックは、少なくとも次を検出対象にする。

| check | 条件 | 期待動作 |
|---|---|---|
| Done task without acceptance | `docs/tasks/*.md` が `status: done` だが acceptance または例外理由がない | fail or warning |
| Completion without source | completion record の項目に正本リンクがない | fail |
| Acceptance orphan | acceptance record が task / release / completion のどれからも参照されない | warning |
| Changelog drift | release/changelog にある完了事項が completion record から辿れない | warning |

この check は `tools/ci/check_completion_trace.py` を実装入口とする。

### 4.12 Agent-tools-hub boundary

`agent-tools-hub` と `workflow-cookbook` の責務境界は次のとおり。

| 判断対象 | 正本 | 内容 |
|---|---|---|
| Agent_tools 全体の repo 選定 | `agent-tools-hub` | どの repo / Skill を使うべきかの横断案内 |
| 複数 repo の初動整理 | `agent-tools-hub` | repo map、入口、既存 Skill へのルーティング |
| Workflow Cookbook 内の作業分割 | `workflow-cookbook/HUB.codex.md` | cookbook 内の docs / Birdseye / Task Seed への分解 |
| Birdseye / Task Seed / Acceptance / CI / Evidence の手順 | `workflow-cookbook` | 実行手順、契約、検収、証跡 |
| 他 repo との接続 | `workflow-cookbook` | plugin config、Evidence、Acceptance、Task state などの連携契約 |

境界ルール:

- `workflow-cookbook/HUB.codex.md` は Agent_tools 全体の routing table を複製しない。
- `workflow-cookbook` から他 repo を説明する場合、連携契約と実行手順に限定する。
- repo 選定や Skill 選定の説明が必要な場合は `agent-tools-hub` を参照する。
- `workflow-cookbook` の改善案は、Birdseye、Task Seed、Acceptance、CI、Evidence、
  release/security operations、self-improvement loop の品質向上に限定する。

### 4.13 Version consistency

release 前の version 情報は、次の artifact 間で整合していること。

| artifact | 期待値 |
|---|---|
| `README.md` badge | 最新 release または明示された current version |
| `pyproject.toml` project.version | package として配布する version |
| `CHANGELOG.md` | release version と日付 |
| `docs/releases/v*.md` | release note の filename と title |
| git tag | `vX.Y.Z` 形式の release tag |
| GitHub Release | tag / changelog / release docs と対応する published release |

整合ルール:

- 同一 release に属する version は `v` prefix の有無を正規化して比較する。
- `CHANGELOG.md` にある release version は、対応する `docs/releases/vX.Y.Z.md`
  を持つこと。
- release docs の title は filename の version と一致すること。
- package 配布を行う release では `pyproject.toml` の version と release version が
  一致すること。
- package 配布を行わない docs-only release では、例外理由を release docs または
  acceptance record に記録すること。

将来の checker は `tools/ci/check_release_evidence.py` の責務を拡張するか、
小さな `check_version_consistency.py` として実装する。

### 4.14 Stable CLI entrypoints

既存の script 入口は後方互換として維持する。

| 既存入口 | 将来 entrypoint 例 | 責務 |
|---|---|---|
| `python -m tools.codemap.update` | `workflow-cookbook birdseye update` | Birdseye / Codemap 更新 |
| `python tools/ci/check_ci_gate_matrix.py` | `workflow-cookbook gate ci-matrix` | logical gate と workflow/docs の整合確認 |
| `python tools/ci/check_acceptance.py` | `workflow-cookbook acceptance check` | acceptance record validation |
| `python tools/ci/generate_acceptance_index.py` | `workflow-cookbook acceptance index` | acceptance index 生成 |
| `python tools/ci/check_release_evidence.py` | `workflow-cookbook release evidence` | release 証跡確認 |

CLI 互換ルール:

- package entrypoint は既存 script と同じ exit code 意味論を維持する。
- 既存 script を削除せず、少なくとも 1 release cycle は wrapper として残す。
- 新旧入口の smoke test を追加し、同一 fixture に対して同等の結果を返すことを確認する。
- help text には対応する docs、主要入力、出力、strict mode の有無を含める。

### 4.15 Docs gate escalation policy

docs gate checker は次の段階を持てる。

| stage | 意味 | CI 動作 |
|---|---|---|
| observe | 導入直後。結果を収集する | pass with notice |
| warn | 既存例外を許容しつつ差分へ注意を出す | pass with warning |
| enforce | 既定違反を merge blocker にする | fail |

昇格条件:

- checker ごとに owner、対象ファイル、既定 stage、次回見直し日を
  `docs/ci-config.md` または `governance/policy.yaml` から追跡できること。
- `warn` から `enforce` へ上げる前に、既存違反の棚卸しと例外理由を記録する。
- false positive が発生した場合は、checker の修正または明示 suppression を優先し、
  gate 全体を無効化しない。
- rollback 条件は「誤検知で通常 docs 更新を阻害する」「既存 release 手順を壊す」
  「downstream reusable workflow が互換性を失う」のいずれかとする。

### 4.16 Plugin capability catalog

workflow plugin capability catalog は、少なくとも次を保持する。

| field | 内容 |
|---|---|
| `capability` | capability 名。例: `docs.resolve` |
| `required_method` | plugin が提供すべき method 名 |
| `input_contract` | 入力 DTO または mapping の最低フィールド |
| `output_contract` | 出力 DTO または mapping の最低フィールド |
| `error_policy` | timeout / retry / fail-open / fail-closed |
| `trace_event` | runtime trace に残す event 名 |
| `schema_ref` | 関連 schema または sample config |

catalog 整合ルール:

- `tools/workflow_plugins/interfaces.py`、`runtime.py`、config schema、
  `tools/workflow_plugins/README.md`、`examples/workflow_plugins*.json` は
  capability 名と required method で一致すること。
- inference plugin についても `schemas/inference-plugin-config.schema.json`、
  sample config、`StructuredLogger` の plugin loader が矛盾しないこと。
- capability 追加時は、sample config と validation test を同時に更新する。
- capability rename は破壊的変更として扱い、migration note と互換 alias の有無を明記する。

### 4.17 Large module split policy

large module は `TECH_DEBT_REGISTER.md` を正本として管理する。

対象基準:

- 500 行を超える Python module
- 20 関数を超える Python module
- CLI、I/O、集計、外部通信、出力が 1 ファイルに集中している module

分割ルール:

- まず `cli.py` と core logic を分け、既存 entrypoint は wrapper として維持する。
- 互換維持のため、既存 import path から public function を re-export できること。
- 分割前後で既存 unit test と CLI smoke test が同じ結果を返すこと。
- 分割計画には対象 module、分割先、優先度、期限、関連テスト、抑制 expiry を含める。
- suppression は永続化せず、期限切れ時に再評価する。

### 4.18 Five-tool run manifest

`five-tool-run-manifest` は RanD、Code-to-gate、HATE、manual-bb-test-harness、QEG の
検収 run を1つの証跡として束ねる JSON artifact である。schema 正本は
`schemas/five-tool-run-manifest.schema.json`、sample config は
`examples/five-tool-chain-manifest.sample.json` とする。

#### 4.18.1 CLI

入口は次の2系統を維持する。

| 入口 | 用途 |
|---|---|
| `python tools/ci/five_tool_manifest.py generate --config <json> --out <json> --validate` | config から manifest を生成し、その場で契約検証する |
| `python tools/ci/five_tool_manifest.py validate --manifest <json> --json` | 既存 manifest を再検証する |
| `wfc-five-tool-manifest` | package entrypoint。既存 script 入口と同じ exit code を返す |

#### 4.18.2 Manifest DTO

manifest は少なくとも次を保持する。

| field | 内容 |
|---|---|
| `schema_version` | manifest schema version。現行は `1.0` |
| `manifest_id` | `five-tool:` prefix を持つ manifest ID |
| `run_id` | 検収 run ID |
| `chain` | `rand`, `code-to-gate`, `hate`, `manual-bb`, `qeg` の順序 |
| `repos[]` | 各 repo の path、branch、commit、upstream、dirty 状態 |
| `artifacts[]` | 入出力 artifact の path、SHA-256、schema version、verdict、trace id、policy hash |
| `qeg_policy_hash` | QEG policy 正本を識別する `sha256:` hash |
| `final_verdict` | `go`, `conditional_go`, `no_go`, `needs_review`, `disqualified` |
| `degraded[]` | 参照のみ、欠落、内包など完全証跡でない場合の理由と影響 |

#### 4.18.3 Contract validation

validation は少なくとも次を検出する。

- required repo に commit が無い場合は fail。
- required artifact が無い場合は fail。
- manifest 内の artifact hash と現在のファイル hash が異なる場合は fail。
- QEG 以外の artifact が直接 `gate_policy` を持つ場合は fail。
- `qeg_policy_hash` が QEG artifact の `policyHash` に存在しない場合は fail。
- QEG artifact が `no_go` または `disqualified` を含むのに final verdict が `go` の場合は fail。
- required repo が dirty の場合、または joinable trace id が無い主要 artifact は warning。

#### 4.18.4 Evidence pack

五ツール検収の evidence pack は、生成された manifest と validation report を
`docs/evidence/five-tool-validation-YYYYMMDD/` に保存できること。
通常の acceptance 正本は既存どおり fixture / cached corpus を優先し、manifest は
cross-repo contract と証跡束ねの観測点として扱う。

## 5. 互換性と変更管理

- ドキュメント、Birdseye 生成物、参照実装、CI テンプレートは相互に矛盾してはならない。
- 公開インターフェース変更時は、関連テストと関連文書を同時に更新すること。
- 変更履歴は `CHANGELOG.md` の `[Unreleased]` に追記すること。
- version、CLI entrypoint、plugin capability、large module split のいずれかを
  変更する場合、互換性影響を `CHANGELOG.md` または該当 task / acceptance に記録すること。

## 6. 検証観点

- 文書整合
  - `requirements` / `spec` / `design` / `CONTRACTS` が矛盾しないこと。
- RUNBOOK slimming
  - RUNBOOK に完了済み詳細が増える変更では、completion record への分離要否が確認されていること。
  - completion record の各項目が正本リンクを持つこと。
- Task / Acceptance / Completion trace
  - `status: done` の task から acceptance または例外理由へ辿れること。
  - completion record が単独の完了正本になっていないこと。
- Agent-tools-hub boundary
  - `workflow-cookbook/HUB.codex.md` が Agent_tools 全体の repo routing を複製していないこと。
  - 横断 repo 選定が必要な文脈では `agent-tools-hub` を参照していること。
- Version consistency
  - README badge、`pyproject.toml`、`CHANGELOG.md`、release docs、git tag が
    release 文脈で矛盾しないこと。
- Stable CLI entrypoints
  - 新しい package entrypoint が既存 script 入口と同じ fixture で同等の結果を返すこと。
  - console script smoke test は `pip` が無い隔離 Python では
    `uv pip install --python <interpreter> -e .` へフォールバックすること。
- Docs gate escalation
  - checker ごとの stage、owner、昇格条件、rollback 条件が追跡できること。
- Plugin capability catalog
  - runtime、schema、README、sample config の capability 定義が一致すること。
- Large module split
  - large module の suppression が `TECH_DEBT_REGISTER.md` の分割計画へリンクしていること。
- Birdseye 整合
  - `README.md`、`docs/BIRDSEYE.md`、`docs/birdseye/README.md`、
    `GUARDRAILS.md`、`RUNBOOK.md` の更新手順が一致すること。
  - `index.json.generated_at` と `hot.json.generated_at` が同じ更新サイクルであること。
- 代表テスト
  - `tests/test_codemap_update.py`
  - `tests/autosave/test_project_lock_service.py`
  - `tests/merge/test_precision_mode_pipeline.py`
  - `tests/test_collect_metrics_cli.py`
  - `tests/perf/test_collect_metrics_autosave_merge.py`
  - `tests/test_structured_logger.py`
  - `tests/test_agent_protocol_evidence.py`
- Python 系ゲート
  - `pytest --cov=. --cov-report=term-missing --cov-fail-under=80`
- 五ツール検収
  - RanD → Code-to-gate → HATE → manual-bb-test-harness → QEG の順で証跡を束ねること。
  - Code-to-gate の `blocked_input` または HATE の `hold` は QEG 相当の最終 Gate へ伝播すること。
  - five-tool run manifest が repo commit、artifact hash、QEG policyHash、final verdict を記録し、
    QEG 以外の direct `gate_policy` と QEG No-Go の見落としを検出すること。
  - HATE `real-repo run` の pytest command は、隔離環境で `pip` / console script に依存しない
    `uv run --with PyYAML --with pytest python -m pytest -q` 形式を推奨すること。

## self-improvement/v1 Gate effectiveness

入力は`ImprovementObservationBundle`、出力は`GateEffectivenessReport`とnon-blocking
`PeriodicNudge`である。`analyze-gates`はbundle schemaを先に検証し、閾値設定を読んで各Gateと
Evidence typeを集計する。旧観測の必須判定fieldが`unknown`なら`insufficient_data`とする。
unused Gateの初回actionは`review`、同じ分類が2期間連続した場合だけ`archive_candidate`とし、
`hard_safety=true`は常にarchiveから除外する。

## 7. 関連資料

- 要件: `docs/requirements.md`
- 設計: `docs/design.md`
- 受入基準: `EVALUATION.md`
- 実行手順: `RUNBOOK.md`
- 境界一覧: `docs/interfaces.md`
- 外部契約: `docs/CONTRACTS.md`
