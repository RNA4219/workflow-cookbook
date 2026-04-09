---
intent_id: INT-001
owner: your-handle
status: active   # draft|active|deprecated
last_reviewed_at: 2026-04-09
next_review_due: 2026-05-09
---

# Runbook

## Environments

- Local / CI / Prod の差分（キー名だけ列挙）

## Execute

- 準備 → 実行 → 確認（最短手順）
- 例）
  - 準備: データ投入 / キャッシュ初期化
  - 実行: コマンド/ジョブ名
  - 確認: 出力の存在・件数・整合
- 検収記録
  - `docs/acceptance/ACCEPTANCE_TEMPLATE.md` を複製し、
    `docs/acceptance/AC-YYYYMMDD-xx.md` を作成する。
  - 実行コマンド、テスト結果、参照ドキュメント、判定を記入する。
  - PR 本文の `Acceptance Record` から該当ファイルへリンクする。
- テスト工程
  - 単体テストと結合テストを分けて `TASK.codex.md` と検収記録へ記載する。
  - Python 系は `pytest --cov=. --cov-report=term-missing --cov-fail-under=80`
    を既定ゲートとして実行する。
  - coverage が 80% 未満なら、検収記録へ理由とフォローアップを残さない限り
    完了扱いにしない。
- CI / Governance 確認
  - `governance/policy.yaml` の `ci.required_jobs` を確認し、論理 gate ID
    (`governance-gate` / `python-ci` / `security-ci`) を基準にする。
  - GitHub 上の実 check 名は [docs/ci-config.md](docs/ci-config.md) の
    「論理 gate ID と実 check 名」を参照する。
  - Phase 判定は [docs/ci_phased_rollout_requirements.md](docs/ci_phased_rollout_requirements.md)
    を参照し、実在する workflow のみを対象にする。
  - branch protection の必須チェック名が、論理 gate ID ではなく
    実 check 名の対応表と一致しているか確認する。
  - GitHub API の取得権限がある場合は
    `python tools/ci/check_branch_protection.py --protection-json <json>` で
    branch protection export を検証する。
  - `python tools/ci/check_ci_gate_matrix.py` を実行し、
    `governance/policy.yaml`、workflow、`docs/ci-config.md` の整合を確認する。
  - `python tools/ci/check_acceptance.py --check` を実行し、
    検収記録の front matter と必須見出しを確認する。
  - `python tools/ci/check_security_posture.py --check --github-repo <owner/name>`
    を実行し、Security docs・Dependabot・GitHub security settings の
    恒常対策が維持されていることを確認する。
  - `python tools/ci/check_release_evidence.py --check --github-repo <owner/name>`
    を実行し、`CHANGELOG.md`、`docs/releases/`、git tag、公開 release の
    証跡が一致していることを確認する。
  - `python tools/ci/check_task_acceptance_sync.py --plugin-config examples/workflow_plugins.cross_repo.sample.json`
    を実行し、Task Seed と Acceptance の対応が plugin 観点でも一致することを確認する。
  - `python tools/ci/generate_acceptance_index.py --plugin-config examples/workflow_plugins.cross_repo.sample.json`
    で `docs/acceptance/INDEX.md` を再生成できることを確認する。
- docs resolve / stale 確認
  - `python tools/context/workflow_docs.py --plugin-config`
    `examples/workflow_plugins.cross_repo.sample.json resolve --task-id <task_id>`
    で読むべき docs を確認する。
  - 読了後は `... ack --task-id <task_id> --doc-id README.md` を実行する。
  - 検収前に `... stale --task-id <task_id> --check` を実行し、stale docs が残っていないことを確認する。
  - 3 repo 連携を確認する場合は `.github/workflows/cross-repo-integration.yml`
    と同じ順序で `workflow-cookbook` / `agent-taskstate` / `memx-resolver`
    の plugin テストを実行する。
- Birdseye / codemap 更新
  - 全体更新: `python tools/codemap/update.py --targets docs/birdseye/index.json,docs/birdseye/hot.json --emit index+caps`
  - 局所更新: `python tools/codemap/update.py --since --radius 1 --emit caps`
  - 確認:
    `docs/birdseye/index.json` / `docs/birdseye/hot.json` /
    `docs/birdseye/caps/*.json` の差分と `generated_at`
    （5 桁ゼロ埋め世代番号）を確認し、必要なら
    `CHECKLISTS.md` の [Hygiene](CHECKLISTS.md#hygiene) を更新する。

## Observability

- ログ/メトリクスの確認点、失敗時の兆候（[ADR-021: メトリクスと可観測性の統合](docs/ADR/ADR-021-metrics-observability.md) を参照）
- KPI の目的・閾値は [EVALUATION.md#KPIs](EVALUATION.md#kpis) を参照し、収集手順と解釈を同期する。
- インシデント発生時は docs/IN-YYYYMMDD-XXX.md に記録し、最新サンプル（[IN-20250115-001](docs/IN-20250115-001.md)）を参照して検知し、ログ・メトリクスの抜粋を添付
- QA メトリクス収集と確認（`tools/perf/` 共通テンプレート準拠）
  - `python -m tools.perf.collect_metrics --suite qa --metrics-url <Prometheus URL> --log-path <StructuredLogger 等の運用ログ>`
    を実行する。`--suite qa` は `.ga/qa-metrics.json` への書き出しを既定とし、Prometheus
    （`workflow_review_latency_*` は `review_latency` に正規化され、旧 `legacy_review_latency_*`
    も互換処理で継続利用可能。`task_seed_cycle_time_*`／`birdseye_refresh_delay_*` も同様に平均化）と
    運用ログ（`checklist_compliance_rate` の比率計算も含む）から統合メトリクスを取得する。
    出力先を変更したい場合は `--output <JSON パス>` を追加指定する。
    `semantic_retention` を取得するには `tools/perf/context_trimmer.trim_messages` へ
    `semantic_options`（例: `{"embedder": <callable>}`）を渡せるよう、ログエミッタ側で埋め込み関数を設定しておく。
    指標と検証手順の詳細は [docs/addenda/D_Context_Trimming.md](docs/addenda/D_Context_Trimming.md) を参照する。
    埋め込み関数はテキストを `Sequence[float]` へ変換できる必要があり、トリミング後の意味保持率は
    このベクトル間のコサイン類似度として集計される。
    `--metrics-url` または `--log-path` のどちらか片方しか利用できない場合は、利用可能な入力のみ指定する。
  - 対話 UI や運用ツールからメトリクスを出力する場合は `tools.perf.structured_logger.StructuredLogger`
    を利用する。例: `from tools.perf.structured_logger import StructuredLogger` →
    `StructuredLogger(name="ops-ui", path="~/.logs/ops/metrics.log")`
    `.inference(metrics={"semantic_retention": 0.9})`。
    こうして生成された JSON ログ行は `collect_metrics --log-path ~/.logs/ops/metrics.log` で取り込まれ、
    `metrics` キー配下の辞書がそのまま運用ログ由来メトリクスとして集計される。
  - LLM 行動追跡を `agent-protocols` の `Evidence` として残したい場合は、
    `StructuredLogger` に Evidence plugin を注入し、`extra.agent_protocol` に
    `evidence_id`、`task_seed_id`、`base_commit`、`head_commit`、`actor`
    を渡す。例:

    ```python
    from tools.perf.structured_logger import StructuredLogger
    from tools.protocols.evidence_bridge import AgentProtocolEvidenceFileSink

    logger = StructuredLogger(
        name="ops-ui",
        path="~/.logs/ops/metrics.log",
        plugins=[
            AgentProtocolEvidenceFileSink(
            path="~/.logs/ops/evidence.jsonl",
            repo_root=".",
            )
        ],
    )
    logger.inference(
        inference_id="run-001",
        model="gpt-5.4",
        prompt={"messages": [{"role": "user", "content": "Ping"}]},
        response={"content": "Pong"},
        extra={
            "agent_protocol": {
                "evidence_id": "EV-200",
                "task_seed_id": "TS-200",
                "base_commit": "abc1234",
                "head_commit": "def5678",
                "actor": "codex",
                "tools": ["StructuredLogger", "Edit", "Shell"],
            }
        },
    )
    ```

    これにより通常ログは維持したまま、`agent-protocols` 互換の Evidence JSON 行が
    `evidence.jsonl` に追加される。`extra.agent_protocol` が無いログ行は通常どおり無視される。
  - plugin をコードではなく設定から組み立てたい場合は、
    `StructuredLogger.from_plugin_specs(...)` と
    `InferencePluginSpec(factory="module:attribute", options={...})` を使う。
    `factory` には
    `tools.protocols.evidence_bridge:create_agent_protocol_evidence_plugin`
    のような import 文字列を指定できる。
  - さらに宣言的にしたい場合は `StructuredLogger.from_plugin_config(...)` を使い、
    次のような JSON config を渡せる。

    ```json
    {
      "inference_plugins": [
        {
          "factory": "tools.protocols.evidence_bridge:create_agent_protocol_evidence_plugin",
          "options": {
            "path": "logs/evidence.jsonl",
            "repo_root": "."
          },
          "enabled": true
        }
      ]
    }
    ```

    `.json` は常に利用でき、`.yaml` / `.yml` は yaml loader が存在する環境で利用できる。
    参照雛形として
    [`examples/inference_plugins.agent_protocol.sample.json`](examples/inference_plugins.agent_protocol.sample.json)
    を利用できる。
  - FastAPI などの Web サービスに組み込む場合は `tools.perf.metrics_registry.MetricsRegistry` を共有シングルトン
    として初期化し、トリミング完了時に `observe_trim` を呼び出す。`compress_ratio=` を直接指定する新 API と、
    既存の `original_tokens=` / `trimmed_tokens=` を渡す後方互換 API のどちらでも動作し、`semantic_retention`
    の有無も任意。`@app.get("/metrics")` エンドポイントで `return PlainTextResponse(registry.export_prometheus())`
    を返すと Prometheus が取得可能となる。収集 CLI は公開 API として `compress_ratio` / `semantic_retention`
    を参照しつつ、Prometheus 上では `trim_compress_ratio_*` / `trim_semantic_retention_*` を優先的に解釈する。
    再オープン率は `workflow_reopen_rate_*` → `docops_reopen_rate` → `reopen_rate` を順に確認し、スペック充足率は
    `workflow_spec_completeness_*` → `spec_completeness_*` → `spec_completeness` をフォールバックとして参照する。
  - 実行後に `.ga/qa-metrics.json` がリポジトリルート配下へ生成されていることを確認する。生成されない場合は
    `--output` に明示したパスと標準出力を突き合わせ、異常がないか確認する。
  - `python - <<'PY'` を実行し、以下を評価して各メトリクスの値を抽出する:

    ```python
    import json

    with open('.ga/qa-metrics.json', encoding='utf-8') as fh:
        data = json.load(fh)

    keys = (
        'checklist_compliance_rate',
        'compress_ratio',
        'semantic_retention',
        'task_seed_cycle_time_minutes',
        'birdseye_refresh_delay_minutes',
        'review_latency',
        'reopen_rate',
        'spec_completeness',
    )
    print({k: data[k] for k in keys})
    ```

    閾値は最新サンプルと突き合わせ、外れた場合は直近成功値との差分と再現条件を記録して共有する。
  - FastAPI 等へ常駐組み込みする際は `tools.perf.metrics_registry.MetricsRegistry` を介し、トリミング結果を逐次記録する:

      ```python
    from fastapi import FastAPI, Response

    from tools.perf.metrics_registry import MetricsRegistry

    registry = MetricsRegistry(default_labels={"service": "workflow"})
    app = FastAPI()

    @app.post("/trim")
    async def record_trim(payload: dict[str, float]) -> dict[str, str]:
        registry.observe_trim(
            compress_ratio=payload["compress_ratio"],
            semantic_retention=payload["semantic_retention"],
            labels={"model": payload.get("model", "unknown")},
        )
        # 旧 API を利用する場合の例（compress_ratio が未計算なときなど）:
        # registry.observe_trim(
        #     original_tokens=payload["original_tokens"],
        #     trimmed_tokens=payload["trimmed_tokens"],
        #     semantic_retention=payload.get("semantic_retention"),
        #     labels={"model": payload.get("model", "unknown")},
        # )
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(registry.export_prometheus(), media_type="text/plain")
      ```

  - `snapshot()` で `{"trim_compress_ratio": [{"labels": {...}, "count": 2, ...}]}` 形式の統計を確認できる。
    Prometheus エクスポートでは `trim_compress_ratio_{count,sum,avg,min,max}` および
    `trim_semantic_retention_{count,sum,avg,min,max}` を同一ラベル集合ごとに出力する。
    例:

    ```text
    # HELP trim_compress_ratio_count Compression ratio observed after trimming. (count).
    trim_compress_ratio_count{model="gpt-5",service="workflow"} 2
    trim_compress_ratio_avg{model="gpt-5",service="workflow"} 0.45
    ```

  - 公開メトリクス名: `trim_compress_ratio` / `trim_semantic_retention`
    （各 `_count`、`_sum`、`_avg`、`_min`、`_max` を同時出力）に加え、後方互換 Gauge
    `compress_ratio` / `semantic_retention` も平均値として公開する。

- 失敗兆候と一次対応
  - `.ga/qa-metrics.json` が生成されない / 壊れている: `python -m tools.perf.collect_metrics --help` でオプションを再確認し、一時ファイルやログ出力設定を洗い直してから再実行。
  - `checklist_compliance_rate` が 95% を下回る:
    実行時のチェックリスト完了ログを抽出し、どの項目が未完了かを
    Birdseye や Git 履歴で確認する。改善作業が必要な場合は Task Seed を
    追加投入する。
  - `task_seed_cycle_time_minutes` が 1440 分を超過: 受付から着手までの待機要因（担当者アサイン、依頼内容不備など）を振り返り、対応 SLA を再共有する。
  - `birdseye_refresh_delay_minutes` が 60 分を超過: Birdseye 更新ジョブの実行ログとスケジューラ状態を確認し、必要に応じて手動更新を実施。

## Outbound Request Approval

- 申請項目: [アウトバウンド通信申請テンプレート](tickets/outbound-request.md#申請テンプレート) を利用し、依頼者・宛先ドメイン/ポート・用途・送信データ分類・想定期間・
  フォールバック手順・再試行上限を記入し、最新の `network/allowlist.yaml` 差分を添付する。
- 承認者: 当番 SRE（一次）とセキュリティ/プライバシー担当（`docs/addenda/G_Security_Privacy.md#4-通信制御とツール実行`）が双方承認して初めて通信開始可とする。
- 記録: 承認完了後に `audit/outbound-requests.log` へ記録し、関連チケットへ決裁ログと実行結果リンクを残す。失敗時はインシデントテンプレ（`docs/IN-YYYYMMDD-XXX.md`）へ転記する。
- 再試行条件: 承認済み通信で 5xx / Timeout が発生した場合のみ、指数バックオフ（初回 2 分、最大 3 回）で自動再試行を許可する。
  4xx や `network/allowlist.yaml` 未反映による失敗は再申請を行い、承認が完了するまで再試行を禁止する。

## Confirm

- Execute 結果を主要メトリクス・アウトプットと突き合わせ、`CHECKLISTS.md` の [Hygiene](CHECKLISTS.md#hygiene) で整合性と未完了項目を再確認
- `docs/acceptance/AC-*.md` の判定と証跡が最新の実行結果と一致していることを確認
- 単体テスト / 結合テスト / coverage 80% の結果が検収記録と一致していることを確認
- CI / Governance 変更時は `.github/workflows/`、`governance/policy.yaml`、
  `docs/ci-config.md`、`docs/ci_phased_rollout_requirements.md` の記述が一致していることを確認
- required jobs を変更した場合は branch protection の必須チェック名と
  `docs/ci-config.md` の対応表も同時に見直し、
  docs だけ先行・設定だけ先行の片寄りを残さない
- インシデント記録を [docs/INCIDENT_TEMPLATE.md](docs/INCIDENT_TEMPLATE.md) に沿って初動報告→確定記録まで更新し、関連 PR / チケットへリンクを共有
- `Observability` で検知したアラート・兆候の解消を運用チャネルへ報告し、残るフォローアップを RUNBOOK / docs/IN-YYYYMMDD-XXX.md に追記

## Rollback / Retry

- どこまで戻すか、再実行条件
- CI Phase を戻す場合は、直前 Phase の required jobs へ戻し、
  `governance/policy.yaml` / `docs/ci-config.md` / `CHANGELOG.md` を同時更新する
- インシデントサマリを更新後、該当PRの説明欄と本RUNBOOKの該当セクションにリンクを追加する
