---
intent_id: DOC-LEGACY
owner: docs-core
status: active
last_reviewed_at: 2026-04-09
next_review_due: 2026-05-09
---

# REQUIREMENTS

## 1. 目的

Workflow Cookbook は、AI エージェント運用とドキュメントガバナンスを一貫して扱うための基盤リポジトリである。  
本リポジトリは次を同時に満たさなければならない。

- 人間とエージェントが同じ入口から仕様・設計・実行手順へ辿れること
- Birdseye により最小読込で依存関係と更新対象を把握できること
- AutoSave / Merge / メトリクス収集などの参照実装を通じて、運用設計をコードとして検証できること
- 下流ソフトウェアが必要に応じて repo 非依存の自己改善ループを
  リリース後運用として導入できるよう、
  reflection / recall / skill evolution / user model の契約を提供すること
- sibling repository `../agent-protocols` の `Evidence` 契約へ接続し、
  LLM 実行の行動証跡を追跡できること
- CI / Security / Governance のテンプレートを派生リポジトリで再利用できること

## 2. 対象利用者

- Workflow Cookbook 自体を保守するメンテナ
- 派生リポジトリへテンプレートや運用ルールを導入する開発者
- Birdseye と Task Seed を使って作業する AI エージェント
- リリース判定、セキュリティレビュー、CI 運用を担う QA / Ops / Security 担当

## 3. スコープ

- 対象
  - `README.md` / `BLUEPRINT.md` / `RUNBOOK.md` / `EVALUATION.md` / `CHECKLISTS.md` / `HUB.codex.md` を中心とした運用ドキュメント群
  - `docs/birdseye/` と `tools/codemap/update.py` による Birdseye 生成・鮮度管理
  - `tools/autosave/` / `tools/merge/` / `tools/perf/` などの参照実装と検証コード
  - 下流ソフトウェア向けの自己改善ループ要件・仕様・設計
  - `agent-protocols` の `Evidence` schema へ変換する LLM 行動追跡ブリッジ
  - `.github/workflows/` と `governance/` による CI / Security / Governance テンプレート
- 非対象
  - 本番サービスのホスティングやデプロイ実体
  - 外部 SaaS の本番運用設定そのもの
  - Cookbook 外部リポジトリの個別要件定義

## 4. 機能要件

### 4.1 ドキュメントハブ

1. 主要ドキュメントは役割ごとに分離され、相互リンクで辿れること。
2. `README.md` は初動入口、`HUB.codex.md` はタスク分割入口、`RUNBOOK.md` は実行手順入口として機能すること。
3. `docs/requirements.md` / `docs/spec.md` / `docs/design.md` は互いに矛盾せず、レビュー時に参照可能であること。
4. `RUNBOOK.md` は完了済みの長い表や詳細証跡を蓄積せず、現在の運用判断、未解決事項、実行手順、参照リンクへ集中すること。
5. 完了済み事項の索引は `docs/completion-record.md` に分離し、正本となる task / acceptance / release 記録へリンクすること。
6. `workflow-cookbook` の `HUB.codex.md` は repo 内のワークフロー運用導線を扱い、Agent_tools 全体の repo 選定や横断ルーティングは `agent-tools-hub` の責務として重複させないこと。

### 4.2 Birdseye / Codemap

1. `docs/birdseye/index.json` はノード一覧・エッジ・カプセル参照を保持すること。
2. `docs/birdseye/caps/*.json` は各ノードの要約、依存、リスク、関連テストを保持すること。
3. `docs/birdseye/hot.json` は主要導線を即時参照できるホットリストとして維持されること。
4. `tools/codemap/update.py` は `--targets` / `--emit` / `--since` / `--radius` により Birdseye の局所更新と全体更新を制御できること。
5. Birdseye の `generated_at` は 5 桁ゼロ埋めの世代番号として一貫して扱われること。

### 4.3 参照実装

1. AutoSave 参照実装はロック検証、スナップショット単調性、ロールアウトガードを検証できること。
2. Merge 参照実装は `baseline` / `strict` の精度モードとロック協調を扱えること。
3. メトリクス収集系は構造化ログと Prometheus 由来データを統合できること。
4. `StructuredLogger` は任意のプラグインを介して `agent-protocols` の `Evidence`
   契約へ推論ログをミラーできること。
5. `Evidence` 連携は `extra.agent_protocol` コンテキストが無い場合に既存ログ出力を壊さず、
   コンテキストがある場合のみ schema 互換の証跡を生成できること。
6. plugin は import 文字列と options から構成でき、外部リポジトリが logger 本体を
   変更せずに差し込めること。
7. plugin は mapping または config file から宣言的に構成できること。
8. 参照実装は docs の契約と矛盾しない最小テストを伴うこと。
9. テスト実装工程は単体テスト・結合テスト・coverage 確認を標準とし、
   Python 系の既定 coverage 下限を 80% とすること。
10. 500 行を超える主要モジュール、または 20 関数を超えるモジュールは
    技術的負債として記録し、分割方針、互換維持方針、優先度を追跡できること。
11. 大きなモジュールを分割する場合、既存 import path と既存 CLI の互換 wrapper を
    維持し、動作変更を伴わない分割から始めること。
12. package metadata、README badge、CHANGELOG、release docs、git tag の
    version 表記が互いに矛盾しないこと。
13. 公開操作は `python tools/...` の既存入口に加え、下流 repo から利用しやすい
    package entrypoint として提供できること。

### 4.4 自己改善ループ

1. `workflow-cookbook` は `hermes-agent` と接続せず、
   そこから得た着想のみを一般化した独自要件として自己改善ループを扱うこと。
2. 自己改善ループは `workflow-cookbook` 自身の内部最適化ではなく、
   下流ソフトウェアが利用する外向き機能として設計されること。
3. 自己改善ループは任意機能であり、未導入の下流ソフトウェアでも
   基本要件の充足を阻害しないこと。
4. 自己改善ループは原則としてリリース後運用で有効化する機能とし、
   実装中・作成途中の通常フローを妨げないこと。
5. 作成途中の変更に対して、reflection / recall / nudges / skill draft 生成を
   必須ゲートとして要求しないこと。
6. 最低限、次の機能契約を提供すること。
   - session reflection
   - memory curation
   - periodic nudges
   - skill evolution
   - cross-session recall
   - user / workspace model
7. これらの機能は repo 非依存の DTO または schema に落とし込めること。
8. 自己改善候補は review 可能な draft 状態を持ち、
   acceptance / evidence と参照関係を残せること。
9. skill 改善や user model 更新は review 無しで即時公開しないこと。
10. 他 repo へ適用できるよう、memory store、search backend、skill registry、
   scheduler は差し替え可能な前提で設計すること。

### 4.5 CI / Governance テンプレート

1. 再利用可能な Python / Security CI ワークフローを提供すること。
2. `governance/policy.yaml` は required jobs と自己改変境界の基準点として機能すること。
3. CI 段階導入の方針は `docs/ci_phased_rollout_requirements.md` と workflow 群で追跡できること。
4. `docs/CONTRACTS.md` に定義された `.ga/qa-metrics.json` と `governance/predictor.yaml`
   の feature detection 契約が維持されること。
5. docs gate は checker ごとに warning / failure の既定値と昇格条件を
   `docs/ci-config.md` または policy から追跡できること。
6. warning から failure へ昇格する checker は、対象範囲、猶予期間、
   既存例外の扱い、rollback 条件を事前に文書化すること。
7. version consistency check は release 前 gate として実行できること。
8. package entrypoint を追加する場合は、既存 script 入口と同じ検証経路で
   CI から疎通確認できること。

### 4.6 Security baseline

1. `.github/dependabot.yml` を維持し、GitHub Actions 依存更新を週次で確認できること。
2. `docs/security/SAC.md` と `docs/security/Security_Review_Checklist.md` を正本として、
   SAST / Secrets / 依存 / Container の 4 種ゲートとレビュー導線が追跡できること。
3. GitHub repository settings では、少なくとも vulnerability alerts、
   Dependabot security updates、secret scanning、push protection を有効化した状態を維持すること。
4. リリース前に security posture を確認し、結果を release 証跡または acceptance 記録から参照できること。

### 4.7 Completion governance

1. `docs/tasks/*.md` が `status: done` になる場合、対応する acceptance record または例外理由を参照できること。
2. `docs/completion-record.md` に記載する完了事項は、単独で完了を主張せず、必ず `docs/tasks/*.md`、
   `docs/acceptance/*.md`、`CHANGELOG.md`、`docs/releases/*.md` のいずれかへリンクすること。
3. Task Seed、Acceptance、Completion Record の関係は一方向に辿れること。
   - Task Seed: 何をなぜ行うか
   - Acceptance: どう検収したか
   - Completion Record: 完了済み事項をどこに集約したか
4. RUNBOOK slimming を維持するため、RUNBOOK に完了済み詳細表を追加する変更は review で検出できること。
5. RUNBOOK に残す完了済み情報は、現在の運用判断に必要な短い参照に限定すること。

### 4.8 Agent-tools-hub 境界

1. Agent_tools 全体の repo 選定、複数 repo の入口案内、既存 Skill へのルーティングは `agent-tools-hub` の責務とすること。
2. `workflow-cookbook` は Birdseye/Codemap、Task Seed、Acceptance、reusable CI、security/release evidence、
   cross-repo workflow plugin、自己改善ループ blueprint の実行手順と契約を扱うこと。
3. `workflow-cookbook` 側の文書は、他 repo の詳細な使い分け表や Agent_tools 全体の routing policy を複製しないこと。
4. `workflow-cookbook` から他 repo に触れる場合は、連携契約、plugin 設定、Evidence/Acceptance/Task の接続点に限定すること。

### 4.9 Plugin capability catalog

1. workflow plugin と inference plugin は、capability 名、必須 method、
   入力 DTO、出力 DTO、timeout / retry / error policy を一覧できること。
2. capability catalog は runtime 実装、schema、README、sample config と
   矛盾しないこと。
3. capability を追加、削除、rename する場合は、互換性影響、
   migration note、検証コマンドを `CHANGELOG.md` または関連 docs に記録すること。
4. capability catalog は手動表だけに依存せず、将来的に runtime から
   自動生成または照合できる形で維持すること。

### 4.10 Five-tool validation manifest

1. RanD、Code-to-gate、HATE、manual-bb-test-harness、QEG を使う五ツール検収では、
   1回の検収 run を `five-tool-run-manifest` として記録できること。
2. manifest は各ツール repo の path、branch、commit、upstream、dirty 状態を保持し、
   検収時点のコード状態を後から確認できること。
3. manifest は入力・出力 artifact の path、存在有無、bytes、SHA-256、schema version、
   verdict、trace id、policy hash を記録できること。
4. QEG の policy 正本は `qeg_policy_hash` として manifest に記録し、
   QEG artifact 内の `policyHash` と一致しなければならない。
5. QEG 以外の artifact が直接 `gate_policy` 正本を持つ場合は契約違反として検出すること。
   下流 artifact は必要に応じて `gate_policy_proposal` または QEG policyHash 参照へ寄せる。
6. QEG の `no_go` / `disqualified` evidence がある場合、manifest の final verdict が
   `go` のまま通過してはならない。
7. Code-to-gate または HATE の成果物を別 artifact に内包して参照する場合は、
   `degraded` 理由と影響を manifest に記録できること。
8. manifest schema、sample config、CLI、unit test、五ツール evidence pack は
   互いに矛盾しないこと。

## 5. 非機能要件

- 保守性
  - 変更は最小差分で行い、ドキュメントと実装が乖離しないこと。
  - 派生リポジトリが再利用しやすいよう、公開テンプレートは後方互換を重視すること。
  - technical debt register は単なる棚卸しではなく、分割優先度、期限、
    抑制理由、関連テストを含むこと。
- 観測性
  - AutoSave / Merge / QA メトリクスは、少なくとも構造化ログまたは集約 JSON から検証可能であること。
  - LLM 実行証跡は `agent-protocols` の `Evidence` として再利用可能な JSON で追跡できること。
- セキュリティ
  - 機密情報はコミットしないこと。
  - Security review 導線と workflow の関係が docs 上で追跡できること。
  - Dependabot / vulnerability alerts / secret scanning の恒常運用が維持されること。
- 可読性
  - エージェントが最小コンテキストで読めるよう、Birdseye とハブ文書を維持すること。
  - 自己改善ループの契約は特定エージェント実装に依存せず、他 repo へ移植可能な粒度で記述すること。
  - RUNBOOK は肥大化を避け、完了済み履歴を索引ドキュメントへ分離すること。

## 6. 受入条件

- Birdseye / Codemap の更新手順が `README.md`、`docs/BIRDSEYE.md`、`docs/birdseye/README.md`、`GUARDRAILS.md`、`RUNBOOK.md` で整合していること。
- `docs/requirements.md`、`docs/spec.md`、`docs/design.md`、`docs/CONTRACTS.md` がこの repo の実態に即した内容になっていること。
- `docs/CONTRACTS.md` に定義された `.ga/qa-metrics.json` と `governance/predictor.yaml` の feature detection 契約が維持されていること。
- `agent-protocols` の `Evidence` 契約に必要な最小フィールド
  （`taskSeedId`、`baseCommit`、`headCommit`、`actor`、`model`、`tools` など）が
  LLM 行動追跡ログから再構成できること。
- PR 本文に Priority Score（値と根拠）が記録されていること。
- `governance/policy.yaml` の `forbidden_paths` を無断で変更しないこと。
- インシデント発生時は `docs/IN-YYYYMMDD-XXX.md` を作成し、該当 PR および `RUNBOOK.md` から相互リンクすること。
- `docs/completion-record.md` に記載された完了項目が、task / acceptance / release / changelog の正本リンクを持つこと。
- `RUNBOOK.md` に完了済み詳細表を追加する場合、`docs/completion-record.md` への分離要否が review で判断されていること。
- `workflow-cookbook` の repo 内 HUB が `agent-tools-hub` の横断 repo routing を複製していないこと。
- README badge、`pyproject.toml`、`CHANGELOG.md`、`docs/releases/*.md`、
  git tag の version が release 時点で整合していること。
- package entrypoint を追加した場合、既存 `python tools/...` 入口と同等の
  smoke test が存在すること。
- plugin capability catalog が runtime / schema / sample config と矛盾していないこと。
- docs gate checker の warning / failure 昇格条件が文書化されていること。
- large module の抑制または分割計画が `TECH_DEBT_REGISTER.md` から追跡できること。
- five-tool run manifest が五ツール repo の commit、artifact hash、QEG policyHash、
  final verdict を束ね、cross-repo contract 違反を検出できること。
- 五ツール検収では Code-to-gate / HATE の `blocked` / `hold` を下流 Gate へ伝播し、
  pytest 単体の成功だけで release approval を出さないこと。
- console script smoke test は、`pip` が無い隔離 Python 環境でも実行できること。
- 最低限の回帰確認として、次のテストが通ること。
  - `tests/test_codemap_update.py`
  - `tests/autosave/test_project_lock_service.py`
  - `tests/merge/test_precision_mode_pipeline.py`
  - `tests/test_collect_metrics_cli.py`
  - `tests/perf/test_collect_metrics_autosave_merge.py`
  - `tests/test_structured_logger.py`
  - `tests/test_agent_protocol_evidence.py`
  - `tests/test_plugin_loader.py`
  - `tests/test_plugin_config.py`
- Python CI では `pytest --cov=. --cov-report=term-missing --cov-fail-under=80`
  を基準として coverage を確認すること。
- `CHANGELOG.md` の `[Unreleased]` に今回の差分が追記されていること。

## 7. トレーサビリティ

- 要件の入口: `docs/requirements.md`
- 仕様の入口: `docs/spec.md`
- 設計の入口: `docs/design.md`
- 自己改善ループ補助設計: `docs/addenda/O_Adaptive_Improvement_Loop.md`
- 外部契約: `docs/CONTRACTS.md`
- 実行手順: `RUNBOOK.md`
- 受入基準: `EVALUATION.md`
- 鮮度管理と最小読込: `GUARDRAILS.md` / `docs/BIRDSEYE.md`
- タスク分割: `HUB.codex.md` / `TASK.codex.md`
