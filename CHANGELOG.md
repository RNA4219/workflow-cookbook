---
intent_id: INT-001
owner: your-handle
status: active   # draft|active|deprecated
last_reviewed_at: 2025-10-14
next_review_due: 2025-11-14
---

# Changelog

## [Unreleased]

## 1.2.0 - 2026-04-10

### Added

- 0076: Improvement Backlog残り実装完了
  - IB-007: Docs Resolve PR gate参照 (workflow追加)
  - IB-009: Sample/docs同期チェック
  - IB-010: Upstream差分抽出CLI
  - IB-012: TaskState export機能
  - IB-014: 用途別docs hub生成
  - IB-016: Security Docs更新チェック
  - IB-017: Release Notes自動生成
  - IB-018: Knowledge Reuse導線
- 0075: Improvement Backlog 実装完了
  - IB-003: Task/Acceptance 双方向整合チェッカー追加
  - IB-008: Acceptance Index 集計機能拡張（status summary 追加）
  - IB-011: Plugin timeout/error policy 実装
  - IB-013: Evidence レポート生成機能追加
  - IB-015: ADR/addenda 索引自動生成
  - 0005: SLO バッジ自動生成ツール追加

### Changed

- 0074: KPI 閾値判定 CLI と Birdseye freshness checker を追加し、
  `RUNBOOK.md` / `CHECKLISTS.md` / `README.md` / backlog docs に
  実行導線と完了状態を反映した
- 0073: 自己改善ループ blueprint を任意機能かつリリース後運用向けへ整理し、
  作成途中の通常フローを妨げない前提を要件・仕様・設計へ明記した
- 0072: `hermes-agent` とは接続せず、着想のみを一般化した独自機能として
  下流ソフトウェア向けの自己改善ループ要件・仕様・設計・境界定義を追加した
- 0070: security posture checker と release evidence checker を追加し、
  security 恒常対策と release 証跡チェックを CI / docs / checklist へ反映した
- 0071: `workflow-cookbook` / `agent-taskstate` / `memx-resolver` を横断する
  cross-repo integration workflow を追加した
- 0069: README 3 言語の冒頭説明に context engineering の位置付けを戻し、
  公開メタデータと repo 実態の整合を取り直した
- 0068: README 3 言語の冒頭説明とタイトルを `Codex` 前提から
  workflow operations 前提へ寄せ、公開向けの位置付けを整理した
- 0067: 既定 README を英語版へ切り替え、日本語版を `README.ja.md` として分離した
- 0066: `README.zh-CN.md` を追加し、日本語版・英語版 README から
  中国語版への切替導線を追加した
- 0065: `README.en.md` を追加し、日本語 README から英語版への切替導線を追加した
- 0064: `README.md` の入口を整理し、重複していた導線を圧縮して
  cross-repo plugin 導線を advanced 扱いへ寄せた
- 0063: 改善候補を `docs/addenda/N_Improvement_Backlog.md` として整理し、
  Security / Release / Acceptance / Cross-Repo plugin の次段 backlog を docs 正本へ追加した
- 0062: workflow plugin dispatcher を `invoke_first` / `invoke_all` へ整理し、
  sample config の統合テスト、renderer / policy 分離、plugin の dataclass 返却へ寄せた
- 0061: workflow plugin interface / capability error / config validate CLI を追加し、
  `agent-taskstate` と `memx-resolver` plugin に store abstraction と resolve cache を導入した
- 0060: `agent-taskstate` / `memx-resolver` 連携向け workflow plugin host、
  Task / Acceptance sync、docs resolve / ack / stale CLI を追加した
- 0058: `tools/ci/check_ci_gate_matrix.py` を追加し、
  `policy.yaml`・workflow・`docs/ci-config.md` の整合を CI で検証できるようにした
- 0059: `task-autosave-project-locks` を検収記録付きで `done` へ進め、
  PR の `Docs matrix` 明示選択を governance gate で検証するようにした
- 0057: テスト実装工程の標準として単体テスト・結合テスト・coverage 80%
  基準を requirements / spec / design / evaluation / runbook / task template /
  Python CI に反映した
- 0056: `docs/acceptance/` の検収記録テンプレートと
  `tools/ci/check_acceptance.py` を追加し、PR 本文からの参照と CI 検証を
  導入した
- 0055: `README.md` にクイックスタート節を追加し、最初の導線を
  3 ステップで辿れるようにした
- 0054: `workflow-agent-evidence` Skill に Claude 向け metadata を追加し、
  README の Skills 節から agent metadata へ辿れるようにした
- 0053: `README.md` を圧縮して導線を整理し、repo 同梱 Skill と
  protocol plugin の入口を明示した
- 0051: CodeQL workflow の解析対象を repo 実態に合わせて `python` のみに絞り、
  JavaScript source 不在による失敗を解消した
- 0050: `labels-sync` workflow のラベル同期 action を現行の
  `EndBug/label-sync@v2` へ置き換え、`README.md` を top-level front matter
  必須チェックの対象外にした
- 0052: `EndBug/label-sync@v2` の input 名を `token` と `config-file` に修正し、
  labels sync workflow の設定不整合を解消した
- 0049: GitHub Actions の Node 24 移行に合わせて
  `actions/checkout` を `v6`、`actions/setup-python` を `v6` へ更新した
- 0048: `README.md` 先頭の front matter を削除し、公開向けの入口として不要な
  管理メタデータを外した
- 0047: repo 内に残っていた絶対パス参照を相対パスへ置き換え、sample config、
  Runbook、Skill references、release docs の再利用性を改善した
- 0046: `workflow-cookbook/skills/workflow-agent-evidence/` を追加し、
  `agent-protocols` の Evidence 連携を扱う repo 同梱 Skill を配置した
- 0044: `CODE_OF_CONDUCT.md` と `SECURITY.md` を廃止し、README / Hub /
  Roadmap / Checklist の参照を `docs/security/` 基準へ整理した
- 0045: ライセンスを Apache-2.0 から MIT へ切り替え、`LICENSE` 本文、
  README 表記、配布チェック、SPDX ヘッダを MIT 前提に更新した
- 0043: `README.md` をクイックナビ・主要導線・参照ガイドの構成へ再整理し、
  Birdseye / LLM 行動追跡 / CI 導線を入口優先で読めるようにした
- 0042: `README.md` の導入説明を Birdseye / Task Seed /
  `agent-protocols` 連携前提へ更新し、公開用 GitHub Topics を
  リポジトリ実態に合わせて見直した
- 0041: `CHANGELOG.md`、`RUNBOOK.md`、`HUB.codex.md`、
  `docs/releases/v1.1.0.md` の Markdown 行長を調整し、
  `markdownlint-cli2` の `MD013` 指摘を解消した
- 0026: `tools/codemap/update.py` に `--radius` を追加し、Birdseye の局所更新 hop 数を制御できるようにした
- 0027: Birdseye / codemap の README・Guardrails・Runbook・Hub・Checklist を `--radius` と 5 桁世代番号の `generated_at` 前提に同期した
- 0028: AutoSave の `autosave.snapshot.commit` テレメトリに `latency_ms` と `lock_wait_ms` を含められるようにし、CI workflow に Phase コメントを追加した
- 0029: `docs/requirements.md` をこのリポジトリ実態に合わせて再定義し、Birdseye・参照実装・CI / Governance テンプレートまで含む要件へ更新した
- 0030: `docs/spec.md` と `docs/design.md` を repo 実態に合わせて再定義し、`docs/requirements.md` の受入条件に `docs/CONTRACTS.md` と最低限の回帰テスト群を明記した
- 0031: `EVALUATION.md` を新しい受入条件へ同期し、`docs/design.md` と `docs/ROADMAP_AND_SPECS.md` の周辺説明を feature detection と現行仕様に合わせて更新した
- 0032: `docs/requirements.md` の受入条件を `EVALUATION.md` と同期し、Birdseye 実生成物の再生成と `docs/ROADMAP_AND_SPECS.md` の現状説明更新を行った
- 0033: `docs/spec.md` を実運用に使える粒度まで詳細化し、`docs/interfaces.md` に codemap / autosave / merge / metrics の責務境界を反映した
- 0034: `docs/design.md` を実装寄りの設計書へ拡張し、コンポーネント責務・状態所有・主要データフロー・拡張ポイント・障害時の扱いを明文化した
- 0035: `docs/AUTOSAVE-DESIGN-IMPL.md` と `docs/MERGE-DESIGN-IMPL.md` を実コード準拠で再定義し、古い UI / CRDT / ロールバック前提を整理した
- 0036: `docs/IMPLEMENTATION-PLAN.md` と
  `docs/tasks/task-autosave-project-locks.md` を参照実装準拠へ更新し、
  段階導入・TDD・ロールバック条件を実コードと telemetry 契約に
  合わせて具体化した
- 0037: `docs/ci_phased_rollout_requirements.md` と `docs/ci-config.md` を
  実在する workflow / required jobs 構成へ再定義し、CI フェーズ文書を
  `policy.yaml` と同期した
- 0038: `RUNBOOK.md` と `CHECKLISTS.md` の CI 運用手順を
  required jobs / Phase 正本に同期し、branch protection との同時確認ルールを追加した
- 0039: `tools/ci/check_branch_protection.py` と対応テストを追加し、
  branch protection export と `policy.yaml` の論理 gate ID / 実 check 名対応を
  検証可能にした
- 0040: `agent-protocols` の `Evidence` 契約へ接続する LLM 行動追跡ブリッジを追加し、
  `StructuredLogger` から任意 plugin / import 文字列 / config file ベースで
  Evidence JSON Lines を出力できるようにし、参照設定サンプル・protocols README・
  plugin config schema を追加した
- 0007: `CHECKLISTS.md` に Development / Pull Request / Ops チェックリストを追加し、Release セクションが新設項目と重複しないよう参照構造へ更新
- 0008: 過去のブランド表現をワークフロー向けの共通名称・リンクへ差し替え、関連チェックリストとメトリクス定義の整合性を再確認
- 0009: `<旧ブランド名>` 参照を中立表現へ整理
- 0010: ブランド非依存の表現へ整理し、`CHANGELOG.md` と `CHECKLISTS.md` の記述を同期
- 0011: `docs/Release_Checklist.md` に上流同期確認のチェック項目を追加し、`docs/UPSTREAM*.md` とのリンクを明示
- 0012: `EVALUATION.md` の KPI 定義を `governance/metrics.yaml` と同期し、`RUNBOOK.md#Observability` と相互参照するリンクを追加
- 0013: `EVALUATION.md` の KPI を目的・収集方法・目標値付きの表形式へ整理し、`RUNBOOK.md#Observability` の手順と整合させた
- 0014: `docs/security/SAC.md` の対象範囲を運用UIカテゴリへ一般化し、特定UI名称への依存を排除
- 0015: `docs/security/SAC.md` の SAC 対象説明を運用系 Web UI として再定義し、特定製品前提を除去
- 0016: `RUNBOOK.md` に外部通信承認フローを追加し、`docs/addenda/G_Security_Privacy.md` の参照内容を同期
- 0017: `datasets/README.md` を新設し、データ保持レビューで参照するデータセット管理手順を整備
- 0018: セキュリティゲート CI（`.github/workflows/security.yml`）を追加し、SAST/Secrets/依存/Container 検証を直列実行で整備
- 0019: `docs/addenda/G_Security_Privacy.md` のセキュリティゲート参照を
  [`.github/workflows/security.yml`](../../.github/workflows/security.yml) へ更新
- 0020: `docs/addenda/G_Security_Privacy.md` から旧 `ci/security.yml` 参照を排除し、CI 手順のリンクを現行構成へ揃えた
- 0021: `security-ci.yml` の Bandit 対象を実在する Python ソースへ更新し、検出時に失敗するゲートへ修正
- 0022: `.github/workflows/reusable/security-ci.yml` の Bandit 監査範囲を `tests` へ拡張し、`safety` ステップの戻り値を尊重
- 0023: `reusable/security-ci.yml` の Bandit 対象ディレクトリを明示し、Secrets/依存診断の失敗を許容しないよう `safety` ゲートを修正
- 0024: `.github/workflows/reusable/security-ci.yml` の Safety 検出で GitHub Actions/`act` が失敗することを次の手順で確認
  1. 脆弱バージョンの依存（例: `pip install 'urllib3==1.25.8'`）を一時的に追加してブランチを作成する
  2. `act -j dep_audit -W .github/workflows/reusable/security-ci.yml --input python-version=3.11`
  3. `safety check --exit-code 1` ステップが非ゼロ終了し、ジョブ全体が失敗することを確認する（GitHub Actions 上でも同様）
- 0025: `.github/workflows/labels-sync.yml` の `concurrency` ブロックをガイド記載の配置へ整え、`.github/workflows/security.yml` と同一形式となることを確認

## 0.1.0 - 2025-10-13

- 0001: 初版（MD一式 / Codex対応テンプレ含む）

## 1.0.0 - 2025-10-16

### Added (v1.0.0)

- 0002: Stable Template API（主要MDの凍結）
- 0003: PR運用の明確化（Intent / EVALUATION リンク / semverラベル）
- 0004: CIワークフロー（links/prose/release）

## 1.1.1 - 2026-04-09

### v1.1.1 Highlights

- 0041: `CHANGELOG.md`、`RUNBOOK.md`、`HUB.codex.md`、
  `docs/releases/v1.1.0.md` の Markdown 行長を調整し、
  `markdownlint-cli2` の `MD013` 指摘を解消した

## 1.1.0 - 2026-04-09

### v1.1.0 Highlights

- 0040: `agent-protocols` の `Evidence` 契約へ接続する LLM 行動追跡ブリッジを追加し、
  `StructuredLogger` から任意 plugin / import 文字列 / config file ベースで
  Evidence JSON Lines を出力できるようにし、参照設定サンプル・protocols README・
  plugin config schema を追加した
- 0039: `tools/ci/check_branch_protection.py` と対応テストを追加し、
  branch protection export と `policy.yaml` の論理 gate ID / 実 check 名対応を
  検証可能にした
- 0038: `RUNBOOK.md` と `CHECKLISTS.md` の CI 運用手順を
  required jobs / Phase 正本に同期し、branch protection との同時確認ルールを追加した
- 0037: `docs/ci_phased_rollout_requirements.md` と `docs/ci-config.md` を
  実在する workflow / required jobs 構成へ再定義し、CI フェーズ文書を
  `policy.yaml` と同期した
- 0036: `docs/IMPLEMENTATION-PLAN.md` と
  `docs/tasks/task-autosave-project-locks.md` を参照実装準拠へ更新し、
  段階導入・TDD・ロールバック条件を実コードと telemetry 契約に
  合わせて具体化した
- 0035: `docs/AUTOSAVE-DESIGN-IMPL.md` と `docs/MERGE-DESIGN-IMPL.md` を
  実コード準拠で再定義し、古い UI / CRDT / ロールバック前提を整理した
- 0034: `docs/design.md` を実装寄りの設計書へ拡張し、
  コンポーネント責務・状態所有・主要データフロー・拡張ポイント・障害時の扱いを明文化した
- 0033: `docs/spec.md` を実運用に使える粒度まで詳細化し、
  `docs/interfaces.md` に codemap / autosave / merge / metrics の責務境界を反映した
- 0032: `docs/requirements.md` の受入条件を `EVALUATION.md` と同期し、
  Birdseye 実生成物の再生成と `docs/ROADMAP_AND_SPECS.md` の現状説明更新を行った
- 0031: `EVALUATION.md` を新しい受入条件へ同期し、`docs/design.md` と
  `docs/ROADMAP_AND_SPECS.md` の周辺説明を feature detection と現行仕様に合わせて更新した
- 0030: `docs/spec.md` と `docs/design.md` を repo 実態に合わせて再定義し、
  `docs/requirements.md` の受入条件に `docs/CONTRACTS.md` と最低限の回帰テスト群を明記した
- 0029: `docs/requirements.md` をこのリポジトリ実態に合わせて再定義し、
  Birdseye・参照実装・CI / Governance テンプレートまで含む要件へ更新した
- 0028: AutoSave の `autosave.snapshot.commit` テレメトリに `latency_ms` と
  `lock_wait_ms` を含められるようにし、CI workflow に Phase コメントを追加した
- 0027: Birdseye / codemap の README・Guardrails・Runbook・Hub・Checklist を
  `--radius` と 5 桁世代番号の `generated_at` 前提に同期した
- 0026: `tools/codemap/update.py` に `--radius` を追加し、
  Birdseye の局所更新 hop 数を制御できるようにした

### Known limitations

- 0005: SLOバッジ自動生成は未実装（README と policy.yaml を手動同期）
- 0006: Canary 連携は任意
