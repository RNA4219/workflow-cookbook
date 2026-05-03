---
intent_id: INT-001
owner: your-handle
status: active   # draft|active|deprecated
last_reviewed_at: 2025-10-14
next_review_due: 2025-11-14
---

# Changelog

## 1.2.0 - 2026-05-03

### Added

- Version consistency checker (INT-IMPROVEMENT-006 step 1):
  - `tools/ci/check_version_consistency.py`: 11 tests
  - RG-007 added to docs/ci-config.md
- Stable CLI entrypoints (INT-IMPROVEMENT-006 step 2):
  - `pyproject.toml`: `[project.scripts]` (wfc-governance-gate, wfc-collect-metrics, wfc-codemap-update, wfc-context-pack)
  - 6 smoke tests
- Docs gate escalation policy (INT-IMPROVEMENT-006 step 3):
  - `governance/policy.yaml`: `ci.checker_stages` (RG-002: enforce, RG-003-007: warn)
  - Escalation Policy section in docs/ci-config.md
  - 4 new tests
- Plugin capability catalog (INT-IMPROVEMENT-006 step 4):
  - `schemas/plugin-capability-catalog.schema.json`
  - `examples/plugin-capability-catalog.sample.json`: 5 capabilities
  - 7 tests

### Changed

- Tech debt resolution: split large modules
  - `tools/perf/collect_metrics/types.py` (572 lines) → rules.py, helpers.py, extractor.py
  - `tools/ci/check_governance_gate.py` (503 lines) → governance_gate/ package
  - docstrings added to evidence_bridge.py, allowlist_guard.py
- pyproject.toml version fixed (0.1.0 → 1.1.3)
- Birdseye freshness threshold: 365 days → 90 days
- Self-improvement loop: nudge checker, propagation checker, sample configs
- docs/releases/v1.2.0.md removed (no git tag) → CHANGELOG section removed

## Unreleased

### Added

- RUNBOOK slimming、Task/Acceptance/Completion trace、agent-tools-hub 境界の要件・仕様を追加。
- 先行改善案として、version consistency、stable CLI entrypoint、docs gate escalation、
  plugin capability catalog、large module split policy を要件・仕様・Blueprint へ追加。
- `tools/ci/check_runbook_slimming.py`: RUNBOOK 完了済み表肥大化検出 checker 追加
- `tools/ci/check_completion_trace.py`: Task/Acceptance/Completion トレース検証 checker 追加
- `tools/ci/check_agent_tools_hub_boundary.py`: agent-tools-hub 境界 checker 追加
- 対応する unit test (`tests/test_check_runbook_slimming.py`, `tests/test_check_completion_trace.py`,
  `tests/test_check_agent_tools_hub_boundary.py`) 追加
- `.github/workflows/markdown.yml`: job 分割 (`docs-gate`, `metrics-gate`, `lint`) + 3 checker 統合 (RG-003, RG-004, RG-005)
- `governance/policy.yaml`: `docs-gate` を required_jobs に追加
- `docs/ci-config.md`: docs-gate, metrics-gate, RG-002〜RG-05 対応表追加
- `tools/ci/check_ci_gate_matrix.py`: docs-gate mapping 追加
- 自己改善ループ schema 定義追加 (spec.md 4.6.2-4.6.6 対応):
  - `schemas/reflection-summary.schema.json`: ReflectionSummary DTO
  - `schemas/skill-draft-record.schema.json`: SkillDraftRecord DTO
  - `schemas/recall-response.schema.json`: RecallResponse DTO
  - `schemas/user-model-snapshot.schema.json`: UserModelSnapshot DTO
  - `schemas/workspace-model-snapshot.schema.json`: WorkspaceModelSnapshot DTO
  - `schemas/periodic-nudge.schema.json`: PeriodicNudge DTO
- `examples/reflection-summary.sample.json`: ReflectionSummary sample config
- `examples/skill-draft-record.sample.json`: SkillDraftRecord sample config
- `tests/test_self_improvement_schemas.py`: schema validation test 追加
- RUNBOOK Operational Readiness Backlog: 4項目の完了状態反映
  - Branch Protection: 完了 (2026-04-17)
  - Release drill: 完了 (2026-04-19)
  - Supply chain: 完了 (2026-04-17)
  - Dependency exceptions: 整備済み
- Birdseye Freshness しきい値 365→90日移行
  - `.github/workflows/markdown.yml`: RG-002しきい値変更
  - `RUNBOOK.md`: 段階計画更新（現行90日、最終30日）
- 自己改善ループ nudge checker 実装:
  - `tools/ci/check_stale_self_improvement.py`: stale reflection/skill draft検出
  - `tests/test_check_stale_self_improvement.py`: nudge checker test (12 tests)
- Task Seed完了propagation checker実装 (SKILL-DRAFT-002):
  - `tools/ci/check_task_completion_propagation.py`: done Task Seedのcompletion-record未反映検出
  - `tests/test_check_task_completion_propagation.py`: propagation checker test (13 tests)
  - `.github/workflows/markdown.yml`: RG-006 task completion propagation gate追加
  - `docs/ci-config.md`: RG-006 checker対応表追加
- 自己改善ループ sample config追加:
  - `examples/user-model-snapshot.sample.json`
  - `examples/workspace-model-snapshot.sample.json`
  - `examples/recall-response.sample.json`
  - `examples/periodic-nudge.sample.json`
- 自己改善ループ nudge checker 実装:
  - `tools/ci/check_stale_self_improvement.py`: stale reflection/skill_draft検出
  - `tests/test_check_stale_self_improvement.py`: nudge checker test (12 tests)
- `.workflow-cache/reflections/SESSION-20260502-002.json`: セッションreflection記録
- `.workflow-cache/skill-drafts/SKILL-DRAFT-002.json`: Task Seed完了自動反映skill draft
- 改善仕様拡充の検収 record `AC-20260503-04` と次実装用 Task Seed
  `task-next-improvement-implementation-20260503` を追加し、RUNBOOK に結果参照を記録。

### Changed

- `docs/tasks/task-release-evidence-operational-drill-20260417.md`: 例外理由追記
- `docs/tasks/task-supply-chain-reproducibility-followup-20260417.md`: 例外理由追記
- `HUB.codex.md`: Evidence 手順参照追加

## 1.1.3 - 2026-04-11

### Added

- Product Readiness P1/P2 完了: acceptance index release mapping、横展開、定期運用導線
- `tools/ci/check_docs_review_due.py`: docs review overdue checker 追加
- `docs/acceptance/INDEX.md`: Release Mapping section 追加

### Changed

- `tools/ci/generate_acceptance_index_standalone.py`: release linkage 拡張

## 1.1.2 - 2026-04-09

### v1.1.2 Highlights

- Security Gate workflow improvements: Bandit skip list expanded, Gitleaks configuration fixed, safety check command updated
- README updates: Connected repositories section added
- CI fixes: Shell injection prevention, markdown quality improvements

### Changed

- 0080: Security Gate workflow修正
  - Bandit: testsディレクトリをscan対象から除外、skipリスト拡張 (B101, B105, B106, B107, B108, B310, B404, B603, B607)
  - Gitleaks: `fetch-depth: 0`追加で完全なgit履歴取得
  - Semgrep: shell injection防止 (env変数使用)
  - Safety: `--full-report`にコマンド修正
  - security-posture job削除
  - reusable workflowをinline化
- 0079: CI/CD修正
  - YAML flow mapping → block mapping変換
  - e2e job条件分岐ロジック修正
  - printf format文字列問題回避
  - memx-resolverテストに`PYTHONPATH=.`設定追加
  - permissions/concurrency設定の構造修正
- 0078: README更新
  - 連携リポジトリセクション追加: agent-taskstate, memx-resolverプラグイン記載
  - 多言語版README整理・再構成
- 0077: テスト・品質向上
  - 12テストファイル追加 → coverage 87.75%
  - Markdownlint修正 (MD022, MD024, MD031, MD032, MD040, MD047)
  - CodeQL code scanning alerts修正

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

- 0006: Canary 連携は任意
