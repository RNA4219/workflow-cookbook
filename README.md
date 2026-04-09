---
intent_id: INT-001
owner: your-handle
status: active   # draft|active|deprecated
last_reviewed_at: 2025-10-14
next_review_due: 2025-11-14
---

# Workflow Cookbook / Codex Task Kit

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repo defines QA/Governance-first workflows, reusable operational docs,
and Codex-oriented task orchestration patterns rather than application code.
It also includes Birdseye / Codemap maintenance flows, Task Seed driven
handoffs, and plugin-based LLM behavior tracking that can emit
`agent-protocols` compatible `Evidence` logs.

<!-- LLM-BOOTSTRAP v1 -->
読む順番:

1. docs/birdseye/index.json  …… ノード一覧・隣接関係（軽量）
2. docs/birdseye/caps/`<path>`.json …… 必要ノードだけ point read（個別カプセル）

フォーカス手順:

- 直近変更ファイル±2hopのノードIDを index.json から取得
- 対応する caps/*.json のみ読み込み

<!-- /LLM-BOOTSTRAP -->

任意のリポジトリに貼って入力時に本リポジトリを参照させるだけで、
**仕様→実装→検収**まで一貫して回せるコンテキストエンジニアリングが
可能な MD 群。

- 人間にもエージェント（Codex等）にも読ませやすい最小フォーマット
- 言語・技術スタック非依存（存在するコマンドだけ使用）

## このリポジトリで扱うもの

- Birdseye / Codemap による Markdown ハブと依存関係の軽量同期
- Task Seed、Runbook、Evaluation、Checklist を中心にした実行導線
- `StructuredLogger` を起点にした plugin 型の LLM 行動追跡
- `agent-protocols` の `Evidence` 契約へ接続できる設定例と consumer sample
- reusable CI / Governance workflow と検証スクリプト

## クイックナビ

- まず全体像をつかむ:
  [`BLUEPRINT.md`](BLUEPRINT.md) /
  [`RUNBOOK.md`](RUNBOOK.md) /
  [`EVALUATION.md`](EVALUATION.md)
- Birdseye を使う:
  [`docs/BIRDSEYE.md`](docs/BIRDSEYE.md) /
  [`tools/codemap/README.md`](tools/codemap/README.md) /
  [`HUB.codex.md`](HUB.codex.md)
- LLM 行動追跡を使う:
  [`tools/protocols/README.md`](tools/protocols/README.md) /
  [`examples/inference_plugins.agent_protocol.sample.json`](examples/inference_plugins.agent_protocol.sample.json) /
  [`examples/agent_protocol_evidence_consumer.sample.py`](examples/agent_protocol_evidence_consumer.sample.py)
- CI / Governance を使う:
  [`CHECKLISTS.md`](CHECKLISTS.md) /
  [`docs/ci-config.md`](docs/ci-config.md) /
  [`docs/ci_phased_rollout_requirements.md`](docs/ci_phased_rollout_requirements.md)

## 使い方（最短）

1. リポジトリ直下に本ファイルをコピー
   - フォークで運用する場合は [`docs/FORK_NOTES.md`](docs/FORK_NOTES.md) をテンプレ通りに初期化し、差分管理を開始
2. `BLUEPRINT.md` で要件と制約を1ページに集約
3. `RUNBOOK.md` で実行手順、`EVALUATION.md` で受入基準を固定
4. タスクごとに `TASK.codex.md` を複製して内容を埋め、エージェントに渡す
5. リリース時は `CHECKLISTS.md` と `CHANGELOG.md` を同期する

## 主要導線

### Birdseye / Codemap

- [`docs/addenda/A_Glossary.md`](docs/addenda/A_Glossary.md)
  …… Intent や Task Seed など頻出用語の定義
- [`GUARDRAILS.md`](GUARDRAILS.md)
  …… 行動指針と Birdseye の最小読込ガードレール
- [`docs/BIRDSEYE.md`](docs/BIRDSEYE.md)
  …… Edges / Hot / 更新手順の説明
- [`tools/codemap/README.md`](tools/codemap/README.md)
  …… Birdseye カプセル再生成と `codemap.update` の使い方
- [`HUB.codex.md`](HUB.codex.md)
  …… 仕様集約とタスク分割のハブ
- [`docs/IN-20250115-001.md`](docs/IN-20250115-001.md)
  …… インシデントログと `deps_out` 照合例

```sh
# 例: main との差分から対象カプセルを自動抽出
python tools/codemap/update.py --since --emit index+caps

# 例: 1 hop に絞って局所更新
python tools/codemap/update.py --since --radius 1 --emit caps

# 例: Birdseye リソースを明示的に指定（従来挙動）
python tools/codemap/update.py --targets docs/birdseye/index.json,docs/birdseye/hot.json --emit index+caps
```

`docs/birdseye/hot.json` が欠落している場合はコマンド実行時に
`FileNotFoundError` が発生し、再生成コマンドがメッセージで提示されます。

### LLM 行動追跡

- [`tools/protocols/README.md`](tools/protocols/README.md)
  …… plugin API、import 文字列 loader、config ベース導入手順
- [`examples/inference_plugins.agent_protocol.sample.json`](examples/inference_plugins.agent_protocol.sample.json)
  …… `StructuredLogger.from_plugin_config(...)` 用の最小サンプル
- [`examples/agent_protocol_evidence_consumer.sample.py`](examples/agent_protocol_evidence_consumer.sample.py)
  …… `Evidence` JSON Lines の受け側たたき台

### タスク運用とリリース

- [`examples/TASK.sample.md`](examples/TASK.sample.md)
  …… `TASK.codex.md` のダミーサンプル
- 完了した `TASK.*` の成果は `[Unreleased](CHANGELOG.md#unreleased)` へ
  通番付きで転記し、Task Seed から成果差分へのリンクを貼る
- リリース時は `CHECKLISTS.md` をなぞり、`CHANGELOG.md` と
  リリースノートへ成果を昇格させる

利用時は `RUNBOOK.md` と `GUARDRAILS.md` を確認し、運用上の前提と制約を
そろえてください。

## 参照ガイド

### 最小導入セット

- [`BLUEPRINT.md`](BLUEPRINT.md) …… Intent と仕様全体の骨子を提示
- [`RUNBOOK.md`](RUNBOOK.md) …… 実装および運用手順を逐次記載
- [`EVALUATION.md`](EVALUATION.md) …… 受入基準と検証観点を定義
- [`GUARDRAILS.md`](GUARDRAILS.md) …… 行動指針と Birdseye 連携の制約を明示
- [`HUB.codex.md`](HUB.codex.md) …… タスク分割と依存グラフの中核ハブ
- [`CHECKLISTS.md`](CHECKLISTS.md) …… リリースとレビューフローのチェックリスト
- [`governance-gate.yml`](.github/workflows/governance-gate.yml)
  …… Intent 検証 CI を常時有効化
- [`docs/security/SAC.md`](docs/security/SAC.md)
  …… セキュリティ非機能要件を契約として明文化
- [`docs/security/Security_Review_Checklist.md`](docs/security/Security_Review_Checklist.md)
  …… セキュリティ審査手順
- [`reusable/python-ci.yml`](.github/workflows/reusable/python-ci.yml) /
  [`reusable/security-ci.yml`](.github/workflows/reusable/security-ci.yml)
  …… 他リポから `workflow_call` で利用できる最小 CI セット

Intent ゲートは
[`tools/ci/check_governance_gate.py`](tools/ci/check_governance_gate.py) により
自動適用されるため、CI の設定だけで運用に組み込めます。

### 変更履歴の更新ルール {#changelog-update-rules}

- **更新タイミング**:
  リリース判定が `CHECKLISTS.md` の [Release](CHECKLISTS.md#release) を通過し、レビュー承認で確定した直後に [`CHANGELOG.md`](CHANGELOG.md)
  を更新する。承認前に書き始めないことで、記録の正確性と監査性を確保する。
- **記載形式**:
  [`CHANGELOG.md`](CHANGELOG.md) ではセマンティックバージョニングに従い、`## x.y.z - YYYY-MM-DD` の見出し配下へ `### Added`・
  `### Changed` などのカテゴリ小見出しを用いて差分を整理する。最新リリースを先頭に追記し、既存節の体裁を崩さない。
  各箇条書きの先頭へ 4 桁ゼロ埋めの通番（例: `0001`）を付与し、既存項目の最大値に 1 を加えて採番する。
- **突合手順**:
  1. `CHECKLISTS.md` の [Release](CHECKLISTS.md#release) を順に確認し、完了済みチェック項目と未了項目を照合する。
  2. チェックリストに記録した内容を [`CHANGELOG.md`](CHANGELOG.md) の該当リリース節へ反映し、必要に応じて `RUNBOOK.md` や関連資料の更新有無をメモする。
  3. 反映後に再度チェックリストへ戻り、記録済みであることをコメントまたは添付リンクで明示してリリース完了とする。

詳細な手順は [`docs/addenda/M_Versioning_Release.md`](docs/addenda/M_Versioning_Release.md) を参照する。

SRC の主要言語に応じて、以下の CI テストセットを組み合わせると、
導入直後から最低限の品質・安全・可搬性が確保できます。

### 再利用CIの呼び出し例（下流リポ側）

```yaml
name: example CI
on: [push, pull_request]
jobs:
  python:
    uses: RNA4219/workflow-cookbook/.github/workflows/reusable/python-ci.yml@v0.1
    with:
      python-version: '3.11'
  security:
    uses: RNA4219/workflow-cookbook/.github/workflows/reusable/security-ci.yml@v0.1
    with:
      python-version: '3.11'
    secrets: inherit
  governance:
    uses: RNA4219/workflow-cookbook/.github/workflows/governance-gate.yml@v0.1
```

#### 言語別 CI テストセット（鉄板構成）

<!-- markdownlint-disable MD013 -->

| 言語 | カテゴリ | コマンド | 目的 |
| :--- | :--- | :--- | :--- |
| Rust | Format | `cargo fmt --all -- --check` | コード整形確認 |
| Rust | Lint | `cargo clippy --all-targets --all-features -D warnings` | 警告・アンチパターン検出 |
| Rust | Test | `cargo test --all-features` | 単体・統合テスト |
| Rust | Build | `cargo build --release` | リリースビルド確認 |
| Rust | Security | `cargo audit` / `cargo deny check` | 依存脆弱性・ライセンス確認 |
| Rust | Docs | `cargo doc -D warnings` | ドキュメント構文確認 |
| Rust (任意) | Coverage | `cargo llvm-cov` | カバレッジ測定 |
| Python | Format | `black --check .` | コード整形確認 |
| Python | Lint | `ruff check .` / `flake8 .` / `pylint src/` | コード品質 |
| Python | Typing | `mypy src/` | 型整合性 |
| Python | Test | `pytest --maxfail=1 --disable-warnings -q` | 単体・統合テスト |
| Python | Security | `bandit -r src/` / `pip-audit` | 静的セキュリティ解析 |
| Python | Coverage | `pytest --cov=src` | カバレッジ |
| Python | Docs | `pydocstyle src/` | docstring 構文確認 |
| Node.js/TS | Lint | `eslint .` / `tsc --noEmit` | コード・型検査 |
| Node.js/TS | Format | `prettier --check .` | 整形検証 |
| Node.js/TS | Test | `npm test` / `vitest run` / `jest --ci` | ユニットテスト |
| Node.js/TS | Build | `npm run build` | 本番ビルド |
| Node.js/TS | Security | `npm audit --audit-level=moderate` | 依存脆弱性 |

> CI 設定全体と最新実行のみを保持する自動キャンセル構成は、[`docs/ci-config.md`](docs/ci-config.md) を参照してください。
| Node.js/TS (任意) | Coverage | `npm run coverage` | カバレッジ |
| Node.js/TS | Docs | `typedoc` / `markdownlint` | API/MD構文確認 |
| Go | Format | `gofmt -l .` | 整形確認 |
| Go | Lint | `golangci-lint run` | 静的解析 |
| Go | Vet | `go vet ./...` | 構文・型安全検査 |
| Go | Test | `go test -v ./...` | 単体テスト |
| Go | Build | `go build ./...` | ビルド保証 |
| Go | Security | `gosec ./...` | 脆弱性スキャン |
| Go (任意) | Coverage | `go test -cover ./...` | カバレッジ測定 |
| Java | Format | `mvn formatter:validate` / `spotless:check` | 整形確認 |
| Java | Lint | `mvn checkstyle:check` / `spotbugs:check` | 静的解析 |
| Java | Test | `mvn test` / `gradle test` | 単体テスト |
| Java | Coverage | `mvn jacoco:report` | カバレッジ |
| Java | Security | `mvn dependency-check:check` | 依存脆弱性 |
| Java | Build | `mvn package` / `gradle build` | 本番ビルド |
| C/C++ | Format | `clang-format --dry-run -Werror` | 整形確認 |
| C/C++ | Lint | `cppcheck --enable=all` | 静的解析 |
| C/C++ | Build | `cmake . && make` / `meson compile` | ビルド確認 |
| C/C++ | Test | `ctest --output-on-failure` / `gtest` | 単体テスト |
| C/C++ | Security | `clang --analyze` / `cppcheck --addon=cert` | セキュリティ |
| C/C++ | Coverage | `lcov` / `gcov` | カバレッジ測定 |

#### 共通モジュール4種（全リポ共通）

| 領域/モジュール | 目的・役割 | 主要仕様書 | 備考 |
| :--- | :--- | :--- | :--- |
| CodeQL | 静的解析・脆弱性検出を CI に組み込む | [docs/spec.md](docs/spec.md) / [docs/ci-config.md](docs/ci-config.md) | `github/codeql-action` ワークフローで SAST ゲートを維持 |
| Dependabot | 依存更新の自動 PR を定期化する | [docs/requirements.md](docs/requirements.md) / [docs/spec.md](docs/spec.md) | 週次スケジュールで依存差分を検知し CI と連動 |
| Pre-commit Hooks | Lint / Format をローカルで再現する | [docs/design.md](docs/design.md) / [docs/requirements.md](docs/requirements.md) | `.pre-commit-config.yaml` でチーム基準を固定 |
| Artifact Upload | テスト結果やログを共有する | [docs/ci-config.md](docs/ci-config.md) / [EVALUATION.md](EVALUATION.md#acceptance-criteria) | CI 実行痕跡をアーカイブしてレビューへ提示 |
| `examples/` | レシピ参照実装と設計・仕様の整合を確認する | [docs/design.md](docs/design.md) / [docs/spec.md](docs/spec.md) | サンプル更新時は Birdseye 同期[^birdseye] |
| `styles/` | QA ルールによる表記統一・禁止用語を管理する | [docs/design.md](docs/design.md) / [docs/requirements.md](docs/requirements.md) | `styles/qa/QA.yml` の用語ルールを適用[^styles] |
| `tools/` | ドキュメント同期と検証スクリプトを運用する | [docs/design.md](docs/design.md) / [RUNBOOK.md](RUNBOOK.md#execute) | `tools/codemap/update.py` で Birdseye を再生成[^birdseye] |
| `docs/security/` | セキュリティレビューと SAC 手順を集約する | [docs/security/Security_Review_Checklist.md](docs/security/Security_Review_Checklist.md) / [docs/security/SAC.md](docs/security/SAC.md) | リリース審査の証跡を更新 |

[^birdseye]: `python tools/codemap/update.py --since --emit index+caps` で Birdseye インデックスとカプセルを更新し、必要に応じて `--radius` で hop 数を調整し、`--targets docs/birdseye/index.json,docs/birdseye/hot.json` などを併用して生成対象を明示する。`--emit` の指定により出力形式を切り替える。詳細は [tools/codemap/README.md](tools/codemap/README.md#実行手順) と [GUARDRAILS.md](GUARDRAILS.md#鮮度管理staleness-handling) を参照。
[^styles]: `styles/qa/QA.yml` の禁止用語・表記揺れルールをレビューで適用し、検知結果を `CHECKLISTS.md` のリリース手順へ反映する。

![lint](https://github.com/RNA4219/workflow-cookbook/actions/workflows/markdown.yml/badge.svg)
![links](https://github.com/RNA4219/workflow-cookbook/actions/workflows/links.yml/badge.svg)
![lead_time_p95_hours](https://img.shields.io/badge/lead__time__p95__hours-24h-blue)
![mttr_p95_minutes](https://img.shields.io/badge/mttr__p95__minutes-30m-blue)
![change_failure_rate_max](https://img.shields.io/badge/change__failure__rate__max-0.20-blue)
<!-- markdownlint-enable MD013 -->

> バッジ値は `governance/policy.yaml` の `slo` と同期。更新時は同ファイルの値を修正し、上記3つのバッジ表示を揃える。

## License

MIT. Unless noted otherwise, files copied from this repo into other projects
remain under the MIT License.

### Commit message guide

- feat: 〜 を追加
- fix: 〜 を修正
- chore/docs: 〜 を整備
- semver:major/minor/patch ラベルでリリース自動分類

### Pull Request checklist (CI 必須項目)

- PR 本文に `Intent: INT-xxx`（例: `Intent: INT-123`）を含めること。
- `EVALUATION` 見出し（例:
  `[Acceptance Criteria](EVALUATION.md#acceptance-criteria)`）へのリンクを本文に
  明示すること。
- 可能であれば [`Priority Score`](docs/addenda/A_Glossary.md#priority-score):
  `<number>` を追記し、`governance/prioritization.yaml` の重み付けを参照する。
- ローカルでゲートを確認する場合は `PR_BODY` に PR 本文を渡してから
  `python tools/ci/check_governance_gate.py` を実行する。
  （任意のディレクトリからは `python /path/to/workflow-cookbook/tools/ci/check_governance_gate.py`
  のように実行しても同様に検証できる。）

  ```sh
  PR_BODY=$(cat <<'EOF'
  Intent: INT-123
  ## EVALUATION
  - [Acceptance Criteria](EVALUATION.md#acceptance-criteria)
  Priority Score: 1
  EOF
  ) python tools/ci/check_governance_gate.py
  ```
