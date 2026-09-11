# 付録G: セキュリティ/プライバシー運用ガイド

> 対象: SACを採用した製品・サービスの運用担当者。各節は対象データ・通信・配布経路がある場合に適用する。
> この付録のパス、KMS、IaC、通知先は構成例。実環境の管理方式・担当・権限をRUNBOOKへ記録する。
> 以下の保持・削除・鍵更新期限を既に採用した運用では期限を維持し、変更は責任者の手続に従う。Gate観測期間とは独立に扱う。

## 1. キー管理

- **原則対応**: SAC-1, SAC-3, SAC-9。
- サーバ側Secretsは環境に合う秘密保管機構で管理する。KMSからの取得や `secrets.{env}.yaml` は構成例であり、平文の秘密をrepoへ保存しない。
  ローカル `.env` はテスト専用で、本番接続情報を含めない。
- APIキー更新は 90 日サイクルを上限とし、失効ログを `audit/security-key-rotation.log` に記録する。
  更新後は旧キーを 1h 内に失効させる。
- モデル・LLMエンドポイントの変更権限と承認対象を定める。IaCのレビュー付きマージは一例。管理UIを使う場合も同等の認可・監査と費用制御を必要とする。

## 2. ログマスキングと監査

- **原則対応**: SAC-1, SAC-2, SAC-8。
- 監査対象と識別に必要な最小情報を定め、Secretsは記録せずPIIはマスク／仮名化する。
  JSONLinesと `/var/log/workflow/audit.log` はこのツールの利用例。OS・保管基盤に応じたパス、アクセス権、保持契約を設定する。
- LLM応答をWeb表示するときは出力文脈に合うエスケープ／サニタイズを行い、別窓リンクは `rel="noopener"` を設定する。ログ保存時の秘匿化と表示時の処理を混同しない。
- `security@workflow-cookbook.example` は送信しないplaceholder。通知先・担当・送信権限を設定した運用だけで通知し、未設定／未送信を実施済みと記録しない。
- 改ざん検知方式を保管基盤に合わせて採用する。以下はHMAC-SHA256の署名チェーン例。日次鍵更新を採用する場合は鍵の識別・検証期間も管理する。
  署名検証は以下のコマンドで実行し、異常検知時は非ゼロ終了コードとともに原因を stderr に出力する。

  ```bash
  python tools/audit/verify_log_chain.py /var/log/workflow/audit.log \
    --hmac-key "$AUDIT_HMAC_KEY" \
    --initial-signature "$(cat /var/log/workflow/audit.seed)"
  ```

  `--secret` は既存ジョブ互換の alias として残すが、新規手順では
  `--hmac-key` を使用する。

  この監査をrelease条件に採用した場合はCIへ組み込み、検証失敗時はリリースを停止して調査する。実行対象パスと鍵の供給方式を実環境に合わせる。

## 3. データ保持と削除

- **原則対応**: SAC-6, SAC-7, SAC-8。
- 監査ログの保持期間は 180 日。期日を超過したファイルは
  `python tools/audit/purge_logs.py /var/log/workflow/ --older-than 180`
  を用いて削除し、削除後は
  [セキュリティ監査ログ削除レポート](../reports/security-retention.md)
  に実行結果を記録する。
- 学習や検証のためのデータセットはバージョン固定し、`datasets/README.md` にハッシュを記録する。
  - テンプレート: `データセット名` / `バージョン / タグ` /
    `取得元 (URL / リポジトリ)` / `ハッシュ値 (SHA256)` の4列を必須とし、取得時点で空欄なく記入する。
  - `記録テンプレート` と `記入例` を同ファイルに保持し、追加登録時はテンプレート行をコピーして追記する。
    依存脆弱性は [Dependency Governanceの重大度別SLA](../security/Dependency_Governance.md#5-脆弱性対応-sla) に従う。Criticalの24h期限とその他の期限を混同しない。
- ユーザ削除リクエスト受領時は 72h 以内に対象データを特定し、削除証跡をチケットへ添付する。
  再生成が必要な場合は匿名化済みスナップショットのみ使用する。

## 4. 通信制御とツール実行

- **原則対応**: SAC-3, SAC-4, SAC-5, SAC-10。
- 外部通信は [`network/allowlist.yaml`](../../network/allowlist.yaml) に登録されたドメインへ限定し、
  `.github/workflows/security.yml` の差分検証で逸脱を検知する。
  `Allowlist Guard` ジョブの `Validate network allowlist changes` ステップが
  `python -m tools.security.allowlist_guard --base-ref "$BASE_REF"` を実行し、
  Pull Request 時は `${{ github.base_ref }}` を基準に比較する。未承認のドメイン追加や目的変更が検出された場合、
  ワークフローが失敗してマージを停止する。ローカル検証は `python -m tools.security.allowlist_guard --base-ref origin/main`
  で実施する。
  ホワイトリスト外の通信要求は RUNBOOK の外部通信承認手順（[`RUNBOOK.md#outbound-request-approval`](../../RUNBOOK.md#outbound-request-approval)）に従い、
  申請テンプレート（[アウトバウンド通信申請テンプレート](../../tickets/outbound-request.md#申請テンプレート)）で申請項目を記入し、承認者・記録方法を満たした場合のみ許可される。
- ツール実行リクエストは JSON Schema [`schemas/tool-request.schema.json`](../../schemas/tool-request.schema.json) を通過し、
  Web表示経路の `connect-src` は承認済みの実接続先へ限定する。
  Schema 違反時は秘密を含まない理由を記録して入力を修正し、同じ不正入力を再送しない。
  再試行は承認済み通信の一時障害に限り、冪等性・処理済み確認・予算を満たすときRUNBOOKのbackoffを使う。検証例:

  ```bash
  jq '.' tool-request.json | jsonschema -i - ../../schemas/tool-request.schema.json
  ```

- Web UI/APIの出力・認証方式に応じてCSRF/CORS/CSPを設定する。
  [`security_headers/middleware.py`](../../security_headers/middleware.py) はFastAPI/Starlette向けのヘッダ設定例であり、認証・CSRF検証そのものを代替しない。
  この実装の検証は `pytest tests/security/test_security_headers.py` で行う。
  運用例:

  ```python
  from fastapi import FastAPI
  from security_headers import SecurityHeadersConfig, SecurityHeadersMiddleware

  app = FastAPI()
  app.add_middleware(
      SecurityHeadersMiddleware,
      config=SecurityHeadersConfig(
          strict_transport_security="max-age=63072000; includeSubDomains",
          content_security_policy="default-src 'self'",
      ),
  )
  ```

- リリース前は [`.github/workflows/security.yml`](../../.github/workflows/security.yml) 等の採用したゲートを確認する。コンテナ配布時はContainer検証も必要。実行ジョブ・証跡・非該当理由を対応付け、適用ゲートの失敗時は承認された例外なしに本番リリースしない。

## 5. 運用レビュー

- 運用リスク・変更頻度に応じたレビュー周期と責任者を定め、逸脱がある場合は `TASK.codex.md` のテンプレートで是正タスクを登録する。
- SAC改訂時は影響を受ける節を同じ変更で整合させ、改訂履歴を `CHANGELOG.md` に追記する。
