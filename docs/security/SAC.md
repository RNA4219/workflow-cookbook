# Security Architecture Contract (SAC) v0.2

対象: 本契約を採用した製品・エージェント・運用サービス。以下の適用条件を実装経路ごとに記録する。
該当条件では必須とし、非該当は理由を残す。既存の承認、通信境界、秘密情報保護を解除する根拠にはしない。

## 原則

1. Secretsは管理対象の秘密保管機構で保護し、ログ・配布物・ブラウザへ露出させない。サーバ資格情報はサーバ側で管理する
2. LLM出力・外部入力は不信任として扱う。Web表示では出力文脈に応じたエスケープ／サニタイズを行い、別窓リンクに `rel="noopener"` を設定する。保存形式と表示処理を分ける
3. 外部通信経路は採用したallowlist・用途・承認に限定する。利用者入力で接続先を決める経路ではループバック・メタデータIP等の内部宛先を拒否する。許可先の追加は既存の承認手続を通す
4. 構造化ツール要求は実際のSchemaと認可検証を通過してから実行する。契約違反は入力修正まで実行しない
5. Web UI/APIでは認証方式に応じたCSRF対策、CORS、CSPを設定する。Cookie認証のCSRF対策をヘッダ追加だけで完了扱いにしない。付録Aは構成検討用の例
6. APIサービス・自動反復処理ではRate/Quotaを設定し、サービス限度・予算に合わせRPS、同時実行、トークン／費用上限、超過時の停止・待機・通知を明文化する
7. ビルド／配布では採用したlockと監査を使う。既知脆弱性はCIの採用済みゲートでブロックし、期限付きの承認例外は台帳で追跡する
8. 監査ログを収集する運用ではPIIを最小化・マスクし、改ざん検知と保管契約を適用する。照合に必要な識別情報と保管権限を定める
9. モデルを利用するサービスでは変更可能な主体・モデル・費用・権限を制御する。サーバ権限や秘密情報に影響する切替を未認可クライアントへ委ねない
10. リリース前に採用したSAST / Secrets / 依存ゲート、およびコンテナを配布する場合のContainerゲートを通過する。失敗を省略扱いにせず、例外は承認と証跡を必要とする
11. GitHub上の本repoではvulnerability alerts・Dependabot security updates・secret scanning・push protectionの採用済み設定を維持し、監視する。別のホスティング先では同等の保護と責任者を定める

### 付録A: CSP設定例

接続先・表示資源を持つWeb構成向けの例。実サービスの必要最小限へ調整し、固定API宛先や `unsafe-inline` を無条件に複製しない。

```text
default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self' https://api.openai.com https://generativelanguage.googleapis.com; frame-ancestors 'none'; base-uri 'none'
```
