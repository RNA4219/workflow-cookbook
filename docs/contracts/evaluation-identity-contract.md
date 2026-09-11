# Evaluation Identity Contract

比較評価・実験・方策復元・ベンチマーク・昇格判定・提出判定の実装・入力・評価条件・結果を結ぶ契約。
共通identityに用途別profileを追加し、検索や性能評価に架空のdeckや対戦数を要求しない。

## 共通の開始条件

Task Seed等の作業記録からmanifestへリンクし、実行前に次を固定する。

- named owner、task ID、目的、freeze日、単一仮説、変更差分（action_delta）、負の対照。
- Legacy/Candidateのsource path・revision・source hash・input hash。dirty変更はbundle hashでも識別する。
- runnerの識別子・hash、データ集合のID・hash、実行設定、model/policy版。
- 測定単位と主指標、外部更新の凍結。モデルを使わない場合は理由付きのnot_usedを記録する。

source hashは実装bundle、input hashは実行入力の同一性を示す。候補名だけで代替しない。
対戦以外の負の対照は、変更条件外の検索・処理・作業に意図しない差がないことを確認する。

## Profileと移行

| profile | 測定単位 | 固有の必須項目 |
|---|---|---|
| game | game | G50/G200/G1000、deck、native runtimeとRNG制御、opponent/cohort/対象deck/先後 |
| document_retrieval | query | 検索集合・主指標・top-kや読み込み予算等の設定 |
| performance | operation | 測定集合・主指標・反復やウォームアップ等の設定 |
| workflow | task | 作業集合・主指標・判定基準等の設定 |

新規manifestはschema_version 1.1と既知のprofileを必須とする。未知profileは失敗となる。
gameは `templates/evaluation-identity-manifest.template.json`、
非gameは `templates/evaluation-measurement-manifest.template.json` から作り、実条件で埋める。
過去の1.0はprofile省略をgameとして読み取る。1.0を非gameとして解釈したり過去artifactを上書きしたりしない。
1.0の読み取り互換を保ち、新規測定で必要なrevision・測定条件を1.1へ記録する。

## 実行前と実行後

```bash
uv run python tools/ci/check_evaluation_identity_manifest.py \
  --manifest docs/evaluation-manifests/EV-YYYYMMDD-01.json --stage preflight --check

uv run python tools/ci/check_evaluation_identity_manifest.py \
  --manifest docs/evaluation-manifests/EV-YYYYMMDD-01.json --stage postrun --check
```

preflightはstatus=frozenと必要なidentity・条件を確認する。成功前に評価を開始しない。
実行後はstatus=completed、outcomeのartifact_path・artifact_sha256・records・metricsを記録する。
metricsには主指標の有限な実測数値を含め、recordsは測定単位に沿う正の整数とする。
非gameはerror_count/crash_count/timeout_count=0を要求する。
gameはtotal_games>0とfallback_count/illegal_action_count/crash_count/timeout_count=0も要求する。
gameの0試合artifactは即Reject。計測障害・空結果をpassや効果ゼロとしない。

1.1のCLI postrunはartifactの実在とSHA-256一致も確認する。相対artifact_pathの基準はmanifestのディレクトリ。
別の基準を使う場合は `--artifact-root <directory>` を指定する。
1.0は旧来の構造検証を維持し、同オプションを指定した場合は実ファイルも照合する。
schemaはデータ形状、checkerはstage条件と主指標の存在・有限値を検証する。
manifestの正しさだけでは測定の妥当性・性能改善・提出可能性を証明しない。

## Gameの判定と保全

- G50は動作・意図したaction delta・負の対照、G200は非劣性、G1000は非劣性と戦略KPI改善を確認する。
- native runtimeの乱数制御可否を記録する。Pythonの同一seedだけで完全な対戦列の同一性を主張しない。
- Pass以外はLegacyを維持する。同一deckで局所候補が2件連続Rejectなら構造・構成・評価設計を再診断する。
- source bundle・manifest・結果artifactを不可分に保存し、Task/Acceptance/registry等をIDとhashで結ぶ。
- postrun成功は証跡整備の条件。registry更新・提出・外部公開は別の操作として扱う。

## 適用境界

read-only診断、文書lint、checkerやcoordinatorの単体fixture試験は比較評価実験に含めずmanifest不要。
評価を走らせていないことを作業記録へ明記し、単体試験の成功を実運用効果や対戦成績として扱わない。
効果や昇格を判断する比較評価を始める場合は、適切なprofileでpreflight/postrunを実施する。
