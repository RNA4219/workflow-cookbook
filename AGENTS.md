---
intent_id: INT-POLICY-20260910
owner: workflow-cookbook
status: active
last_reviewed_at: 2026-09-10
next_review_due: 2026-10-10
---

# 評価・実験タスクの開始ゲート

このリポジトリで、比較評価、実験、方策復元、ベンチマーク、昇格判定、提出判定を
含む作業を始めるエージェントは、実装変更・対戦実行・外部提出より先に、必ず
[`docs/contracts/evaluation-identity-contract.md`](docs/contracts/evaluation-identity-contract.md)
を読む。

以下は **MUST** とする。

1. `templates/evaluation-identity-manifest.template.json` から評価manifestを作成する。
2. 共通identity（Legacy/Candidateのsource/input hash・revision、runner、freeze日、owner、
   単一仮説、変更差分、負の対照）と用途別profile・測定単位・データ集合・版を固定する。
   deck/native runtime/opponent setとG50等のgateはgame profileに適用する。
3. G50など最初の評価を開始する前に、
   `uv run python tools/ci/check_evaluation_identity_manifest.py --manifest <path> --stage preflight --check`
   を通す。
4. 実行後は同じmanifestに結果artifact hash・実測値・`records`・障害数を記録し、
   `--stage postrun --check` を通す。gameでは `total_games > 0` とfallback/illegal action=0も必要。
   非gameはquery/operation/taskの正の件数とerror/crash/timeout=0を要求し、架空の対戦数を記録しない。
5. `--stage postrun` が成功するまで、Champion registry更新、提出、外部公開を行わない。

read-only診断、文書lint、checkerの単体fixture試験は比較評価実験に当たらずmanifest不要。
評価を走らせていないことを作業記録へ明記し、単体試験を実運用の効果測定として報告しない。
非gameは `templates/evaluation-measurement-manifest.template.json` を使用する。
同一seedをnative runtimeを含む完全な対戦列の同一性証明として扱ってはならない。
