---
task_id: 20260912-03
intent_id: INT-ADOPTION-EDGE-CASES
owner: workflow-cookbook
status: done
---

# コピーと導入診断の境界条件を修正する

[検査契約](../contracts/adoption-validation.md)と[コピー契約](../contracts/full-workflow-copy.md)に従い、
追加調査で見つかった7項目を修正する。[受入](../acceptance/AC-20260912-03.md)に結果を記録する。

## Scope

- A1: `.JSON`を含むcapsuleの候補列挙と役割に基づくJSON構造検査をOS間で一致させる。
- A2: onboardingでも空・空白・不正要素を持つrepo-listを拒否し、検査0件を成功させない。
- A3: `.yml`と`.yaml`のworkflowを検出する。
- A4: 引用符・コメントを持つ同じtemplate_versionを同版として扱う。
- A5: パス成分の大小文字とファイル・ディレクトリの衝突をコピー前に拒否する。
- A6: Git indexのmodeを読み取りで検査し、WindowsからGitで共有する際の実行属性を確認する。
- A7: CLI構文エラーも終了コード1とJSONで返し、helpの正常動作を保持する。

コピー更新機能、外部接続doctor、ブランチ保護設定の変更は含めない。
既存AGENTS・Git index・Git設定を自動変更せず、既存manifest v1/v2の検査互換性を保つ。
本作業はcheckerの単体fixtureとCLIの機能検証であり、比較評価実験・モデル性能測定は行わない。

## Verification

7項目それぞれに正常条件と失敗条件を設け、コピー・Tier・onboardingの回帰、全体pytest、
Ruff/mypy、文書・Birdseyeのゲートを確認する。
Windowsで作ったGit bundleをLinuxでcloneするCIを追加し、同梱verifierと実行属性を確認する。
未実施の環境での結果は成功として記録しない。
