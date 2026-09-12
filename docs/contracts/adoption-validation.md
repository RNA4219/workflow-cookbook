---
intent_id: INT-ADOPTION-VALIDATION
owner: workflow-cookbook
status: active
last_reviewed_at: 2026-09-12
next_review_due: 2026-12-12
---

# コピー後の検査と導入段階の判定

## Git checkout と整合性

コピー manifest v2 は元の全ファイルの SHA-256 と mode を保持する。
各v2レコードの `checkout_sha256` は必須で、元hashとの集合が実ファイルの完全な
改行形式のhash集合と一致することまで単独検査する。欠損・不足・無関係なhashは拒否する。
改行が LF または CRLF に統一された UTF-8 テキストに限り、もう一方の改行形式の
SHA-256 もコピー時に記録し、その完全一致を認める。バイナリ、単独 CR、混在改行は元の
バイト一致だけを認める。本文や末尾改行の追加・削除、部分的な改行変更は検出する。
AGENTS の管理ブロックも LF 版か CRLF 版の完全一致を要求する。
既存 AGENTS、Git 設定、属性ファイルは書き換えない。任意の filter や文字コード変換の
出力を同一内容と認定しない。実行属性の検査は従来どおり維持する。

検査は Git とコピー元がなくても動く。v1 manifest は従来のバイト一致で検査する。
v1 の同梱 verifier を自動更新したり、旧コピーの manifest を黙って書き換えたりしない。

## Tier の構造検査

各パスについて `exists` と `kind` に加え、`valid` と不適合理由を返す。
必須 Markdown は通常ファイルで空白以外の本文が必要。Task/Acceptance のディレクトリには
有効な Markdown、caps には有効な capsule JSON が必要。
index は非空の nodes 辞書、hot は非空の nodes 配列、caps は id と summary を持つ
JSON オブジェクトを要求する。パス欠落、種類違い、空、破損、読み取り不能は合格させない。

候補の拡張子は大小文字を区別せずに列挙し、capsuleは役割としてJSON構造を必須にする。
`.JSON`という表記で本文だけの検査へ切り替えない。

`current_tier` は内容を伴う文書構造の段階であり、実運用の合格証明ではない。
`operational_compliance` は `not_evaluated`。未作成の記録は実績として補わず、初期導入では
不足理由と次段階への必要項目を返す。上流コピーを導入先自身の文書として数えない。

## テンプレートの版検査

比較対象は既存の対象文書とする。存在しない文書は Tier 検査で扱う。
対象文書があるのにテンプレートがない、片方の版がない、読み取れない場合は `unknown`。
比較対象がゼロの場合も全体を `unknown` にする。
全体の `drift_status` は `current / drifted / unknown / not_checked`。
版違いがあれば `drifted`、版違いがなくても不明が残れば `unknown`。
従来の `drifted` bool は実際に版違いが分かった場合だけ true とし、不明を表す欄とは分ける。
`--check-drift --check` は `current` の場合だけ成功する。

引用符の外側にある末尾コメントを版番号へ含めず、同じ文字列値を同版として扱う。
引用符内の文字とコメントを区別し、閉じていない引用符などの不正な値は不明とする。
1行のplain・単一引用符・二重引用符の値に対応する。二重引用符はJSON互換のescape、
単一引用符は引用符の二重化を解釈する。複数行、alias、複合値など未対応のYAML表記は不明とする。

## 複数repoとCI診断

Tierとonboardingのrepo-listは、非空文字列または非空の`repo`文字列を持つオブジェクトの
非空配列を要求する。不正な要素を黙って除外せず、入力全体をエラーとして返す。
検査0件を成功として扱わない。onboardingは`.github/workflows/`の`.yml`と`.yaml`を読む。

## Gitへ共有する実行属性

Windowsのファイルシステムへコピーしただけでは、Git indexに実行属性が記録されるとは限らない。
通常のコピー・`--check`はローカルファイルの検査であり、Gitへの共有準備を認定しない。
`--check-git-modes`を明示した場合のみGitを使い、manifestの全ファイルについてstage 0のmodeを
照合する。未登録・競合・modeの不一致は失敗する。indexとGit設定は変更しない。
利用者が必要な属性を明示的に設定し、再検査してからcommitする手順を
[導入ガイド](../adoption-guide.md)へ記載する。

## 受入

- Git の commit/clone と autocrlf、既存 attributes の組合せを Windows/Linux の fixture で検査する。
- clone 後の単独 verifier と再実行は成功し、本文・バイナリ・部分改行の変更は失敗する。
- 空、型違い、破損した JSON、記録なしでは Tier 3 に合格しない。有効な文書構造では合格する。
- 同版、異版、版欠落、テンプレート不在、比較対象ゼロ、未検査を区別する。
- 本タスクは checker とコピーの機能検証であり、モデル性能や長期ドリフト率の比較実験は行わない。
