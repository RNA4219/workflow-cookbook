# 共有workspaceの検証記録と配布ファイルの照合

[validation.json](validation.json)は2026-07-19の検証記録であり、後の改修版を再検証した記録ではない。
[historical-hash-manifest.json](historical-hash-manifest.json)は当時の照合表を保存したもの。
その参照先は以後変更されており、現在のcheckoutとの一致を保証しない。
当時の全ファイルのbytesをこのパックだけで復元することもできない。

[hash-manifest.json](hash-manifest.json)は2026-09-11の配布ファイルをGit内のbytesで照合した表。
`root` はrepo rootを表す。旧検証結果を現在のソースへ付け替える意味ではなく、
公開するファイルの整合性を確認するために使用する。
Windowsでcheckout時の改行変換がある場合は `git show HEAD:<path>` のbytesで照合する。

今回の改修版の自動検証は[PR #504のCI](https://github.com/RNA4219/workflow-cookbook/pull/504/checks)を参照する。
leaseとfinishするjobの一致を必須にし、job未指定のleaseではjobを完了扱いにしない。
既存のterminal jobの再利用確認は、同じjob keyを指定したacquireで行う。
