---
task_id: 20260411-07
intent_id: INT-001
owner: docs-core
status: done
last_reviewed_at: 2026-04-11
next_review_due: 2026-05-11
---

# Task Seed: Acceptance Normalization for AC-20260411-01

## 背景

- [AC-20260411-01](../acceptance/AC-20260411-01.md) は
  `RG-002` から `RG-006` に scope を絞り直しているが、
  文書の先頭では依然として `status: approved` が目立つ。
- 後続の
  [AC-20260411-02](../acceptance/AC-20260411-02.md) と
  [AC-20260411-03](../acceptance/AC-20260411-03.md)
  が追加されていても、
  release 判定や acceptance index では
  まず `AC-20260411-01` の `approved` が見えるため、
  初回タスクが `RG-001` まで含めて完了したように誤読されやすい。
- レビュー指摘の本質は
  「本文で限定していても、record 全体の読み味が強すぎる」
  という点にある。

## ゴール

- `AC-20260411-01` を単独で読んでも、
  `RG-001` まで完了したと誤認しない表現にする。
- acceptance index / release 判定 / docs 読者に対して、
  `AC-20260411-01`、`AC-20260411-02`、`AC-20260411-03` の役割分担を明示する。
- approval の意味が「この scope に対してのみ承認」であることを
  文書全体で一貫させる。

## 修正対象

1. `AC-20260411-01` の表現調整
   - `status: approved` を維持するなら、
     タイトル直下または Verification Result に
     「RG-002〜RG-006 のみ承認対象」であることを
     1 行で即読できる形で追記する。
   - それでも誤読余地が強い場合は、
     状態表現を `reviewing` / `superseded` 相当へ見直す案も検討する。

2. 後続 acceptance との関係明示
   - `AC-20260411-01` から
     `AC-20260411-02` と `AC-20260411-03` への参照を強める。
   - 少なくとも
     `RG-001 は AC-20260411-02`,
     `RG-002/RG-003 の最終調整は AC-20260411-03`
     と分かる記述を残す。

3. index / docs の整合
   - 必要なら [docs/acceptance/INDEX.md](../acceptance/INDEX.md) の見え方も確認する。
   - acceptance record の一覧だけ見た人が誤読しないかを確認する。

## TDD / 検証

1. docs review
   - `AC-20260411-01` 単独で読んだ場合に、
     `RG-001` まで承認済みと読めないことを確認する。
   - `AC-20260411-02` / `AC-20260411-03` を含めて読んだ場合に、
     役割分担が明確であることを確認する。

2. acceptance index
   - `docs/acceptance/INDEX.md` に並んだとき、
     `AC-20260411-01` が過大な完了表現に見えないことを確認する。

3. formatting
   - `markdownlint-cli2` を通す。

## 完了条件

- `AC-20260411-01` を単独で見ても `RG-001` まで完了したように読めない
- `AC-20260411-02` / `AC-20260411-03` との役割分担が明示されている
- acceptance index から見たときの誤読余地が下がっている

## 検収観点

- `approved` が scope を越えた完了表現として読めないか
- 初回 acceptance と後続 acceptance の責務分担が明記されているか
- release 判定時に誤った evidence として扱われないか

## 参照

- [AC-20260411-01](../acceptance/AC-20260411-01.md)
- [AC-20260411-02](../acceptance/AC-20260411-02.md)
- [AC-20260411-03](../acceptance/AC-20260411-03.md)
- [docs/acceptance/INDEX.md](../acceptance/INDEX.md)
