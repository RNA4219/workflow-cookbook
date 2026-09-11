# 解消済み質問が復元入力から欠ける理由

この文書は実測中のコード診断。最終件数と効果判定はRESULTS.mdとaudit.jsonを参照する。
本試験のproduction code、凍結入力、採点基準は変更していない。

## 観測

最初のlong-03/scripted/反復0では、16〜128イベントの全8回答でresolved_questionsだけが不一致だった。
目的・採用案・制約・判断理由・未回答質問・許可などは、その実行では正しい。
公開question listではQUESTION-3-0などのstatus=answeredを取得できる。
標準context buildのsnapshot.open_questionsでは、その回答済み項目が除外される。
最初の16イベント時点ですでに発生しているため、この観測だけで「履歴が長くなるほど悪化する」とは判断できない。
長さによる追加の悪化は、別途8測定点の項目別結果で確認する。

## 根拠

- `Agent_tools/agent-taskstate/src/agent_taskstate/context_bundle.py:668`:
  decisionsの対象はaccepted/proposedで、rejectedの経緯は標準の決定一覧に入らない。
- `Agent_tools/agent-taskstate/src/agent_taskstate/context_bundle.py:679`:
  質問の抽出条件が `task_id = ? AND status = 'open'`。
- `Agent_tools/agent-taskstate/docs/src/agent-taskstate_requirements_one_pager.md:214`:
  初期の生成契約は「open な open_questions」を常時含める。
- `Agent_tools/workflow-cookbook/docs/contracts/task-context-continuity.md`:
  workflow側は解決済み情報も保持するという広い目的を記載している。

したがってDBの消失とは別の、標準復元が対象とする情報範囲の問題である。
MVPのopen質問限定契約には沿っているが、回答済み事項の経緯を要求する今回の用途には不足する。
rejected決定の除外も静的に確認したが、その過去決定一覧の完全復元率は今回の採点対象ではない。

## 改修する場合の方向

回答済み質問と回答内容を、status・根拠・更新時刻を持つ独立した復元項目として含める仕様を定める。
open質問へ戻す形で混ぜると、未解決事項を誤って増やすので状態の区別を保持する。
既存APIとの互換性を保ちながら、書き込み、bundle、workflow入力、採点を一貫して検証する必要がある。
この方向は今回の調査結果からの改修案であり、本測定内では実装していない。

## 評価範囲の読み方

今回の目的・制約・判断理由は識別コードで判定し、解消済み質問はID一覧で判定している。
自由記述の回答本文や判断理由を意味まで完全に復元できるかは、この正解率に含まれない。
回答モデルは標準snapshotを入力として回答し、自律的に追加のquestion listを呼び出す構成ではない。
そのため標準復元の不足を測った結果であり、追加取得を行う別のエージェント構成へそのまま一般化しない。
6課題は共通のイベント構造を持つ合成シナリオであり、6分野の実務を代表する標本ではない。
同一設定での2反復は再現性の観測であり、独立した課題を12種類検証したとは数えない。

128イベントの蓄積とプロセス再接続を試すが、コンテキスト上限まで負荷を増やす試験ではない。
暦日で数か月の保存、版の移行、同時更新、停電時の復旧も今回のモデル測定の対象外である。
今回確認できた保存・復元の挙動と、これら未検証の運用条件を分けて扱う。
