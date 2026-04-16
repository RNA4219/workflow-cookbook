---
task_id: 20260415-01
intent_id: INT-SEC-001
owner: security
status: done
last_reviewed_at: 2026-04-17
next_review_due: 2026-05-17
completed_at: 2026-04-17
---

# Task Seed: Security Priority Response

## 背景

- `workflow-cookbook` の脆弱性診断により、
  まず対処すべき 3 つの論点が確認された。
- 現状の
  [`tools/perf/collect_metrics.py`](../../tools/perf/collect_metrics.py)
  は、`metrics_url` / `pushgateway_url` を十分に検証せず
  `urlopen` に渡しており、
  SSRF とローカルファイル読取の攻撃面を持つ。
- 現状の
  [`.github/workflows/security.yml`](../../.github/workflows/security.yml)
  では `Bandit` の `B310` `B603` `B607` などが広く除外されており、
  上記のような危険なコードパターンを CI で拾いにくい。
- 依存脆弱性監査についても、
  lockfile / requirements ベースで固定された監査ではなく、
  環境差分のノイズを拾いやすい構成になっている。

## ゴール

- SSRF / ローカルファイル読取の成立余地を最優先で閉じる。
- CI セキュリティ検査が、今回の種別の問題を再び見逃さない状態に戻す。
- 依存脆弱性監査を、継続運用できる低ノイズな方式へ寄せる。

## 優先順位

1. `P0`: `tools/perf/collect_metrics.py` の SSRF / ローカルファイル読取対策
   - 直接 exploit に近いリスクであり、コード上で攻撃面が成立している。
   - まず入力制限とプロトコル制限を入れ、危険な到達先を遮断する。

2. `P1`: `security.yml` の Bandit 除外是正
   - 既存の危険な呼び出しを CI が見逃しやすい状態であり、
     再発防止の観点で優先度が高い。
   - 全面解除ではなく、必要最小限の suppress に寄せる。

3. `P2`: 依存脆弱性監査の運用是正
   - 直接の exploit 面ではないが、
     監査ノイズが高いと継続的なセキュリティ運用が崩れる。
   - lockfile / requirements ベースへ寄せて、再現性のある監査にする。

## 最小の実施順

1. `collect_metrics.py` に対して、
   許可スキームと許可到達先の制限を導入する。
2. `security.yml` と Bandit 設定を見直し、
   今回の種別が CI で検出される状態へ戻す。
3. 依存脆弱性監査の方式を
   `pip-audit` などの固定入力ベースへ移す。

## 修正対象

1. `P0`: URL 入力制限
   - [`tools/perf/collect_metrics.py`](../../tools/perf/collect_metrics.py)
   - [`tests/test_perf.py`](../../tests/test_perf.py)
   - `http` / `https` 以外のスキームを拒否する。
   - `file://` を明示的に拒否する。
   - `localhost`、loopback、private network、link-local など
     危険な到達先を禁止する。
   - 既存テストに `file://` 許容がある場合は、
     期待値を拒否側へ更新する。

2. `P1`: CI セキュリティ検知の回復
   - [`.github/workflows/security.yml`](../../.github/workflows/security.yml)
   - 必要なら `Bandit` 設定ファイルや引数定義も見直す。
   - `B310` `B603` `B607` の除外をそのまま維持せず、
     根拠のある個別 suppress へ寄せる。
   - 今回の `urlopen` リスクを CI で再検出できることを確認する。

3. `P2`: 依存脆弱性監査の低ノイズ化
   - [`requirements.txt`](../../requirements.txt)
   - [`pyproject.toml`](../../pyproject.toml)
   - [`.github/workflows/security.yml`](../../.github/workflows/security.yml)
   - `pip-audit` など、
     固定された依存定義に基づく監査手法へ切り替える。
   - CI 実行時の入力が環境依存にならないよう整える。

## TDD / 検証

1. `P0`
   - `file://` が拒否されることをテストで確認する。
   - private / loopback / link-local 到達先が拒否されることを確認する。
   - 正常な `https` URL は従来どおり許可されることを確認する。

2. `P1`
   - `Bandit` が `urlopen` 系の危険な利用を再検出することを確認する。
   - suppress が必要な箇所は、理由付きで限定的に残っていることを確認する。

3. `P2`
   - 依存監査が lockfile / requirements ベースで再現可能に動くことを確認する。
   - ローカル環境差分によって結果が大きく揺れないことを確認する。

## 完了条件

- `collect_metrics.py` が `file://` や危険な到達先を受け付けない
- SSRF / ローカルファイル読取の再発防止テストが追加されている
- `Bandit` が今回の種別を CI で見逃しにくい構成になっている
- 依存脆弱性監査が低ノイズで再現可能な方式に更新されている
- 変更内容に対応するテストまたは CI 検証が通る

## 検収観点

- URL バリデーションがスキームだけでなく到達先まで見ているか
- `file://` を黙って通す経路が残っていないか
- `Bandit` の除外が「広く黙らせる設定」のまま残っていないか
- 依存監査が実行環境依存のノイズを引きずっていないか

## 参照

- [`tools/perf/collect_metrics.py`](../../tools/perf/collect_metrics.py)
- [`tests/test_perf.py`](../../tests/test_perf.py)
- [`.github/workflows/security.yml`](../../.github/workflows/security.yml)
- [`requirements.txt`](../../requirements.txt)
- [`pyproject.toml`](../../pyproject.toml)
