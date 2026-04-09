# codemap ツール

`codemap.update` は Birdseye のインデックスおよびカプセルを再生成するコマンドです。
現行の `run_update` は指定したターゲットを起点に、既定では ±2 hop のカプセルを探索し、
探索範囲の `deps_in` / `deps_out` のみを再計算して Birdseye のトポロジーを同期します。
以下の手順で最新化します。

## 依存

- Python 3.11 以上
- 追加の外部ライブラリは不要（標準ライブラリのみで実行できます）

## 実行手順

1. （任意）仮想環境を作成し、有効化します。
2. リポジトリルートで次のコマンドを実行します。

   ```bash
   # 例: 直近の main との差分から対象カプセルを推測
   python tools/codemap/update.py --since --emit index+caps

   # 例: 明示的にターゲットを指定（従来挙動）
   python tools/codemap/update.py --targets docs/birdseye/index.json,docs/birdseye/caps --emit index+caps

   # 例: 1 hop のみ再計算して局所更新
   python tools/codemap/update.py --since --radius 1 --emit caps
   ```

   - `--since` を指定すると `git diff --name-only <参照>...HEAD` を用いて Birdseye 配下の変更ファイルから対象を自動推定します。参照を省略すると `main` が使われます。
   - `--targets` には再生成したい Birdseye リソースをカンマ区切りで指定します。
     ルート（`docs/birdseye/`）や `index.json` / `hot.json` / `caps/` ディレクトリをターゲットに含めた場合は、
     すべてのカプセルが探索の起点となり、±2 hop の再計算が全体へ波及します。
     明示的にターゲットを限定することで、±2 hop のカプセル範囲を利用者が制御できます。
   - `--emit` には出力したい成果物（`index` / `caps` / `index+caps`）を指定します。
   - `--radius` には探索する hop 数を指定します。既定値は `2` で、`0` を指定するとターゲット自身のみを更新します。
3. 実行後、以下の成果物が更新されます。
   - `docs/birdseye/index.json`
   - `docs/birdseye/hot.json`（`index` を出力する場合に含まれます）
   - `docs/birdseye/caps/*.json`
   - `generated_at` は 5 桁ゼロ埋めの世代番号として更新され、`index.json` と `hot.json` は同じ更新サイクルの番号へ揃います。

## Birdseye 再生成スクリプト

`update.py` は Birdseye の再生成処理を司るエントリーポイントです。
各ターゲットの解析や JSON 生成ロジックは `run_update` 内で完結し、インデックス・ホットリスト・カプセルの依存情報を同期します。
詳細な処理を追加する際は、既存の例外設計・型安全方針に従って実装してください。
Birdseye ドキュメント（`docs/BIRDSEYE.md` / `docs/birdseye/README.md`）と整合するよう、手順の更新が必要な場合は併せてメンテナンスしてください。

- CLI エントリ: `python tools/codemap/update.py ...`
- 追加の機能を導入する場合は、Birdseye ドキュメント（`docs/BIRDSEYE.md` / `docs/birdseye/README.md`）と整合するよう手順を更新してください。

### Birdseye 再生成の観点

- `run_update` はターゲット起点に指定された hop 数までのカプセルを探索し、範囲内の `deps_in` / `deps_out` を再計算します。既定値は ±2 hop です。
- ルート（`docs/birdseye/`）や `index.json` / `hot.json` / `caps/` ディレクトリをターゲットに含めた場合は、全カプセルが探索対象に含まれます。
- `docs/birdseye/index.json` を更新する際は、同一ルートにある `hot.json` も同じ `generated_at` 世代番号へ揃うよう自動で書き換えられます。
