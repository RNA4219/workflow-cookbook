---
intent_id: INT-TEST-REVIEW
owner: docs-core
status: active
last_reviewed_at: 2026-07-11
next_review_due: 2026-08-10
---

# Test Review Checklist

テスト品質を確保するためのレビュー手順書。

## 概要

AIが作成したテストには以下のリスクが存在する：

1. **ローカル/本番環境乖離** - 環境変数や実行環境の違いによるエラー
2. **テスト通過詐欺** - モック、スキップ、ダミーデータによる偽の成功
3. **テストケース不足** - エッジケースの漏れ

本手順書はこれらのリスクを軽減するためのレビュープロセスを定義する。

---

## テスト作成フロー

変更した挙動の確認 → 必要な試験の作成・実行 → リスクとrepo契約に応じたレビュー → マージ判定。
独立レビューや業務判断が必要なら担当者・対象・結果を記録する。全変更に別AIと人間の二段レビューを要求しない。

---

## 1. AI作成フェーズ

### 作成者AIが守るべき原則

| 原則 | 内容 | NG例 |
|------|------|------|
| 実際の処理をテスト | モックは最終手段 | 全てモックで置換 |
| スキップ条件の明示 | スキップする場合は理由をコメント | `pytest.skip("temp")` |
| エッジケース網羅 | 境界値、空値、異常値 | ハッピーパスのみ |
| 環境依存の排除 | 環境変数はfixtureで注入 | `os.environ.get("API_KEY")` |

### 作成時チェックリスト

- [ ] 単体試験は関数を直接検証し、CLI・配布の試験は必要に応じsubprocessや隔離installで境界を通しているか
- [ ] モックは最小限か
- [ ] スキップ条件に正当な理由があるか
- [ ] 環境変数を使用していないか、使用する場合はfixture化しているか
- [ ] 正常系と異常系の両方をテストしているか

---

## 2. 独立レビューフェーズ

公開契約、権限、データ移行、重大な業務ロジック等のリスクやrepoのレビュー契約に応じて実施する。
担当者は作成者と独立した判断を行い、対象範囲と所見を記録する。別AIの利用自体を必須にしない。

### レビュー観点

#### 2.1 スキップ/モックの妥当性

```python
# NG: 存在しない環境変数でスキップ
@pytest.mark.skipif(not os.environ.get("FAKE_VAR"), reason="no FAKE_VAR")
def test_something():
    ...

# OK: 明確な理由と条件
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix socket not supported on Windows"
)
def test_unix_socket():
    ...
```

#### 2.2 ダミーデータ返却の確認

```python
# NG: 常に成功を返すモック
mock_api.return_value = {"status": "success"}

# OK: 実際のレスポンス形式を模倣
mock_api.return_value = {
    "status": "success",
    "data": {"id": 1, "name": "test"},
    "timestamp": "2026-04-10T00:00:00Z"
}
```

#### 2.3 アサーションの有効性

```python
# NG: 常にTrueになる
assert len(result) >= 0

# OK: 具体的な値を検証
assert len(result) == 3
assert result[0].id == "expected-id"
```

#### 2.4 テストカバレッジの確認

- [ ] 正常系（ハッピーパス）
- [ ] 異常系（エラーケース）
- [ ] 境界値（空、最大、最小）
- [ ] 型エラー（文字列/数値の混在）

### レビューチェックリスト

| チェック項目 | 確認内容 |
|-------------|---------|
| スキップ条件 | 不自然なスキップがないか |
| モック範囲 | テスト対象までモックしていないか |
| アサーション | 意味のある検証をしているか |
| 環境依存 | ローカル/CI/本番で差分がないか |
| カバレッジ | 主要な分岐をカバーしているか |

---

## 3. 人間レビューフェーズ

業務判断・リスク受容・責任者承認が必要な場合は、該当する観点を人間が確認する。

### 3.1 ドメイン知識の確認

- [ ] ビジネスロジックが正しく反映されているか
- [ ] 実際のユーザー操作パターンをカバーしているか
- [ ] 業務上の重要なエッジケースが含まれているか

### 3.2 現場での使われ方

- [ ] 実際の運用環境でのシナリオを想定しているか
- [ ] 既知の障害パターンをテストしているか
- [ ] パフォーマンス要件を考慮しているか

### 3.3 セキュリティ観点

- [ ] 認証・認可のテストがあるか
- [ ] 入力バリデーションをテストしているか
- [ ] 機密情報の漏洩がないか

---

## 4. レビュー結果の記録

### レビューレポート形式

```markdown
## Test Review Report

**Reviewer**: [AI名/人間名]
**Date**: YYYY-MM-DD
**Test File**: tests/test_xxx.py

### Findings

| Category | Issue | Severity | Status |
|----------|-------|----------|--------|
| Skip | 存在しない環境変数でスキップ | High | Fixed |
| Mock | テスト対象をモックしている | Critical | Fixed |
| Assertion | 常にTrueになるアサーション | Medium | Fixed |

### Recommendation

- [ ] Approve
- [ ] Request Changes
- [ ] Block
```

---

## 5. CI/CDパイプラインでの自動チェック

### 採用済みゲートとレビュー用の観測

本repoの coverage 80% ゲートは維持する。導入先は対象モジュールと基準を定義する。

- skipは実行結果の理由・未検証範囲・実行対象に対する比率と推移を確認する。
- mockは外部境界に限定され、検証対象を置換していないか内容を確認する。
- grepによるskip/mock語の総数だけでCIを失敗させない。
- 実行結果のレポートを保存し、必要な境界試験が実行されていなければ未検証として扱う。

---

## 付録: テストアンチパターン集

### A. 環境変数スキップ詐欺

```python
# 検出方法
grep -r "os.environ.get.*skip" tests/

# 対策
# 環境変数はfixtureで明示的に管理
@pytest.fixture
def api_key():
    return os.environ.get("TEST_API_KEY", "default-test-key")
```

### B. 常に成功するテスト

```python
# 検出方法
# assert True, assert len(x) >= 0 などを検索
grep -r "assert True\|assert len.*>= 0" tests/

# 対策
# 具体的な期待値を設定
assert result.status == "completed"
```

### C. テスト対象のモック化

```python
# NG: テスト対象をモック
@patch("myapp.core.process")
def test_process(mock_process):
    mock_process.return_value = "ok"
    result = process()  # テスト対象がモック！
    assert result == "ok"  # 常に通る

# OK: 依存のみモック
@patch("myapp.api.external_call")
def test_process(mock_api):
    mock_api.return_value = {"data": "test"}
    result = process()  # テスト対象は実物
    assert result.processed == True
```

---

## 変更履歴

| 日付 | 変更内容 | 作成者 |
|------|---------|--------|
| 2026-04-10 | 初版作成 | AI |
