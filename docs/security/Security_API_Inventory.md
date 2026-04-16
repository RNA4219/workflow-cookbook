---
intent_id: INT-SEC-007
owner: security
status: active
last_reviewed_at: 2026-04-17
next_review_due: 2026-05-17
---

# Security API Inventory

本ドキュメントは URL / subprocess / file access の
危険パターン利用箇所を repo 横断で棚卸しした結果である。
個別修正で終わらせず、継続的な再点検を可能にする。

## 棚卸し方針

1. **定期実行**: 月次または major release 前に grep で再棚卸し
2. **新規追加時**: PR で危険 API 利用を明示し、台帳へ追記
3. **検証必須**: 外部入力を受ける箇所は到達先制限・入力検証を必須とする

## URL / Network Access

### urlopen / urllib.request

| 箇所 | ファイル | 入力種別 | 検証手段 | nosec | リスク評価 |
| --- | --- | --- | --- | --- | --- |
| collect_metrics.py:1238 | tools/perf/collect_metrics.py | 外部入力 URL | `_validate_url()` で危険スキーム拒否 | B310 | 低（検証済み） |
| collect_metrics.py:1287 | tools/perf/collect_metrics.py | 外部入力 URL | `_validate_url()` で危険スキーム拒否 | B310 | 低（検証済み） |
| check_security_posture.py:48 | tools/ci/check_security_posture.py | 固定 host | GitHub API 固定、token auth | B310 | 低（固定 host） |
| check_security_posture.py:63 | tools/ci/check_security_posture.py | 固定 host | GitHub API 固定、token auth | B310 | 低（固定 host） |
| check_release_evidence.py:116 | tools/ci/check_release_evidence.py | 固定 host | GitHub API 固定、token auth | B310 | 低（固定 host） |

### URL 検証要件

外部入力 URL を受ける箇所は以下を満たす必要がある:

- **スキーム制限**: `http://`, `https://` 以外は拒否
- **危険スキーム拒否**: `file://`, `ftp://`, `data://` 等
- **loopback 拒否**: `localhost`, `127.0.0.1`, `::1`
- **private network 拒否**: `10.x`, `172.16-31.x`, `192.168.x`
- **link-local 拒否**: `169.254.x`

`collect_metrics.py` の `_validate_url()` はこれらを実装済み。

## subprocess

| 箇所 | ファイル | コマンド種別 | shell | 入力種別 | nosec | リスク評価 |
| --- | --- | --- | --- | --- | --- | --- |
| codemap/update.py:129 | tools/codemap/update.py | git diff | False | 固定 args | B603,B607 | 低 |
| aggregate_int.py:84 | tools/ci/aggregate_int.py | git | False | 固定 args | B603,B607 | 低 |
| check_governance_gate.py:22 | tools/ci/check_governance_gate.py | git | False | 固定 args | B603,B607 | 低 |
| check_release_evidence.py:89 | tools/ci/check_release_evidence.py | git | False | 固定 args | B603,B607 | 低 |
| allowlist_guard.py:273 | tools/security/allowlist_guard.py | git | False | 固定 args | B603,B607 | 低 |

### subprocess 安全要件

- **shell=False**: 必須（shell injection 防止）
- **固定 args**: 外部入力を args に渡さない
- **git command のみ**: 現状は git に限定
- **新規追加時**: 台帳へ追記、レビューで shell=False を確認

## file access

| 箇所 | ファイル | 操作種別 | 入力種別 | リスク評価 |
| --- | --- | --- | --- | --- |
| test_audit_scripts.py:25 | tests/tools/audit/test_audit_scripts.py | write | テスト用固定 path | 低（テスト内） |

### file access 安全要件

- **外部入力を path に渡さない**: path traversal 防止
- **書き込み先制限**: repo 内または許可された directory
- **削除操作**: 明示的な承認が必要

現状はテストコード内のみで、外部入力由来の path 操作なし。

## 検出ツール

- **Bandit**: B310, B603, B607 を検出
- **Semgrep**: 追加 rule で URL / subprocess / file を検出可能
- **grep 棚卸し**: 月次で以下を実行:
  ```sh
  grep -r "subprocess" --include="*.py" tools/
  grep -r "urlopen" --include="*.py" tools/
  grep -r "nosec" --include="*.py" tools/
  ```

## 定期棚卸し手順

1. 月次または release 前に grep 検索を実行
2. 新規利用箇所を台帳へ追記
3. 既存箇所の検証手段が有効か確認
4. 不要箇所を削除、台帳を更新

## 関連資料

- [Security Exception Registry](./Security_Exception_Registry.md)
- [Enterprise Readiness Checklist](./Enterprise_Readiness_Checklist.md)
- [Security Review Checklist](./Security_Review_Checklist.md)