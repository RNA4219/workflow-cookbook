---
intent_id: INT-SEC-006
owner: security
status: active
last_reviewed_at: 2026-04-17
next_review_due: 2026-05-17
---

# Security Exception Registry

本ドキュメントは `nosec` / `Bandit skip` / `Semgrep skip` などの
セキュリティ検出例外を追跡管理する台帳である。
「広く黙らせる」のではなく、「理由・責任・期限」を持つ形で継続管理する。

## 例外追加の承認プロセス

1. 新規例外追加時は PR で以下を提示する:
   - 対象コード箇所（ファイル:行番号）
   - 除外対象の rule ID（B310, B603 等）
   - 除外理由（安全側デフォルト、固定入力、到達先制限 等）
   - 責任者（レビュー承認者）
   - 見直し期限（最大 1 年、高リスクは 3 月）
2. `docs/security/Security_Exception_Registry.md` の台帳へ追記する。
3. CI で台帳と実際の nosec / skip を照合する（将来実装予定）。

## 例外台帳

### nosec B310 (urlopen)

| 箇所 | ファイル | 行 | 理由 | 責任者 | 見直し期限 | 状態 |
| --- | --- | --- | --- | --- | --- | --- |
| collect_metrics.py:1238 | tools/perf/collect_metrics.py | 1238 | URL 検証済み（`_validate_url()` で危険スキーム拒否） | security team | 2026-07-17 | active |
| collect_metrics.py:1287 | tools/perf/collect_metrics.py | 1287 | URL 検証済み（`_validate_url()` で危険スキーム拒否） | security team | 2026-07-17 | active |
| check_security_posture.py:48 | tools/ci/check_security_posture.py | 48 | 固定 host（`api.github.com`）、token auth | security team | 2026-07-17 | active |
| check_security_posture.py:63 | tools/ci/check_security_posture.py | 63 | 固定 host（`api.github.com`）、token auth | security team | 2026-07-17 | active |
| check_release_evidence.py:116 | tools/ci/check_release_evidence.py | 116 | 固定 host（`api.github.com`）、token auth | security team | 2026-07-17 | active |

### nosec B603, B607 (subprocess without shell)

| 箇所 | ファイル | 行 | 理由 | 責任者 | 見直し期限 | 状態 |
| --- | --- | --- | --- | --- | --- | --- |
| codemap/update.py:129 | tools/codemap/update.py | 129 | git diff、固定 args、`shell=False` | security team | 2026-07-17 | active |
| aggregate_int.py:84 | tools/ci/aggregate_int.py | 84 | git command、固定 args、`shell=False` | security team | 2026-07-17 | active |
| check_governance_gate.py:22 | tools/ci/check_governance_gate.py | 22 | git command、固定 args、`shell=False` | security team | 2026-07-17 | active |
| check_release_evidence.py:89 | tools/ci/check_release_evidence.py | 89 | git command、固定 args、`shell=False` | security team | 2026-07-17 | active |
| allowlist_guard.py:273 | tools/security/allowlist_guard.py | 273 | git command、固定 args、`shell=False` | security team | 2026-07-17 | active |

### Bandit workflow skip

| 除外 rule | workflow 箇所 | 理由 | 責任者 | 見直し期限 | 状態 |
| --- | --- | --- | --- | --- | --- |
| B101 (assert_used) | security.yml:52 | test code 用、pytest で安全 | security team | 2026-07-17 | active |
| B105-108 (hardcoded_password*) | security.yml:52 | 既知制約、認証系では未使用 | security team | 2026-07-17 | review |
| B404 (import_subprocess) | security.yml:52 | subprocess 利用箇所は B603 で個別管理 | security team | 2026-07-17 | active |
| B603 (subprocess_without_shell) | security.yml:52 | git command 固定 args、個別 nosec あり | security team | 2026-07-17 | active |

**注**: B310 (urlopen) は workflow で除外されていない。
urlopen 系はコード内で個別 nosec と URL 検証を併用している。

## 定期見直し手順

1. `next_review_due` 到達時に以下を実施:
   - 各例外の妥当性再確認
   - コード変更による影響評価
   - 不要例外の削除
2. 見直し結果を台帳へ記録（状態: active / review / removed）
3. `Security_Exception_Registry.md` の front matter の
   `last_reviewed_at` / `next_review_due` を更新

## 関連資料

- [Enterprise Readiness Checklist](./Enterprise_Readiness_Checklist.md)
- [Security Review Checklist](./Security_Review_Checklist.md)
- [SAC.md](./SAC.md)
- [ci-config](../ci-config.md)
