# Manual Black-Box Brief: Five Tool Validation 2026-07-02

## 1. 根拠付き観点

- Release readiness report: `docs/acceptance/AC-20260702-01.md` の criteria と
  `generate_evidence_report.py` のテスト証跡に基づき、release/security/metrics 入力が
  readiness summary に反映されること。
- Security and audit CLI: `tools/audit/verify_log_chain.py` の `--hmac-key` 推奨と `--secret` 互換 alias により、運用互換と静的解析上の誤検知低減が両立すること。
- HATE isolated execution: `tests/test_cli_entrypoints.py` が `pip` 不在の隔離 Python でも console script smoke を実行できること。
- Traceability: Task Seed、Acceptance、requirements、spec、RUNBOOK が five-tool Gate と同じ期待値を示すこと。

## 2. リスク

- P1: workstation-local `.tmp` evidence は長期保全ではないため、release 時は CI artifact へ昇格が必要。
- P2: `--secret` alias は互換のため残るため、将来 release note で非推奨化方針を明示する余地がある。

## 3. 優先度

- P0: なし。
- P1: release approval 前に `docs/evidence/five-tool-validation-20260702/qeg-workflow-cookbook/output-record.json` を確認する。
- P2: 次回 CI 整備で five-tool evidence pack を artifact として保存する。

## 4. 手動テストケース

| ID | 観点 | 手順 | 期待結果 | 根拠 |
| --- | --- | --- | --- | --- |
| MBB-WFC-001 | Audit CLI互換 | `python tools/audit/verify_log_chain.py --help` を確認 | `--hmac-key` と `--secret` alias が表示される | `docs/addenda/G_Security_Privacy.md` |
| MBB-WFC-002 | HATE隔離 | HATE roster の `uv run --with PyYAML --with pytest python -m pytest -q` を実行 | `overall_status: pass`、573 records | HATE report |
| MBB-WFC-003 | QEG専用fixture | `node dist/cli.js validate <fixture>` を実行 | `Validation: PASS` | QEG fixture |

## 5. 工数

- 再実行: 20-30分。
- human spot review: 15分。
- CI artifact 昇格設計: 30-60分。

## 6. Gate 判定

`go`

## 7. Go/No-Go brief

No-Go blocker は残っていない。Code-to-gate readiness、HATE real-repo、pytest、acceptance
trace、QEG workflow-cookbook fixture が揃ったため、このローカル検収範囲では Go とする。
