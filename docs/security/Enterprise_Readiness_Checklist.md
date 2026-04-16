---
intent_id: INT-SEC-002
owner: security
status: active
last_reviewed_at: 2026-04-17
next_review_due: 2026-05-17
---

# Enterprise Readiness Checklist

このチェックリストは、`workflow-cookbook` が
「個別修正が入っている」状態から、
「企業利用で説明可能な品質・統制・証跡を継続できる」状態へ
近づいているかを判定するための実務用チェックリストです。

上場企業相当の利用を想定する場合、
単一の脆弱性修正だけでは不十分です。
少なくとも、
安全な実装、CI の検出力、供給網の再現性、リリース証跡、
インシデント運用の 5 系統で継続的に説明できる必要があります。

## 評価基準

- `A`: 標準化され、CI / docs / 証跡まで揃い、例外も説明可能
- `B`: 主要な仕組みはあり、運用できるが、一部は手作業または証跡不足
- `C`: 個別対策はあるが、継続運用や監査説明に弱い
- `D`: 仕組みや責任分界がほぼ未整備

## 1. Secure Coding

- [ ] SSRF / local file read / command execution など危険パターンに対し、
  安全側デフォルトの実装が入っている
- [ ] 同種パターンを repo 横断で再点検し、
  個別修正で終わらせていない
- [ ] 危険 API 利用には suppress ではなく、
  妥当性検証・到達先制限・理由コメントがある
- [ ] 失敗系テストが実装され、
  `allowed` / `blocked` の境界条件が明文化されている

完了条件:
- 個別の脆弱性修正だけでなく、
  同カテゴリの利用箇所を一覧化して説明できる

## 2. CI Security Gates

- [ ] `Bandit` / `Semgrep` / secret scan / dependency audit の
  役割分担が明確
- [ ] 重大リスクに関する rule が広く除外されていない
- [ ] `nosec` や skip の理由をレビューで追跡できる
- [ ] branch protection 上の必須チェックと docs が一致している
- [ ] security gate 失敗時の扱い
  （fail / warn / 例外承認）が説明可能

完了条件:
- 「なぜその check が必要か」「なぜその例外が許されるか」を
  docs と設定の両方で説明できる

## 3. Supply Chain / Dependency Governance

- [ ] 依存監査が固定入力
  （`requirements.txt` / lockfile 等）ベースで再現可能
- [ ] lockfile または同等の再現性担保がある
- [ ] `Dependabot` 等で継続的な更新導線がある
- [ ] SBOM 生成または同等の依存可視化がある
- [ ] 重大脆弱性の修正 SLA / 運用責任が決まっている

完了条件:
- 依存関係の「今の構成」「更新方法」「脆弱性対応ルール」を
  第三者へ説明できる

## 4. Release / Change Management

- [ ] リリースチェックリストが存在し、更新されている
- [ ] 変更点、既知制約、受入記録、リリースノートが連動している
- [ ] rollback 手順が実行可能な粒度で書かれている
- [ ] release evidence を検証する手段がある
- [ ] タグ / changelog / docs/releases の整合が確認できる

完了条件:
- いつ、何を、誰が、どの証跡で出したかを追跡できる

## 5. Ops / Incident Readiness

- [ ] `RUNBOOK.md` に日常運用・異常時確認・検収導線がある
- [ ] incident template / sample incident があり、
  再発防止へつながる
- [ ] 監視・メトリクスの見方が docs 化されている
- [ ] 重大障害時の rollback / communication 導線がある
- [ ] 定期レビュー期限や docs freshness を確認する仕組みがある

完了条件:
- 障害やセキュリティ事象が起きたときに、
  属人的な判断だけでなく手順と証跡で追える

## 6. Documentation / Auditability

- [ ] 実装、Task Seed、Acceptance、Runbook、Release docs が相互参照されている
- [ ] `last_reviewed_at` / `next_review_due` が維持されている
- [ ] 重要方針が README だけでなく、
  運用 docs に落ちている
- [ ] 監査時に必要な証跡の所在が分かる

完了条件:
- 担当者が変わっても、
  docs から判断経路と運用責任を復元できる

## 判定の目安

- `A` 判定:
  6 セクション中 5 以上が `A` または `B` で、
  `C` が 1 以下
- `B` 判定:
  重大リスクに対する対策と最低限の運用証跡はあるが、
  supply chain / release / incident のどこかに宿題が残る
- `C` 判定:
  個別対策はあるが、
  継続運用・監査説明・再現性に弱い

## 関連資料

- [Security Review Checklist](./Security_Review_Checklist.md)
- [Security Exception Registry](./Security_Exception_Registry.md)
- [Security API Inventory](./Security_API_Inventory.md)
- [Branch Protection Operation](./Branch_Protection_Operation.md)
- [CHECKLISTS](../../CHECKLISTS.md)
- [RUNBOOK](../../RUNBOOK.md)
- [Release Checklist](../Release_Checklist.md)
