---
task_id: 20260913-02
intent_id: INT-PACKAGE-VERSION-1.3.0
owner: workflow-cookbook
status: done
---

# パッケージ版を1.3.0へ更新する

[受入](../acceptance/AC-20260913-02.md) / [リリースノート](../releases/v1.3.0.md)

## Scope

v1.2.0以後の後方互換な機能追加をminor releaseとして整理し、パッケージ、lockfile、
README、CHANGELOG、リリースノートを1.3.0へ揃える。正式なtag公開前でも、全version sourceが
一致した単一のrelease candidateをCIで検査できるようにする。

公開tag、GitHub Release、外部package registryへの配布はrelease承認後の工程とする。
本作業では比較評価実験やモデル性能測定を行わず、単体試験を効果測定として扱わない。

## Verification

version consistency checkerのtag済み、tag待ち、不整合、古い版、無関係な未tag文書を試験する。
wheelとsource archiveをbuildし、wheel名、METADATA、LICENSE、公開CLIのsmokeを確認する。
