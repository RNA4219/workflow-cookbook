---
intent_id: INT-SEC-014
owner: security
status: active
last_reviewed_at: 2026-04-20
next_review_due: 2026-05-20
---

# Dependency Tree

本ドキュメントは workflow-cookbook の依存関係（transitive dependencies）を可視化する。

生成コマンド: `python tools/security/visualize_dependencies.py`

## Runtime Dependencies

直接依存（requirements.txt）:

| Package | Version |
| --- | --- |
| pyyaml | pinned |

### Transitive Dependencies

(pipdeptree で transitive deps を取得)

## Dev Dependencies

直接依存（pyproject.toml dev）:

| Package | Version |
| --- | --- |
| pytest | pinned |
| pytest-cov | pinned |
| bandit | pinned |
| pip-audit | pinned |

### Transitive Dependencies

- **bandit** `1.9.4` (required: `1.9.4`)
  - colorama `0.4.6`
  - PyYAML `6.0.3`
  - rich `14.3.3`
    - markdown-it-py `4.0.0`
      - mdurl `0.1.2`
    - Pygments `2.19.2`
  - stevedore `5.7.0`
- **pip_audit** `2.10.0` (required: `2.10.0`)
  - CacheControl `0.14.4`
    - msgpack `1.1.2`
    - requests `2.32.5`
      - certifi `2026.2.25`
      - charset-normalizer `3.4.6`
      - idna `3.11`
      - urllib3 `2.6.3`
  - cyclonedx-python-lib `11.7.0`
    - license-expression `30.4.4`
      - boolean.py `5.0`
    - packageurl-python `0.17.6`
    - py-serializable `2.1.0`
      - defusedxml `0.7.1`
    - sortedcontainers `2.4.0`
    - typing_extensions `4.15.0`
  - packaging `26.0`
  - pip-api `0.0.34`
    - pip `26.0.1`
  - pip-requirements-parser `32.0.1`
    - packaging `26.0`
    - pyparsing `3.3.2`
  - platformdirs `4.9.6`
  - requests `2.32.5`
    - certifi `2026.2.25`
    - charset-normalizer `3.4.6`
    - idna `3.11`
    - urllib3 `2.6.3`
  - rich `14.3.3`
    - markdown-it-py `4.0.0`
      - mdurl `0.1.2`
    - Pygments `2.19.2`
  - tomli `2.4.1`
  - tomli_w `1.2.0`
- **pytest-cov** `7.0.0` (required: `7.0.0`)
  - coverage `7.13.4`
  - pluggy `1.6.0`
  - pytest `9.0.2`
    - colorama `0.4.6`
    - iniconfig `2.3.0`
    - packaging `26.0`
    - pluggy `1.6.0`
    - Pygments `2.19.2`

## 可視化方法

### pipdeptree

```bash
pip install pipdeptree
pipdeptree --json-tree > .ga/dependency-tree.json
```

### requirements.txt ベース

本プロジェクトは `requirements.txt` を lockfile 相当として扱う。
transitive dependencies は pipdeptree で可視化し、
docs/security/Dependency_Tree.md に記録。

## 参照

- [Dependency Governance](./Dependency_Governance.md)
- [Enterprise Readiness Assessment](../reports/enterprise-readiness-assessment-20260417.md)


