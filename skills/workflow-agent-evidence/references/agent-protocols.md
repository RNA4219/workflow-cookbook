# agent-protocols Evidence Notes

## Source Files

| File | Content |
|------|---------|
| `agent-protocols/schemas/Evidence.schema.json` | Evidence JSON schema |
| `agent-protocols/docs/protocol.md` | Protocol specification |
| `agent-protocols/docs/operations.md` | Operational details |
| `agent-protocols/examples/evidence.sample.json` | Sample Evidence record |

## Required Fields

### Top-Level

| Field | Type | Description |
|-------|------|-------------|
| `taskSeedId` | string | Task Seed identifier |
| `baseCommit` | string | Base commit SHA |
| `headCommit` | string | Head commit SHA |
| `inputHash` | string | Input content hash |
| `outputHash` | string | Output content hash |
| `diffHash` | string | Diff content hash |
| `model` | string | Model identifier |
| `tools` | array | Tools used |
| `environment` | object | Runtime environment |
| `staleStatus` | string | Stale detection result |
| `mergeResult` | string | Merge outcome |
| `startTime` | string | ISO 8601 timestamp |
| `endTime` | string | ISO 8601 timestamp |
| `actor` | string | Actor identifier |
| `policyVerdict` | string | Policy check result |

### Environment Object

| Field | Type | Description |
|-------|------|-------------|
| `os` | string | Operating system |
| `runtime` | string | Runtime version |
| `containerImageDigest` | string | Container digest (if applicable) |
| `lockfileHash` | string | Lockfile hash |

## Review Checklist

- [ ] All required fields present
- [ ] Commit references are valid SHA
- [ ] Hashes are reproducible
- [ ] Timestamps are ISO 8601 format
- [ ] Actor matches expected identity
- [ ] Environment reflects actual runtime

## Schema Alignment

When Evidence schema changes:

1. Update `tools/protocols/evidence_bridge.py`
2. Update `schemas/inference-plugin-config.schema.json`
3. Update `examples/inference_plugins.agent_protocol.sample.json`
4. Update `examples/agent_protocol_evidence_consumer.sample.py`
5. Update tests in `tests/test_agent_protocol_evidence.py`
6. Sync `docs/requirements.md`, `docs/spec.md`, `docs/design.md`

## Validation

```sh
# Validate Evidence record structure
python -c "
import json
from jsonschema import validate
schema = json.load(open('schemas/inference-plugin-config.schema.json'))
instance = json.load(open('examples/inference_plugins.agent_protocol.sample.json'))
validate(instance, schema)
print('Valid')
"
```
