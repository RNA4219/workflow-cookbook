agent-protocols Evidence notes

Source files
- ../agent-protocols/schemas/Evidence.schema.json
- ../agent-protocols/docs/protocol.md
- ../agent-protocols/docs/operations.md
- ../agent-protocols/examples/evidence.sample.json

Required fields summary
- taskSeedId
- baseCommit
- headCommit
- inputHash
- outputHash
- model
- tools
- environment
- staleStatus
- mergeResult
- startTime
- endTime
- actor
- policyVerdict
- diffHash

Environment object requires
- os
- runtime
- containerImageDigest
- lockfileHash

Review points
- Preserve Evidence required fields.
- Keep actor, commit references, and hashes reproducible.
- If schema expectations change, update workflow-cookbook docs and tests together.
