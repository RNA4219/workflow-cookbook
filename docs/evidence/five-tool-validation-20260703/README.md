# Five-tool validation evidence - 2026-07-03

This evidence pack records a cross-repo five-tool validation manifest for
`workflow-cookbook`.

## Scope

- Skill: `five-tool-validation-gate`
- Chain: RanD -> Code-to-gate -> HATE -> manual-bb-test-harness -> QEG
- Target: `workflow-cookbook` chain run manifest and cross-repo contract evidence
- Source evidence: `docs/evidence/five-tool-validation-20260702/`

## Commands

```sh
uv run python tools/ci/five_tool_manifest.py generate --config examples/five-tool-chain-manifest.sample.json --out docs/evidence/five-tool-validation-20260703/five-tool-run-manifest.json --validate
uv run python tools/ci/five_tool_manifest.py validate --manifest docs/evidence/five-tool-validation-20260703/five-tool-run-manifest.json --json
```

## Result

- Manifest validation: `ok`
- Errors: none
- Warnings:
  - `hate` repo was dirty with 44 changes at capture time.
  - `manual-bb:go_no_go_brief` is Markdown evidence and has no joinable trace IDs.

The warnings are recorded as observability signals. They do not override the
QEG `go` verdict or the policy hash check for this fixture-backed validation.
