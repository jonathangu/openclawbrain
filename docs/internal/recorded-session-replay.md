# Recorded Session Replay Proof Bundles

This lane freezes recorded-session replay results as a curated proof bundle instead of leaving the replay run spread across scratch activation roots.

## Why this exists

`runRecordedSessionReplay()` already computes the semantic replay artifacts:

- `traceHash`
- `fixtureHash`
- `scoreHash`
- `bundleHash`

What it did not do was publish a narrow, reproducible bundle on disk. The proof writer now runs replay in a temporary scratch root and only writes the durable proof surface into the requested artifact directory.

## Artifact layout

Every replay proof bundle uses the same fixed layout:

```text
manifest.json
trace.json
fixture.json
bundle.json
environment.json
summary.md
summary-tables.json
hashes.json
modes/
  no_brain.json
  seed_pack.json
  learned_replay.json
```

The layout is intentionally fixed. `manifest.json` must keep these exact relative paths so the validator can detect drift instead of silently accepting widened output.

## File roles

- `trace.json`: the sanitized recorded-session source trace.
- `fixture.json`: the deterministic replay fixture derived from the trace.
- `bundle.json`: the replay result across `no_brain`, `seed_pack`, and `learned_replay`.
- `environment.json`: narrow writer/runtime facts for the proof run. This is observational metadata, not part of the semantic replay hash contract.
- `summary.md`: short human-readable proof summary.
- `summary-tables.json`: ranking plus per-mode and per-turn table rows.
- `modes/*.json`: exact per-mode reports lifted from `bundle.json`.
- `hashes.json`: semantic hashes plus file-content digests for the curated artifact set.

## Hash contract

Two hash classes matter:

1. Semantic hashes

- `traceHash`
- `fixtureHash`
- `scoreHash`
- `bundleHash`

These prove the replay content itself.

2. File digests

`hashes.json` also records content digests for the written proof files plus `manifest.json`. This lets the validator catch on-disk drift even when the semantic bundle object still parses.

`hashes.json` does not hash itself. That avoids a self-reference cycle.

## Validation path

Repo helper:

```bash
tsx scripts/validate-recorded-session-replay.ts --trace path/to/recorded-trace.json
```

The script writes the proof bundle, writes `validation-report.json`, and exits nonzero if:

- the manifest layout drifts
- fixture rebuild no longer matches the trace
- bundle or score hashes fail verification
- summary tables drift from `bundle.json`
- per-mode files drift from `bundle.json`
- file digests no longer match the written artifacts

## Determinism boundary

The published proof bundle is deterministic with respect to the replay inputs and the fixed writer layout.

The replay execution scratch root is temporary and deleted after the run. That scratch state is intentionally not part of the durable proof bundle.

`environment.json` records the local writer runtime (`nodeVersion`, `platform`, `arch`). That file is expected to vary if the proof is regenerated on a different machine, but the semantic replay hashes should remain stable when the replay inputs are unchanged.
