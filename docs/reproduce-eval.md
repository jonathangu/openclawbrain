# Reproduce Recorded Replay Eval

This repo's recorded-session replay proof lane is the reproducible evaluation surface for sanitized session traces.

## Inputs

You need one sanitized trace JSON with contract `recorded_session_trace.v1`.

The trace must already include:

- stable timestamps
- sanitized privacy notes
- seed cues
- replay turns with expected context phrases

## Run the proof writer

From the repo root:

```bash
tsx scripts/validate-recorded-session-replay.ts --trace path/to/recorded-trace.json
```

Optional explicit output dir:

```bash
tsx scripts/validate-recorded-session-replay.ts \
  --trace path/to/recorded-trace.json \
  --artifact-dir docs/evidence/2026-03-26/<git-sha>/recorded-session-replay/<trace-id>
```

Without `--artifact-dir`, the script writes to:

```text
docs/evidence/YYYY-MM-DD/<git-sha>/recorded-session-replay/<trace-id>/
```

## What to inspect

Start with:

- `summary.md`
- `validation-report.json`
- `hashes.json`
- `coverage-snapshot.json`
- `hardening-snapshot.json`

Then inspect:

- `bundle.json` for the full replay result
- `summary-tables.json` for compact ranking and per-turn tables
- `modes/learned_route.json` when the learned lane is the interesting delta

## Expected success conditions

A healthy run should leave `validation-report.json` with:

- `ok: true`
- `fileHashesMatch: true`
- `bundleHashMatches: true`
- `scoreHashMatches: true`

## Regression workflow

1. Re-run the same trace through the script.
2. Compare `hashes.json`, `summary-tables.json`, `coverage-snapshot.json`, and `hardening-snapshot.json` with the previous proof bundle.
3. If semantic hashes changed, inspect `bundle.json` and the per-mode files before claiming the replay contract moved.
4. If only file digests changed, inspect `summary.md`, `manifest.json`, and `modes/*.json` for writer drift.

## Fast contract test

The focused package test for this lane is:

```bash
node --test packages/cli/dist/test/recorded-session-replay-proof-bundle.test.js
```

That test freezes the curated artifact layout, reproducibility across output roots, and validator failure on per-mode drift.
