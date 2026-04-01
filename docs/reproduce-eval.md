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

## Run the lane writer for multiple traces

When you want a rerunnable corpus-level proof surface with aggregate tables and worked traces:

```bash
npm run proof:replay-lane
```

That defaults to the canonical frozen manifest at:

`evals/recorded-session-replay/canonical-frozen-20/manifest.json`

The canonical set is currently equivalent-only, not first-party real-trace-backed. Keep that truth boundary attached to any claims made from the replay lane.

You can still override the input set explicitly:

```bash
tsx scripts/build-recorded-session-replay-lane.ts \
  --trace path/to/trace-a.json \
  --trace path/to/trace-b.json
```

or with an explicit manifest:

```bash
tsx scripts/build-recorded-session-replay-lane.ts \
  --trace-manifest path/to/replay-manifest.json
```

The lane writer keeps the per-trace bundles under `recorded-session-replay/<trace-id>/` and writes aggregate artifacts under `recorded-session-replay/_lane/`.

## Run the frozen eval gate

To score the same canonical frozen set as one eval gate:

```bash
npm run proof:frozen-eval-gate
```

That defaults to the same canonical manifest and writes:

```text
docs/evidence/YYYY-MM-DD/<git-sha>/frozen-recorded-session-eval/manifest/
```

By default, the gate hard-checks replay validity and quality. It still reports `qualityAdjustedPromptSavingsUsd`, but treats that prompt-cost proxy as observational unless you explicitly set `--min-quality-adjusted-prompt-savings-usd`. That matches the product framing: this equivalent-only replay lane does not prove long-run task-level economics or fewer raw LLM/API calls over time.

Override the manifest or output path if needed:

```bash
tsx scripts/run-frozen-recorded-session-eval-gate.ts \
  --manifest path/to/replay-manifest.json \
  --output-dir path/to/out
```

If you want an explicit prompt-cost floor for a local experiment, add:

```bash
tsx scripts/run-frozen-recorded-session-eval-gate.ts \
  --min-quality-adjusted-prompt-savings-usd 0
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

For the corpus lane, inspect:

- `_lane/README.md`
- `_lane/summary-tables.json`
- `_lane/pairwise-deltas.json`
- `_lane/win-rate-matrix.json`
- `_lane/worked-traces.md`
- `_lane/generation-report.json`

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

For the corpus lane:

1. Re-run `build-recorded-session-replay-lane.ts` with the same trace list or manifest.
2. Compare `_lane/summary-tables.json`, `_lane/pairwise-deltas.json`, and `_lane/win-rate-matrix.json`.
3. Use `_lane/worked-traces.md` to inspect the highest-spread traces first.
4. Check `_lane/generation-report.json` before trusting the aggregate view if any trace bundle failed validation.

## Fast contract test

The focused package test for this lane is:

```bash
node --test packages/cli/dist/test/recorded-session-replay-proof-bundle.test.js
```

That test freezes the curated artifact layout, reproducibility across output roots, and validator failure on per-mode drift.

The lane-level aggregate test is:

```bash
npx vitest run test/replay-proof-lane.test.ts
```
