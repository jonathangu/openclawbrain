# Canonical Frozen 20 Recorded-Session Inputs

This directory freezes the canonical 20-slot recorded-session replay input surface for downstream eval work.

## What is here

- `manifest.json`: canonical slot manifest with category, provenance, sanitization notes, and stable paths
- `manifest.schema.json`: machine-readable schema for the manifest
- `traces/<category>/<slot-id>/trace.json`: the actual replayable `recorded_session_trace.v1` input for each slot

## Truthfulness boundary

As of 2026-04-01, this repo does not contain any checked-in `recorded_session_trace.v1` input with provenance strong enough to call it a verified first-party real production trace.

This freeze therefore uses replayable equivalents only:

- 2 published proof-bundle fixtures already checked into `docs/evidence`
- 5 checked-in test fixtures lifted directly or normalized from dynamic temp-workspace tests
- 13 newly frozen replayable equivalents derived from checked-in docs/tests where no stronger source existed

Every slot records that gap explicitly in `manifest.json` via `realTraceSourceAvailable: false` and its `sourceKind` / `notes` fields.

## Category contract

The set keeps exactly:

- 5 direct-answer traces
- 5 plan/execution traces
- 5 retrieval/memory-heavy traces
- 5 correction/follow-up-heavy traces

## Path contract

The canonical path shape is fixed:

`traces/<category-dir>/<slot-id>/trace.json`

Where `<category-dir>` is one of:

- `direct-answer`
- `plan-execution`
- `retrieval-memory-heavy`
- `correction-follow-up-heavy`

Downstream lanes should treat the slot ids and these paths as stable.

## Verification

The focused regression is:

`npx vitest run test/canonical-frozen-trace-set.test.ts`

That test checks the manifest shape, category counts, provenance summary, and that every frozen trace can round-trip through the recorded-session proof-bundle writer.

For a single trace, you can also write a proof bundle with:

`tsx scripts/validate-recorded-session-replay.ts --trace <path-to-trace.json>`

