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
coverage-snapshot.json
hardening-snapshot.json
hashes.json
modes/
  no_brain.json
  vector_only.json
  graph_prior_only.json
  learned_route.json
```

The layout is intentionally fixed. `manifest.json` must keep these exact relative paths so the validator can detect drift instead of silently accepting widened output.

## File roles

- `trace.json`: the sanitized recorded-session source trace.
- `fixture.json`: the deterministic replay fixture derived from the trace.
- `bundle.json`: the replay result across `no_brain`, `vector_only`, `graph_prior_only`, and `learned_route`.
- `environment.json`: narrow writer/runtime facts for the proof run. This is observational metadata, not part of the semantic replay hash contract.
- `summary.md`: short human-readable proof summary.
- `summary-tables.json`: ranking plus per-mode and per-turn table rows.
- `coverage-snapshot.json`: aggregate compile/phrase-hit coverage across replay modes.
- `hardening-snapshot.json`: aggregate warnings, compile failures, promotions, and export/attribution counts across replay modes.
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
- coverage or hardening snapshots drift from `bundle.json`
- per-mode files drift from `bundle.json`
- file digests no longer match the written artifacts

## Determinism boundary

The published proof bundle is deterministic with respect to the replay inputs and the fixed writer layout.

The replay execution scratch root is temporary and deleted after the run. That scratch state is intentionally not part of the durable proof bundle.

`environment.json` records the local writer runtime (`nodeVersion`, `platform`, `arch`). That file is expected to vary if the proof is regenerated on a different machine, but the semantic replay hashes should remain stable when the replay inputs are unchanged.

## Lane-level aggregate artifacts

When you need to rerun a corpus instead of a single trace, use the lane writer:

```bash
tsx scripts/build-recorded-session-replay-lane.ts \
  --trace path/to/trace-a.json \
  --trace path/to/trace-b.json
```

That keeps the existing per-trace bundle contract intact under:

```text
docs/evidence/YYYY-MM-DD/<git-sha>/recorded-session-replay/<trace-id>/
```

and adds one aggregate surface under:

```text
docs/evidence/YYYY-MM-DD/<git-sha>/recorded-session-replay/_lane/
  README.md
  index.json
  summary-tables.json
  pairwise-deltas.json
  win-rate-matrix.json
  worked-traces.md
  generation-report.json
```

The `_lane/` directory is intentionally separate from the per-trace bundle roots so proof-cron and other bundle scanners do not confuse the aggregate view with a single replay bundle.

### Aggregate file roles

- `README.md`: compact operator-facing overview of mode totals and pairwise results.
- `index.json`: machine-readable lane index, bundle listing, and assumptions.
- `summary-tables.json`: stable cross-trace mode, trace, and per-turn tables in fixed mode order.
- `pairwise-deltas.json`: left-minus-right aggregate deltas plus per-trace pairwise records.
- `win-rate-matrix.json`: trace-level and turn-level win/loss/tie matrices for every mode pairing.
- `worked-traces.md`: short human-readable worked examples sorted by score spread.
- `generation-report.json`: rerun report with the exact bundle validation outcomes and any failed traces.

### Canonical manifest contract

The canonical replay corpus now lives at:

`evals/recorded-session-replay/canonical-frozen-20/manifest.json`

That manifest uses contract `canonical_recorded_session_trace_set_manifest.v1` and stable trace paths from `entries[].path`.

The lane writer defaults to that manifest when no explicit `--trace` or `--trace-manifest` input is provided:

```bash
npm run proof:replay-lane
```

The frozen set is still truthfully equivalent-only, not first-party real-trace-backed. That boundary is recorded in the manifest itself under `realTraceCoverage.summary`.

Compatibility fallback for ad hoc manifests remains available, but the canonical lane-c manifest is the primary replay/eval contract.
