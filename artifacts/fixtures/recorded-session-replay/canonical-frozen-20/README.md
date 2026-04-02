# Frozen Fixture Manifest Scaffold

This directory freezes the canonical 20-slot fixture selection for comparative replay work without checking in 20 `fixture.json` payloads yet.

## What is here

- `manifest.json`: frozen scaffold keyed to the canonical recorded-session trace manifest
- `manifest.schema.json`: machine-readable schema for the scaffold contract

## What is not here

The actual `fixtures/<family>/<slot-id>/fixture.json` files are intentionally absent in this lane.

This scaffold only freezes:

- the exact 20 slots
- the fixed 5/5/5/5 trace-family split
- the canonical trace paths and `traceHash` values
- the per-slot `selectionHash` values for scaffold metadata
- the deterministic future fixture-hash rule
- the stable future fixture path template

## Immutability contract

Any change to slot membership, family assignment, trace content, expected fixture hash, or future fixture path requires a new manifest regeneration rather than an in-place edit.

## Commands

Write the checked-in scaffold:

```bash
node --experimental-transform-types scripts/eval/frozen-fixture-manifest.ts write
```

Validate the scaffold against the canonical trace set:

```bash
node --experimental-transform-types scripts/eval/frozen-fixture-manifest.ts validate
```
