# Graphify scheduler and registry

Status: repo-native off-path scheduler note.

This document defines the periodic Graphify scheduler tranche that sits on top of the shipped Graphify bridge.

The scheduler is intentionally **off the serve path**. It exists to make Graphify runs periodic, inspectable, replayable, and registry-linked without turning Graphify into current-truth authority.

## 1) Cadences

Graphify scheduler runs come in two explicit cadences:

### Delta cadence
- targets recent new material
- prefers short feedback cycles
- useful for fresh source bundles, new import slices, and quick drift capture

### Reorg cadence
- targets older and broader material
- prefers slower, wider review cycles
- useful for maintenance reorganization, stale coverage, and long-tail provenance cleanup

Both cadences stay in the same off-path Graphify family. The difference is the review window, not the truth boundary.

## 2) Required entrypoints

The scheduler tranche exposes two operator entrypoints:

- `npm run graphify:delta-cadence`
- `npm run graphify:reorg-cadence`

Each entrypoint should write a per-run bundle under the scheduler output root and update the persistent registry.

## 3) Persistent registry

The registry is the authoritative pointer table for scheduler runs.

Expected registry fields:
- cadence
- run id
- generated timestamp
- source bundle link
- Graphify run bundle link
- compiled-artifact pack link
- import-slice / candidate-pack-input link
- deterministic lint link
- maintenance-diff link
- truth boundary flags
- downstream replayability notes

Registry location:
- `artifacts/graphify-scheduler/registry.json`

Each cadence run root should also carry its own `retention-policy.json` and `retention-policy.md` so the run can be replayed without depending on a later overwrite.

The registry must keep the downstream artifacts linked so a run can be inspected and replayed without reconstructing provenance from scratch.

## 4) Retention rules

Retention is registry-linked, not serve-path-driven.

### source bundles
- replay inputs
- retained while referenced by the registry
- removable only after an explicit vacuum step once unlinked

### import slices
- candidate-pack inputs
- retained while referenced by the registry
- removable only after an explicit vacuum step once unlinked

### candidate-pack inputs
- replay seeds
- retained while referenced by the registry
- removable only after an explicit vacuum step once unlinked

### lint / diff outputs
- diagnostic records
- retained while referenced by the registry
- removable only after an explicit vacuum step once unlinked

The scheduler should never quietly prune a referenced artifact.

## 5) Truth boundary

Scheduler outputs stay below stronger truth layers:

1. runtime truth
2. proof truth
3. docs truth
4. Graphify derived artifacts

That means scheduler outputs can be inspected, replayed, and reviewed, but they do not become current truth on their own.

## 6) Output shape

A run root should expose the usual off-path surfaces, ideally with direct child directories named:

- `source-bundle/`
- `run/`
- `compiled/`
- `import/`
- `lints/`
- `maintenance/`

The run root should also contain bounded human and machine summaries such as:

- `summary.md`
- `status.json`
- `registry-entry.json`
- `retention-policy.json`
- `retention-policy.md`

Those surfaces should point back to the registry and the upstream source bundle hash.

## 7) Inspection rule

If a Graphify scheduler run cannot be replayed from the registry plus its linked downstream artifacts, the scheduler surface is incomplete.

That is the bar for this tranche: not cleverness, but stable provenance.
