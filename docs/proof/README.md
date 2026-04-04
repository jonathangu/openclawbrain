# Proof packaging

This page packages the repo-side public/operator proof surfaces, including the shipped operator proof lane and the target-state Teacher v3 bundle.

It is a map, not a claim: **shipped** surfaces are current truth, **target** surfaces are review-only Teacher v3 outputs, and **example** surfaces are docs-only scaffolding.

## Legend

- **shipped** — already exists in the repo or in checked-in evidence; safe to point at as current truth.
- **target** — derived review surface; useful for packaging and review, not live authority.
- **example** — intentionally synthetic scaffold; do not treat it as proof.

## Shipped proof surfaces

| State | Surface | What it shows | Notes |
| --- | --- | --- | --- |
| shipped | `openclawbrain status --openclaw-home ~/.openclaw --detailed` | live runtime truth | canonical runtime snapshot |
| shipped | `openclawbrain proof --openclaw-home ~/.openclaw` | host-anchored operator proof bundle | durable bundle with `summary.md`, `steps.json`, `verdict.json`, and runtime-load-proof truth |
| shipped | `docs/evidence/YYYY-MM-DD/<git-sha>/` | frozen proof bundle snapshots | checked-in evidence tree; see `docs/evidence/README.md` |
| shipped | `scripts/verify-proof-smoke.mjs` | proof freshness gate | only enforces when the repo still advertises frozen proof claims |
| shipped | `docs/internal/recorded-session-replay.md` | replay proof bundle layout | shows the stable proof-bundle contract and worked-trace lane |
| shipped | `docs/architecture/teacher-v3-proof.md` | Teacher v3 proof contract | explicitly maps shipped vs target-state truth |

## Target-state Teacher v3 packaging

| State | Surface | What it shows | Notes |
| --- | --- | --- | --- |
| target | `scripts/teacher-v3-proof-bundle.mjs` | Teacher v3 bundle writer | code exists, but the emitted bundle is a derived review surface, not live authority |
| target | `artifacts/teacher-v3-proof/<run-id>/` | workspace output root | development bundle root for review runs |
| target | `summary.md` | human summary | bounded, comparative, publication-safe |
| target | `status.json` | machine summary | counts and state only; no raw payload dump |
| target | `surface-map.json` | shipped-vs-target inventory | each referenced surface is explicitly labeled |
| target | `proposal-report.json` | machine-readable proposal report | includes lineage, evidence links, and recommendations |
| target | `verdict.json` | review verdict | reviewable, shadow-only, promotable, rejected, or expired |

The current writer can seed the bundle from runtime capture when Gate 1 proposal persistence is absent. That is a seam, not a shipped claim, and the bundle should say so plainly in `gate1Seam.note`.

## Worked examples

These examples are for packaging and review only.

### Docs-only scaffold pack

- `artifacts/fixtures/compiled-artifacts/target-state-scaffold/README.md`
- `artifacts/fixtures/compiled-artifacts/target-state-scaffold/pack.manifest.json`
- `artifacts/fixtures/compiled-artifacts/target-state-scaffold/artifacts/ca_example_concept_substrate_01/artifact.md`
- `artifacts/fixtures/compiled-artifacts/target-state-scaffold/artifacts/ca_example_map_of_territory_01/artifact.md`
- `artifacts/fixtures/compiled-artifacts/target-state-scaffold/artifacts/ca_example_provenance_gap_report_01/artifact.md`

This pack is derived from docs only. It is **not** a runtime proof bundle, **not** a promoted pack, and **not** a replacement for live authority.

### Frozen host/operator proof example

- `docs/evidence/2026-03-16/4ccd71a22418b9170128b8d948f5a95801a10380/`
- `docs/evidence/README.md`
- `docs/EVIDENCE.md`

This is the checked-in shipped operator proof lane. Use it as the honest example of what a frozen public proof bundle looks like today.

## Honest boundary

If a page, bundle, or artifact is listed under **target** or **example**, do not present it as shipped operator truth.

Shipped operator truth comes from runtime status, operator proof, and frozen evidence bundles. The Teacher v3 package is derived from those surfaces and should stay explicit about the boundary.
