# Teacher v3 proposal reporting / proof surfaces

This document defines the **operator-reporting and proof surfaces** for Teacher v3 proposals.

It is intentionally **design-only**. It does **not** claim a new shipped runtime surface exists yet. It maps the future proposal-reporting lane onto the truth surfaces that already ship today, and it keeps the boundary explicit between:

- **already shipped truth surfaces** — the runtime, proof, and docs surfaces that are real today
- **target-state proposal surfaces** — the Teacher v3 bundle/status/reporting surfaces we want to add

The core rule is simple:

> Teacher v3 reporting may summarize and cross-reference truth, but it must not become a new source of truth for the live runtime.

## Surface hierarchy

Teacher v3 should read truth in this order:

1. **runtime truth** — what the live runtime actually reports
2. **proof truth** — what the operator proof bundles capture from the exercised host
3. **docs truth** — what the repo publicly claims and freezes
4. **proposal truth** — what Teacher v3 proposes for compiler/lint/mutation/forgetting lanes

The target-state proof bundle is an overlay on top of the first three surfaces, not a replacement for them.

## Already shipped surfaces

These are the existing surfaces this design must anchor to.

| State | Exact surface | What it already proves | How Teacher v3 should use it |
|---|---|---|---|
| shipped | `src/brain-runtime/service.ts` + `src/brain-cli.ts` via `openclawbrain status --openclaw-home ~/.openclaw --detailed` | live runtime truth such as `serveState`, `currentPackVersion`, `currentPackMetadata`, `teacherConfigured`, `teacherProvider`, `teacherModel`, `teacherConfigError`, `operatorHealth`, `learningHealth`, `lastCompileReportSummary`, `lastAssemblyDecision`, and `lastPrefetchDecision` | use as the canonical runtime snapshot, never as a proposal store |
| shipped | `scripts/capture-openclawbrain-operator-proof.mjs` via `openclawbrain proof --openclaw-home ~/.openclaw` | operator proof bundle truth, including `summary.md`, `steps.json`, `verdict.json`, `extracted-startup-breadcrumbs.log`, and `runtime-load-proof.json` | use as the host-anchored proof bundle that Teacher v3 can cite |
| shipped | `scripts/proof-cron.mjs health` and `scripts/proof-cron.mjs nightly` | proof-health snapshots and aggregates with `status.json`, `snapshot.json`, `summary.md`, `aggregate.json`, `bundle-index.json`, `manifest.json`, and `smoke.json` | use as freshness/health aggregation inputs for proposal reporting |
| shipped | `scripts/verify-proof-smoke.mjs` | frozen proof freshness gate for the public operator proof claim | use as the claim guardrail, not as a Teacher v3 report generator |
| shipped | `docs/EVIDENCE.md`, `docs/RELEASE_CONTRACT.md`, `CLAIMS.md` | public truth boundaries and what is or is not claimed today | use as the docs truth layer when deciding whether proposal surfaces may be described as shipped |
| shipped | `README.md`, `docs/lifecycle.md`, `docs/getting-started/quick-start.md`, `docs/architecture/overview.md`, `docs/architecture/deep-dive.md` | operator-facing narrative for install / status / proof / recovery | use as the human-facing path map that proposal reporting should stay consistent with |

## Target-state surfaces

These are the proposal-reporting surfaces this lane should define, but which are not shipped yet.

| State | Proposed surface | Purpose | Notes |
|---|---|---|---|
| target | `teacher-v3` proof bundle | one reviewable bundle per proposal run | contains both human-readable summary and machine-readable metadata |
| target | `live-proof-rung-1` | first publication-safe proof overlay | before/after evidence surfaces plus token / latency / truth checks; still target-state only |
| target | `summary.md` | concise operator summary | should explain what the proposal is, what truth surfaces it read, and what changed relative to those surfaces |
| target | `status.json` | thin machine status | should stay bounded and report counts/state, not dump raw source payloads |
| target | `proposal-report.json` | machine-readable proposal report | should include proposal lane, proposal class, lineage, status, replay gate dimensions, evidence refs, counterevidence refs, and recommendations |
| target | `canary-rollout.json` | bounded canary plan for proposal classes / candidate packs | must stay off by default, carry `surfaceState: "target"`, and remain separate from the replay gate and live serve path |
| target | `surface-map.json` | shipped-vs-target inventory | should make explicit which referenced surfaces are already shipped and which are target-state only |
| target | `evidence-links.json` | normalized source references | should point back to runtime status, operator proof, proof-cron outputs, and docs truth surfaces |
| target | `verdict.json` | review verdict | should say whether the proposal bundle is reviewable, shadow-only, promotable, rejected, or expired |

## Proposed bundle shape

The proposal bundle should be small, bounded, and explicit about provenance.

```ts
type TeacherV3SurfaceRef = {
  id: string;
  state: "shipped" | "target";
  kind: "runtime_truth" | "proof_truth" | "docs_truth" | "proposal_truth";
  source: string; // command, file, or bundle path
  note?: string;
};

type TeacherV3ProofBundleV1 = {
  bundleId: string;
  proposalId: string;
  lane: "compiler" | "lint" | "mutation" | "forgetting" | "correction";
  status: "draft" | "reviewable" | "shadow" | "promotable" | "promoted" | "rejected" | "expired";
  lineage: {
    basePackId?: string;
    baseGraphHash?: string;
    producerVersion: string;
    promptHash?: string;
    scope: string;
    idempotencyKey: string;
  };
  surfaceMap: TeacherV3SurfaceRef[];
  evidenceLinks: Array<{ refId: string; kind: string; path: string }>;
  counterevidenceLinks?: Array<{ refId: string; kind: string; path: string }>;
  runtimeTruthSnapshot?: Record<string, unknown>;
  proofTruthSnapshot?: Record<string, unknown>;
  docsTruthSnapshot?: Record<string, unknown>;
  recommendations: string[];
  createdAt: string;
};
```

### Bundle rules

- **bounded**: keep the bundle small enough to inspect in one pass
- **read-only**: bundles report on truth surfaces; they do not mutate them
- **explicit**: every surfaced claim must cite a source surface
- **comparative**: the bundle should say what is already shipped vs what is only target-state
- **replayable**: lineage must be strong enough to regenerate or diff the bundle later

Promoted proposal bundles should keep the proposal id, proposal class, rollback key,
replay suites, and an inspectable `surfaceMap` that labels each referenced
surface as `shipped` or `target`. For operator review, the bundle should stay
small enough to inspect without reconstructing proof ad hoc from raw logs.

### First live-proof rung

The first live-proof rung is still **target-state only**. It sits underneath the shipped operator proof lane instead of replacing it.

The rung should add a compact overlay with:

- before and after evidence surfaces, each tagged as `shipped` or `target`
- explicit `token`, `latency`, and `truth` checks with bounded `pass` / `warn` / `fail` status
- publication-safe artifacts only; raw logs and secret-bearing captures stay out of the public bundle
- a surface map that makes the shipped-vs-target split obvious at a glance

Suggested shape:

```ts
type TeacherV3LiveProofRungV1 = {
  rungId: "live-proof-rung-1";
  summary: string;
  beforeSurfaces: TeacherV3SurfaceRef[];
  afterSurfaces: TeacherV3SurfaceRef[];
  checks: Array<{ kind: "token" | "latency" | "truth"; status: "pass" | "warn" | "fail"; summary: string }>;
  publicationSafeArtifacts: Array<{ artifactId: string; kind: string; path: string; redactions: string[] }>;
  shippedStateNotes: string[];
  targetStateNotes: string[];
};
```

## Status semantics

Teacher v3 status should stay thin and operational.

### Shipped status semantics

The current shipped runtime status surface is the canonical source for live truth. It already reports:

- runtime health and serve state
- promoted pack identity
- teacher configuration and error state
- learning / operator health summaries
- compile / assembly / prefetch decision summaries

That status is the right place to answer: **what is the runtime doing right now?**

### Target-state Teacher v3 status semantics

Teacher v3 status should answer different questions:

- which proposal lane is this bundle about?
- which source surfaces were used?
- which surfaces are already shipped vs target-state?
- is the proposal bundle reviewable, shadow-only, promotable, or rejected?
- what follow-on code changes are recommended?

It should **not** try to restate the entire runtime status payload.

## Mapping to the existing surfaces

A Teacher v3 report should be able to say, in one glance, where each fact came from.

### Runtime truth mapping

Use the runtime status surfaces for:

- `serveState`
- `currentPackVersion`
- `currentPackMetadata`
- `teacherConfigured` / `teacherProvider` / `teacherModel` / `teacherConfigError`
- `operatorHealth`
- `learningHealth`
- `lastCompileReportSummary`
- `lastAssemblyDecision`
- `lastPrefetchDecision`
- `routeTraceCount` / `supervisionCount` / other live counters already surfaced by status

### Proof truth mapping

Use the operator proof surfaces for:

- install / restart / status choreography
- startup breadcrumbs
- runtime-load-proof snapshots
- exact bundle verdicts
- host truth for the exercised OpenClaw profile

### Docs truth mapping

Use the frozen docs surfaces for:

- what the repo claims publicly today
- whether a claim is already frozen or still target-state
- whether a new Teacher v3 report may be described as shipped, preview, or design-only

## Recommended storage layout

For development, a Teacher v3 bundle can live under a workspace artifact path such as:

```text
artifacts/teacher-v3-proof/<run-id>/
```

For any frozen/public proof surface, the same bundle shape should mirror the repo's existing evidence layout and live under the evidence tree, with the date and commit SHA made explicit.

The exact path is less important than the contract:

- bundle name is stable
- bundle fields are stable
- source surfaces are cited by exact command or file path
- shipped vs target-state is explicit

## What should be visible in the report

At minimum, the human report should answer:

1. What proposal lane was reviewed?
2. What existing truth surfaces were referenced?
3. Which referenced surfaces are already shipped?
4. Which surfaces are target-state only?
5. What is the recommendation?
6. What follow-on code work should happen next?

## Non-goals

- Do not make Teacher v3 reporting a new live truth source.
- Do not hide shipped vs target-state behind generic wording like “done” or “supported.”
- Do not expand the runtime `status` payload into an unbounded proof dump.
- Do not imply the proposal bundle is already part of the shipped operator proof lane.

## Bottom line

Teacher v3 needs a proof/reporting surface that is explicitly **derived** from the existing runtime, proof, and docs truth layers.

The shipped surfaces already give us the raw ingredients.
The target-state bundle should organize them into a reviewable proposal report without pretending to be the live runtime itself.
