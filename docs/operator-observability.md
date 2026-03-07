# Operator Observability

This is the operator-facing diagnostics contract for attached OpenClawBrain installs.

## What operators must be able to prove

The public observability surface should answer six questions clearly and continuously:

- **health**: are the active and candidate slots activation-ready?
- **promotion safety**: can a pointer move happen safely right now?
- **freshness**: what exact snapshot/export/build is active right now, and is a staged candidate fresher?
- **supervision freshness**: what local source streams and teacher signals most recently shaped the learned pack?
- **route evidence**: did the served compile actually use the promoted pack's learned `route_fn` when required?
- **fallback**: if selection fell back, is that fact explicit?

## Repo proof

Run the dedicated observability smoke after install:

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm observability:smoke
pnpm observability:report
```

Those repo-local lanes exercise the public package surface only on temporary activation state and assert:

- `inspectActivationState()` reports healthy active and candidate slots
- `describeActivationObservability()` surfaces promotion freshness, learned `route_fn` freshness/version, and graph-dynamics freshness
- `describeNormalizedEventExportObservability()` surfaces supervision freshness by source plus the freshest local teacher signal
- `describeActivationTarget()` surfaces the promoted pack id, workspace snapshot, workspace revision, event range, export digest, route policy, and build time
- `compileRuntimeFromActivation()` plus `describeCompileFallbackUsage()` emit stable route evidence and explicit fallback usage

`pnpm observability:report` is intentionally local-only: it proves temporary export, pack, activation, and compile state created inside the repo lane. It does not claim live production telemetry coverage or remote supervision recency outside those local artifacts.

## Health and promotion checks

Use `@openclawbrain/activation` as the narrow operator-facing inspection surface:

```ts
import { describeActivationObservability, describeActivationTarget, inspectActivationState } from "@openclawbrain/activation";

const inspection = inspectActivationState("/runtime/activation", new Date().toISOString());

if (!inspection.active?.activationReady) {
  throw new Error(`active slot unhealthy: ${(inspection.active?.findings ?? []).join("; ")}`);
}

if (inspection.candidate !== null && !inspection.promotion.allowed) {
  throw new Error(`candidate cannot be promoted: ${inspection.promotion.findings.join("; ")}`);
}

const freshness = describeActivationTarget("/runtime/activation", "active", {
  requireActivationReady: true
});
const observability = describeActivationObservability("/runtime/activation", "active", {
  requireActivationReady: true
});

console.log({
  activePackId: inspection.active?.packId ?? null,
  candidatePackId: inspection.candidate?.packId ?? null,
  promotionAllowed: inspection.promotion.allowed,
  rollbackAllowed: inspection.rollback.allowed,
  freshness,
  promotionFreshness: observability.promotionFreshness,
  learnedRouteFnFreshness: observability.learnedRouteFn,
  graphDynamicsFreshness: observability.graphDynamics
});
```

What this proves:

- `activationReady` and `findings` show basic health or the exact blocker
- `promotion.allowed` and `rollback.allowed` show whether pointer movement is currently safe
- `freshness.workspaceSnapshot`, `freshness.eventRange`, `freshness.eventExportDigest`, `freshness.routePolicy`, `freshness.routerIdentity`, and `freshness.builtAt` show exactly what promoted state is being served
- `promotionFreshness` makes it explicit whether the active pack is behind a promotion-ready candidate and which freshness dimensions advanced
- `learnedRouteFnFreshness` exposes the local `route_fn` identity, version, checksum, and training/build timestamps
- `graphDynamicsFreshness` exposes the local graph checksum, build freshness, and structural/hebbian/decay settings being served

## Supervision freshness checks

Use `@openclawbrain/event-export` when you have the local normalized export that fed a learned pack:

```ts
import { describeNormalizedEventExportObservability } from "@openclawbrain/event-export";

const report = describeNormalizedEventExportObservability(normalizedEventExport);

console.log({
  supervisionFreshnessBySource: report.supervisionFreshnessBySource,
  teacherFreshness: report.teacherFreshness
});
```

What this proves:

- `supervisionFreshnessBySource` tells you which local source stream most recently contributed supervision and how many human/self labels it carried
- `teacherFreshness` tells you the freshest local human-supervision event visible in that export
- these are export-local proofs only; they do not imply anything about supervision that has not yet been exported locally

## Learned-route checks

Use `@openclawbrain/compiler` to prove what happened at compile time:

```ts
import { compileRuntimeFromActivation, describeCompileFallbackUsage } from "@openclawbrain/compiler";
import { CONTRACT_IDS } from "@openclawbrain/contracts";

const compile = compileRuntimeFromActivation(
  "/runtime/activation",
  {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-1",
    userMessage: "route feedback through the learned pack path",
    maxContextBlocks: 2,
    maxContextChars: 320,
    modeRequested: "learned",
    compactionMode: "native"
  },
  {
    expectedTarget: {
      workspaceSnapshot: "workspace-1@snapshot-42",
      eventExportDigest: "sha256-abc123",
      routePolicy: "requires_learned_routing",
      routerIdentity: "pack-1:route_fn"
    }
  }
);
const fallbackUsage = describeCompileFallbackUsage(compile.response);

console.log({
  slot: compile.slot,
  packId: compile.target.packId,
  workspaceSnapshot: compile.target.workspaceSnapshot,
  usedLearnedRouteFn: compile.response.diagnostics.usedLearnedRouteFn,
  routerIdentity: compile.response.diagnostics.routerIdentity,
  selectionDigest: compile.response.diagnostics.selectionDigest,
  notes: compile.response.diagnostics.notes,
  fallbackUsage
});
```

What this proves:

- `expectedTarget` rejects stale pack/view mismatches before serving context
- `modeRequested`, `modeEffective`, `usedLearnedRouteFn`, and `routerIdentity` prove whether learned routing was actually in effect
- `selectionDigest` gives a stable fingerprint for the selected context set
- `notes` tells you both which activation target was served and whether compilation used token matching or deterministic priority fallback
- `fallbackUsage` makes priority fallback explicit even when token matches filled only part of the selection budget

The most important notes to look for are:

- `activation_slot=...`
- `target_pack_id=...`
- `target_route_policy=...`
- `target_router_identity=...` when learned routing is active
- `target_workspace_snapshot=...`
- `target_workspace_revision=...` when the pack is pinned to a revision
- `target_event_range=START-END#COUNT`
- `target_event_export_digest=...`
- `target_built_at=...`
- `selection_mode=priority_fallback` when no token hit occurs

## What healthy steady state looks like

In an attached deployment, healthy steady state looks like this:

- first value appears from the fast-boot pack before any full replay finishes
- live events keep landing while passive replay keeps improving candidate packs in the background
- freshness fields move forward with newer snapshots and export digests as promotions happen
- learned-route evidence stays explicit whenever the active pack requires learned routing
- fallback diagnostics stay explicit instead of silently hiding that compile served a priority-ranked context set

If any of those conditions are not true, activation inspection and compile diagnostics are the first place to look.
