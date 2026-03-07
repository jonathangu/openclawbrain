# Operator Observability

This is the operator-facing diagnostics contract for attached OpenClawBrain installs.

## What operators must be able to prove

The public observability surface should answer five questions clearly and continuously:

- **health**: are the active and candidate slots activation-ready?
- **promotion safety**: can a pointer move happen safely right now?
- **freshness**: what exact snapshot/export/build is active right now?
- **route evidence**: did the served compile actually use the promoted pack's learned `route_fn` when required?
- **fallback**: if selection fell back, is that fact explicit?

## Repo proof

Run the dedicated observability smoke after install:

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm observability:smoke
```

That smoke exercises the public package surface only on temporary activation state and asserts:

- `inspectActivationState()` reports healthy active and candidate slots
- `promotion.allowed` becomes `true` before promotion and `rollback.allowed` becomes `true` after promotion
- `describeActivationTarget()` surfaces the promoted pack id, workspace snapshot, workspace revision, event range, export digest, route policy, and build time
- `compileRuntimeFromActivation()` emits stable compile diagnostics including route evidence and fallback notes

## Health and promotion checks

Use `@openclawbrain/activation` as the narrow operator-facing inspection surface:

```ts
import { describeActivationTarget, inspectActivationState } from "@openclawbrain/activation";

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

console.log({
  activePackId: inspection.active?.packId ?? null,
  candidatePackId: inspection.candidate?.packId ?? null,
  promotionAllowed: inspection.promotion.allowed,
  rollbackAllowed: inspection.rollback.allowed,
  freshness
});
```

What this proves:

- `activationReady` and `findings` show basic health or the exact blocker
- `promotion.allowed` and `rollback.allowed` show whether pointer movement is currently safe
- `freshness.workspaceSnapshot`, `freshness.eventRange`, `freshness.eventExportDigest`, `freshness.routePolicy`, `freshness.routerIdentity`, and `freshness.builtAt` show exactly what promoted state is being served

## Learned-route checks

Use `@openclawbrain/compiler` to prove what happened at compile time:

```ts
import { compileRuntimeFromActivation } from "@openclawbrain/compiler";
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

console.log({
  slot: compile.slot,
  packId: compile.target.packId,
  workspaceSnapshot: compile.target.workspaceSnapshot,
  usedLearnedRouteFn: compile.response.diagnostics.usedLearnedRouteFn,
  routerIdentity: compile.response.diagnostics.routerIdentity,
  selectionDigest: compile.response.diagnostics.selectionDigest,
  notes: compile.response.diagnostics.notes
});
```

What this proves:

- `expectedTarget` rejects stale pack/view mismatches before serving context
- `modeRequested`, `modeEffective`, `usedLearnedRouteFn`, and `routerIdentity` prove whether learned routing was actually in effect
- `selectionDigest` gives a stable fingerprint for the selected context set
- `notes` tells you both which activation target was served and whether compilation used token matching or deterministic priority fallback

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
