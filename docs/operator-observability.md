# Operator Observability

This is the operator-facing diagnostics contract for attached OpenClawBrain installs.

The attach path must prove four things clearly and continuously:

- **health**: the active and candidate slots are activation-ready or they explain exactly why not
- **promotion safety**: promotion and rollback readiness are explicit before any pointer move happens
- **freshness**: the pack currently being served is pinned to a concrete workspace snapshot, event range, export digest, and build timestamp
- **fallback**: runtime compilation tells you when it matched request tokens and when it had to fall back deterministically

The install posture behind these checks does not change:

- do **not** wait for a full history replay before first value
- materialize a fast-boot pack from current workspace state and recent normalized events
- learn fresh live events first while older history catches up passively in the background
- keep passive background learning on continuously after attach
- fail open in OpenClaw when brain artifacts are unavailable or mid-refresh

## Repo proof

Run the dedicated observability smoke after install:

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm observability:smoke
```

That smoke exercises the public package surface only and asserts:

- `inspectActivationState()` reports healthy active and candidate slots
- `promotion.allowed` becomes `true` before promotion and `rollback.allowed` becomes `true` after promotion
- `describeActivationTarget()` surfaces the promoted pack id, workspace snapshot, workspace revision, event range, export digest, and build time
- `compileRuntimeFromActivation()` emits compile diagnostics with a stable `selectionDigest`, explicit served-target notes, and an explicit `selection_mode=priority_fallback` note when token matching does not hit

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
- `freshness.workspaceSnapshot`, `freshness.eventRange`, `freshness.eventExportDigest`, and `freshness.builtAt` show exactly what runtime state is being served

## Runtime fallback checks

Use `@openclawbrain/compiler` to prove what happened at compile time:

```ts
import { compileRuntimeFromActivation } from "@openclawbrain/compiler";
import { CONTRACT_IDS } from "@openclawbrain/contracts";

const compile = compileRuntimeFromActivation(
  "/runtime/activation",
  {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-1",
    userMessage: "zebra nebula quartz",
    maxContextBlocks: 2,
    maxContextChars: 320,
    modeRequested: "heuristic",
    compactionMode: "native"
  },
  {
    expectedTarget: {
      workspaceSnapshot: "workspace-1@snapshot-42",
      eventExportDigest: "sha256-abc123"
    }
  }
);

console.log({
  slot: compile.slot,
  packId: compile.target.packId,
  workspaceSnapshot: compile.target.workspaceSnapshot,
  selectionDigest: compile.response.diagnostics.selectionDigest,
  notes: compile.response.diagnostics.notes
});
```

What this proves:

- `expectedTarget` rejects stale pack/view mismatches before serving runtime context
- `selectionDigest` gives a stable fingerprint for the selected context set
- `notes` tells you both which activation target was served and whether compilation used token matching or deterministic priority fallback
- `modeRequested`, `modeEffective`, and `usedLearnedRouteFn` tell you whether learned routing was actually in effect

The most important notes to look for are:

- `activation_slot=...`
- `target_pack_id=...`
- `target_route_policy=...`
- `target_workspace_snapshot=...`
- `target_workspace_revision=...` when the pack is pinned to a revision
- `target_event_range=START-END#COUNT`
- `target_event_export_digest=...`
- `target_built_at=...`
- `target_router_identity=...` when learned routing is active
- `selection_mode=priority_fallback` when no keyword hit occurs

## What operators should expect

Healthy steady state looks like this:

- first value appears from the fast-boot pack before any full history replay finishes
- live events keep landing while passive replay keeps improving candidate packs in the background
- freshness fields move forward with newer snapshots and export digests as promotions happen
- fallback diagnostics stay explicit instead of silently hiding that runtime served a priority-ranked context set

If any of those conditions are not true, the activation inspection and compile diagnostics above are the first place to look.
