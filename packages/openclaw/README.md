# @openclawbrain/openclaw

OpenClaw integration helpers for promoted-pack compile consumption and normalized event emission.

Use this package when OpenClaw needs a narrow, typed bridge over promoted packs:

- resolve the active promoted pack from activation pointers
- consume `runtime_compile.v1` through an activation-aware serve-path helper
- surface learned-route diagnostics alongside compiled context
- emit normalized interaction and feedback events for learner handoff
- optionally write learner-facing event-export bundles on disk
- keep the post-attach loop real with canonical supervision, bounded learner refresh, candidate promotion, and later compile freshness

```ts
import { compileRuntimeContext, runContinuousProductLoopTurn, runRuntimeTurn } from "@openclawbrain/openclaw";

const compileResult = compileRuntimeContext({
  activationRoot: "/var/openclawbrain/activation",
  message: "feedback scanner route gating",
  runtimeHints: ["feedback scanner"]
});

if (compileResult.ok) {
  console.log(compileResult.compileResponse.diagnostics.usedLearnedRouteFn);
  console.log(compileResult.compileResponse.diagnostics.routerIdentity);
}

const turnResult = runRuntimeTurn(
  {
    sessionId: "session-123",
    channel: "whatsapp",
    userMessage: "feedback scanner route gating",
    compile: {
      createdAt: "2026-03-07T17:01:00.000Z"
    },
    delivery: {
      createdAt: "2026-03-07T17:02:00.000Z",
      messageId: "msg-123"
    },
    export: {
      rootDir: "/var/openclawbrain/exports/session-123"
    }
  },
  {
    activationRoot: "/var/openclawbrain/activation"
  }
);

const productLoopTurn = runContinuousProductLoopTurn({
  activationRoot: "/var/openclawbrain/activation",
  loopRoot: "/var/openclawbrain/runtime-loop",
  packLabel: "post-attach-loop",
  workspace: {
    workspaceId: "workspace-1",
    snapshotId: "workspace-1@snapshot-2",
    capturedAt: "2026-03-07T18:00:30.000Z",
    rootDir: "/workspace/openclawbrain",
    revision: "runtime-loop-rev-2"
  },
  turn: {
    sessionId: "session-123",
    channel: "whatsapp",
    userMessage: "Compile the fresher learned route artifact after promotion.",
    feedback: [{ content: "Prefer the fresher learned route artifact after promotion." }]
  }
});

console.log(productLoopTurn.compileActiveVersion);
console.log(productLoopTurn.learning.promoted);
console.log(productLoopTurn.state.currentActivePack?.routerIdentity);
```

This package stays fail-open for non-learned-required compile misses, and event-export write failures do not erase successful compile output.

The learned-required serve path is stricter:

- `compileRuntimeContext()` now forwards the active-slot diagnostics from `@openclawbrain/compiler.compileRuntimeFromActivation()`
- learned-required active packs return `hardRequirementViolated=true` and `fallbackToStaticContext=false` when route artifacts drift or disappear
- `runRuntimeTurn()` throws instead of silently serving through those learned-required failures
- event-export write failures still do not erase successful compile output
