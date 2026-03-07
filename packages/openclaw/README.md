# @openclawbrain/openclaw

OpenClaw-facing runtime integration helpers for the TypeScript-first OpenClawBrain stack.

Use this package when OpenClaw needs a narrow, typed bridge over promoted packs without giving up runtime ownership:

- resolve the active promoted pack from activation pointers
- consume `runtime_compile.v1` through an activation-aware serve-path helper
- emit normalized interaction and feedback events for learner handoff
- optionally write learner-facing event-export bundles on disk

```ts
import { compileRuntimeContext, runRuntimeTurn } from "@openclawbrain/openclaw";

const compileResult = compileRuntimeContext({
  activationRoot: "/var/openclawbrain/activation",
  message: "feedback scanner route gating",
  runtimeHints: ["feedback scanner"]
});

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
```

This package stays fail-open for non-learned-required compile misses, and event-export write failures do not erase successful compile output.

The learned-required serve path is stricter:

- `compileRuntimeContext()` now forwards the active-slot diagnostics from `@openclawbrain/compiler.compileRuntimeFromActivation()`
- learned-required active packs return `hardRequirementViolated=true` and `fallbackToStaticContext=false` when route artifacts drift or disappear
- `runRuntimeTurn()` throws instead of silently serving through those learned-required failures
- event-export write failures still do not erase successful compile output
