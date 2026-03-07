# `@openclawbrain/compiler`

Deterministic promoted-pack compilation with learned `route_fn` enforcement and native structural compaction.

Install this after `@openclawbrain/pack-format` when you need the narrow compile boundary over promoted packs.

## Install

```bash
pnpm add @openclawbrain/compiler
```

## Includes

- pack loading for compile-time use
- activation-aware compilation from active or promoted pack slots
- compile-target expectation checks over pack id, workspace snapshot, event range, event-export digest, route policy, and router identity via `expectedTarget`
- deterministic context ranking over graph blocks and vector keywords
- learned-routing enforcement when a pack manifest requires it
- larger-context budget enforcement via max-block and max-character limits
- native structural compaction when selection pressure exceeds the compile budget
- operator-facing compile notes for activation slot, served target provenance, learned-route evidence, and explicit fallback

## Example

```ts
import { compileRuntimeFromActivation } from "@openclawbrain/compiler";
import { CONTRACT_IDS } from "@openclawbrain/contracts";

const { target, response } = compileRuntimeFromActivation(
  "/runtime/activation",
  {
    contract: CONTRACT_IDS.runtimeCompile,
    agentId: "agent-1",
    userMessage: "compile manifest routing",
    maxContextBlocks: 3,
    maxContextChars: 1800,
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

console.log(target.packId, response.packId);
console.log(response.diagnostics.usedLearnedRouteFn);
console.log(response.diagnostics.routerIdentity);
console.log(response.diagnostics.notes);
```

`response.diagnostics` explicitly tells operators which target was served, whether learned routing actually ran, and whether compilation had to fall back.
