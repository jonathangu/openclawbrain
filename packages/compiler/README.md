# `@openclawbrain/compiler`

Deterministic pack-backed context selection and native structural compaction for OpenClaw.

Install this after `@openclawbrain/pack-format` when OpenClaw needs a narrow compile boundary that stays separate from runtime ownership.

## Install

```bash
pnpm add @openclawbrain/compiler
```

## Includes

- pack loading for compile-time use
- activation-aware compilation from promoted runtime slots
- compile-target expectation checks over pack id, workspace snapshot, event range, and event-export digest
- deterministic context ranking over graph blocks and vector keywords
- larger-context budget enforcement via max-block and max-character limits
- native structural compaction when selection pressure exceeds the runtime budget
- learned-routing enforcement when a pack manifest requires it

## Example

```ts
import { compileRuntimeFromActivation } from "@openclawbrain/compiler";
import { CONTRACT_IDS } from "@openclawbrain/contracts";

const { target, response } = compileRuntimeFromActivation("/runtime/activation", {
  contract: CONTRACT_IDS.runtimeCompile,
  agentId: "agent-1",
  userMessage: "compile manifest routing",
  maxContextBlocks: 3,
  maxContextChars: 1800,
  modeRequested: "heuristic",
  compactionMode: "native"
}, {
  expectedTarget: {
    workspaceSnapshot: "workspace-1@snapshot-42",
    eventExportDigest: "sha256-abc123"
  }
});
```
