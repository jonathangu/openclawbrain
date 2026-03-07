# `@openclawbrain/compiler`

Deterministic pack-backed context selection and native structural compaction for OpenClaw.

Install this after `@openclawbrain/pack-format` when OpenClaw needs a narrow compile boundary that stays separate from runtime ownership.

## Install

```bash
pnpm add @openclawbrain/compiler
```

## Includes

- pack loading for compile-time use
- deterministic context ranking over graph blocks and vector keywords
- larger-context budget enforcement via max-block and max-character limits
- native structural compaction when selection pressure exceeds the runtime budget
- learned-routing enforcement when a pack manifest requires it

## Example

```ts
import { compileRuntime } from "@openclawbrain/compiler";
import { CONTRACT_IDS } from "@openclawbrain/contracts";

const response = compileRuntime("/packs/pack-123", {
  contract: CONTRACT_IDS.runtimeCompile,
  agentId: "agent-1",
  userMessage: "compile manifest routing",
  maxContextBlocks: 3,
  maxContextChars: 1800,
  modeRequested: "heuristic",
  compactionMode: "native"
});
```
