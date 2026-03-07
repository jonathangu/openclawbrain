# `@openclawbrain/compiler`

Deterministic runtime compilation over immutable OpenClawBrain packs.

Install this after `@openclawbrain/pack-format` when OpenClaw needs a narrow compile boundary that stays separate from runtime ownership.

## Install

```bash
pnpm add @openclawbrain/compiler
```

## Includes

- pack loading for compile-time use
- deterministic context ranking over graph blocks and vector keywords
- learned-routing enforcement when a pack manifest requires it

## Example

```ts
import { compileRuntime } from "@openclawbrain/compiler";
import { CONTRACT_IDS } from "@openclawbrain/contracts";

const response = compileRuntime("/packs/pack-123", {
  contract: CONTRACT_IDS.runtimeCompile,
  agentId: "agent-1",
  userMessage: "compile manifest routing",
  maxContextBlocks: 2,
  modeRequested: "heuristic"
});
```
