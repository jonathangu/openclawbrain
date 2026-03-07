# `@openclawbrain/activation`

Activation inspection, staging, promotion, and rollback helpers for TypeScript-first OpenClawBrain packs.

## Install

```bash
pnpm add @openclawbrain/activation
```

## Includes

- activation pointer loading and inspection
- compile-target inspection over active, candidate, and previous slots
- candidate staging, promotion, and rollback helpers with manifest-pinned safety checks
- activation-readiness checks surfaced as a package-first API

## Example

```ts
import { describeActivationTarget, inspectActivationState } from "@openclawbrain/activation";

const inspection = inspectActivationState("/runtime/activation", new Date().toISOString());

if (!inspection.active?.activationReady) {
  throw new Error(`active slot unhealthy: ${(inspection.active?.findings ?? []).join("; ")}`);
}

console.log({
  promotionAllowed: inspection.promotion.allowed,
  rollbackAllowed: inspection.rollback.allowed,
  activeTarget: describeActivationTarget("/runtime/activation", "active", {
    requireActivationReady: true
  })
});
```
