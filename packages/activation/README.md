# `@openclawbrain/activation`

Activation inspection, staging, promotion, rollback, and freshness helpers for promoted OpenClawBrain packs.

## Install

```bash
pnpm add @openclawbrain/activation
```

## Includes

- activation pointer loading and inspection
- compile-target inspection over active, candidate, and previous slots
- candidate staging, promotion, and rollback helpers with manifest-pinned safety checks
- freshness inspection over pack id, snapshot, event range, route policy, and router identity
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
