# `@openclawbrain/pack-format`

Immutable pack layout helpers for OpenClawBrain.

Install this after `@openclawbrain/contracts` if you need to read, verify, write, stage, promote, or roll back pack artifacts on disk.

If you only need the activation-facing API, use `@openclawbrain/activation` as the narrower package surface.

## Install

```bash
pnpm add @openclawbrain/pack-format
```

## Includes

- canonical pack file layout constants
- checksum-validated pack loading and JSON payload writing
- activation-pointer helpers for active, candidate, and previous slots
- manifest-gated promotion, rollback, and activation inspection helpers

## Example

```ts
import { loadPack, PACK_LAYOUT } from "@openclawbrain/pack-format";

const pack = loadPack("/packs/pack-123");

console.log(pack.manifest.packId);
console.log(PACK_LAYOUT.manifest);
```
