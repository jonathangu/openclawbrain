# V5 Command Map

| Canonical command | Implementation |
|---|---|
| `pnpm ocb:traces:validate` | `node scripts/traces/validate.mjs --mode smoke` |
| `pnpm ocb:traces:validate:smoke` | `node scripts/traces/validate.mjs --mode smoke` |
| `pnpm ocb:traces:validate:production` | `node scripts/traces/validate.mjs --mode production` |

Production validation is expected to fail closed until 40 admitted real product-evidence traces are available.
