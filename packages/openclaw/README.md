# @openclawbrain/openclaw

Staged package split note:

- `@openclawbrain/openclaw` is the plugin/runtime payload for OpenClaw.
- The operator CLI surface is moving to `@openclawbrain/cli`.
- Public docs are not flipped yet; treat this package split as repo-stage work until the release lane is ready.

## What stays here

- the OpenClaw plugin manifest
- the installed extension/runtime guard
- `compileRuntimeContext()`
- runtime load-proof helpers used by the extension

## What moves out

- `openclawbrain` / `openclawbrain-ops` bins
- daemon management
- import/export helpers
- install/status/rollback/operator management helpers

## Runtime usage

```ts
import {
  compileRuntimeContext,
  recordOpenClawProfileRuntimeLoadProof
} from "@openclawbrain/openclaw";
```

## Operator usage

Use `@openclawbrain/cli` for:

- `openclawbrain install --openclaw-home <path>`
- `openclawbrain status --openclaw-home <path> --detailed`
- daemon, import/export, and other operator commands

If you install this plugin package into OpenClaw before the CLI package is installed, the extension will fail open and tell you to install `@openclawbrain/cli` before pinning the activation root.
