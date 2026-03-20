# @openclawbrain/openclaw

`@openclawbrain/openclaw@0.4.0` is the published OpenClawBrain plugin/runtime payload for OpenClaw.

Use it with the published operator CLI:

```bash
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.1 openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.1 openclawbrain status --openclaw-home ~/.openclaw --detailed
```

Current host/plugin caveat: some hosts still warn about a plugin id mismatch because the plugin manifest id is `openclawbrain` while the package/entry hint is `openclaw`. The install still works; treat that warning as currently cosmetic.

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

- `npx @openclawbrain/cli@0.4.1 openclawbrain install --openclaw-home <path>`
- `npx @openclawbrain/cli@0.4.1 openclawbrain status --openclaw-home <path> --detailed`
- daemon, import/export, and other operator commands

If you install this plugin package into OpenClaw before the CLI package is installed, the extension will fail open and tell you to install `@openclawbrain/cli` before pinning the activation root.
