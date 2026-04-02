# OpenClawBrain 0.4.21

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/cli@0.4.21`
- plugin/runtime remains `@openclawbrain/openclaw@0.4.6`

## Why this release exists

This release makes the single-product story truer in the real product, not just the copy.

The main goal is simple:

- one visible OpenClawBrain version
- one install path
- less legacy seam confusion
- more truthful operator proof for the selected repaired host

## What changed

### Public release changes

- retired compatibility binary now fails closed and points operators back to the canonical install path instead of silently invoking the wrong lane
- install/daemon guardrails now detect stale legacy compatibility runtime seams and refresh them onto the durable current CLI path
- compatibility migration onto the canonical plugin lane now replaces the wrong seam instead of drifting into `plugin already exists` style confusion
- generated shadow extension package metadata now resolves the real runtime dependency correctly
- proof now accepts generated shadow hook sources as valid and no longer degrades the target repaired profile just because unrelated attached profiles are only partially covered
- first-read docs now keep the product story on one install path and treat manual/plugin surgery as maintainer-only background detail

### Internal package note

- the runtime/plugin package stays on `@openclawbrain/openclaw@0.4.6`
- the main shipped delta in this release is in the operator/front-door surface, migration/guardrail behavior, and proof classification logic

## Important caveats

- this does not make backwards compatibility a strategic product goal again
- internal split-package mechanics still exist, but they are not the public product story
- `proof` remains a separate follow-up command today even though the long-run product direction is one unified install/proof lane

## Verification

- `npx vitest run test/install-converge-seam.test.ts test/shadow-extension-deps.test.ts test/compat-cli-guard.test.ts` passed
- `node --test packages/cli/dist/test/install-converge.test.js packages/cli/dist/test/daemon.test.js packages/cli/dist/test/proof-cli-surface.test.js` passed
- `node scripts/release-plan.mjs --json` passed after version/changelog/release-note updates
- `git diff --check` passed for repo/site truth-surface updates

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

If you are already on the canonical install lane, rerun the same lane. Do not use the retired compatibility package path.
