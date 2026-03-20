# OpenClawBrain 0.4.2 release notes

Planned packages for this patch wave:

- plugin/runtime payload stays at `@openclawbrain/openclaw@0.4.0`
- operator CLI advances to `@openclawbrain/cli@0.4.2`

This is a CLI-only patch release.
The split-package architecture does not change; the patch closes the remaining high-signal status/package seams left after `0.4.1`.

## What changed

- `openclawbrain install --shared` and `attach --shared` now persist declared attachment policy under `activation-root/attachment-truth/policy-declaration.json`
- later `openclawbrain status` reads use that persisted declaration, so a rerun no longer falls back to `policy=null` / `undeclared` after a truthful shared install
- the canonical brain-store traced-learning bridge remains the preferred status truth, and the CLI tarball now explicitly ships the traced-learning bridge plus the operator modules its entrypoint imports
- tarball verification now proves those release-surface requirements instead of relying on repo-only drift

## Canonical operator lane after this patch

```bash
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.2 install --openclaw-home ~/.openclaw --shared
openclaw gateway restart
npx @openclawbrain/cli@0.4.2 status --openclaw-home ~/.openclaw --detailed
```

## Why this matters

`0.4.1` fixed the false repin failure for shared installs, but the operator story still had two high-signal seams:

1. status could underreport the declared shared policy immediately after a truthful `install --shared`
2. the repo carried canonical traced-learning and operator-module fixes that were not frozen into a ready-to-verify CLI tarball surface

`0.4.2` makes both seams explicit and shippable.
