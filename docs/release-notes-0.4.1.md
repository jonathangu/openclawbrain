# OpenClawBrain 0.4.1 release notes

Published packages for this patch wave:

- plugin/runtime payload stays at `@openclawbrain/openclaw@0.4.0`
- operator CLI advances to `@openclawbrain/cli@0.4.1`

This is a CLI-only patch release.
The public split-package architecture from `0.4.0` stays the same; the patch fixes one real operator seam in the shared-home attach path.

## What changed

- rerunning `openclawbrain install --shared` against a native package plugin that is already pinned to the requested activation root now succeeds as a no-op
- the installer still throws if the installed loader entry truly does not expose a patchable `ACTIVATION_ROOT` constant
- the shared attach declaration can now be rerun safely on the real shared Mac mini without forcing a fake repin failure

## Canonical operator lane after this patch

```bash
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.1 openclawbrain install --openclaw-home ~/.openclaw --shared
openclaw gateway restart
npx @openclawbrain/cli@0.4.1 openclawbrain status --openclaw-home ~/.openclaw --detailed
```

## Why this matters

`0.4.0` proved the split package story on a real host.
`0.4.1` makes the shared-home declaration idempotent too.
That matters on this Mac because one OpenClaw home serves multiple profiles, so the truthful operator behavior is “already pinned, nothing to change” rather than “throw even though the install is already correct.”

## Truthful remaining caveat

Some hosts still warn that the plugin manifest id is `openclawbrain` while the package/entry hint is `openclaw`.
That warning remains cosmetic.
This patch does not attempt to hide it.
