# OpenClawBrain 0.4.0 release notes

Published packages for the `0.4.0` split:

- plugin/runtime payload: `@openclawbrain/openclaw@0.4.0`
- operator CLI: `@openclawbrain/cli@0.4.0`
- compatibility holdover for older installs: `@jonathangu/openclawbrain@0.3.5`

The split landed on `main` in commit `b3ada81`.

## What changed

- The package split is now the real public surface rather than staged repo work.
- The canonical operator lane is now:
  1. `openclaw plugins install @openclawbrain/openclaw@0.4.0`
  2. `npx @openclawbrain/cli@0.4.0 install --openclaw-home ~/.openclaw`
  3. `openclaw gateway restart`
  4. `npx @openclawbrain/cli@0.4.0 status --openclaw-home ~/.openclaw --detailed`
- That exact public-registry flow already passed on the real host `redogfood`.
- The repo and package READMEs now lead with the split plugin-plus-CLI story instead of the older combined-package/global-install story.

## Remaining caveat

Some hosts still warn about a plugin id mismatch during plugin install because the manifest uses `openclawbrain` while the package/entry hint uses `openclaw`.

That warning is still real. It is currently cosmetic, not evidence that the install or attach flow failed, and the docs keep it visible on purpose.

## Why it matters

0.4.0 turns the split package lane into the truthful outside-operator story.
The plugin/runtime payload and the operator CLI are now separately published, the public-registry flow has already passed on a real host, and the remaining host/plugin warning is documented plainly instead of being papered over.
