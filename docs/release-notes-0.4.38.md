# OpenClawBrain 0.4.38

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.38`
- `@openclawbrain/cli@0.4.38`

## Why this release exists

`0.4.38` is the install-hotfix follow-up to the ongoing-learning release.

`0.4.37` shipped the bounded continuous-learning loop, operator controls, and replay/eval hardening, but the published CLI package still had a packaging seam on the real host install lane: some CLI surfaces reached outside the installed package for `openclawbrain-contracts.js` and crashed under global npm install.

`0.4.38` fixes that seam without changing the public product story.

## What changed

- makes the published CLI self-contained by shipping the canonical JSON helper inside the CLI package and routing Graphify/import-export surfaces to that local file
- hardens CLI tarball verification so the packed `import-export.js` surface is imported directly from the extracted tarball before release passes
- keeps the shipped continuous-learning loop, operator controls, and replay/eval hardening from `0.4.37`
- keeps the public install lane unchanged: install, restart, status, proof

## Operator truth

This is a hotfix release, not a new operator workflow.

The canonical lane is still:

- install or upgrade with `openclawbrain install --openclaw-home ...`
- restart the gateway
- verify with `status --detailed`
- capture durable evidence with `proof`

## Focused verification

- `npm run release:verify`
- `node scripts/verify-release-docs-drift.mjs`
- packed CLI tarball import smoke for `dist/src/import-export.js`
- real-host install lane on `~/.openclaw`

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

If you already run the canonical install lane, rerun the same lane. This release restores that lane on the published package pair.
