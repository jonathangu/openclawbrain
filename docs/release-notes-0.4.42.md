# OpenClawBrain 0.4.42

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.42`
- `@openclawbrain/cli@0.4.42`

## Why this release exists

`0.4.42` exists to close the last ugly install seam in the public lane.

By `0.4.41`, the shipped operator surfaces already told the truth about cold-start-rooted lineage and the canonical install lane was the right public story. But a real host could still hit a stale nested duplicate OpenClawBrain surface, for example under `~/.openclaw/.openclaw/extensions/openclawbrain`, and fail plugin-manager converge even though the fix was mechanically obvious.

This release teaches the install lane to repair that class of skew directly.

## What changed

- the packaged install helpers now detect stale nested duplicate OpenClawBrain extension surfaces before plugin-manager converge
- those stale nested duplicates are quarantined automatically so the install lane can continue instead of failing on a duplicate-surface blocker
- the canonical `install -> gateway restart -> status --detailed -> proof` lane stays the same, but it now repairs this real-world nested-home skew instead of merely surfacing it
- the `0.4.41` cold-start-lineage, bounded-anytime, route-quality, economics, and provenance operator story stays intact

## Operator truth

This is still not a new install workflow.

The canonical lane stays the same:

- run `openclawbrain install --openclaw-home ...`
- restart the gateway
- verify with `status --detailed`
- capture durable evidence with `proof`

What changes is the repair behavior. If a host carries a stale nested duplicate extension copy from an older or mis-shaped install, the install helper now quarantines that duplicate surface first so converge can succeed honestly.

## What success looks like

On a previously skewed host, rerunning the canonical lane should now:

1. detect the nested duplicate surface
2. quarantine it before plugin-manager converge
3. finish install on the intended OpenClaw home
4. leave `status --detailed` showing a converged surface again

## Focused verification

- `node --test packages/openclaw/dist/test/lifecycle-layout-truth.test.js`
- `npm run release:verify:openclaw`
- `npm run release:verify:cli`

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

If the host is already healthy, this release should behave like a no-drama rerun of the same lane. If the host still carries a stale nested duplicate extension surface, this release should repair it instead of stopping at a duplicate blocker.
