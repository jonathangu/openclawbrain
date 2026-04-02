# OpenClawBrain 0.4.22

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.22`
- `@openclawbrain/cli@0.4.22`

## Why this release exists

This release fixes the post-`#6` reliability seam captured in issue `#7`.

Before this fix, `0.4.21` could:

- install cleanly
- attach cleanly
- show green proof

while the async watch / teacher loop was still unhealthy.

That is exactly the kind of gap the public product should not hide.

This release makes two truths cleaner at the same time:

- the passive watch / teacher path is materially more reliable
- the public OpenClawBrain version is singular again instead of leaking two different live split-package numbers into the product story

## What changed

### Runtime / learner reliability

- startup replay now skips the full historical export rewalk when restored teacher state already tracks the seen export digests
- no-op always-on learner cycles stop rebuilding the runtime graph when `selectedSlices=0`
- session-tail cursor state now self-heals:
  - missing session-path states stabilize
  - missing session-file states stabilize
  - stale cursor entries prune instead of reappearing forever as fake changed sessions
- session parsing now accepts current real session shapes:
  - top-level `compaction` records
  - string-valued `message.content`
- install-time local teacher autodetect now prefers larger compatible Qwen lanes instead of biasing toward smaller ones

### Product/version truth

- the split publish lane is version-aligned again at `0.4.22`
- the public release story stays: **OpenClawBrain 0.4.22**
- the canonical install lane stays one front door: `openclawbrain install --openclaw-home ...`

## Important caveats

- this release improves the underlying async watch / teacher reliability, but it does **not** yet fully tighten proof/status verdicts around every possible stale-watch edge case
- the next clean follow-up is stronger status/proof degradation when the async watch lane is unhealthy even though runtime serve-state is still green
- internal split packages still exist as an implementation detail, but they should not be treated as competing public product versions

## Verification

- `npm run release:plan -- --json`
- `npm run release:verify`
- `npm view @openclawbrain/openclaw version`
- `npm view @openclawbrain/cli version`

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

If you are already on the canonical install lane, rerun the same lane. Do not use the retired compatibility package path.
