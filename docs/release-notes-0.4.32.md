# OpenClawBrain 0.4.32

Canonical install lane:

```bash
openclawbrain install --openclaw-home ./openclaw-cormorantai
openclaw gateway restart
openclawbrain status --openclaw-home ./openclaw-cormorantai --detailed
openclawbrain proof --openclaw-home ./openclaw-cormorantai
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.32`
- `@openclawbrain/cli@0.4.32`

## Why this release exists

`0.4.32` is the trust-surfaces hardening release.

The live runtime was already healthier than some of the operator surfaces made it look. This release tightens those surfaces without reopening the host-churn mistakes:

- converge now avoids a no-op plugin-manager refresh when the authoritative split-package plugin is already installed
- traced-learning/status now prefers stronger active-pack truth when a promoted pack is clearly serving
- persisted traced-learning status summaries now no-op when the normalized payload is unchanged

The point is simple: if nothing semantic changed, the operator surfaces should stop acting like something changed.

## Install and upgrade story

Fresh homes now default to the cold-start prior. Existing homes rerun the same install lane and keep their earned preferences on top while OpenClawBrain rebuilds the stronger generic prior underneath.

That is the public promise of this release:

- new users get a strong default on day one
- existing users keep their corrections, habits, and personal layer
- upgrades refresh the generic prior without resetting the user's brain

## What changed

- makes converge treat an already-authoritative native plugin install as a no-op instead of blindly refreshing plugin-manager state
- upgrades false-null / false-negative pack truth in traced-learning status when active-pack and watch-snapshot evidence prove a promoted pack is serving
- avoids rewriting traced-learning persisted status summaries when the normalized payload is identical
- keeps the hardening work repo-only and read-path focused rather than reintroducing live-host proof/install churn

## Operator truth

OpenClaw itself should stay on Codex GPT-5.4. OpenClawBrain remains the separate local teacher/runtime lane.

Use the same exact `--openclaw-home` path through install, status, proof, rollback, and troubleshooting. That path can be:

- the default `~/.openclaw`
- a profile-specific home like `~/.openclaw-Tern`
- an explicit nonstandard path like `./openclaw-cormorantai`

The important thing is not the shape of the path; it is pinning the same chosen home consistently.

This release does **not** require re-dogfooding the shared live host to be useful. It is primarily about making the status/truth surfaces less noisy and more trustworthy.

## Upgrade

```bash
openclawbrain install --openclaw-home ./openclaw-cormorantai
openclaw gateway restart
openclawbrain status --openclaw-home ./openclaw-cormorantai --detailed
openclawbrain proof --openclaw-home ./openclaw-cormorantai
```
