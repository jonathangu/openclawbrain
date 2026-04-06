# OpenClawBrain 0.4.33

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.33`
- `@openclawbrain/cli@0.4.33`

## Why this release exists

`0.4.33` makes the learned cold-start prior the default OpenClawBrain story.

That is the point of the release.

Fresh installs should not start from the weaker old fallback path. They should start from the learned prior. Existing installs should not lose earned value when they upgrade. They should rebuild the stronger generic prior underneath the user layer they already earned.

This release makes that the default behavior and makes the public docs say it plainly.

## Install and upgrade story

Fresh homes now default to the cold-start prior.

Existing homes rerun the same install lane. OpenClawBrain rebuilds the stronger generic prior underneath the current home while preserving the user's saved preferences, corrections, and recent overlay on top.

The rule is simple:

- new users get a stronger default on day one
- existing users keep the value they already earned
- install, upgrade, status, and proof stay one coherent lane

## What changed

- makes the learned serve-time `v2` prior the default path instead of falling back to the older heuristic/v1 seam on fresh installs
- preserves existing-user baseline state on upgrade so the user layer stays on top of the new base prior
- ships the broader governed QA cold-prior tranche as the default learned prior story:
  - `44` approved rows
  - `20` `STOP_LOCAL` rows
  - replay `44/44`
- keeps the repaired `STOP_LOCAL` behavior in the broadened tranche and on the refreshed held-out QA slice
- carries the same-family route function story through runtime learning: `STOP_LOCAL` and tool actions remain first-class actions, online `toolActionPriors` update in the same family, and direct teacher-action distillation is in place
- clears ToolMind for governed second-wave use and verifies a minimal export/training/runtime smoke end to end
- aligns the public site and repo docs around one install / upgrade story
- archives the retired `0.3.8` compatibility release note so it no longer reads like the active lane

## Operator truth

Use the same chosen `--openclaw-home` path through install, status, proof, rollback, detach, and troubleshooting.

The current public promise is not “start over.”
It is “start stronger, or upgrade without losing your brain.”

That means:

- a fresh home starts from the learned cold-start prior
- an existing home keeps its learned preferences on top
- `status --detailed` is the fast live check
- `proof` is still the durable record

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

If the home already has learned preferences, rerunning `install` preserves that user layer and rebuilds the generic prior underneath it.
