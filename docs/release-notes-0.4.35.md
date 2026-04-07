# OpenClawBrain 0.4.35

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.35`
- `@openclawbrain/cli@0.4.35`

## Why this release exists

`0.4.35` closes the tool-action symmetry pass.

Before this release, the system could still blur together two different things across trace, learning, and scoring:

- a generic tool capability
- the concrete bound tool instance that was actually available and chosen

That distinction matters. OpenClawBrain should learn from the real tool action it took and prefer the real bound tool when it is present, not fall back to a fuzzier capability-level story.

This release makes that boundary more explicit and more consistent.

## What changed

- splits decision traces so tool capabilities and bound tool-instance actions are recorded separately
- updates seed-phase learning so chosen toolcard traversals reinforce concrete `tool_action` priors instead of leaving that reinforcement behind the traced decision
- scores explicit tool-instance bindings above generic capability matches when both are available, so retrieval prefers the real bound tool action
- keeps the route-function action family more coherent across trace, learn, and serve surfaces

## Operator truth

This is a runtime/learning coherence release.

It does **not** change the public install, proof, or rollback lane.
The canonical operator path is still:

- install or upgrade with `openclawbrain install --openclaw-home ...`
- restart the gateway
- verify with `status --detailed`
- capture durable evidence with `proof`

The release value is inside the learned routing behavior and its debugging/proof surfaces, not a new operator workflow.

## Focused verification

- `npm exec vitest run test/release-docs-drift.test.ts test/brain-core/trace.test.ts test/brain-core/update.test.ts test/brain-core/policy.test.ts`
- `node scripts/verify-release-docs-drift.mjs`

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

If you already run the canonical install lane, rerun the same lane. This release is about making tool-action learning and scoring agree more cleanly with the actual tool instance that was in play.
