# OpenClawBrain 0.4.46

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.46`
- `@openclawbrain/cli@0.4.46`

## Why this release exists

`0.4.46` publishes the bounded selective-intervention reset tranche that landed after `0.4.45`.

The point of this release is narrow and evidence-backed: protect current-choice fidelity, prove broader specificity/restraint without harm, keep the one-home operator proof clean, and add the first route-level tool-capability choice proof without claiming live tool execution end to end.

## What changed

- keeps the later-preference/current-choice lane protected at `full-ocb 5/5`, with `regret=0` and `harm=0`
- broadens the specificity/restraint proof cohort to `full-ocb 12/12`, with `regret=0` and `harm=0`
- accepts the current `Connectivity probe: ok` gateway-health wording in the operator proof lane
- preserves a clean one-home operator proof: verdict `success_and_proven`, severity `none`, warnings `0`
- adds a bounded route-level capability-choice lane for `weather.current_conditions`
- proves the paired weather lane: the current weather/rain prompt selects `tool_capability`; the weather-definition prompt selects `stop_local`
- keeps provider `tool_instance` as a hard negative rather than counting it as a success
- restores the tiny checked cold-start router sample fixture required by the release verification suite

## Operator truth

The public lane is still:

- run `openclawbrain install --openclaw-home ...`
- restart the gateway
- verify with `status --detailed`
- capture durable evidence with `proof`

The release does not ask operators to learn a new front door. It tightens what the current lane can honestly prove.

## Honest boundary

This release is bounded selective-intervention proof plus one route-level capability-choice proof.

It does **not** claim:

- broad memory is solved
- broad online answer quality is proven
- live weather-tool execution is proven end to end
- every tool-capability family is generalized
- Teacher v3 runtime/proof adoption is complete

## What success looks like

After upgrading, a healthy host should still converge through the same install / restart / detailed-status / proof lane.

The new release-level proof story is:

- current-choice fidelity stays protected
- specificity/restraint improves without recorded regret or harm
- operator proof is clean on the exercised home
- the first capability-choice family has paired must-fire and must-not-fire route-level evidence

## Focused verification

- operator proof bundle: `success_and_proven`, severity `none`, warnings `0`
- `npm run release:verify:proof` → `proof smoke: ok`
- `npx vitest run test/brain-core/tool-capability-choice-lane.test.ts --reporter=dot` → passed
- `npx tsx scripts/validate-tool-capability-choice-lane.ts --family weather.current_conditions --output-dir ...` → `tool capability choice proof: ok (weather.current_conditions)`
- `npx vitest run test/brain-core/tool-capability-choice-lane.test.ts test/brain-core/route-rows.test.ts test/brain-core/cold-start-router-trainer.test.ts test/brain-core/cold-start-router-runtime.test.ts --reporter=dot` → 4 files / 24 tests passed
- `npx vitest run test/brain-runtime/assembler-extension.test.ts --reporter=dot` → 1 file / 26 tests passed

Repo-wide typecheck note: `npx tsc -p tsconfig.json --noEmit --pretty false` still reports pre-existing repo-wide type errors unrelated to this lane.

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Keep the claim boundary narrow: this release proves selective intervention and the first capability-choice route lane, not broad autonomous memory.
