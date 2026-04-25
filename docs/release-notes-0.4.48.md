# OpenClawBrain 0.4.48

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.48`
- `@openclawbrain/cli@0.4.48`

## Why this release exists

`0.4.48` is a bounded routing/proof tranche. It makes three measured improvements without changing the live install front door:

1. cold-start candidate-artifact prior selection no longer under-selects a single block in replay override use
2. learned-route replay proof now reports explicit activation-usefulness labels instead of hiding wins/ties/harm in aggregate quality
3. Teacher v3 now has one narrow durable, shadow-only graph-maintenance proposal lifecycle with replay and rollback evidence

The live learned `route_fn` path remains central. Cold-start priors and graph-maintenance artifacts stay subordinate to governed replay/proof boundaries.

## What changed

- adds a deterministic cold-start scorecard comparing `no_brain`, `graph_prior_only`, `cold_start_prior_single`, `cold_start_prior`, and `learned_route`
- lets cold-start candidate-artifact replay override select multiple close-score candidates within the caller's block budget instead of always selecting one block
- adds per-turn learned-route activation usefulness labels: fired, should-have-fired, beneficial, harmful, neutral, missed opportunity, correct abstention, and proxy cost deltas
- adds a scorecard summary for unique beneficial learned-route wins, harmful activations, neutral ties, missed opportunities, and fired prompt/context cost deltas
- adds a narrow Teacher v3 graph-maintenance lifecycle for shadow-only `add_edge` proposals, including evidence refs, subject ids, expected effect, replay suites, rollback key, safe class mode, and durable replay summary
- hardens Teacher proposal storage so mutation/forgetting/correction classes cannot be promoted through the shadow-only path
- records integration/release guardrails under `task-artifacts/T-20260425-287/`

## Measured evidence

Cold-start scorecard on the frozen replayable trace set:

- old `cold_start_prior_single`: mean quality `92.05`, phrase hits `64/74`, selected blocks `45`, selected chars `12,501`
- new `cold_start_prior`: mean quality `92.8`, phrase hits `65/74`, selected blocks `48`, selected chars `13,101`
- `graph_prior_only`: mean quality `92.8`, phrase hits `65/74`, selected blocks `135`, selected chars `26,125`
- `learned_route`: mean quality `97.3`, phrase hits `71/74`, selected blocks `135`, selected chars `27,592`

Learned-route activation-usefulness fixture:

- observed activation labels: `5/5` comparable turns
- fired learned routing: `5`
- unique beneficial learned-route wins: `1`
- harmful activations: `0`
- neutral activation ties: `4`
- missed beneficial opportunities: `0`
- fired prompt-token proxy delta vs graph prior: `+93`

Teacher graph-maintenance proof:

- example proposal: `prop_graph_add_edge_01`
- class/kind: `mutation` / `add_edge`
- safe mode: `shadow_only`
- replay outcome: applied in a cloned candidate graph
- rollback: restored
- promotion bypass: false
- live self-editing: false

## Operator truth

The public lane is unchanged:

- run `openclawbrain install --openclaw-home ...`
- restart the gateway
- verify with `status --detailed`
- capture durable evidence with `proof`

This release does not introduce a new install front door.

## Honest boundary

This release proves bounded improvements on named scorecards, fixtures, and shadow replay artifacts.

It does **not** claim:

- broad memory is solved
- broad online answer quality is proven
- fresh homes are equivalent to trained homes
- cold-start beats the served learned router
- learned routing is now universally better than graph/vector baselines
- Graphify or Teacher graph maintenance is live runtime truth authority
- graph mutations promote or edit live graph truth automatically
- live tool execution behavior changed

## Focused verification

Release-prep verification included:

- `npm test` — `116` files / `748` tests passed
- `npm run test:learned-route-mission` — `8` files / `45` tests passed
- focused cold-start scorecard/loader/runtime/trainer/replay tests — `6` files / `32` tests passed
- focused graphify/Teacher graph-maintenance tests — `6` files / `13` tests passed
- integrated lane tests documented in `task-artifacts/T-20260425-287/integration-merge-report.md`
- `git diff --check`
- `npm run release:verify:docs-drift`
- `npm run release:plan`

Repo-wide `npx tsc --noEmit` still reports pre-existing fixture/type drift outside this release lane; it is not the publish gate for `0.4.48`.

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Keep the claim boundary narrow: this release strengthens bounded cold-start selection, learned-route activation accounting, and shadow-only graph-maintenance proof. It is not a broad-memory or live graph self-editing release.
