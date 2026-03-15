# Graph Schema

## Node Kinds

- `chunk`
- `workflow`
- `correction`
- `toolcard`
- `episode_anchor`
- `summary_bridge`

## Edge Kinds

- `sibling`
- `semantic`
- `learned`
- `inhibitory`
- `bridge`
- `seed`

`seed` edges are virtual or persisted parameters from `__START__` into candidate seed regions.

## Trust

- `human`
- `self`
- `scanner`
- `teacher`

## Serving Rules

- Promoted packs are immutable snapshots.
- Runtime reads only the current promoted pack.
- Mutable state is used only by the learner and candidate-pack builder.

## Mutation Rules

Phase order:

1. `connect`
2. `prune`
3. `inject`
4. `split`
5. `merge`

Only `connect`, `prune`, and `inject` may be auto-evaluated before the replay harness is strong enough for `split` and `merge`.

## Teaching Rules

- A taught correction must be embedded immediately.
- It must connect to the triggering route and, when identifiable, to the chosen seed region.
- If the mistaken target is known, add inhibitory structure against that path.
