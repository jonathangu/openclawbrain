# RL Contract

## State

`TraversalState` is the routing state for a single episode:

- current node
- query embedding
- visited set
- fired nodes
- remaining budget
- hop count
- max hops

At the seed phase, the current node is the virtual `__START__` policy head.

## Actions

The action set is:

- `traverse(targetNodeId)`
- `stop`

At seed time, the candidates are `__START__ -> seed_i` plus `STOP`.

## Policy

- Softmax over the available actions.
- Candidate scores combine learned parameters with structural priors and semantic relevance.
- `STOP` is always available and increases with budget and hop pressure.

## Reward

- Reward is terminal only.
- Signed reward comes from trust-ranked labels.
- The baseline is an EMA of recent reward.

## Update Rule

- Use full-trajectory REINFORCE with the baseline.
- Every chosen action in the episode contributes to the gradient.
- Seed actions update `__START__ -> seed_i` parameters exactly like later traversal decisions.

## Constraints

- No teacher cheating: teacher evaluation only sees what the router saw.
- No intermediate reward shaping that overrides terminal outcome.
- No deterministic scorer disguised as a policy.
