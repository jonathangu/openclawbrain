# OpenClawBrain End State

OpenClawBrain is a native OpenClaw context-engine plugin with three hard boundaries:

1. Transcript substrate
   Lossless transcript persistence, compaction, search, and expansion stay in the inherited LCM layer.
2. Learned routing graph
   Corrections, toolcards, workflows, episode anchors, and summary bridges are learned over a separate graph.
3. Immutable serving packs
   Live queries read only the currently promoted pack. Learning mutates candidate graphs off-path and promotes only after replay and health gates pass.

## Runtime Shape

- Plugin runtime owns the `contextEngine` slot.
- Query path embeds the latest query, decides whether to use the brain, traverses the promoted pack, and assembles four ranked sections:
  - correction cards
  - route-selected evidence
  - toolcards and workflows
  - transcript support
- Episode and trace data are recorded on every qualifying routed turn.
- Harvesting attaches labels to exact episodes when possible.
- Worker logic applies updates, decay, teacher labels, mutation proposals, replay, and promotion off the hot path.

## Learning Contract

- Finite-horizon stochastic traversal with `STOP`.
- Full-trajectory REINFORCE with a baseline.
- Seed choice is part of the learned policy, not a pure semantic prefilter.
- Reward sources are trust-ranked: `human > self > scanner > teacher`.
- Structural changes are candidate-graph proposals, not live mutations.

## Operator Contract

- `init`, `status`, `trace`, `replay`, `promote`, `rollback`, `disable`, `enable`, and `doctor` must expose enough state to explain routing and promotion decisions.
- Every routed turn must either use the brain or emit a concrete skip reason.
- Every promotion must explain why it passed.

## Release Contract

- Mechanism proof on toy graphs.
- Recorded-session replay benchmark.
- Full OpenClaw integration validation.
- Documentation and website copy must match the actual shipped behavior.
