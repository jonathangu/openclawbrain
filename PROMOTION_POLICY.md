# OpenClawBrain: Promotion Policy

## Pack Promotion Gates

A candidate pack is promoted to serving only if ALL gates pass:

| Gate | Threshold | Blocks Promotion If |
|---|---|---|
| firedPerQuery | ≥ 1.0 | Brain fires too few nodes per query |
| dormantPercent | < 30% | Too many nodes never fire |
| orphanCount | < 10 | Too many disconnected nodes |
| humanEpisodesRegressed | = 0 | Any human-confirmed episode would change routing |

## Replay Gate

Before promotion, replay the last 100 episodes against the candidate graph.
Compare routing decisions: if human-positive episodes would now route differently, reject.

## Rollback

Any promoted pack can be rolled back via `brain_status` → rollback.
The previous promoted pack becomes current again.

## Evidence Ladder

Do not claim a rung you haven't measured.

| Rung | What | Exit Criterion |
|---|---|---|
| 1. Mechanism proof | Unit tests prove REINFORCE matches paper | Tests pass with full-trajectory credit |
| 2. Recorded-session benchmark | Replay 20 real sessions | Learned policy beats random baseline |
| 3. Shadow eval | Run brain in parallel with LCM-only | No regression in context quality |
| 4. Narrow online rollout | Enable for single user | Health metrics stable for 1 week |
| 5. General availability | Enable by default | Evidence from rungs 1-4 documented |

## Mutation Validation

Every structural mutation (split/merge/prune/connect/inject) is a proposal.
Proposals are validated against the replay suite before application.
A mutation ledger tracks all proposals with evidence and outcomes.
