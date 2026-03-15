# OpenClawBrain v2 — Release Contract

This document is the fast truth surface for the repo.

Use these three public labels consistently:

- **paper-faithful core**
- **live-path implemented**
- **operationally validated**

Current truthful state:

- **paper-faithful core:** yes
- **live-path implemented:** yes
- **operationally validated:** not yet

That is the whole point of this contract: the repo is already past “foundation only,” but it is not yet at a fully proven 1.0 operating state.

## 1. Algorithmic claims true now (`paper-faithful core`)

These are true in the current repo and can be claimed now.

- **Finite horizon traversal**
  - The traversal path is bounded by explicit hop limits and a `STOP` action in the policy/traversal code.
  - See: `src/brain-core/traverse.ts`, `test/brain-core/traverse.test.ts`
- **Terminal reward rather than per-step reward shaping**
  - Episodes are recorded first, then rewards/labels are attached afterward and consumed by the worker during update.
  - See: `src/brain-core/episode.ts`, `src/brain-worker/worker.ts`, `test/brain-core/integration.test.ts`
- **Stochastic policy over actions**
  - The route policy samples traversal decisions from a distribution rather than hard-coding a deterministic next edge only.
  - See: `src/brain-core/policy.ts`, `src/brain-core/traverse.ts`, `test/brain-core/policy.test.ts`
- **Full-trajectory REINFORCE updates**
  - Updates are computed across the recorded trajectory, not as a single one-step chosen-edge patch.
  - See: `src/brain-core/update.ts`, `src/brain-worker/worker.ts`, `test/brain-core/update.test.ts`
- **Learned seed routing**
  - The repo already learns from the virtual `__START__` / `START_NODE_ID` head, so seed selection is part of the learnable policy surface.
  - See: `src/brain-core/types.ts`, `src/brain-core/traverse.ts`, `src/brain-core/update.ts`, `test/brain-core/update.test.ts`
- **Immutable promoted packs**
  - Serving happens from promoted snapshots; mutable working state is promoted into immutable pack snapshots.
  - See: `src/brain-core/pack.ts`, `src/brain-store/store.ts`, `src/brain-runtime/service.ts`, `test/brain-runtime/service.test.ts`

### Claims this section does **not** make

Do **not** collapse these truths into claims like:

- “the full learning product is finished”
- “the repo is operationally proven”
- “the worker/process boundary is already production-safe”

Those belong to the operational section below, and they are not done yet.

## 2. Product claims true now (`live-path implemented`)

These product-path behaviors are already wired into the real OpenClawBrain runtime.

- **Recurrence gate + explicit skip reasons**
  - The assembler decides whether to use brain routing or bypass with explicit reasons such as `skip_short_static_lookup`, `skip_no_embedding`, `skip_uninitialized`, and `skip_budget_too_small`.
  - See: `src/brain-runtime/assembler-extension.ts`, `test/brain-runtime/assembler-extension.test.ts`
- **Shadow mode**
  - Shadow mode records route episodes/traces without injecting the brain context block into the prompt.
  - See: `src/brain-runtime/assembler-extension.ts`, `src/brain-runtime/service.ts`
- **Correction-first assembly**
  - Injected brain context is structured with correction cards first, then route-selected evidence, then toolcards/playbooks, then transcript support.
  - See: `src/brain-runtime/assembler-extension.ts`
- **Immediate `brain_teach` retrieval path**
  - `brain_teach` embeds the taught node immediately, links it to recent route context/seed region, applies an inhibitory edge toward the misroute when applicable, and promotes a new pack.
  - See: `src/brain-runtime/service.ts`, `src/brain-runtime/tools.ts`, `test/brain-runtime/service.test.ts`
- **Replay-gated promotion**
  - The worker runs a replay gate before promotion and blocks promotion when health/replay conditions fail.
  - See: `src/brain-worker/worker.ts`, `src/brain-core/pack.ts`, `test/brain-core/replay.test.ts`
- **Episode + trace recording on the live path**
  - `BrainService.query()` records both episodes and traces against the active promoted pack.
  - See: `src/brain-runtime/service.ts`, `test/brain-runtime/service.test.ts`
- **Serve-from-last-promoted-pack fallback**
  - The serving path reads from the current promoted snapshot and does not require mutable-worker success to continue serving.
  - See: `src/brain-runtime/service.ts`, `README.md#fallback-behavior`

### How to describe the repo publicly right now

Short honest version:

> OpenClawBrain v2 already has a paper-faithful routing core and a real live runtime path. What is still unfinished is the operational hardening and proof layer.

That sentence is safe. Stronger “fully done / fully proven / production validated” wording is not.

## 3. Operational claims not yet true (`operationally validated`)

These are **not** true yet and should be described as active work, not delivered fact.

- **Out-of-process learner**
  - The learner still runs in-process via `setInterval` inside the plugin runtime.
  - Current evidence: `src/brain-worker/worker.ts`, `src/brain-runtime/service.ts`
- **Frozen benchmark and proof artifacts**
  - The repo does not yet ship a frozen evidence ladder that ties mechanism tests, replay benchmarks, shadow-mode benchmarks, and live-install proofs into one reproducible artifact structure.
- **Full install validation matrix**
  - There is not yet a disposable host-app harness that another machine can run to validate linked install, init, routing decisions, teach retrieval, shadow behavior, and worker-down fail-open behavior.
- **Green full-repo typecheck**
  - Full `npx tsc --noEmit` is still affected by upstream `openclaw/plugin-sdk` type drift.
- **Structured evidence harvesting**
  - Harvesting still leans heavily on pattern detection rather than the target structured human/self/scanner evidence flow.
  - Current evidence: `src/brain-runtime/harvester-extension.ts`
- **Bundle-based mutation evaluation**
  - Mutation replay is still closer to proposal-by-proposal evaluation than the intended clustered bundle evaluation with explicit pass/fail explanations.
  - Current evidence: `src/brain-worker/worker.ts`

## What 1.0 means from here

The repo reaches an honest 1.0 only when all three labels are true at once:

1. **paper-faithful core** — already true
2. **live-path implemented** — already true
3. **operationally validated** — still to finish

The shortest path from the current state is:

1. freeze this contract in the README/site
2. build the disposable OpenClaw install validation harness
3. move the learner out of process
4. replace heuristic harvesting with structured evidence flow
5. upgrade mutation replay to bundle evaluation
6. freeze evidence artifacts and packaging/install recovery

Until then, the correct public framing is:

- **paper-faithful core:** yes
- **live-path implemented:** yes
- **operationally validated:** not yet
