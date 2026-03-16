# OpenClawBrain v2 — Release Contract

This is the fast truth surface for the repo.

Use these public labels consistently:

- **paper-faithful core**
- **live-path implemented**
- **operationally validated**

Current truthful state:

- **paper-faithful core:** yes
- **live-path implemented:** yes
- **operationally validated:** not yet

That is the contract. The repo is already beyond "foundation only," but it is not yet at an honest 1.0 operating state.

## 1. True in code now

These are safe public claims today.

### Paper-faithful routing core
- **Finite-horizon traversal with `STOP`**
  - Code: `src/brain-core/traverse.ts`, `test/brain-core/traverse.test.ts`
- **Terminal reward with baseline, not shaping rewards**
  - Code: `src/brain-core/episode.ts`, `src/brain-core/update.ts`, `src/brain-worker/worker.ts`, `test/brain-core/update.test.ts`
- **Stochastic policy over actions**
  - Code: `src/brain-core/policy.ts`, `src/brain-core/traverse.ts`, `test/brain-core/policy.test.ts`
- **Full-trajectory REINFORCE updates**
  - Code: `src/brain-core/update.ts`, `src/brain-worker/worker.ts`, `test/brain-core/integration.test.ts`
- **Learned seed routing as part of the policy surface**
  - Code: `src/brain-core/types.ts`, `src/brain-core/traverse.ts`, `src/brain-core/update.ts`, `src/brain-store/store.ts`, `test/brain-core/seed-policy.test.ts`
- **Immutable promoted packs for serving**
  - Code: `src/brain-core/pack.ts`, `src/brain-runtime/service.ts`, `src/brain-store/store.ts`, `test/brain-runtime/service.test.ts`

### Live runtime path
- **Explicit runtime decisions** (`use_brain`, `shadow`, named skip modes)
  - Code: `src/brain-runtime/assembler-extension.ts`, `test/brain-runtime/assembler-extension.test.ts`
- **Correction-first assembly**
  - Code: `src/brain-runtime/assembler-extension.ts`
- **Immediate `brain_teach` retrieval path**
  - Code: `src/brain-runtime/service.ts`, `src/brain-runtime/tools.ts`, `test/brain-runtime/service.test.ts`
- **Episode and trace recording on the live path**
  - Code: `src/brain-runtime/service.ts`, `src/brain-core/trace.ts`, `test/brain-runtime/service.test.ts`
- **Serve from the last promoted pack even when the worker is unavailable**
  - Code: `src/brain-runtime/service.ts`, `test/brain-runtime/service.test.ts`, `scripts/validate-brain-runtime-behavior.ts`
- **Child-worker mode exists and is real**
  - Code: `openclaw.plugin.json`, `src/brain-runtime/service.ts`, `src/brain-worker/child-runner.ts`, `test/brain-runtime/service.test.ts`

## 2. Implemented but not frozen

These are real enough to build on, but not frozen enough to oversell.

- **Host-surface validation harness**
  - Current files: `scripts/validate-openclaw-install.mjs`, `scripts/validate-brain-runtime-behavior.ts`, `scripts/validate-short-static-classification.ts`
  - Truth: recurrent routing, shadow mode, and current host checks run inside a dedicated sterile validation lane with per-run diagnostic artifacts; deterministic session-bound `brain_teach` proof now exists, but the current raw host lane is blocked by stale OpenClaw seam drift (`plugins.slots.contextEngine` rejected, `api.registerContextEngine` removed) and the final narrow worker-down host claim is still incomplete.
  - Boundary: raw prompt-driven `openclaw agent --local` is **not** the release proof boundary for `brain_teach`; that claim is now closed by the deterministic session-bound harness rather than raw host prompting.
  - Boundary: short-static host drift is currently truth-frozen as stale current-OpenClaw host seam drift, not as a resolved semantic behavior claim.
  - Boundary: worker-down host proof is claimed only at the exact host-visible boundary actually proven (continued serving from the last promoted pack + unhealthy worker status / exit truth), not as a stronger deterministic crash-observation claim.
- **Child-worker serving boundary**
  - Current files: `src/brain-runtime/service.ts`, `src/brain-runtime/worker-supervisor.ts`, `src/brain-worker/child-runner.ts`, `src/brain-worker/protocol.ts`, `src/brain-cli.ts`
  - Truth: the child worker now runs behind a dedicated supervisor boundary with explicit protocol messages, restart accounting, reload acknowledgements, lease protection, and stronger status/doctor truth. `in_process` mode remains available only as a dev-only fallback and must not be treated as the production operator boundary.
- **Raw evidence → resolved labels flow**
  - Current files: `src/brain-runtime/harvester-extension.ts`, `src/brain-runtime/evidence-detectors.ts`, `src/brain-harvest/*.ts`, `src/brain-worker/worker.ts`, `src/brain-store/store.ts`
  - Truth: explicit evidence tables and trust-ordered resolution are real, but source extraction still leans heavily on heuristics.
- **Replay-gated promotion**
  - Current files: `src/brain-core/replay.ts`, `src/brain-core/pack.ts`, `src/brain-worker/worker.ts`
  - Truth: promotion gates exist, but mutation evaluation is still closer to proposal-by-proposal than bundle-level replay decisions.

## 3. Not done yet

These are still active work and must not be described as complete.

- **Frozen host-surface proof for worker-down fail-open on the current host seam**
  - Primary files: `scripts/validate-openclaw-install.mjs`, `scripts/validate-brain-teach-session-bound.ts`, `scripts/validate-short-static-classification.ts`, `src/brain-runtime/tools.ts`, `src/brain-runtime/service.ts`
  - Required truth before this is marked done: keep deterministic session-bound `brain_teach` proof frozen, adapt the current OpenClaw host seam, and then land a narrow host worker-down claim that matches the actual artifact bundle.
- **Resolved short-static-lookup host-surface semantics on the adapted current host seam**
  - Primary files: `src/brain-runtime/assembler-extension.ts`, `scripts/validate-openclaw-install.mjs`, `scripts/validate-short-static-classification.ts`
- **Bundle-based mutation evaluation with clear pass/fail explanations**
  - Primary files: `src/brain-core/mutator.ts`, `src/brain-worker/worker.ts`, `src/brain-store/store.ts`, `src/brain-store/migrations.ts`
- **Frozen proof ladder with dated release artifacts**
  - Primary files: `docs/EVIDENCE.md`, `docs/evidence/`, `scripts/validate-openclaw-install.mjs`
- **Green full-repo `npx tsc --noEmit`**
  - Primary files: `tsconfig.json`, `package.json`, SDK-boundary imports
- **Boring install / recovery path for another operator**
  - Primary files: `README.md`, `docs/configuration.md`, `openclaw.plugin.json`, future release workflow/evidence files

## Safe public summary

> OpenClawBrain v2 already has a paper-faithful routing core and a real live runtime path. What remains is the operational hardening, host-surface proof, mutation-bundle evaluation, and release-evidence layer.
