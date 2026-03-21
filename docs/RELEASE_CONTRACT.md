# OpenClawBrain v2 — Release Contract

This is the sharp truth surface for the repo.

Use these public labels consistently:
- **paper-faithful core**
- **live-path implemented**
- **operationally validated**

Current truthful state:
- **paper-faithful core:** yes
- **live-path implemented:** yes
- **operationally validated:** sterile harness passes 7/7; full host-surface proof bundle pending host-seam adaptation

That is the contract. The repo is beyond "foundation only," with a paper-faithful core and real live runtime. The sterile host harness passes all 7 runtime assertions. The remaining gap is the full end-to-end host-surface proof bundle capture.

## 1. True in code now

These are safe public claims today.

### Paper-faithful routing core
- **Finite-horizon traversal with `STOP`**
  - Code: `src/brain-core/traverse.ts`, `test/brain-core/traverse.test.ts`
- **Terminal reward with baseline rather than shaping rewards**
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
- **Correction-first assembly and summary-aware routing guidance**
  - Code: `src/brain-runtime/assembler-extension.ts`, `src/brain-runtime/summary-routing-policy.ts`, `src/assembler.ts`
- **Immediate `brain_teach` retrieval path and explicit user-correction commit lane**
  - Code: `src/brain-runtime/service.ts`, `src/brain-runtime/tools.ts`, `src/brain-runtime/user-memory-proposals.ts`, `test/brain-runtime/service.test.ts`
- **Episode and trace recording on the live path**
  - Code: `src/brain-runtime/service.ts`, `src/brain-core/trace.ts`, `test/brain-runtime/service.test.ts`
- **Serve from the last promoted pack even when the worker is unavailable**
  - Code: `src/brain-runtime/service.ts`, `test/brain-runtime/service.test.ts`, `scripts/validate-brain-runtime-behavior.ts`
- **Child-worker mode is real**
  - Code: `openclaw.plugin.json`, `src/brain-runtime/service.ts`, `src/brain-runtime/worker-supervisor.ts`, `src/brain-worker/child-runner.ts`, `test/brain-runtime/service.test.ts`
- **Structured raw evidence and worker-side trust resolution are real**
  - Code: `src/brain-runtime/harvester-extension.ts`, `src/brain-runtime/evidence-detectors.ts`, `src/brain-harvest/*.ts`, `src/brain-worker/worker.ts`, `src/brain-store/store.ts`

## 2. Implemented but not frozen

These are real enough to build on, but not frozen enough to oversell.

### Host-surface validation harness
- Current files: `scripts/validate-openclaw-install.mjs`, `scripts/validate-brain-runtime-behavior.ts`, `scripts/validate-brain-teach-session-bound.ts`, `scripts/validate-short-static-classification.ts`
- Truth:
  - deterministic session-bound `brain_teach` proof exists
  - deterministic runtime proof for teach retrieval and worker-down fail-open exists
  - OpenClawBrain now includes a hook-based compatibility bridge for hosts where `api.registerContextEngine` is gone
  - the sterile harness no longer writes the dead `plugins.slots.contextEngine` slot
- Boundary:
  - raw prompt-driven `openclaw agent --local` is **not** the release proof boundary for `brain_teach`
  - the full sterile host harness is still **not frozen end to end** because the host-turn proof bundle still stalls before completion on the sterile boundary
  - until that host lane is frozen, short-static host semantics and the final narrow worker-down host claim are still not closed at the host boundary

### Child-worker serving boundary
- Current files: `src/brain-runtime/service.ts`, `src/brain-runtime/worker-supervisor.ts`, `src/brain-worker/child-runner.ts`, `src/brain-worker/protocol.ts`, `src/brain-cli.ts`
- Truth: the child worker now runs behind a dedicated supervisor boundary with explicit protocol messages, restart accounting, reload acknowledgements, lease protection, and stronger status/doctor truth. `in_process` remains a dev-only fallback rather than the operator boundary.

### Raw evidence → resolved labels flow
- Current files: `src/brain-runtime/harvester-extension.ts`, `src/brain-runtime/evidence-detectors.ts`, `src/brain-harvest/*.ts`, `src/brain-worker/worker.ts`, `src/brain-store/store.ts`, `src/engine.ts`
- Truth: multiple concurrent raw signals can be persisted before worker-side resolution; structured tool/function-output parts feed self-evidence detection; scanner guidance can bind to structured message parts; and same-trust scanner conflicts now prefer structured extractors over heuristic-only scanner signals.
- Boundary: source extraction still leans too heavily on heuristics outside the structured cases already covered.

### Replay-gated promotion
- Current files: `src/brain-core/replay.ts`, `src/brain-core/pack.ts`, `src/brain-worker/worker.ts`
- Truth: promotion gates exist and matter.
- Boundary: mutation evaluation is still closer to proposal-level checks than the intended bundle-level replay contract.

### Packaging and release boundary
- Current files: `package.json`, `packages/openclaw/package.json`, `packages/cli/package.json`, `README.md`, `docs/lifecycle.md`, `docs/configuration.md`, `docs/EVIDENCE.md`, future CI/release workflow surfaces
- Truth: the split packages are published and the public-registry operator lane is real:
  - `openclaw plugins install @openclawbrain/openclaw@0.4.0`
  - `npx @openclawbrain/cli@0.4.3 install --openclaw-home ~/.openclaw`
  - `openclaw gateway restart`
  - `npx @openclawbrain/cli@0.4.3 status --openclaw-home ~/.openclaw --detailed`
- Boundary: some hosts still warn about a plugin id mismatch because the manifest uses `openclawbrain` while the package/entry hint uses `openclaw`; release verification and docs alignment still need to keep that caveat visible rather than implying the seam is fixed.

## 3. Not done yet

These are still active work and must not be described as complete.

- **Frozen end-to-end host-surface proof on the current host seam**
  - Required truth before done: the sterile host harness must complete again, and the resulting artifacts must freeze the actual current host claims rather than older seam failures.
- **Bundle-based mutation evaluation with clear pass/fail explanations**
  - Primary files: `src/brain-core/mutator.ts`, `src/brain-worker/worker.ts`, `src/brain-store/store.ts`, `src/brain-store/migrations.ts`
- **CI-enforced proof ladder / release gates**
  - Primary files: future workflow surfaces, `package.json`, `docs/EVIDENCE.md`
- **Host/plugin id alignment for the split packages**
  - Primary files: `packages/openclaw/openclaw.plugin.json`, `packages/openclaw/package.json`, `README.md`, docs packaging boundary
- **Boring install / validation / recovery path for another operator**
  - Primary files: `README.md`, `docs/configuration.md`, `openclaw.plugin.json`, validation scripts

## Safe public summary

> OpenClawBrain v2 already has a paper-faithful routing core, a real live runtime path, and a published split-package operator lane. The remaining work is mainly host-surface proof, release engineering, bundle-level mutation evaluation, plugin-id seam cleanup, and cleaner operator truth.
