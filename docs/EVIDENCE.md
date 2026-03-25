# OpenClawBrain v2 — Evidence Ladder

This document defines what proof must exist before public claims are treated as frozen.

The point is not to accumulate logs for their own sake. The point is to make the repo's public claims auditable.

## What counts as evidence

Evidence should answer four questions clearly:
1. **What exact claim was being tested?**
2. **What command or harness produced the result?**
3. **What environment/model/config did it run with?**
4. **What remains open after this run?**

If a bundle cannot answer those questions quickly, it is not a good release artifact yet.

## Artifact layout

Store release and benchmark artifacts under:

```text
docs/evidence/YYYY-MM-DD/<git-sha>/
```

Each serious bundle should contain at minimum:
- `summary.md`
- `validation-report.json`
- `status.json`
- `doctor.json`
- `config-snapshot.json`
- `logs.txt`

If a routed path is part of the claim, include:
- `trace.json`

For Level 4 host-install runs, also include the pre-run diagnostic ladder outputs:
- `status-all.txt`
- `gateway-probe.txt`
- `gateway-status.txt`
- `channels-status.txt`

If a run is partial, `summary.md` must say exactly what was and was not proven.

## Reading evidence correctly

Not every bundle under `docs/evidence/` is a frozen release proof.

Three categories matter:

### 1. Frozen proof bundles
Use these when the repo is claiming a result publicly.

### 2. Partial proof bundles
Useful for tracking progress, but the summary must explicitly say the run was partial and what boundary remains open.

### 3. Historical failure bundles
Useful when they truthfully capture seam drift or operator failures, but they must not be mistaken for the current success boundary.

In practice, a lot of recent evidence is still in category 2 or 3.

## Proof ladder

### Level 1 — Mechanism proofs

Purpose: prove the runtime and learning primitives in isolation.

Primary surfaces:
- `test/brain-core/policy.test.ts`
- `test/brain-core/traverse.test.ts`
- `test/brain-core/update.test.ts`
- `test/brain-core/seed-policy.test.ts`
- `test/brain-runtime/service.test.ts`
- `test/brain-runtime/harvester.test.ts`
- `src/brain-runtime/worker-supervisor.ts`
- `src/brain-worker/protocol.ts`
- `scripts/validate-brain-runtime-behavior.ts`

Required claims:
- finite-horizon traversal with `STOP`
- full-trajectory update behavior
- seed routing is learnable
- immediate `brain_teach` retrieval works
- serve-from-last-promoted-pack survives worker crash at runtime level
- child-worker supervision records restart truth, reload acknowledgements, stale-lease takeover, and second-writer refusal
- raw harvesting preserves multiple concurrent evidence signals before worker-side label resolution collapses them
- structured self/scanner evidence preserves richer raw metadata when available
- episode attribution and resolver attribution are audited rather than implied

### Level 2 — Recorded replay proofs

Purpose: prove candidate changes do not regress protected behavior.

Primary surfaces:
- `src/brain-core/replay.ts`
- `src/brain-core/pack.ts`
- `src/brain-worker/worker.ts`
- `test/brain-core/replay.test.ts`

Required claims:
- promotion replay gate blocks regressions
- human-positive episodes do not regress silently
- candidate packs explain why they passed or failed
- mutation evaluation can be audited at the bundle boundary once that work lands

### Level 3 — Shadow proofs

Purpose: prove the runtime can record decisions safely without injecting learned context.

Primary surfaces:
- `src/brain-runtime/assembler-extension.ts`
- `src/brain-runtime/service.ts`
- `scripts/validate-openclaw-install.mjs`

Required claims:
- shadow mode records episode/trace ids
- shadow mode does not visibly inject the brain block
- host-surface status/trace stay truthful

### Level 4 — Disposable host-install proofs

Purpose: prove the plugin on the real OpenClaw host surface.

Primary surfaces:
- `scripts/validate-openclaw-install.mjs`
- `scripts/validate-brain-teach-session-bound.ts`
- future CI workflow surfaces
- `openclaw.plugin.json`
- `README.md`

Required claims:
- recurrent route used
- static lookup bypassed when appropriate, or remaining host-surface drift explicitly classified and truth-frozen
- shadow mode recorded
- `brain_teach` proven through the deterministic session-bound harness, or honestly scoped out of the raw host-prompt boundary
- worker-down host proof stays narrow: serving continues from the last promoted pack and host-visible worker health/exit truth remains visible
- `skip_no_embedding` and `skip_uninitialized` asserted explicitly

## Release checklist

Do not claim a release candidate is fully proven unless the bundle includes:
- exact commit SHA
- exact validation command(s)
- model + embedding configuration used
- pass/fail results for host harness assertions
- status and doctor snapshots
- at least one trace proving the routed path being claimed
- a short summary of what remains open

For an operator-grade release, the proof ladder should also be enforced by CI or another repeatable release gate rather than living only as prose.

## Current proof truth

As of the current trunk:

- **Level 1:** ✅ real — mechanism tests pass
- **Level 2:** ✅ real — replay gate exists, mutation bundles implemented
- **Level 3:** ✅ real — shadow mode recorded with episode/trace ids
- **Level 4:** ✅ frozen — the current public install / attach / `status --detailed` / `proof` lane is frozen on the exercised host surface

**Evidence bundles available:**
- `docs/evidence/2026-03-16/4ccd71a22418b9170128b8d948f5a95801a10380/` — sterile lane run with all 7 assertions passing:
  - `teachRetrieval`: PASS (taught node retrieved correctly)
  - `workerDownFailOpen`: PASS (serving continues after worker crash)
  - `recurrentQuery`: PASS
  - `shortLookup`: PASS (bypass evidence captured)
  - `shadowMode`: PASS (no injected context visible)
  - `noEmbedding`: PASS
  - `uninitialized`: PASS
- Current operator proof lane: `npx @openclawbrain/cli proof --openclaw-home ~/.openclaw --skip-install --skip-restart`
  - captures `summary.md`, `steps.json`, `verdict.json`, raw step logs, startup breadcrumbs, and runtime-load-proof snapshots
  - freezes install/runtime/reporting truth for the exercised host surface

**Remaining boundaries (honestly scoped):**
- Raw prompt-driven `openclaw agent --local` is **not** the release proof boundary for `brain_teach` — the deterministic session-bound harness is the proof surface
- Frozen operator proof does **not** yet imply universal dated citations, exact attribution on every learning/supervision path, same-gateway multi-profile support, or broad live answer-quality gains
- Exact promoted-pack ids and traced counts are environment-specific proof details, not stable public constants

The repo is now past theory-only and has frozen evidence for the operator install/runtime lane. The next proof rungs are about attribution clarity and live gain, not basic host proof capture.

## What CI should eventually enforce

The intended release gate should eventually require at least:
- tests
- package verification (`npm pack --dry-run` or stronger equivalent)
- evidence-ladder checks appropriate to the release claim
- host/runtime validation checks that match the repo's public contract

Until that exists, docs must stay honest that the evidence ladder is partly documented discipline rather than a fully enforced release boundary.
