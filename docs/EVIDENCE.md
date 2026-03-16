# OpenClawBrain v2 — Evidence Ladder

This document defines what proof must exist before public claims are treated as frozen.

## Artifact layout

Store release and benchmark artifacts under:

```text
docs/evidence/YYYY-MM-DD/<git-sha>/
```

Each bundle should contain at minimum:

- `status.json`
- `doctor.json`
- `trace.json`
- `validation-report.json`
- `config-snapshot.json`
- `logs.txt`
- `summary.md`

For Level 4 host-install runs, the bundle should also include the pre-run diagnostic ladder outputs:

- `status-all.txt`
- `gateway-probe.txt`
- `gateway-status.txt`
- `channels-status.txt`

If a proof run is partial, the `summary.md` should say exactly what was and was not proven.

## Proof ladder

### Level 1 — Mechanism proofs

Purpose: prove the math/runtime primitives in isolation.

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
- raw harvesting preserves multiple concurrent evidence signals with extractor metadata before worker-side trust resolution collapses them into labels
- structured tool-result/function-output parts can generate self-evidence even when flattened stored text is empty
- structured self-evidence records richer raw metadata when available (`messageId`, tool/call identity, command/exit-code/file/artifact hints)
- scanner guidance can also record structured metadata when a runbook/checklist message is backed by explicit message parts, including tool chains and non-tool guidance parts like file/snapshot/subtask blocks
- explicit episode attribution, resolver attribution, and recent-conversation fallback are all audited rather than implied

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
- future: `.github/workflows/validate-openclaw-install.yml`
- `openclaw.plugin.json`
- `README.md`

Required claims:
- recurrent route used
- static lookup bypassed when appropriate, or the remaining host-surface drift is explicitly classified/truth-frozen
- shadow mode recorded
- `brain_teach` proven by a deterministic session-bound harness (`scripts/validate-brain-teach-session-bound.ts`) with 20/20 identical passes, or honestly classified as out of scope for raw prompt-driven host proof
- worker-down host proof stays narrow: last-promoted-pack serving continues and host status surfaces unhealthy/exit truth
- `skip_no_embedding` and `skip_uninitialized` asserted explicitly

## Release checklist

Do not claim a release candidate is fully proven unless the artifact bundle includes:

- the exact commit SHA
- the validation command(s)
- the model + embedding configuration used
- pass/fail results for host harness assertions
- status and doctor snapshots
- at least one trace proving the routed path being claimed
- a short markdown summary of what remains open

## Current proof truth

As of the current trunk:

- **Level 1:** materially real
- **Level 2:** present but not yet bundle-complete
- **Level 3:** partially real on the host surface
- **Level 4:** not frozen; deterministic session-bound `brain_teach` proof now exists under `docs/evidence/YYYY-MM-DD/<git-sha>/brain-teach-session-bound/`, short-static host classification is currently truth-frozen as stale current-OpenClaw host seam drift under `docs/evidence/YYYY-MM-DD/<git-sha>/short-static-classification/`, and the final narrow worker-down host claim still remains open

That means the repo is already beyond theory-only, but it does **not** yet have a frozen release-evidence ladder.
