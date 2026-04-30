# OpenClawBrain State and Next Path

Date: 2026-04-30
Repo state: clean-room V5 rebuild from a blank local repo.
Current decision: **CONTINUE**.

## Current state

OpenClawBrain is now an evidence-first rebuilt project, not yet a broadly installable product.

What exists and is verified:

- V5 evidence scoreboard and result contracts.
- Real trace admission gate with privacy/redaction checks.
- Runtime event capture and candidate export path.
- Deterministic runtime decision interface for redacted agent turns.
- Four-backend eval harness:
  - `none`
  - `correction-only`
  - `correction+heuristics`
  - `full-ocb`
- Blind packet generation and judgment import.
- Generated `/results` and 30-day decision memo.
- Production evidence run from real privacy-scrubbed historical session logs across profiles:
  - `pelican`: 23 traces
  - `main` / GuClaw: 8 traces
  - `bountiful`: 8 traces
  - `family`: 1 trace
- Required slice coverage:
  - direct-answer: 6
  - continuation: 6
  - correction-follow-up: 8
  - retrieval-heavy: 6
  - tool-heavy: 6
  - stale-memory-conflict: 8
- Product policy scoring over one selected backend per trace:
  - direct-answer -> `none`
  - correction-follow-up / stale-memory-conflict -> `correction-only`
  - continuation / retrieval-heavy / tool-heavy -> `full-ocb`
- Latest production result:
  - selected product policy wins: `40/40`
  - selected product policy harms: `0/40`
  - evidence E2E: `true`
  - decision: `CONTINUE`

Canonical verification commands:

```bash
pnpm ocb:e2e:smoke
pnpm ocb:e2e:production
pnpm ocb:traces:production-status
```

## What OpenClawBrain is not yet

OpenClawBrain is **not yet installable or widely usable**.

Missing product pieces:

- No public npm package/install path.
- `package.json` is still `private: true`.
- No stable CLI for normal users.
- No OpenClaw gateway/profile plugin integration.
- No live profile-bound runtime loop that automatically influences real assistant turns.
- No package-level config schema, migrations, or upgrade path.
- No user-facing status/proof command for “How is the brain doing?”
- No release docs, onboarding docs, or public claim boundary.

The project has proof that the selected policy is worth continuing; it does not yet have the distribution/runtime shell that makes it useful to many people.

## Product principle for the next phase

The public product should be simple:

> One OpenClaw gateway per machine, many OpenClaw profiles, brain behavior attached at the profile layer, one public OpenClawBrain install path.

Do not expose competing public package identities or multiple version stories. Internals can be modular, but users should experience one product.

## Next path to installable and widely used

### Phase 1 — Runtime integration

Goal: make the selected policy run inside real OpenClaw turns.

Deliverables:

1. Define the live intervention contract:
   - input: redacted current turn, profile identity, available memory candidates, tool/context metadata
   - output: `stay_silent`, correction/context intervention, full-context intervention, or proof-only event
2. Implement the selected policy from the evidence run:
   - direct-answer -> stay silent / no intervention
   - correction-follow-up and stale-memory-conflict -> correction-only intervention
   - continuation, retrieval-heavy, tool-heavy -> full-context intervention
3. Attach the policy at the OpenClaw profile boundary, not globally.
4. Preserve strict profile separation across `main`, `pelican`, `bountiful`, etc.
5. Emit redacted runtime proof events for every decision.
6. Add a kill switch and conservative default.

Exit gate:

- A real OpenClaw profile can run with OpenClawBrain enabled.
- A served turn can be improved by the policy without manual repo commands.
- `stay_silent` is a first-class successful outcome.

### Phase 2 — Install surface

Goal: a new user can install and enable OpenClawBrain without cloning this repo as a developer.

Deliverables:

1. Public package plan with one user-visible version.
2. CLI commands:
   - `openclawbrain install`
   - `openclawbrain enable --profile <profile>`
   - `openclawbrain disable --profile <profile>`
   - `openclawbrain status --profile <profile>`
   - `openclawbrain proof --profile <profile>`
3. OpenClaw integration docs and config patching path.
4. First-run wizard or one-shot setup command.
5. Safe defaults:
   - profile-local storage
   - private logs stay local
   - no raw transcript upload
   - redacted proof artifacts only
6. Upgrade/uninstall path.

Exit gate:

- Fresh machine / fresh OpenClaw profile can install, enable, ask status, run a turn, and see proof.

### Phase 3 — Dogfood proof

Goal: prove the installable product works on real profiles.

Deliverables:

1. Dogfood on `main`, `pelican`, and `bountiful` profiles.
2. Many-profile separation proof on one gateway.
3. Correction-follow-up demonstration:
   - user corrects preference
   - later turn gets better
   - proof shows why
4. Continuation demonstration:
   - user says “continue”
   - OpenClawBrain provides bounded state/context
   - assistant avoids redundant questions
5. Tool-heavy demonstration:
   - OpenClawBrain suggests or gates read-only verification before claims
6. Honest status report generated from live data.

Exit gate:

- A non-author user can reproduce one “it remembered/helped at the right time” moment and inspect proof.

### Phase 4 — Release and adoption

Goal: make OpenClawBrain usable by many people without overclaiming.

Deliverables:

1. README rewritten for users, not just rebuild notes.
2. Install docs.
3. Configuration docs.
4. Troubleshooting docs.
5. Privacy and local-data docs.
6. Examples:
   - personal assistant profile
   - project profile
   - family/admin profile
   - coding/project profile
7. Public claim boundary:
   - say it is a selective intervention layer
   - do not claim generic intelligence or universal memory improvement
   - show evidence and caveats
8. First tagged release.
9. CI gate for smoke + production scoreboard commands.

Exit gate:

- A GitHub visitor can understand what it does, install it, enable it, and verify it helped.

## Recommended immediate next task

Start **Phase 1: Runtime integration**.

The detailed productization plan is now in [`docs/PRODUCTIZATION_PLAN.md`](./PRODUCTIZATION_PLAN.md). Treat it as the canonical next-phase build order and release shape.

First concrete milestone:

> Build a profile-bound OpenClawBrain runtime adapter that applies the selected policy to real OpenClaw turns and writes redacted proof events, behind an opt-in profile config flag.

Suggested implementation order:

1. Add `packages/runtime-policy` with the selected-policy function and tests.
2. Add `packages/openclaw-integration` or equivalent adapter boundary.
3. Add a local config format for profile enablement.
4. Add `openclawbrain status` and `openclawbrain proof` stubs backed by runtime events.
5. Dogfood on one profile only, then expand to `main`, `pelican`, and `bountiful`.

## Current bottom line

OpenClawBrain has crossed the evidence decision gate.

The next job is not more scoreboard work. The next job is to turn the selected policy into an installable, profile-bound runtime product that normal OpenClaw users can enable, inspect, trust, and recommend.
