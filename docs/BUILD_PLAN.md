# OpenClawBrain Build Plan

Date: 2026-04-30
Status: post-evidence, pre-productization
Decision state: **CONTINUE**

## 0. Executive summary

OpenClawBrain should stop trying to win as a broad research artifact and start winning as a **small trustworthy runtime product**.

The product to build is:

- **one install path**
- **one public version story**
- **per-profile enablement**
- **conservative default behavior**
- **visible local proof of when it helped**
- **easy disable / uninstall**

The current repo already proves the selected policy is worth continuing:

- selected product policy wins: `40/40`
- selected product policy harms: `0/40`
- `evidence_e2e_complete=true`
- decision: `CONTINUE`

What is missing is not more scoreboard work. What is missing is the **runtime + install shell** that lets a normal OpenClaw user install it, turn it on for one profile, get one better turn, and inspect why.

---

## 1. Product goal

### What OpenClawBrain should be

> A local, profile-bound selective intervention layer for OpenClaw.

It should help a profile:

- remember corrections,
- continue bounded work,
- add relevant context for retrieval-heavy and tool-heavy turns,
- stay silent on direct answers,
- show local redacted proof of what it did.

### What it should not be

Do **not** position it as:

- general memory for everything
- a general intelligence upgrade
- a cloud memory system
- a product that uploads raw transcripts
- something that fires on every turn

The winning UX is: **quiet most of the time, useful at the right moments, inspectable after the fact**.

---

## 2. Current repo truth and code anchors

These are the current assets to build on instead of replacing blindly.

### Evidence and command surface

- `package.json`
  - canonical scripts already exist for trace admission, eval, runtime decision capture, evidence runs, and results generation.
  - important current truth: root package is still `private: true`, so publishing is not ready.

- `docs/results/COMMANDS.md`
  - canonical contract for the current command surface.
  - already defines the runtime evidence lane, fail-closed behavior, and required artifacts.

### Current runtime seam

- `scripts/runtime/decide.mjs`
  - current deterministic runtime decision interface.
  - already enforces redacted inputs, deterministic output, `fire` vs `stay_silent`, reproducibility metadata, and candidate export.
  - this is the strongest seed for the future runtime-policy package.

- `scripts/runtime/capture-event.mjs`
  - current redacted runtime event normalization and capture.
  - already writes candidate-only runtime events and manifest entries.

- `scripts/runtime/export-candidate.mjs`
  - converts runtime events into trace-admission candidates.
  - already fail-closes on raw/unredacted/secret-like fields.

### Eval semantics to preserve

- `packages/eval-harness/src/backend-types.ts`
  - defines the current backend contract and intervention vocabulary.

- `packages/eval-harness/src/backends/full-ocb.ts`
  - current full-context eval adapter.
  - useful as the semantic model for what `full_context` means in the product runtime.

### Product decision logic already proven

- `packages/results-site/src/decision.ts`
  - critical: scores only `product_selected=true` rows when computing product thresholds.
  - this is the code embodiment of the current product truth.

- `docs/STATE_AND_NEXT.md`
- `docs/PRODUCTIZATION_PLAN.md`
  - current repo docs describing the transition from evidence gate to productization.

### Trace and evidence pipeline to keep intact

- `scripts/traces/admit.mjs`
- `scripts/traces/status.mjs`
- `scripts/ocb-e2e-production.mjs`
- `scripts/evidence/build-session-log-production-traces.mjs`
- `scripts/evidence/judge-blind-packets.mjs`

These should remain as the evidence spine while the product shell is built around them.

---

## 3. Durable product constraints

These are already learned constraints and should become explicit product rules.

1. **One single public install path and version story.**  
   Source: `memory/2026-04-02.md#L27`.

2. **Safest install shape is per-profile activation roots.**  
   Example: `~/.openclawbrain/activation/<ProfileName>`.  
   Source: `memory/2026-03-11.md#L18`.

3. **Install should use a normal shell, not a live agent session.**  
   Source: `memory/2026-03-11.md#L28`.

4. **OpenClawBrain should attach at the profile boundary, not leak across profiles.**

5. **The product must be allowed to stay silent.** Silence is success on direct-answer turns.

6. **No raw transcript upload.** Redacted proof only.

7. **A truthful plugin should expose at least one first-class service beyond hook-only behavior.**  
   Source: `memory/2026-04-02.md#L27`.

---

## 4. Target product architecture

Build toward this repo shape:

```text
openclawbrain/
  packages/
    runtime-policy/
    openclaw-integration/
    proof-store/
    cli/
    installer/
  docs/
  scripts/
  eval/
```

### 4.1 `packages/runtime-policy`

Purpose: pure deterministic decision engine.

Owns:

- turn classification to selected product slices
- selected policy mapping
- confidence / abstain behavior
- proof event generation
- zero side effects

Primary exported function:

```ts
decideOpenClawBrainIntervention(input): InterventionDecision
```

### 4.2 `packages/openclaw-integration`

Purpose: bind OpenClawBrain to real OpenClaw profile turns.

Owns:

- reading profile config
- receiving redacted turn input
- reading candidate memories / bounded context
- calling runtime-policy
- injecting intervention payloads back into the runtime boundary
- writing proof events
- exposing a small first-class capability/service for runtime integrity/status

### 4.3 `packages/proof-store`

Purpose: local proof trail storage and rendering.

Owns:

- proof event schema
- append-only local event writes
- rotation / retention
- per-profile event queries
- `status` and `proof` summaries

### 4.4 `packages/cli`

Purpose: single user-facing command: `openclawbrain`.

Owns:

- install / enable / disable / status / proof / doctor / uninstall
- human-readable output
- `--json` machine output where useful

### 4.5 `packages/installer`

Purpose: installation and profile patching logic.

Owns:

- activation-root creation
- OpenClaw profile patching
- extension/link placement
- restart instructions / helper execution
- uninstall and cleanup behavior

---

## 5. Phase-by-phase build plan

## Phase 1 — Extract the runtime policy core

### Goal

Turn the selected evidence policy into a reusable pure package.

### Why first

Right now the repo has a minimal deterministic runtime seam in `scripts/runtime/decide.mjs`, but it is not yet the real selected policy engine. Productization starts by making that policy explicit, typed, tested, and reusable.

### Build steps

#### 1.1 Create `packages/runtime-policy`

Add:

```text
packages/runtime-policy/
  package.json
  src/types.ts
  src/classify-turn.ts
  src/select-policy.ts
  src/decide.ts
  src/proof-event.ts
  test/*.test.ts
```

#### 1.2 Define runtime types

Recommended types:

```ts
export type RuntimeMode = "off" | "proof-only" | "conservative" | "active";

export type TurnSlice =
  | "direct-answer"
  | "continuation"
  | "correction-follow-up"
  | "retrieval-heavy"
  | "tool-heavy"
  | "stale-memory-conflict"
  | "unknown";

export type InterventionDecision =
  | { kind: "stay_silent"; slice: TurnSlice; proof: ProofEvent }
  | { kind: "correction_only"; slice: TurnSlice; message: string; proof: ProofEvent }
  | { kind: "full_context"; slice: TurnSlice; context: string; proof: ProofEvent }
  | { kind: "proof_only"; slice: TurnSlice; proof: ProofEvent };
```

#### 1.3 Port current deterministic discipline from existing code

Reuse design constraints from:

- `scripts/runtime/decide.mjs`
- `scripts/runtime/capture-event.mjs`
- `scripts/runtime/export-candidate.mjs`

Preserve:

- redacted-only input
- deterministic reproducibility metadata
- fail-closed raw/secret rejection
- explicit `stay_silent`
- candidate-only proof/event lane

#### 1.4 Implement the selected policy mapping

Hard-code the current product policy first:

| Slice | Product action |
|---|---|
| `direct-answer` | `stay_silent` |
| `correction-follow-up` | `correction_only` |
| `stale-memory-conflict` | `correction_only` |
| `continuation` | `full_context` |
| `retrieval-heavy` | `full_context` |
| `tool-heavy` | `full_context` |
| `unknown` | `stay_silent` |

#### 1.5 Add tests

Mirror the style of existing repo tests:

- `scripts/runtime/decide.test.mjs`
- `scripts/runtime/capture-event.test.mjs`
- `scripts/runtime/export-candidate.test.mjs`
- `packages/results-schema/test/*.test.ts`

Required test cases:

- direct-answer => silent
- correction-follow-up => correction_only
- stale-memory-conflict => correction_only
- continuation => full_context
- retrieval-heavy => full_context
- tool-heavy => full_context
- unknown / low-confidence => silent
- proof-only mode => no intervention, proof emitted
- malformed / raw input => rejection

### Exit gate

- `packages/runtime-policy` exists and is pure.
- Product policy is encoded once.
- Tests cover all six slices plus fail-closed behavior.

---

## Phase 2 — Build the proof event model

### Goal

Make the proof trail a first-class product surface, not an afterthought.

### Why

Without proof, users cannot trust selective intervention. The proof command is the trust interface.

### Build steps

#### 2.1 Create `packages/proof-store`

Add:

```text
packages/proof-store/
  package.json
  src/schema.ts
  src/write-event.ts
  src/read-events.ts
  src/render-status.ts
  src/render-proof.ts
  test/*.test.ts
```

#### 2.2 Define proof schema

Recommended event shape:

```ts
type ProofEvent = {
  schemaVersion: "ocb.proof.event.v1";
  profileId: string;
  eventId: string;
  timestamp: string;
  slice: TurnSlice;
  mode: RuntimeMode;
  decisionKind: "stay_silent" | "correction_only" | "full_context" | "proof_only";
  reasonCode: string;
  usedMemoryIdsRedacted: string[];
  rawTranscriptStored: false;
  containsRealUserData: false;
};
```

#### 2.3 Reuse current runtime-event discipline

Borrow from:

- `scripts/runtime/capture-event.mjs`
- `scripts/runtime/export-candidate.mjs`

But split product proof from eval-trace export:

- **proof events** = product trust surface
- **trace candidates** = evidence pipeline inputs

These are related but should not be conflated.

#### 2.4 Add per-profile local storage

Recommended layout:

```text
~/.openclawbrain/
  profiles/
    main/
      proof-events.jsonl
      status.json
    pelican/
      proof-events.jsonl
```

Or if activation-root discipline remains canonical:

```text
~/.openclawbrain/activation/<ProfileName>/
  proof-events.jsonl
  status.json
```

That shape is consistent with the prior install truth about dedicated activation roots per profile. Source: `memory/2026-03-11.md#L18`.

#### 2.5 Add summary renderers

Need two product renderings:

1. `status`
2. `proof`

`status` answers:

- is it enabled?
- which mode?
- did the adapter load?
- last decision?
- are proof events being written?

`proof` answers:

- what decisions happened recently?
- which ones fired?
- why?
- was raw text stored? (must be no)

### Exit gate

- A profile can accumulate proof events.
- Proof output is human-readable.
- Storage is local and separated by profile.

---

## Phase 3 — Build the OpenClaw runtime adapter

### Goal

Turn the pure policy into real runtime behavior on real turns.

### Why
n
This is the actual bridge from research repo to product.

### Build steps

#### 3.1 Create `packages/openclaw-integration`

Add:

```text
packages/openclaw-integration/
  package.json
  src/config.ts
  src/profile-context.ts
  src/redaction.ts
  src/build-input.ts
  src/apply-decision.ts
  src/runtime-adapter.ts
  src/service.ts
  test/*.test.ts
```

#### 3.2 Build a redacted turn input builder

Use the runtime decision contract as seed from:

- `scripts/runtime/decide.mjs`

The adapter should construct a `RuntimePolicyInput` from:

- profile id
- redacted current turn
- candidate memories
- tool-context summary
- runtime mode

#### 3.3 Build injection semantics

Map decisions into real runtime behavior:

- `stay_silent` => inject nothing
- `correction_only` => inject only the correction payload
- `full_context` => inject bounded context summary
- `proof_only` => inject nothing, write proof

Keep this conservative:

- bounded text only
- no giant transcript dumps
- no profile mixing
- no automatic mutation of external systems

#### 3.4 Add first-class service capability

This matters because earlier learned truth says a hook-only plugin is not the right end state. Source: `memory/2026-04-02.md#L27`.

Add a small service owned by the integration package, for example:

- runtime status / heartbeat
- proof-store integrity check
- adapter registration and health

That service should be the truthful answer to “what is the plugin actually providing beyond a hook?”

#### 3.5 Add opt-in profile config

Target shape:

```json
{
  "openclawbrain": {
    "enabled": true,
    "mode": "conservative",
    "activationRoot": "~/.openclawbrain/activation/main",
    "proofEvents": true,
    "rawTranscriptUpload": false
  }
}
```

#### 3.6 Dogfood with one profile first

Do **not** go broad immediately.

Order:

1. `main`
2. `pelican`
3. `bountiful`
4. `family`

### Code pointers

Current files that should directly shape this phase:

- `scripts/runtime/decide.mjs`
- `scripts/runtime/capture-event.mjs`
- `scripts/runtime/export-candidate.mjs`
- `packages/eval-harness/src/backend-types.ts`
- `packages/eval-harness/src/backends/full-ocb.ts`

### Exit gate

- One real OpenClaw profile can enable OCB.
- A real turn can be improved.
- A proof event is written.
- Direct-answer turns stay silent.

---

## Phase 4 — Build the CLI shell

### Goal

Create the one user-facing command surface that normal users interact with.

### Why

Without CLI, the product is still a repo, not a tool.

### Build steps

#### 4.1 Create `packages/cli`

Add:

```text
packages/cli/
  package.json
  src/main.ts
  src/commands/install.ts
  src/commands/enable.ts
  src/commands/disable.ts
  src/commands/status.ts
  src/commands/proof.ts
  src/commands/doctor.ts
  src/commands/uninstall.ts
  test/*.test.ts
```

#### 4.2 Minimum commands

```bash
openclawbrain install
openclawbrain enable --profile <profile>
openclawbrain disable --profile <profile>
openclawbrain status --profile <profile>
openclawbrain proof --profile <profile>
openclawbrain doctor
openclawbrain uninstall --profile <profile>
```

#### 4.3 Support `--json`

Needed for:

- scripting
- tests
- future operator tooling
- easier diagnosis

#### 4.4 `doctor` checks

`doctor` should verify:

- Node version compatibility
- OpenClaw home path
- target profile exists
- activation root exists / writable
- integration files linked/installed
- proof-store writable
- adapter service loadable
- restart required / not required

#### 4.5 `status` should be backed by real runtime facts

Not hand-wavy config only. It should include:

- enabled state
- mode
- adapter loaded
- proof events writing
- last event timestamp
- last decision kind

### Code pointers

Use these current surfaces as the source of truth for behavior and naming:

- `package.json`
- `docs/results/COMMANDS.md`
- `scripts/runtime/capture-event.mjs`
- `scripts/runtime/export-candidate.mjs`

### Exit gate

- A user can install and inspect the product with one command family.
- `status` and `proof` are real, not placeholder text.

---

## Phase 5 — Build the installer and uninstall path

### Goal

Make install / enable / disable / uninstall a coherent operator story.

### Why

Install confusion kills trust faster than model quality problems.

### Build steps

#### 5.1 Create `packages/installer`

Add:

```text
packages/installer/
  package.json
  src/install.ts
  src/enable-profile.ts
  src/disable-profile.ts
  src/uninstall.ts
  src/restart.ts
  src/layout.ts
  test/*.test.ts
```

#### 5.2 Canonical install shape

Use:

- one global package install path
- per-profile activation roots
- explicit profile enablement

This matches prior install lessons: `memory/2026-03-11.md#L18` and `memory/2026-04-02.md#L27`.

#### 5.3 Install behavior

`openclawbrain install` should:

- validate environment
- create base directories
- install/link extension/runtime components
- set up any required local model dependency checks
- print exact next step: `openclawbrain enable --profile main`

#### 5.4 Enable behavior

`openclawbrain enable --profile main` should:

- create profile activation root
- patch profile config safely
- register runtime adapter/service
- set default mode to conservative
- print whether restart is required

#### 5.5 Disable behavior

Should:

- turn off the profile cleanly
- preserve proof unless user explicitly purges
- avoid breaking unrelated profiles

#### 5.6 Uninstall behavior

Should:

- remove integration
- optionally remove proof / activation data only with explicit confirmation flag
- avoid destructive global cleanup by default

### Exit gate

- Fresh install works.
- Disable is reversible.
- Uninstall is safe.

---

## Phase 6 — Make publishing real

### Goal

Turn the repo into one real public package.

### Why

Today `package.json` still says `private: true`; that blocks the actual product.

### Build steps

#### 6.1 Pick one public identity

Choose exactly one:

- `openclawbrain`
- or `@openclaw/openclawbrain`

Do **not** expose multiple competing public package identities.

#### 6.2 Convert the root package into a publishable release story

Root decisions needed:

- public package name
- `bin` field
- `files` field
- semver strategy
- release script
- provenance

#### 6.3 Internal package versioning

Workspaces may have separate internal packages, but the user should perceive **one version**. That is a durable product rule from prior work. Source: `memory/2026-04-02.md#L27`.

#### 6.4 Release CI

CI should at minimum run:

```bash
pnpm ocb:e2e:smoke
pnpm ocb:e2e:production
pnpm ocb:traces:production-status
pnpm test
```

And only publish when release gates pass.

#### 6.5 Publish with provenance

```bash
npm publish --provenance --access public
```

### Exit gate

- `npm install -g <public-package>` works.
- `openclawbrain --help` works from a clean machine.

---

## Phase 7 — Fresh-machine product test

### Goal

Prove the tool works without repo knowledge.

### Test flow

```bash
npm install -g openclawbrain
openclawbrain install
openclawbrain enable --profile main
openclawbrain status --profile main
# run one real OpenClaw turn
openclawbrain proof --profile main
openclawbrain disable --profile main
```

### Required truths

- install does not require editing repo files manually
- proof shows a real event
- direct-answer can still show `stay_silent`
- no raw transcript upload
- uninstall / disable are obvious

### Exit gate

A non-author user can do the above from docs only.

---

## Phase 8 — Dogfood and demo preparation

### Goal

Produce the first honest demo that ordinary users care about.

### Demo cases

#### 8.1 Correction follow-up

- user corrects a preference
- later turn gets correction-only help
- proof explains why

#### 8.2 Continuation

- user says “continue”
- OCB injects bounded context
- assistant avoids redundant clarifying question

#### 8.3 Tool-heavy verification

- turn implies tool verification is appropriate
- OCB contributes bounded tool-context / verification bias
- proof shows why

#### 8.4 Silence

- user asks a direct-answer question
- OCB stays silent
- proof shows `stay_silent`

### Exit gate

You can demo all four without hand-waving or hidden operator patching.

---

## 6. Docs to write

These markdown files should exist before public push:

```text
docs/BUILD_PLAN.md
docs/INSTALL.md
docs/CONFIG.md
docs/PROOF.md
docs/PRIVACY.md
docs/TROUBLESHOOTING.md
docs/UNINSTALL.md
```

### README rewrite priorities

Current `README.md` is still rebuild-oriented. It should become user-oriented.

Top section should answer:

1. What is it?
2. Why would I want it?
3. How do I install it?
4. How do I enable it for one profile?
5. How do I inspect proof?
6. How do I disable or uninstall it?

---

## 7. Repo-level implementation order

This is the concrete recommended order in the current repository.

1. Add `packages/runtime-policy`
2. Add `packages/proof-store`
3. Add `packages/openclaw-integration`
4. Add `packages/cli`
5. Add `packages/installer`
6. Flip root package from private workspace shell to publishable release shell
7. Rewrite README and install docs
8. Run fresh-machine install test
9. Dogfood on one profile
10. Expand to many profiles

This order minimizes wasted work because it builds the pure core first, then the trust surface, then the runtime bridge, then the user shell.

---

## 8. Explicit non-goals for this cycle

Do **not** spend the next product cycle on:

- adding more eval backends
- polishing dashboards further
- benchmarking wars
- cloud sync
- remote SaaS control plane
- broad “memory for everything” claims
- more evidence abstractions before install works

The scoreboard is done enough. Distribution and trustworthy runtime behavior are the bottleneck.

---

## 9. Release definition of done

OpenClawBrain becomes “installable and usable by many people” only when all of the following are true:

1. One public install path exists.
2. One public version story exists.
3. A user can enable it for one profile.
4. It stays silent on direct answers.
5. It helps on at least correction-follow-up, continuation, and one tool/retrieval-heavy case.
6. `status` tells the truth from runtime state.
7. `proof` shows a redacted local explanation.
8. Raw transcripts are not uploaded.
9. Disable and uninstall are safe.
10. Fresh-machine docs are enough for a non-author user.

---

## 10. Immediate next task

If I were implementing this now, I would start here:

### Next coding milestone

Build `packages/runtime-policy` and make it the single owner of:

- slice classification
- selected policy mapping
- decision object generation
- proof event generation
- conservative abstention rules

### Exact code to use as seeds

- `scripts/runtime/decide.mjs`
- `scripts/runtime/capture-event.mjs`
- `scripts/runtime/export-candidate.mjs`
- `packages/eval-harness/src/backend-types.ts`
- `packages/eval-harness/src/backends/full-ocb.ts`
- `packages/results-site/src/decision.ts`

That is the cleanest bridge from the evidence-complete repo you have now to the installable product you actually want.
