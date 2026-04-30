# OpenClawBrain Productization Plan

Date: 2026-04-30
Status: next phase after V5 evidence gate

## Thesis

OpenClawBrain wins by stopping at “evidence-first research project” and shipping the smallest trustworthy runtime product:

- one install command
- one profile enable command
- one visible “it helped here” proof trail
- conservative behavior that stays silent most of the time

The evidence gate already says the selected policy is worth continuing:

- selected product policy wins: `40/40`
- selected product policy harms: `0/40`
- Evidence E2E: `true`
- decision: `CONTINUE`

But OpenClawBrain is not yet installable or widely usable. It still lacks:

- public package/install path
- stable CLI
- OpenClaw gateway/profile integration
- live runtime loop
- config and upgrade path
- user-facing status/proof commands

## Winning product shape

OpenClawBrain should be experienced as:

> One OpenClawBrain package. One OpenClaw gateway. Many OpenClaw profiles. Brain behavior enabled per profile. Local proof users can inspect.

OpenClaw itself is the right host shape: the Gateway is the control plane, while users interact through normal OpenClaw channels and profiles. OpenClawBrain should attach at the profile boundary, not as a global behavior that bleeds across profiles.

## User-facing promise

Narrow and strong:

> OpenClawBrain helps your OpenClaw profile remember corrections, continue work, and use the right context — without dumping raw private transcripts or meddling when it should stay silent.

Do **not** market it as:

- “agent memory”
- “a smarter brain”
- “AI that knows you”
- broad generic intelligence

Market it as a **selective intervention layer with proof**.

## First-run experience

The target first-run path should be:

```bash
npm install -g openclawbrain
openclawbrain install
openclawbrain enable --profile main
openclawbrain status --profile main
```

After real usage:

```bash
openclawbrain proof --profile main
```

Example proof output:

```text
OpenClawBrain proof for profile: main

Enabled: yes
Mode: conservative
Raw transcript upload: never
Last 24h decisions:
 stay_silent: 42
 correction_only: 3
 full_context: 5
 proof_only: 9

Recent useful intervention:
 Turn type: stale-memory-conflict
 Decision: correction_only
 Why: user corrected preference on 2026-04-25
 Raw text stored: no
 Redacted proof event: yes
```

That proof surface is how users trust it.

## Immediate build order

### 1. Build `packages/runtime-policy`

Create the smallest reusable core.

It should export one pure function:

```ts
decideOpenClawBrainIntervention(input): InterventionDecision
```

Input shape:

```ts
type RuntimePolicyInput = {
  profileId: string;
  redactedTurn: RedactedTurn;
  candidateMemories: MemoryCandidate[];
  toolContext?: ToolContextSummary;
  runtimeMode: "off" | "proof-only" | "conservative" | "active";
};
```

Output shape:

```ts
type InterventionDecision =
  | { kind: "stay_silent"; proof: ProofEvent }
  | { kind: "correction_only"; message: string; proof: ProofEvent }
  | { kind: "full_context"; context: string; proof: ProofEvent }
  | { kind: "proof_only"; proof: ProofEvent };
```

Hard-code the selected evidence policy first:

| Slice / turn type | Decision |
|---|---|
| direct-answer | `stay_silent` |
| correction-follow-up | `correction_only` |
| stale-memory-conflict | `correction_only` |
| continuation | `full_context` |
| retrieval-heavy | `full_context` |
| tool-heavy | `full_context` |
| unknown / low-confidence | `stay_silent` |

The key product rule: `stay_silent` is a successful behavior, not a failure.

### 2. Build `packages/openclaw-integration`

This package should sit at the profile boundary and answer one question:

> For this OpenClaw profile and this redacted turn, should OpenClawBrain inject anything?

Hard requirements:

- never mix profiles
- treat `main`, `pelican`, `bountiful`, `family`, etc. as separate local stores
- default to conservative mode
- do not upload raw transcripts
- write redacted proof events

Target profile config:

```json
{
  "openclawbrain": {
    "enabled": true,
    "mode": "conservative",
    "storage": "profile-local",
    "rawTranscriptUpload": false,
    "proofEvents": true
  }
}
```

Do not start with global install magic that touches every profile. Start with explicit profile-local enablement.

### 3. Ship the CLI before adding intelligence

The CLI is the product shell. Without it, normal users cannot use the evidence.

Minimum commands:

```bash
openclawbrain install
openclawbrain enable --profile <profile>
openclawbrain disable --profile <profile>
openclawbrain status --profile <profile>
openclawbrain proof --profile <profile>
openclawbrain doctor
openclawbrain uninstall --profile <profile>
```

`doctor` matters because users will have different Node versions, package managers, profiles, permissions, services, and channel setups.

Example status:

```text
OpenClawBrain status

Profile: main
Enabled: yes
Mode: conservative
Runtime adapter: connected
Proof events: writing
Raw transcript upload: disabled
Profile isolation: pass
Last decision: stay_silent, 2 minutes ago
```

### 4. Make package publishing real

Current repo state is still not package-ready:

- root `package.json` is `private: true`
- no public user-facing package identity is locked
- no `bin`, `files`, release metadata, or provenance story is ready

Pick one public identity:

- `openclawbrain`, or
- `@openclaw/openclawbrain` only with official OpenClaw alignment

Do not publish multiple user-facing packages. Internals can be workspaces, but users should see one installable tool and one visible version.

When CI is ready, publish with provenance:

```bash
npm publish --provenance --access public
```

For a tool touching assistant context, supply-chain trust is part of the product.

## What to cut

Do not spend the next cycle on:

- more benchmark polish
- more backend variants
- more dashboards
- generic memory marketing
- complicated remote service
- cloud sync story
- public leaderboard
- marketplace extension before local install works

The scoreboard already did its job. The next question is:

> Can a normal OpenClaw user install it, enable it on one profile, get one better turn, and inspect why?

## User win moments

### Moment 1 — Correction follow-up

User corrects a preference once. Later, OpenClawBrain injects only the relevant correction.

Proof should say:

```text
Decision: correction_only
Reason: later turn conflicts with prior correction
Raw text stored: no
Profile: family
```

### Moment 2 — Continue without re-explaining

User says “continue”. Instead of asking “continue what?”, OpenClawBrain supplies bounded context.

Proof should say:

```text
Decision: full_context
Reason: continuation turn with available bounded state
```

### Moment 3 — Tool-heavy verification

User asks for a claim that should be checked. OpenClawBrain nudges the assistant toward read-only verification instead of inventing.

Proof should say:

```text
Decision: full_context
Reason: tool-heavy turn
```

### Moment 4 — Silence

User asks a simple direct question. OpenClawBrain does nothing.

Proof should say:

```text
Decision: stay_silent
Reason: direct-answer turn
```

This is critical. Users should not feel the brain breathing over every turn.

## Release shape

First public release:

> `v0.1.0` — profile-bound local proof release

Release promise:

- OpenClawBrain can be enabled per OpenClaw profile.
- It selectively injects correction/context only when the runtime policy says it should.
- It writes redacted local proof events.
- It does not upload raw transcripts.
- It can be disabled or uninstalled.

README top section should say:

```md
# OpenClawBrain

A local, profile-bound selective intervention layer for OpenClaw.

It helps an OpenClaw profile:
- remember user corrections,
- continue bounded work,
- supply relevant context,
- stay silent on direct answers,
- show local proof of what it did.

It is not a general intelligence layer.
It is not a cloud memory service.
It does not upload raw transcripts.
```

Install section:

```bash
npm install -g openclawbrain
openclawbrain install
openclawbrain enable --profile main
openclawbrain status --profile main
```

Proof section:

```bash
openclawbrain proof --profile main
```

Uninstall section:

```bash
openclawbrain disable --profile main
openclawbrain uninstall --profile main
```

## Trust model

Make these defaults visible:

- private logs stay local
- raw transcripts are not uploaded
- proof events are redacted
- profiles are isolated
- default mode is conservative
- kill switch exists
- product can stay silent

The proof command is not a nice-to-have. It is the trust surface.

## Technical acceptance gates

Before calling it usable by many people, require:

```bash
pnpm ocb:e2e:smoke
pnpm ocb:e2e:production
pnpm ocb:traces:production-status
pnpm test packages/runtime-policy
pnpm test packages/openclaw-integration
openclawbrain doctor
openclawbrain status --profile main
openclawbrain proof --profile main
```

Fresh-machine release test:

```bash
npm install -g openclawbrain
openclawbrain install
openclawbrain enable --profile main
openclawbrain status --profile main
# run one OpenClaw turn
openclawbrain proof --profile main
openclawbrain disable --profile main
```

A release is not real until that path works without repo knowledge.

## Adoption strategy

Start with three audiences:

1. Power OpenClaw users who already use multiple profiles.
2. People burned by bad memory; show correction-follow-up, stale-memory-conflict, and continuation demos.
3. Privacy-sensitive users; lead with local proof, redaction, no raw transcript upload, and uninstallability.

Launch demo should not be a benchmark. It should be:

> I corrected my OpenClaw profile once. Later, it used the correction. It showed me exactly why. It did not upload my raw transcript. And on simple questions, it stayed silent.

That is the wedge.

## Next concrete milestone

Build a profile-bound OpenClawBrain runtime adapter that:

1. reads an opt-in profile config flag,
2. receives redacted real OpenClaw turns,
3. applies the selected product policy,
4. injects correction/full-context only when appropriate,
5. writes redacted proof events,
6. exposes status/proof through the CLI,
7. defaults to conservative mode,
8. can be disabled instantly.

That is the shortest path from “evidence says continue” to “users can install this and feel the win.”
