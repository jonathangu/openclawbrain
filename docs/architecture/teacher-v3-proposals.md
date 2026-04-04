# Teacher v3 proposal envelope and lineage contract

Teacher v3 needs one stable container for every off-path output that can influence compilation, linting, mutation, forgetting, or replay-gated promotion.

The goal is not just to store a proposal payload. The goal is to make the proposal replayable, auditable, deduplicable, and reversible.

## Scope

This contract covers proposal-like outputs from the Teacher v3 lanes:

- compiler proposals
- lint proposals
- mutation proposals
- forgetting proposals
- correction proposals that are generated off-path before being committed

For the forgetting lane, the payload should carry a retention action and state
transition. Prefer soft retention first (`retained` → `demoted` → `archived`
→ `tombstoned`) and only allow `hard_delete` after tombstoning. Teacher-driven
forgetting must not hard-delete `user_explicit` correction memory.

It does **not** make teacher output authoritative truth.

Current-truth authority remains:

1. `user_explicit`
2. `raw_source`
3. `operator_policy`

Everything else is derivation or retrieval support.

## Authority vs derivation

Keep the split explicit:

- **authority** = where truth comes from
- **derivation** = how a surface was produced

Recommended derivation roles:

- `summary_navigation`
- `teacher_inference`
- `teacher_compilation`
- `teacher_lint`
- `teacher_mutation_proposal`
- `teacher_forgetting_proposal`

Notes:

- `teacher_inference` is derivation, not authority.
- `summary_navigation` is a retrieval role, not truth.
- explicit user quotes and typed correction memories stay on the authority side.

## Canonical contract

### Evidence reference

```ts
type EvidenceRef = {
  sourceKind: "user_turn" | "tool_trace" | "file" | "repo" | "summary" | "correction";
  sourceId: string;
  span?: { start: number; end: number };
  authority: "user_explicit" | "raw_source" | "operator_policy";
  derivation?:
    | "summary_navigation"
    | "teacher_inference"
    | "teacher_compilation"
    | "teacher_lint"
    | "teacher_mutation_proposal"
    | "teacher_forgetting_proposal";
  excerpt?: string;
  sourceHash?: string;
};
```

Rules:

- `sourceId` must point to a durable source record, not just a display label.
- `authority` describes the substrate, not the proposal.
- `derivation` is optional because raw evidence may be reused in multiple proposal classes.
- `excerpt` is for operator readability; it is not identity.

### Proposal lineage

```ts
type ProposalLineage = {
  proposalClass: "compiler" | "lint" | "mutation" | "forgetting" | "correction";
  basePackVersion?: number;
  baseGraphHash?: string;
  producerVersion: string;
  producerBuildId?: string;
  promptHash?: string;
  templateId?: string;
  scope: string;
  profile?: string;
  idempotencyKey: string;
  sourceBundleId?: string;
  parentProposalIds?: string[];
};
```

Required meaning:

- `basePackVersion` = the promoted pack version the proposal was generated against, when a pack is the base reference.
- `baseGraphHash` = the graph snapshot hash used for replay / structural comparison, when available.
- `producerVersion` = the code or agent version that produced the proposal.
- `promptHash` = canonical hash of the prompt/template payload, so reruns can be recognized.
- `scope` = a stable human-readable boundary, e.g. `docs/architecture`, `release-drift`, `mutation-shadow`, `correction-auto`.
- `profile` = optional policy profile or lane profile name.
- `idempotencyKey` = deterministic replay identity for duplicate suppression.

Identity rule:

- `proposalId` identifies a stored proposal row.
- `idempotencyKey` identifies the logical proposal instance.
- do **not** use timestamps as part of proposal identity.

Suggested idempotency inputs:

- proposal class
- normalized scope
- normalized subject ids
- normalized evidence fingerprints
- `basePackVersion`
- `baseGraphHash`
- `producerVersion`
- `promptHash`
- `templateId`
- `profile`

A stable canonical hash over those inputs is enough. The exact hash function can be implementation detail, but the canonicalization rule must be fixed.

### Proposal envelope

```ts
type ProposalEnvelope = {
  proposalId: string;
  proposalClass: "compiler" | "lint" | "mutation" | "forgetting" | "correction";
  status: "proposed" | "validated" | "shadow_scored" | "promotable" | "promoted" | "rejected" | "expired" | "rolled_back";
  lineage: ProposalLineage;
  subjectIds: string[];
  evidence: EvidenceRef[];
  counterevidence?: EvidenceRef[];
  payload: unknown;
  expectedEffect?: {
    retrieval?: "better" | "same" | "uncertain";
    truthRisk?: "low" | "medium" | "high";
    tokenBudget?: "lower" | "same" | "higher";
  };
  confidence: number;
  replaySuites: string[];
  rollbackKey: string;
  expiresAt?: string;
  createdAt: string;
  resolvedAt?: string;
};
```

Minimum requirements:

- `proposalId` is unique per stored proposal row.
- `proposalClass` must be explicit.
- `lineage` must be recorded at creation time.
- `subjectIds` must be stable ids, not raw text snippets.
- `evidence` must include the source authority for every non-trivial claim.
- `payload` may be class-specific, but the outer envelope must remain stable.
- `replaySuites` must name the gate(s) that can validate the proposal.
- `rollbackKey` must identify the exact reversible path.

## Lifecycle contract

Recommended states:

- `proposed`
- `validated`
- `shadow_scored`
- `promotable`
- `promoted`
- `rejected`
- `expired`
- `rolled_back`

State rules:

- `proposed` = created, not yet checked.
- `validated` = structurally sane and minimally complete.
- `shadow_scored` = run through a replay / comparison surface.
- `promotable` = passed the relevant gate(s).
- `promoted` = affected the promoted pack or durable store.
- `rejected` = failed a gate or was declined by policy.
- `expired` = stale before promotion.
- `rolled_back` = once-promoted proposal was reversed.

## Replay gate dimensions

Every proposal class should expose the same four replay-gate dimensions in an
inspectable surface with class-specific promotion mode:

- truth invariants
- attribution floor
- boundedness
- reversibility

The gate is a review contract, not a canary/live rollout switch.
If a later tranche adds canary rollout discipline, it should live in a separate
proposal/candidate-pack plan object, default to off, and stay target-state only
until explicitly shipped.

Suggested class emphasis:

- `compiler` — keep derived artifacts subordinate to explicit correction memory, require evidence-backed claims, and retain base pack / graph identity for replay.
- `lint` — stay report-only, cite the triggering evidence, and keep findings bounded to a single-pass review bundle.
- `mutation` — remain candidate-graph only, require evidence and rollback identity, and preserve the pre-mutation base state.
- `forgetting` — protect `user_explicit` correction memory, prefer demote/archive/tombstone over delete, and preserve the supersession chain.
- `correction` — keep explicit typed corrections above summaries and teacher inference, require source turns or raw quotes, and keep the correction scope small and auditable.

The code surface now exposes this as an inspectable `TeacherProposalReplayGateV1`
profile. In this tranche, `compiler` and `lint` are `reviewMode: "promotable"`
while `mutation`, `forgetting`, and `correction` stay `reviewMode: "shadow_only"`.

## Replay and rollback rules

A proposal is not promotable unless it can answer:

1. what base state it was generated against
2. what evidence justified it
3. what replay suites evaluated it
4. how to reverse it if needed

For structural proposals, the minimum replay identity should include:

- `basePackVersion`
- `baseGraphHash`
- `idempotencyKey`
- `rollbackKey`

For bundle-level promotions, the bundle must retain the proposal ids that it promoted, not just an aggregate summary.

## Storage guidance

The current repo already has a mutation-specific store shape (`brain_mutations`, `brain_mutation_bundles`, `brain_learning_journal`). That is a useful first home, but it is not yet a full proposal identity model.

Follow-on implementation should either:

- extend the existing mutation tables with indexed lineage columns, or
- normalize into a new proposal table and keep mutation/bundle rows as views or children

Either way, these fields should become queryable:

- `proposalId`
- `proposalClass`
- `idempotencyKey`
- `basePackVersion`
- `baseGraphHash`
- `promptHash`
- `scope`
- `profile`
- `rollbackKey`

## Concrete follow-on code surfaces

These are the places that will need later changes for lineage / proposal identity.

### Core types

- `src/brain-core/types.ts`
  - define shared `EvidenceRef`, `ProposalLineage`, and `ProposalEnvelope` types
  - add proposal identity fields to mutation and journal records
  - keep authority / derivation tags typed, not free-form

### Persistence

- `src/brain-store/migrations.ts`
  - add lineage columns / indexes for proposal identity and dedupe
  - decide whether to normalize or extend existing mutation tables
- `src/brain-store/store.ts`
  - write and read proposal identity fields
  - support lookup by `idempotencyKey` and base-state fields
  - store replay / rollback metadata alongside proposal payloads

### Worker and learning journal

- `src/brain-worker/worker.ts`
  - attach proposal envelope metadata when recording mutation and bundle events
  - persist bundle-to-proposal relationships explicitly
- `src/brain-core/bundle-evaluator.ts`
  - thread lineage into bundle evaluation records and verdicts
- `src/brain-core/pack.ts`
  - preserve base pack identity in promotion / rollback transitions
- `src/brain-core/replay.ts`
  - carry the base graph / pack identity into replay comparisons

### Runtime proposal emitters

- `src/brain-runtime/service.ts`
  - emit explicit correction proposal metadata with source quote, source message id, and proposal lane tags
  - preserve lineage when auto-correction is proposed or committed
- `src/brain-runtime/tools.ts`
  - standardize tool output so teacher / correction proposals carry the same envelope metadata

### Operator surfaces

- `src/brain-runtime/promotion-story.ts`
  - surface proposal lineage and rollback identity in the operator story
- any future status / proof surfaces
  - report proposal class, base pack, prompt hash, and replay outcome

## Non-goals

- Do not treat teacher output as current-truth authority.
- Do not use mutable timestamps as identity.
- Do not hide provenance in opaque JSON when an indexed column is needed.
- Do not require live-path learning to answer proposal identity questions.
- Do not delete `user_explicit` correction memory as part of forgetting proposals.

## Recommended next step

Implement the envelope first, then thread lineage into mutation storage and promotion reporting.

That sequencing keeps Teacher v3 honest:

> off-path proposals first, promoted effects second, live serving last.
