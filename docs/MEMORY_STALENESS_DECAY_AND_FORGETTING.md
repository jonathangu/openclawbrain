# Memory Authority, Staleness, Decay, Override, and Forgetting in OpenClawBrain

## Why this note exists

OpenClawBrain's next hard memory problem is not merely "retrieve better
memories." It is deciding whether a retrieved memory still has authority.

The key distinction:

> Relevance is not authority.

A memory can match the current turn and still be wrong, expired, unsafe to use,
superseded, too broad, inappropriate to inject, or no longer part of the user's
active preference set.

Examples:

- "Use pnpm in this repo" may be relevant to a test command but wrong after the
  repo migrates package managers.
- "Keep answers short" may be a useful default, but not authoritative when the
  user explicitly asks for deep discussion and critique.
- "Route design work to agent X" may be true for one project phase and stale
  after responsibilities change.
- "Remember this codeword" may be authorized for explicit recall, but never
  appropriate for proactive injection.
- "Forget that" may mean suppress, soft-delete, tombstone, or hard-delete,
  depending on sensitivity and user intent.

The old mental model is too simple:

```text
relevant + high score => inject
```

The product contract should become:

```text
relevant
  + authorized
  + current
  + applicable
  + safe
  + non-superseded
  + compatible with the current instruction
  => inject, weakly adapt, verify, confirm, abstain, or never use
```

This note updates the earlier staleness/decay proposal. The important revision
is that OpenClawBrain should not model memory governance as one overloaded enum
like:

```text
active | confirm-needed | suppressed | superseded | expired | tombstoned
```

Those labels are useful outcomes, but they mix different categories. A better
design is a **memory authority layer** that computes route-time authority from
several independent dimensions.

## Short answer

OpenClawBrain is not purely additive in its schema. It already has useful
primitives:

- `freshness`
- `confidence`
- `importance`
- `last_seen_at`
- `last_used_at`
- `superseded_by`
- `deleted_at`
- memory edges including `contradicts` and `supersedes`
- soft-delete behavior for explicit forget/delete/suppress requests
- outcome learning that lowers memory confidence/importance after corrections
- maintenance decay, duplicate consolidation, and pruning

But the current behavior is still too additive at the product level.

The current runtime mostly behaves as though:

```text
retrieve relevant memory
rank it
inject top candidates
```

The next version should behave more like:

```text
retrieve candidate memory
resolve authority for this turn
decide whether to inject, weakly adapt, verify, confirm, abstain, audit only,
or never use
```

The design center should be:

```text
Never confuse relevance with authority.
Never let old memory silently override current instruction.
Never use private or suppressed memory proactively.
Never re-capture tombstoned memory without explicit reauthorization.
Prefer verification over confirmation when the environment can answer.
Preserve auditability unless privacy requires redaction or deletion.
Ask for confirmation only when the memory is user-owned, material, uncertain,
and not cheaply verifiable.
```

## Current implementation: what exists today

### Schema primitives

The `memory_nodes` table includes:

- `importance`
- `freshness`
- `confidence`
- `use_count`
- `useful_count`
- `capture_count`
- `last_seen_at`
- `last_used_at`
- `superseded_by`
- `deleted_at`

The `memory_edges` table supports:

- `related`
- `contradicts`
- `supersedes`
- `extends`
- `used_with`
- `supports_workflow`

This means the database can represent decay, override, and forgetting. The gap
is not raw storage. The gap is authority policy.

Relevant files:

- `packages/openclaw-plugin/src/memory-store.ts`
- `packages/openclaw-plugin/src/memory-operations.ts`
- `packages/openclaw-plugin/src/context-selector.ts`
- `packages/openclaw-plugin/src/learning.ts`
- `packages/openclaw-plugin/src/route-learning.ts`
- `packages/openclaw-plugin/src/feedback-distiller.ts`

### Search excludes deleted and superseded memories

The main memory search path filters out:

- `deleted_at IS NOT NULL`
- `superseded_by IS NOT NULL`

That means a deleted or superseded memory should not normally be injected.

This is good. It proves the system is not literally append-only.

But:

- freshness is not a hard authority gate
- active search is not the same as "safe to use now"
- graph inspection can still expose inactive memories for audit
- low freshness only reduces ranking weight; it does not produce a different
  route behavior

### Context selection treats freshness as a score term

The context selector currently ranks candidates roughly like this:

```text
score = importance * 0.4 + confidence * 0.4 + freshness * 0.2
```

Then it adds type-specific boosts:

- corrections get a large boost
- preferences get a boost
- workflows get a boost
- routing rules and tool conventions get boosts

This means freshness can affect ranking, but it does not change the use mode.
A stale memory is still just a lower-ranked memory.

That is the wrong abstraction. A stale memory should sometimes trigger a
different behavior:

```text
current -> inject
stale but material -> verify or confirm
expired -> abstain
suppressed -> never proactively inject
tombstoned -> do not re-capture
```

### Explicit contradiction can supersede memory

The feedback distiller schema allows a candidate to declare:

```json
{
  "contradictions": [
    {
      "existingMemoryId": "...",
      "reason": "...",
      "action": "supersede_existing"
    }
  ]
}
```

The applier then marks the old memory as superseded.

This path exists, but it is not robust enough:

- The LLM has to identify the contradiction and supply the existing memory ID.
- The applier does not appear to run deterministic conflict lookup first.
- Same-key upsert can blur reinforcement, revision, and replacement.
- Supersession proof is not yet rich enough for a human to understand why an
  old memory lost authority.

The biggest risk is same-key update:

```text
key = preferred_package_manager
old value = npm
new value = pnpm
```

If this is updated in place, the system loses the historical fact that a
reversal happened. Better behavior:

```text
same key + same value => reinforce
same key + changed value => revision or supersession candidate
same key + narrower scope => scoped exception candidate
```

### Explicit forget/delete/suppress exists

Capture intent recognizes text like:

```text
forget
delete
remove
do not remember
don't remember
do not store
stop using
suppress
```

When the distiller returns `feedbackType="delete_or_suppress"`, the applier
searches for matching memories and soft-deletes them.

This is useful, but incomplete:

- query-based matching can under-match or over-match
- "do not bring this up" is different from "delete content"
- soft delete does not prevent future re-capture
- sensitive forget requests may need redaction or hard delete
- the system should preserve different levels of audit depending on privacy

The missing primitive is a tombstone.

### Outcome learning adjusts scores, not authority events

When injected memories later receive outcomes:

- accepted/helped/tool_success increases importance/confidence
- ignored slightly lowers importance
- user_corrected lowers importance/confidence
- harmful/tool_failure lowers importance/confidence more

This is valuable, but not enough.

Bad outcome does not always mean "bad memory."

Example:

```text
Memory: User prefers concise answers.
Assistant: gives terse answer.
User: No, I need detail here.
```

The memory may still be true as a default. The mistake was over-application in
a context where the user explicitly asked for depth.

Outcome learning should produce structured events:

```text
memory_used
user_corrected
correction_subject_detected
possible_over_application
possible_wrong_scope
memory_authority_reduced_for_this_pattern
```

Then the system can learn whether the memory was false, stale, too broad, or
misapplied.

### Freshness decay exists, but it is generic

Maintenance calls `decayFreshness(agentId)`.

Current behavior:

- if `last_seen_at` is older than seven days, freshness drops by roughly `0.07`
- memories below very low freshness are counted
- pruning happens separately when memory count exceeds the configured cap

This is a start, but:

- decay is not type-aware
- decay is not based on volatility
- decay is not based on environment changes
- decay does not create `verify_before_use` or `confirm_before_use`
- decay is anchored to `last_seen_at`, not richer timestamps like
  `last_confirmed_at`, `last_verified_at`, or `last_successful_use_at`
- retention emitted by the distiller is not stored as a first-class memory field

## Authority, not freshness

Staleness matters, but it should not be the central abstraction.

A memory has authority only when several things are true:

- it is true enough
- it is still current
- it applies to this scope
- the user still wants it used
- the system is allowed to use it this way
- it is compatible with the current instruction
- using it will not create disproportionate risk or annoyance

Freshness is only one input.

For example:

```text
Use pnpm in this repo.
```

This can fail because it is stale, but also because:

- the current repo is different
- the current branch/tooling changed
- package files now indicate npm
- the user is asking conceptually, not asking to run commands
- the user explicitly said to ignore old workflow memory for this task

Likewise:

```text
Keep answers short.
```

This may still be true as a default preference, but it has low authority when
the user asks for:

```text
Deep discussion and critique.
```

So the question is not:

```text
Is this memory stale?
```

It is:

```text
Does this memory have enough authority in this turn to influence behavior?
```

## Do not make memory state one enum

The earlier model proposed states like:

- active
- confirm-needed
- suppressed
- superseded
- expired
- tombstoned
- hard-deleted

Those words are useful, but they mix several categories:

- temporal validity: current, stale, expired
- behavior policy: injectable, confirm-needed, suppressed
- graph relation: canonical, duplicate, superseded
- retention/privacy state: stored, redacted, hard-deleted, tombstoned

These are not mutually exclusive.

A memory could be:

```text
retained for audit
superseded by a newer memory
not injectable
visible only during explicit graph inspection
covered by a tombstone that prevents re-capture
```

Trying to encode that as one final `state` will produce brittle enum sprawl.

Better: split authority into orthogonal axes, then compute the route-time
decision.

## Better decomposition: orthogonal axes

### 1. Retention state

Question:

```text
Does the content still exist?
```

Values:

- `stored`
- `redacted`
- `soft_deleted`
- `hard_deleted`

### 2. Behavioral availability

Question:

```text
May the assistant use this memory proactively?
```

Values:

- `injectable`
- `weak_context_only`
- `confirm_before_use`
- `verify_before_use`
- `explicit_request_only`
- `never_use`

### 3. Temporal validity

Question:

```text
Is the memory still likely to be true now?
```

Values:

- `current`
- `stale`
- `expired`
- `unknown`

### 4. Graph relation

Question:

```text
How does this memory relate to other memories?
```

Values:

- `canonical`
- `duplicate`
- `revised_by`
- `superseded_by`
- `exception_to`
- `supports`
- `contradicts`

### 5. Scope applicability

Question:

```text
Where does this memory apply?
```

Scope levels:

- current explicit instruction
- current task/session
- project
- repo
- branch
- app
- channel
- agent
- user/global

### 6. Privacy and consent policy

Question:

```text
What use restrictions apply?
```

Values:

- `normal`
- `sensitive`
- `recall_only`
- `do_not_restore`
- `do_not_reveal_proactively`
- `do_not_recapture_without_reauthorization`

The authority resolver can then compute:

- `inject`
- `weak_context`
- `verify_before_use`
- `confirm_before_use`
- `abstain`
- `audit_only`
- `never_use`
- `delete_or_redact`

## Stored validity vs runtime authority

A stored memory cannot fully know whether it is authoritative. Authority depends
on the current turn.

Example:

```text
Memory: User prefers concise answers.
Current turn: "Deep discussion and critique."
```

The memory may remain historically valid. It should not be deleted or marked
false. But runtime authority is low because the current instruction is more
specific and more recent.

Another example:

```text
Memory: Use pnpm in this repo.
Current context: same repo, pnpm-lock.yaml exists.
```

Authority is high after environment verification.

Same memory, different context:

```text
Memory: Use pnpm in this repo.
Current context: same repo, package-lock.json exists, no pnpm-lock.yaml.
```

Authority is uncertain. The system should verify more, ask, or abstain.

So OpenClawBrain needs two layers:

```text
Stored validity:
  What do we know about this memory historically?

Runtime authority:
  Should this memory influence this turn?
```

## Do not decay evidence confidence

The current model has fields like:

- `confidence`
- `importance`
- `freshness`

The authority model should avoid corrupting epistemic confidence with temporal
decay.

These are different statements:

```text
I am confident the user said this.
I am confident this was useful in the past.
I am confident this is still true.
I am confident this should be used now.
```

A memory can be highly reliable as history but low-authority as current
guidance.

Example:

```text
User wanted Decagon interview prep on May 1.
```

The system may be fully confident the user said it. But it should have near-zero
authority outside that task context.

Split confidence into at least:

```text
evidence_confidence:
  How sure are we that this memory was correctly captured?

current_validity:
  How likely is it still true?

behavioral_authority:
  How appropriate is it to use now?
```

Decay should primarily affect `current_validity` and
`behavioral_authority`, not `evidence_confidence`.

## Confirmation is necessary, but not enough

The strongest missing behavior is a middle path between injection and omission.

Current bad choices:

```text
silently use memory
silently ignore memory
```

Better choice:

```text
surface uncertainty when the memory materially matters
```

Example:

```text
I have an older repo memory that says to use pnpm here. Do you still want that,
or should I infer from the current project files?
```

But confirmation can become annoying. A memory system that constantly asks "do
you still want this?" feels less intelligent.

Confirmation should require all of these:

- memory is relevant
- memory would materially change the answer or action
- memory is old, contradicted, volatile, or high-impact
- no cheaper deterministic verification is available
- the user has not already given a newer instruction
- cost of being wrong is meaningfully higher than cost of asking

That last condition matters.

## Prefer verification when the environment can answer

Some memories should be verified instead of confirmed.

Examples:

- "Use pnpm in this repo."
- "Tests are run with npm test."
- "Project uses Next.js."
- "Main branch is called master."
- "Deployment uses script X."

For these, the agent should often inspect:

- lockfiles
- `package.json`
- workspace files
- config files
- recent command outcomes
- repository state
- docs

Then update authority based on evidence.

This suggests a field:

```text
validation_strategy:
  user_confirm
  environment_check
  tool_success
  document_check
  time_expiry
  explicit_request_only
  never_proactive
```

Rule of thumb:

```text
User-owned truth -> ask user.
Environment-owned truth -> verify environment.
System-owned policy -> inspect config.
Sensitive recall -> explicit request only.
```

## Scope precedence should beat raw recency

A newer global memory should not automatically override an older repo-specific
memory.

Example:

```text
Old repo-specific memory: In repo A, use pnpm.
New global memory: I usually use npm.
```

The global memory is fresher, but less specific. It should not supersede the
repo-specific memory.

Similarly:

```text
Old global preference: Keep answers concise.
Current task instruction: Go deep.
```

The current task instruction wins because it is narrower and more immediate,
not because the global preference is false.

A useful precedence hierarchy:

```text
current explicit instruction
> current task/session memory
> active project/repo memory
> durable user preference
> inferred habit
> historical observation
```

This hierarchy may matter more than temporal decay.

## Avoid preference absolutism

A common memory failure is treating preferences as commands.

```text
Keep answers short.
```

This should not mean:

```text
Always be short forever.
```

It should mean:

```text
Use concision as a default prior unless the current task suggests otherwise.
```

Distinguish:

### Hard user constraints

Examples:

- "Never mention X."
- "Do not store Y."
- "Always use British spelling in this project."

These should not decay silently.

### Soft user preferences

Examples:

- "I like concise answers."
- "Prefer examples."
- "Usually use pnpm."

These should be soft priors that current task instructions can override.

### Task instructions

Examples:

- "For this one, go deep."
- "Ignore previous style preferences."

These should have a fresh instruction advantage.

## Override handling needs a conflict model

OpenClawBrain should distinguish at least six relation types.

### Reinforcement

The new evidence says the same thing.

Behavior:

- update `last_confirmed_at`
- increase `evidence_confidence` or support count slightly
- update `last_seen_at`
- preserve the same canonical node

### Revision

The new evidence changes details but keeps the same subject.

Example:

```text
Old: summaries should be under 5 bullets
New: summaries should be under 3 bullets
```

Behavior:

- preserve old value in audit
- create new canonical value or update with revision proof
- add `revised_by` or `supersedes` edge

### Contradiction

The new evidence conflicts with the old memory.

Example:

```text
Old: use npm
New: use pnpm
```

Behavior:

- create new active memory
- mark old memory superseded
- add `new -> old` supersedes edge
- record contradiction proof

### Scoped exception

The new evidence is narrower than the old memory.

Example:

```text
Old: prefer concise answers
New: for this research memo, go deep
```

Behavior:

- keep broad preference
- create task/session exception
- route by scope specificity
- expire exception automatically

### Scope shift

The new evidence is about a different scope.

Example:

```text
Old: in repo A, use pnpm
New: in repo B, use npm
```

Behavior:

- keep both
- never let the broader or different-scope memory erase the narrower one

### Temporary mode

The new evidence is an interaction mode, not a durable preference change.

Example:

```text
Usually be concise, but today I want detailed reasoning.
```

Behavior:

- create temporary mode memory
- expire it after session/task boundary
- do not alter durable global preference

### Suggested conflict pipeline

```text
1. Extract candidate memory.
2. Retrieve possible conflicts by subject, type, normalized key, scope, and
   nearby graph edges.
3. Let the LLM classify semantic relation:
   reinforce | revise | contradict | exception | scope_shift | temporary_mode
   | unrelated | ambiguous
4. Code validates allowed transition.
5. Store proof edge and transition reason.
```

The LLM can classify semantics. Code must own the graph transition.

## Forgetting needs precision

"Forget" should not mean only one thing.

Separate:

- use
- visibility
- storage
- recapture

When the user says:

```text
Forget that.
```

They may mean:

- do not use it proactively
- do not mention it
- do not show it in normal memory search
- delete the stored content
- do not ever re-learn it

### Suppress

Meaning:

```text
Keep it for audit/search, but do not proactively inject it.
```

Use when:

```text
Don't bring that up unless I ask.
```

### Soft delete

Meaning:

```text
Remove from active memory and normal search, but keep audit trail.
```

Use when ordinary low-sensitivity memory should stop influencing behavior.

### Tombstone

Meaning:

```text
Do not re-capture this same memory without explicit reauthorization.
```

Use when:

```text
Do not store this.
Do not remember this again.
Forget that codeword.
```

Tombstones should preserve only enough information to prevent re-capture.

Example:

```text
tombstone_key = hash(agent_id + scope + normalized_subject + memory_type)
policy = do_not_recapture_without_explicit_reauthorization
```

For secrets, do not store the secret in the tombstone. Store a safe fingerprint
or category-level block.

### Hard delete

Meaning:

```text
Erase stored content as much as technically possible.
```

For privacy-sensitive memories, hard deletion should be normal enough to be
trusted. It does not need to be frequent, but it should be easy and obvious.

## Auditability versus privacy

The earlier stance said:

```text
Never lose auditability by accident.
```

That remains right, but it needs a privacy caveat.

For normal workflow memories, auditability is valuable:

```text
Why did you use pnpm?
Because you corrected me, and that correction superseded the npm memory.
```

For sensitive memories, auditability can become a liability:

```text
Remember this codeword.
Delete that codeword.
```

If the content remains visible in an audit graph, the user may reasonably feel
the system did not forget it.

Use audit levels:

### Full audit

Content and transition are visible.

Use for ordinary workflow/project memories.

### Redacted audit

Transition is visible, content hidden.

Use for sensitive recall, private values, or deleted values.

### Minimal audit

Only deletion/suppression event is visible.

Use when retaining details creates privacy risk.

### No retained audit

Hard deletion where required.

Use when privacy intent outweighs auditability.

## Memory as policy, not content

A memory is not just a stored fact. It is a policy object with permissions.

Example:

```text
User prefers short answers.
```

This implies:

```text
When no stronger instruction is present, bias toward brevity.
Do not override explicit requests for depth.
Do not ask for confirmation every time.
```

Another example:

```text
Do not store the codeword.
```

This implies:

```text
Prevent capture.
Prevent proactive recall.
Delete or redact prior content.
Block future re-learning unless reauthorized.
```

Memory types should have policy templates.

| Memory type | Default use | Overridden by | Decay | Revalidate when |
| --- | --- | --- | --- | --- |
| preference | weak context | current task instruction | medium | old plus high-impact |
| workflow | verify or inject | environment config | fast | stale plus cannot verify |
| recall_rule | explicit request only | forget/delete policy | special | sensitive policy requires |
| correction | strong context | newer explicit correction | slow | old plus high-risk |
| project_fact | supporting context | repo/docs/current files | medium | environment contradiction |
| routing_rule | strong if explicit | newer assignment/instruction | medium | high-impact and old |

This makes memory behavior predictable.

## Proposed architecture: Memory Authority Resolver

Introduce a dedicated component:

```text
MemoryAuthorityResolver
```

Its job is not retrieval. Its job is adjudication.

Input:

- current turn
- candidate memories
- route/task type
- scope stack
- recent explicit instructions
- environment signals
- memory graph edges
- user privacy policy
- tool outcomes

Output:

```json
[
  {
    "memoryId": "...",
    "relevanceScore": 0.82,
    "authorityDecision": "inject",
    "authorityScore": 0.91,
    "reasons": [
      "explicit_user_correction",
      "repo_scope_match",
      "recently_verified"
    ],
    "requiredAction": null,
    "proofEdges": ["..."]
  }
]
```

Possible `authorityDecision` values:

- `inject`
- `weak_context`
- `verify_before_use`
- `confirm_before_use`
- `abstain`
- `audit_only`
- `never_use`
- `delete_or_redact`

Current shape:

```text
search -> rank -> inject
```

Better shape:

```text
search -> candidate expansion -> authority resolution
  -> injection / verification / confirmation / abstention
```

The `/explain-last` route should be able to say:

```text
Found memory M.
Did not inject it because it was superseded by M2.

Found memory N.
Injected it because it was explicit, repo-scoped, recently verified, and
matched the current repo.

Found memory P.
Treated it as weak context because the current user instruction overrode it.
```

## Proposed minimal schema refinement

Do not start by stuffing everything into `memory_nodes`.

Add a side table:

```text
memory_validity
---------------
memory_id
base_availability
retention_state
privacy_class
decay_policy
validation_strategy
valid_from
valid_until
expires_at
last_confirmed_at
last_verified_at
last_successful_use_at
last_failed_use_at
last_contradicted_at
revalidate_after
half_life_days
evidence_confidence
current_validity_score
behavioral_authority_score
state_reason
updated_at
```

Add another table:

```text
memory_authority_events
-----------------------
event_id
memory_id
event_type
source
turn_id
route_id
evidence_id
old_value
new_value
reason
created_at
```

Useful event types:

- `captured`
- `reinforced`
- `used`
- `verified`
- `user_confirmed`
- `user_corrected`
- `suppressed`
- `soft_deleted`
- `tombstoned`
- `hard_deleted`
- `superseded`
- `expired`
- `revived`
- `abstained_due_to_staleness`
- `confirmation_requested`
- `confirmation_declined`
- `verification_succeeded`
- `verification_failed`
- `overridden_by_current_instruction`

## Runtime authority algorithm

```text
1. Retrieve candidate memories by semantic relevance and scope.

2. Remove hard-blocked memories:
   - hard_deleted
   - tombstoned
   - never_use
   - superseded unless audit requested

3. Expand graph:
   - find superseding memories
   - find contradictions
   - find scoped exceptions
   - find tombstones
   - find recent outcomes

4. Compute authority features:
   - evidence confidence
   - source authority
   - scope specificity
   - current instruction compatibility
   - age since confirmation
   - age since verification
   - volatility
   - outcome reliability
   - risk of wrong use
   - cost of confirmation
   - cost of silent abstention
   - availability of verification
   - privacy sensitivity

5. Decide use mode:
   - inject
   - weak_context
   - verify_before_use
   - confirm_before_use
   - abstain
   - audit_only
   - never_use

6. Emit proof:
   - why used
   - why ignored
   - why verified
   - why confirmation was requested
   - which memory superseded which
```

Retrieval can over-include. Authority resolution should be strict.

## Decision matrix

Do not use a single score threshold. Use a policy matrix.

| Authority | Risk | Verification available | Decision |
| --- | --- | --- | --- |
| high | low | not needed | inject |
| high | high | yes | verify_before_use |
| high | high | no | confirm_before_use |
| medium | material | yes | verify_before_use |
| medium | material | no | confirm_before_use |
| medium | low materiality | not needed | weak_context |
| low | high | any | abstain |
| suppressed | any | any | audit_only or never_use |
| tombstoned | any | any | never_use and do not recapture |

## Product behavior examples

### Example A: stale tool memory verified by environment

Memory:

```text
Use pnpm in repo X.
```

Current context:

```text
same repo
memory last confirmed 4 months ago
pnpm-lock.yaml exists
```

Decision:

```text
verify silently through repo files
use after verification
update last_verified_at
```

No need to ask.

### Example B: stale tool memory contradicted by files

Memory:

```text
Use pnpm in repo X.
```

Current context:

```text
same repo
package-lock.json exists
no pnpm-lock.yaml
```

Decision:

```text
do not silently use pnpm
infer npm from files, or ask if ambiguity remains
mark memory confirm-needed or contradicted_by_environment
```

### Example C: global preference overridden by task

Memory:

```text
User prefers concise answers.
```

Current user says:

```text
Deep discussion and critique.
```

Decision:

```text
do not delete memory
do not ask
locally override
produce detailed answer
record current_instruction_overrode_preference if useful
```

### Example D: explicit preference override

Old:

```text
Prefer concise answers.
```

New:

```text
Actually, I prefer detailed answers by default now.
```

Decision:

```text
create new preference
supersede old preference
preserve edge and proof
use new default later
```

### Example E: local exception

Old:

```text
Prefer concise answers.
```

New:

```text
For this design doc, go deep.
```

Decision:

```text
create task/session exception
do not supersede global preference
expire exception after context ends
```

### Example F: recall-sensitive memory

Memory:

```text
Codeword is X.
```

Policy:

```text
explicit_request_only
never_proactive
```

User asks:

```text
What was the codeword?
```

Decision:

```text
reveal only if explicit recall policy allows it
```

User says:

```text
Forget that codeword.
```

Decision:

```text
delete or redact content
create tombstone
prevent re-capture
```

## Priority implementation plan

### Phase 1: Add authority decisions to context selection

Do not overhaul the schema first. Add a resolver that returns:

- `inject`
- `weak_context`
- `confirm_before_use`
- `verify_before_use`
- `abstain`
- `audit_only`
- `never_use`

This creates the behavioral slot where authority can live.

### Phase 2: Split confidence from current validity

Add or approximate:

- `evidence_confidence`
- `current_validity`
- `behavioral_authority`

Stop treating stale memories as merely lower-confidence memories.

### Phase 3: Make supersession deterministic for same-key value changes

Change upsert behavior:

```text
same key + same value => reinforce
same key + changed value => revision/supersession candidate
same key + narrower scope => exception candidate
```

This is likely the single biggest correctness win.

### Phase 4: Add tombstones

Soft delete is not enough. Tombstones are essential for trust.

Add safe tombstone keys that prevent re-capture without storing sensitive
content.

### Phase 5: Add validation strategies

For each memory type, define whether stale use should trigger:

- user confirmation
- environment verification
- tool outcome validation
- silent abstention
- explicit request only

### Phase 6: Add proof surfaces

Every injected, suppressed, superseded, verified, or confirm-needed memory
should be explainable.

Proof should show not only:

```text
why this memory matched
```

but:

```text
why this memory had authority
```

or:

```text
why this relevant memory was not used
```

## Test cases to add

- relevant but stale preference triggers weak context or confirm, not blind
  injection
- current explicit instruction overrides old soft preference
- environment-verifiable memory triggers verification instead of asking
- stale package-manager memory updates `last_verified_at` after lockfile check
- same-key changed value creates revision/supersession lineage
- scoped exception does not erase global preference
- forget request soft-deletes ordinary memory
- suppress request prevents proactive injection but keeps audit
- tombstone prevents re-capture
- hard delete/redacted audit path works for sensitive recall memory
- harmful/corrected injection creates authority event, not just score change
- superseded memories are visible in proof but not injected
- lower-scope memory beats newer broader memory when both match
- stale low-risk preference becomes weak context, not confirmation spam

## Open questions for advice

1. What is the right balance between stored validity and runtime authority?

2. Which memory types should be user-confirmed versus environment-verified?

3. How should OpenClawBrain distinguish true override from scoped exception?

4. Should explicit user preferences decay at all, or only decay into
   confirm-needed/weak-context?

5. What should the default "forget that" behavior be for ordinary memory?

6. What should the default "forget that" behavior be for sensitive recall
   memory?

7. Should tombstones be visible in graph/proof surfaces, and at what redaction
   level?

8. How much confirmation friction is acceptable before memory feels annoying?

9. Should authority events feed route-policy-v3 directly, or should they first
   live as memory-layer evidence?

10. What should the resolver do when the current user instruction conflicts
    with a high-authority hard constraint?

## Recommended product stance

The updated stance:

```text
Never confuse relevance with authority.
Never let old memory silently override current instruction.
Never use private or suppressed memory proactively.
Never re-capture tombstoned memory without explicit reauthorization.
Prefer verification over confirmation when the environment can answer.
Preserve auditability unless privacy requires redaction or deletion.
Ask for confirmation only when the memory is user-owned, material, uncertain,
and not cheaply verifiable.
```

This is the humane version of agent memory.

If OpenClawBrain gets this right, memory will feel less like a pile of recalled
facts and more like responsible judgment:

```text
It remembers.
It checks whether the memory still has authority.
It respects current instructions.
It verifies environment facts.
It asks only when asking is worth it.
It forgets in ways that match user intent.
```

The next architecture step is therefore:

> Add a memory authority layer between retrieval and injection.
