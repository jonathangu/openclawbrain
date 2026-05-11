# The Ultimate Guide to OpenClawBrain

**Evidence, not vibes, for agent memory.**

OpenClawBrain is local, accountable memory for OpenClaw agents. It remembers durable corrections, preferences, workflows, project facts, routing rules, and outcomes, then learns when that memory should affect a future turn.

The short version is:

> An agent should not remember everything all the time. It should learn when memory actually matters.

The current release, `openclawbrain@0.2.29`, is the result of several failed and partially-successful attempts to build that idea. The final architecture is not what the first plan expected. The project started with a strong belief in graph memory and simulation proof. It went through Python mechanism experiments, an eval-heavy V5 system, a native OpenClaw plugin, a flat-file v0.1 memory injector, a v0.2 SQLite graph, an aggressive capture loop, a route teacher, route-policy-v2, route-policy-v3 as the production route brain, Memory Authority resolution between retrieval and injection, Codex continuity owned entirely by OpenClawBrain rather than OpenClaw core, and now Memory Graph Maintenance for long-term graph health.

The most important lesson is not "use a graph." It is:

> Memory quality is a routing problem before it is a storage problem.

The graph matters. SQLite matters. Full-text search matters. But the core product is the route function that decides whether memory should participate at all, what kind of memory is allowed, how much context budget it gets, and when the correct action is silence.

```text
feedback and outcomes
  -> redacted route frames
  -> SQLite evidence graph
  -> shadow decisions and replay cases
  -> calibrated candidate snapshots
  -> active route-policy-v3 route_fn
  -> Memory Authority resolver
  -> bounded context injection or abstention
```

## The Core Claim

OpenClawBrain is not a note-taking feature. It is not a bigger RAG index. It is not a prompt that tells the model to "remember better."

It is a local learning loop around an AI agent:

1. Observe normal work.
2. Capture durable signals such as corrections, accepted help, route misses, tool outcomes, and handoffs.
3. Redact and scope those signals.
4. Store them as a SQLite-backed graph plus evidence tables.
5. Search and rank candidate memories only when the route function says memory is likely to help.
6. Inject a small context block, or abstain.
7. Record what happened.
8. Use outcomes, teacher critiques, counterfactuals, shadow decisions, replay, and calibration to improve the route function.

The invariant is the spine of the whole system:

> **LLM decides semantic meaning. Code enforces trust boundaries. SQLite stores the graph and evidence.**

The LLM is useful for semantic compression. It can notice that a messy sentence is really a correction, that a failed retrieval was a route miss, or that a tool outcome should become a workflow lesson. But the LLM does not get to write directly to memory or production routing. Code validates, redacts, scopes, dedupes, thresholds, stores, replays, calibrates, promotes, and rolls back.

## Why Memory Is Harder Than It Looks

The naive version of agent memory sounds simple:

```text
save useful facts
retrieve relevant facts later
put them in the prompt
```

That breaks down quickly.

Some memories are true only for a repo, a channel, a user, or a project. Some are temporary. Some are stale. Some are dangerous to store as plaintext. Some are correct but irrelevant most of the time. Some should be searched but not injected. Some should only be used if the user explicitly asks. Some should supersede older memories. Some should never be stored at all.

The failures are not edge cases. They are the product.

The agent has to answer questions like:

- Is this an explicit correction or a casual preference?
- Is this user-authored or assistant-invented?
- Does this apply to this repo, this agent, this channel, or globally?
- Does the current turn really need memory?
- Is the memory fresh enough?
- Would injecting it help or pollute the prompt?
- Is confidence high enough for a visible action?
- Should a low-confidence route fall back or stay silent?
- Can we prove why this memory was captured, rejected, retrieved, injected, or ignored?

OpenClawBrain exists because these decisions need durable machinery, not a larger prompt.

## The Earlier Path: Simulation First

Before the production plugin settled into the current architecture, there was a separate Python proof track called Brain Ground Zero. It tested the mechanism in isolation: graph memory, route functions, policy-gradient updates, structural plasticity, RAG baselines, recurring workflows, relational drift, and sparse feedback.

The Python benchmark was useful. It showed that a "full brain" mechanism could beat simpler RAG and partial-brain baselines in controlled long-lived memory tasks while using much less context. The proof bundles measured relational drift, recurring workflows, and sparse feedback across multiple seeds and baselines.

But it also taught a painful lesson:

> A benchmark can prove the mechanism and still not ship the product.

The simulation world made the hidden variables clean. Production agent memory does not. Real agent turns have messy language, private data, tool failures, latency limits, scanner constraints, plugin loading problems, auth, packaging, stale route handlers, and real users who get annoyed when memory appears at the wrong time.

The benchmark answered "could this mechanism work?" The product still had to answer:

- Can it load as a native OpenClaw plugin?
- Can it run under the actual Gateway Node runtime?
- Can it avoid blocking every turn with another LLM call?
- Can it store useful evidence without leaking raw transcript text?
- Can it pass ClawHub packaging and scan expectations?
- Can a friend install it from ClawHub?
- Can the operator inspect what happened?
- Can memory be quiet by default?

That is why the current OpenClawBrain is not a Python simulator. It is a native OpenClaw plugin with hooks, routes, services, SQLite storage, local LLM paths, proof surfaces, package verification, and public install instructions.

## The Failed Mental Models

### 1. "The graph is the product"

The first strong instinct was graph-first. That was not wrong. Memory really does need nodes, edges, supersession, scope, confidence, freshness, FTS, and audit records.

But a graph by itself does not decide when to speak.

If a graph is always retrieved, it becomes prompt pollution. If it is never retrieved, it is a museum. If retrieval is just keyword search, it misses durable corrections and over-fires on incidental terms.

The final lesson:

> The graph is the evidence substrate. The route function is the product behavior.

### 2. "More context is safer"

More context feels safe because it gives the model more information. In practice, more context often makes the agent worse. It distracts, increases latency, creates stale instruction conflicts, and gives the model extra ways to rationalize the wrong action.

OpenClawBrain therefore optimizes for sparse injection:

```xml
<openclawbrain_context>
Relevant memory:
- Must follow: Use pnpm instead of npm in this repo.
</openclawbrain_context>
```

That is the ideal shape: one or two pieces of context at the exact turn where they matter.

### 3. "Capture every useful thing"

Automatic capture is dangerous. Assistants often say things like "I'll remember that," but assistant output is weak evidence. A casual user sentence can look like a rule. A private value can look like a useful memory. A one-time task can look like a workflow.

The current stance:

- Capture broadly enough to notice candidates.
- Store narrowly after validation.
- Treat user-authored corrections as high authority.
- Treat assistant claims as weak support.
- Keep raw transcript storage off.
- Keep proof rows for accepts, rejects, abstentions, and injections.

### 4. "The LLM can own the memory loop"

LLMs are good at semantic grouping and summarization. They are not trustworthy database writers. They can over-generalize, forget scope, invent confidence, or transform a private one-time event into a global rule.

The boundary became:

```text
LLM:
  propose semantic meaning
  summarize redacted evidence
  critique route decisions
  propose candidate labels

Code:
  redact
  validate schema
  enforce scope
  dedupe
  threshold
  store
  search
  replay
  calibrate
  promote
  roll back
```

### 5. "A cleaner plan will solve it"

OpenClawBrain had many plans. The plans helped, but the real progress came when every plan had to produce an inspectable artifact:

- a route
- a table
- a proof event
- a test
- a package
- a temp-HOME install
- a live endpoint
- a public page

The meta-lesson for AI-assisted building is brutal but useful:

> Ask the AI for evidence surfaces, not just implementation surfaces.

## The Evolution

### Stage 0: Mechanism Proof

The Brain Ground Zero work tested whether "full brain" memory mechanisms could beat RAG-like baselines under long-lived memory pressure. The important families were:

- relational drift
- recurring workflows
- sparse feedback
- recorded head-to-head fixture replay

This track was valuable because it forced the project to compare:

- no brain
- vector RAG
- vector rerank
- static graph
- heuristic stateful
- route function only
- graph plus route policy
- full brain

It also introduced ideas that later mattered in production:

- route_fn as the served policy
- teacher feedback
- background labeling
- policy updates
- structural graph memory
- decay
- connect, split, merge, prune
- context efficiency as a product metric

What did not transfer directly was the neatness. Production memory needed privacy, latency, packaging, and operator proof.

### Stage 1: V5 Eval and Evidence Pipeline

The earliest OpenClawBrain repository history was dominated by V5 runtime evidence and evaluation infrastructure. It had trace admission, blind judge packets, ledgers, thresholds, product decision trees, tool-trace safety, and result pages.

This was rigorous, but it was not yet the shipped plugin. In fact, the repo later had a major revamp that deleted hundreds of files and reset the product around a smaller native OpenClaw plugin.

The lesson:

> Evaluation scaffolding is only useful if it is connected to the runtime that users actually install.

### Stage 2: Native Plugin and v0.1

The next shape was a native OpenClaw plugin. That mattered because the product had to live inside OpenClaw, not beside it.

The first native version was closer to a static context injector:

- classify the turn
- read activation files
- inject context on selected slices
- expose status/proof routes

That proved the plugin path. It did not prove real memory.

The failure:

> A notepad with a turn classifier is not memory.

v0.1 made memory visible, but it still required manual notes and did not learn from outcomes.

### Stage 3: v0.2 Plan and SQLite Graph

The v0.2 plan moved from flat files into a real local runtime:

- `memory_nodes`
- `memory_edges`
- `memory_search` FTS5
- `memory_injections`
- `route_decisions`
- `route_examples`
- `route_policy_snapshots`
- `distillation_runs`
- `background_jobs`
- `proof_events`
- `capture_audit`

This was the point where the graph became durable. It also separated evidence from memory:

- evidence plane: what happened
- memory plane: what the system currently believes
- recall plane: what the model can search or see

The implementation added the modules that still define the current product:

- `memory-store.ts` for SQLite schema, FTS, graph, jobs, route decisions, proofs, and audit rows
- `feedback-distiller.ts` for structured feedback extraction
- `memory-operations.ts` for validated memory changes
- `route-fn.ts` for prompt-time route planning
- `context-selector.ts` for bounded context formatting
- `learning.ts` for outcome resolution and score updates
- `route-learning.ts` for route examples and policy snapshots
- `search.ts` for route payloads and memory supplements
- `index.ts` for hooks, routes, and service wiring

The key lesson:

> Store every decision you will later need to debug.

`use_count` on a memory is not enough. The system needed injection rows, route decision rows, distillation audit rows, proof rows, and later route-policy-v3 warehouse rows.

### Stage 4: Packaging Reality

The first "working in repo" version was not the same as "working when installed."

The live extension hit native dependency problems: `better-sqlite3` lacked the correct binding under the Gateway Node runtime. The package could pass tests in the workspace and still fail in the installed extension path.

That forced the SQLite fallback work:

- try `better-sqlite3` first
- fall back to Node's `node:sqlite`
- add a `/doctor` route
- smoke-test SQLite and FTS under the real runtime
- verify temp-HOME installs, not just repo tests

The lesson:

> If the install path is different from the dev path, the install path is the product.

### Stage 5: Scanner and Public Release Reality

ClawHub scan issues taught another lesson. A local LLM URL, config wording, or package description can look suspicious when interpreted by a security scanner. A `SKILL.md` that makes sense to an operator can look like prompt override language to a broad scanner.

That forced copy and metadata changes:

- rename `allowPromptInjection` to `allowPromptContext`
- move config under `plugins.entries.openclawbrain.config`
- remove scanner-sensitive language from packaged files
- describe localhost LLM paths as config, not install sources
- separate static scan status from LLM review status
- publish source-linked packages with clear tags and release archives

The lesson:

> Public software is not just code. It is how the code is described to tools that do not share your intent.

### Stage 6: Aggressive Memory Loop

Once install worked, the project pushed into "aggressive but safe" memory. Explicit remember requests, capture intent, recall rules, tool conventions, agent assignments, and route rules became more important.

This stage exposed the capture/retrieval split:

- capture can be broad
- retrieval must be careful
- plaintext recall values require explicit policy
- secret-like values are not the same as user-approved recall facts
- the route function should not block capture just because retrieval is conservative

The lesson:

> Capture and retrieval are different products. Conflating them makes memory either timid or unsafe.

### Stage 7: Route Teacher and Counterfactuals

The project then added a route teacher. It could critique actual route decisions and generate counterfactuals:

- What if no memory had been used?
- What if a different memory type had been used?
- What if graph depth had widened?
- What if the system stayed silent?
- What if the planner had been allowed?

This mattered because route learning needs negative examples. A memory system that only records successes never learns when to abstain.

The lesson:

> The system must learn from the memory it did not use.

### Stage 8: Route-Policy-v2

Route-policy-v2 was the first compact learned route function. It moved beyond policy text and into a deterministic schema:

- supported route gates
- memory type gates
- broad-rule rejection
- sync-budget enforcement
- offline scoring
- activation/shadow states
- matched rule IDs
- proof payloads exposing active policy and counterfactual summary

This was a real product step. It made the learned route function inspectable.

But v2 was still too rule-like. It could route, but it did not have enough warehouse structure for calibration, replay, candidate comparison, action-family stats, and champion/challenger promotion.

The lesson:

> A learned rule list is a bridge. The production system needs a learning warehouse behind it.

### Stage 9: Route-Policy-v3

Route-policy-v3 is the current production route brain.

It changed the target from "learn some routing rules" to a full route-learning system:

```text
turn outcomes + critiques + counterfactuals + shadow proposals
  -> normalized learning warehouse
  -> prototype/action learning + calibration + replay eval
  -> candidate policy distillation
  -> champion/challenger promotion gates
  -> compact deterministic route_fn snapshot at runtime
```

The hard cutover decision was:

- v3-first serving
- v2 fallback and rollback only
- legacy heuristics as last-resort fallback
- abstention as production behavior
- `gated_active` as the default target mode
- champion/challenger promotion
- shadow decisions as evidence
- rollback lineage as mandatory
- stricter gates for sync/planner action families
- redacted-only learning storage

This is the architecture that finally matches the product claim.

## What OpenClawBrain Stores

OpenClawBrain stores memory and route evidence locally in SQLite.

The basic graph objects are:

- `memory_nodes`: corrections, preferences, workflows, project facts, tool conventions, routing rules, agent assignments, recall rules, outcomes, and context
- `memory_edges`: related, contradicts, supersedes, extends, used_with, supports_workflow
- `memory_search`: SQLite FTS5 search over content, tags, and normalized keys
- `memory_injections`: what memory was injected, where, with what rank/score, and what outcome later resolved
- `route_decisions`: route, confidence, latency tier, selected/omitted memory IDs, policy snapshot, outcome, reward
- `proof_events`: auditable records for capture, reject, retrieve, inject, abstain, and learn
- `distillation_runs`: structured LLM proposal audit with input hashes and validation status

Route-policy-v3 adds a learning warehouse:

- `route_frames_v3`
- `route_shadow_decisions_v3`
- `route_calibration_examples_v3`
- `route_eval_cases_v3`
- `route_eval_case_labels_v3`
- `route_action_family_stats_v3`
- `route_policy_candidate_reports_v3`
- action prototypes, pair examples, bandit feedback, and bandit state

The point is not to collect data for its own sake. The point is to be able to answer:

- Why did this memory exist?
- Why did this route retrieve memory?
- Why did this route abstain?
- Which policy snapshot was active?
- Which rule matched?
- Which memory IDs were selected or omitted?
- What would the candidate policy have done?
- Why was a candidate promoted, shadowed, rejected, or rolled back?

## The Runtime Path

A normal turn follows this shape:

```text
before_prompt_build
  -> redact current turn
  -> route-policy-v3 decides whether memory should participate
  -> if v3 abstains, v2 and heuristics can fall back
  -> SQLite FTS + graph search finds candidates
  -> ContextSelector chooses a tiny memory set
  -> bounded context is injected or nothing is injected
  -> route decision and proof rows are recorded

agent_end / after_tool_call / background service
  -> resolve outcomes
  -> distill durable feedback
  -> update memory scores
  -> produce route frames
  -> run teacher and counterfactuals
  -> update shadow/eval/calibration evidence
  -> generate candidate snapshots
  -> promote or keep shadow/rejected
```

Most prompt-time work is local and deterministic. Heavier semantic work happens after the turn.

The latency model has four tiers:

| Tier | Meaning | Prompt-time model call |
| --- | --- | --- |
| Tier 0 | local route decision only | no |
| Tier 1 | cached route plus SQLite retrieval | no |
| Tier 2 | one bounded memory planner call for ambiguous/high-signal turns | yes, limited |
| Tier 3 | distillation, learning, pruning, replay, calibration, promotion | no, background |

This design exists because memory cannot become a tax on every turn. If memory makes the agent slower every time, the user will disable it. If memory only acts when the evidence supports it, it can become useful without becoming noisy.

## The Route-Policy-v3 Loop

Route-policy-v3 is a compact deterministic route function served from a learned snapshot.

At runtime:

```text
active valid route-policy-v3 snapshot
  -> calibrated family-aware match
  -> bounded context injection + proof

v3 abstains / no safe match / invalid snapshot
  -> route-policy-v2 fallback

v2 misses or rollback required
  -> legacy heuristics as last resort
```

The important word is "abstain."

Abstention is not a failure. It is a production route. If the policy lacks support, confidence, scope, or safe match, it should keep the prompt clean. That is how memory remains trustworthy.

Candidate policies are promoted only after gates:

- schema validation
- broad-rule rejection
- replay/eval performance
- family-specific calibration
- cold-start support
- sync-budget checks
- harm-rate checks
- activation cooldowns
- rollback lineage
- candidate report generation

The current active snapshot is the champion. New snapshots are challengers. A challenger must show enough improvement and safety to replace the champion.

## Why SQLite

SQLite is boring in the best possible way.

OpenClawBrain needs local-first storage with:

- durable tables
- transactions
- FTS5
- indexes
- typed rows
- inspectable files
- no hosted dependency
- no external graph database
- easy backups
- compatibility with local plugin runtime

SQLite lets OpenClawBrain be a local product. The graph is not a hosted service and not a magic vector database. It is an ordinary file under the activation/runtime directory, with memory nodes, edges, search, route decisions, proof rows, and v3 learning tables.

This is also why proof is credible. The system can expose exactly what happened through local routes:

- `/plugins/openclawbrain/status`
- `/plugins/openclawbrain/doctor`
- `/plugins/openclawbrain/proof`
- `/plugins/openclawbrain/graph`
- `/plugins/openclawbrain/learn`
- `/plugins/openclawbrain/search`
- `/plugins/openclawbrain/route-teacher`
- `/plugins/openclawbrain/route-counterfactuals`
- `/plugins/openclawbrain/route-policy`
- `/plugins/openclawbrain/audit`
- `/plugins/openclawbrain/explain-last`

## Why the Current Architecture Works

It works because it splits the problem along the right boundaries.

### Boundary 1: Meaning vs authority

The LLM can propose meaning. It cannot grant authority. A memory becomes authoritative only when code validates the source, scope, evidence, confidence, and safety.

### Boundary 2: Storage vs serving

The graph can store many things. The prompt should see very few things. The storage layer is rich; the serving layer is sparse.

### Boundary 3: Evidence vs belief

Evidence rows are append-only facts about what happened. Memory nodes and policy snapshots are current beliefs derived from evidence. That separation makes debugging possible.

### Boundary 4: Runtime vs update path

Runtime must be compact and low-latency. The update path can be richer, slower, and more analytical. That is why route-policy-v3 distills a compact snapshot for serving.

### Boundary 5: Active policy vs challenger policy

The active policy serves. Challenger policies shadow, replay, and report. Promotion is measured, not assumed.

### Boundary 6: Recall vs action

Retrieving a memory is not the same as injecting it. Injecting is not the same as acting. Acting is not the same as being correct. OpenClawBrain records these separately.

## What a Useful Turn Looks Like

Suppose the user says:

```text
Actually, use pnpm in this repo.
```

OpenClawBrain should not merely append that sentence to a notes file. It should:

1. Detect a high-confidence correction.
2. Redact and scope it to the current repo/project.
3. Store a memory node.
4. Update FTS.
5. Record a proof event.
6. Later, when the user asks to run tests, route the turn.
7. Decide that a repo workflow/tool convention memory is relevant.
8. Search and select the correction.
9. Inject a bounded context block.
10. Record the route decision and injection.
11. Observe whether the agent used pnpm and whether the command worked.
12. Update the memory score and route evidence.

The visible result should be tiny:

```xml
<openclawbrain_context>
Relevant memory:
- Must follow: Use pnpm instead of npm in this repo.
</openclawbrain_context>
```

The invisible result should be rich:

- route decision
- selected memory IDs
- omitted memory IDs
- injection row
- proof row
- outcome
- reward
- teacher/counterfactual evidence if useful
- route-policy training example

## Lessons Learned Building It

### Lesson 1: The first working demo is usually the wrong abstraction

The v0.1 plugin proved that context could be injected. That was necessary. It was not the product. The product needed evidence, learning, and routing.

### Lesson 2: Plan for proof before you plan for features

Every major feature needs a proof surface:

- How do I know it loaded?
- How do I know it captured?
- How do I know it rejected something?
- How do I know it searched?
- How do I know it injected?
- How do I know it learned?
- How do I know it abstained?
- How do I know I can roll back?

### Lesson 3: Store the rejected decisions

If you only store memories, you cannot debug why bad memories did not become memories. If you only store injections, you cannot debug missed recall. If you only store successes, you cannot learn abstention.

### Lesson 4: Public packaging is part of architecture

The Node runtime, native SQLite bindings, ClawHub scan wording, manifest fields, install command, temp-HOME install, and gateway restart behavior all shaped the final design. A plugin that works only in the repo does not exist for users.

### Lesson 5: "Aggressive" and "safe" are not opposites if the stages are separated

OpenClawBrain can be aggressive in noticing evidence and conservative in serving it. This is possible because capture, storage, retrieval, injection, and promotion are separate stages.

### Lesson 6: Correct silence must be rewarded

A memory system that only rewards retrieval will retrieve too much. The route learner needs examples where `no_memory` was the right answer.

### Lesson 7: The LLM should be a teacher, not the runtime

The LLM can critique, summarize, label, and propose. The runtime should serve deterministic snapshots, not call a teacher on every turn.

### Lesson 8: Scoping is not metadata, scoping is safety

A repo rule, channel rule, agent assignment, or project preference can be correct in one place and harmful in another. Scope is part of the truth value.

### Lesson 9: A graph without a route function is a liability

Graphs make it easy to connect things. They also make it easy to over-connect. Route-policy-v3 is what prevents graph expansion from becoming context sprawl.

### Lesson 10: The final design should make failure boring

If v3 is missing, fall back. If v3 abstains, fall back. If v2 misses, use heuristics. If the LLM is unavailable, keep the main agent working. If proof JSONL is torn, use SQLite. If the native SQLite driver fails, use fallback. If a candidate policy is risky, keep it shadow.

## Meta-Lessons: How To Ask AI To Build Something Like This

OpenClawBrain was built with AI help. That worked only when the requests became more concrete, more evidence-based, and more hostile to hand-wavy completion.

Here are the lessons.

### 1. Do not ask for "the ultimate system" first

Ask for:

- the smallest runtime hook
- the first status route
- the first proof row
- the first schema
- the first test
- the first temp install
- the first live verification

Then ask for the next layer.

### 2. Ask the AI to preserve a single invariant

For OpenClawBrain, the invariant was:

> LLM decides semantic meaning. Code enforces trust boundaries. SQLite stores the graph and evidence.

This helped keep the implementation from drifting into model-owned memory, prompt-owned storage, or unverifiable magic.

### 3. Ask for failure tables

Good prompt:

```text
List every way this memory route can be wrong: over-capture, under-capture,
stale retrieval, scope leak, prompt pollution, latency regression, install failure,
scanner failure, rollback failure. For each, name the row/table/proof/test that catches it.
```

Bad prompt:

```text
Make the memory smarter.
```

### 4. Ask for public claims to be smaller than private ambition

The public claim should lag the internal dream. That prevents the project from publishing "AI memory brain" language before install, proof, and rollback are real.

### 5. Force the AI to work from repo history

Ask:

```text
Use git log, tags, closeout notes, tests, package versions, and live endpoints.
Do not invent the history. Separate what shipped from what was planned.
```

That keeps the writeup honest.

### 6. Require install-path verification

For a plugin, tests are not enough. Ask for:

- package build
- temp-HOME install
- enable
- config validate
- gateway restart or reload
- runtime inspect
- live route checks
- public package inspect

### 7. Ask for "what should stay boring"

Not every part should be clever. In OpenClawBrain:

- storage should be SQLite
- prompt context should be small
- runtime route snapshots should be deterministic
- failure should be fallback/abstention
- proof should be inspectable rows

The AI will often add cleverness unless asked to protect boring parts.

### 8. Ask the AI to keep rejected ideas in the narrative

The rejected ideas are the guide. They explain why the current shape exists. For OpenClawBrain, the rejected ideas were not wasted: simulation-only proof, flat-file v0.1, broad graph retrieval, model-owned memory, and scan-sensitive package wording all taught something.

### 9. Demand "done means"

Every task should end with a definition of done:

```text
Done means: tests pass, diff check clean, package packs, temp install works,
runtime inspect shows loaded, site deploy succeeds, live URL contains the new copy.
```

Without that, the AI will stop at "implemented."

### 10. Ask for the learning loop in one diagram

If the AI cannot draw the learning loop, it probably does not understand the system. For OpenClawBrain, the loop is:

```text
observe -> redact -> store evidence -> route -> retrieve -> inject/abstain
  -> observe outcome -> teacher/counterfactual -> shadow/replay/calibrate
  -> promote or roll back -> serve a compact route_fn
```

## The Memory Authority Upgrade

The next hard problem was not "retrieve better memories." It was deciding
whether a retrieved memory still had the right to influence the current turn.

OpenClawBrain adds a `MemoryAuthorityResolver` between retrieval and
context selection:

```text
search finds relevant candidates
  -> authority resolver checks validity, scope, privacy, supersession, risk
  -> context selector injects, weakens, verifies, confirms, abstains, or suppresses
  -> proof rows explain what happened
```

The new invariant is:

```text
relevant != authorized
```

A memory can be semantically relevant and still be wrong, expired, unsafe,
superseded, too broad, tombstoned, or overridden by the user's current
instruction. For example, "prefer concise answers" remains a useful default, but
it should not override a turn that explicitly asks for deep critique.

The resolver supports these decisions:

| Decision | Meaning |
|---|---|
| `inject` | The memory is relevant, current, scoped, safe, and strong enough to use. |
| `weak_context` | The memory is a soft prior, not a command. |
| `verify_before_use` | The memory is environment-owned and should be checked before acting. |
| `confirm_before_use` | The memory is user-owned, material, uncertain, and not cheaply verifiable. |
| `abstain` | The memory is relevant but should not affect this turn. |
| `audit_only` | The memory stays visible for inspection, not prompt influence. |
| `never_use` | The memory is deleted, tombstoned, private, or otherwise blocked. |

This adds two new SQLite surfaces:

- `memory_validity`: retention state, behavioral availability, temporal
  validity, privacy class, decay policy, validation strategy, and authority
  scores.
- `memory_authority_events`: records of memories being used, weakened, verified,
  confirmed, suppressed, tombstoned, superseded, or withheld.

It also changes update behavior. Same-key same-value captures reinforce an
existing memory. Same-key changed-value captures create lineage instead of
overwriting history. Sensitive "forget" requests create tombstones so the system
does not delete a memory and then quietly recapture it later.

## The 0.2.25 Upgrade: Codex Continuity

The practical workflow changed too. Jonathan now uses Codex UI as the high-bandwidth coding workbench and OpenClaw/Telegram as the mobile operator surface. OpenClawBrain should therefore remember the operating context and expose a quiet control plane without turning Telegram into a second coding UI.

The 0.2.25 bridge is deliberately OpenClawBrain-owned:

```text
Codex UI does deep local work
  -> OpenClawBrain reads local Codex state
  -> Memory Authority decides what matters
  -> Telegram/OpenClaw gets status, watched terminal events, or handoff briefs
  -> OpenClaw core stays stock and upgradeable
```

The bridge provides:

- `/brain codex status`
- `/brain codex threads`
- `/brain codex watch`
- `/brain codex handoff`
- read-only HTTP routes under `/plugins/openclawbrain/codex/*`

It stores only redacted bridge audit events and durable operating truths. It does not store raw Codex messages, full command output, full diffs, or temporary telemetry as durable memory. Telegram-to-Codex writes are disabled by default and must stay feature-flagged behind trusted sender, repo allowlist, provenance, risk classification, and confirmation controls.

## The 0.2.29 Upgrade: Memory Graph Maintenance

Memory Authority fixed the runtime question:

```text
This memory is relevant, but can it influence this turn?
```

Memory Graph Maintenance fixes the long-term graph question:

```text
After many turns, corrections, stale facts, tombstones, and outcomes,
how should the graph itself evolve?
```

The important boundary is that graph maintenance can provide features and proposals, but it cannot inject memory or certify authority. `MemoryAuthorityResolver` still recomputes turn-level use every time.

The first implementation is deliberately conservative:

- health metrics for duplicates, bad edges, stale high-authority nodes, tombstones, and scoped exception candidates
- dry-run proposals before mutation
- exact duplicate consolidation with canonical lineage
- bad edge retirement with edge observations
- stale high-authority detection as review-gated proposals
- tombstone recapture detection without leaking tombstoned content
- scoped exception proposals for repeated current-instruction overrides
- route-teacher feedback recorded as behavioral observations only, not truth evidence

Operator commands:

```text
/brain graph health
/brain graph dry-run
/brain graph proposals
/brain graph apply <proposalId>
/brain graph reject <proposalId>
/brain graph stale
/brain graph clusters
/brain graph tombstones
/brain graph explain <proposalId>
```

The product lesson is simple:

> Generic memory retrieves old context. OpenClawBrain governs memory as evidence: provenance, scope, validity, correction, forgetting, and proof.

## The Practical Operator Model

Install or upgrade:

```bash
openclaw plugins install clawhub:openclawbrain@0.2.29 --force
openclaw plugins enable openclawbrain
openclaw gateway restart
```

Verify runtime, not just package metadata:

```bash
openclaw plugins inspect openclawbrain --runtime
openclaw doctor
```

Then inspect the memory surfaces through the authenticated OpenClaw client:

```text
/plugins/openclawbrain/status
/plugins/openclawbrain/doctor
/plugins/openclawbrain/proof?limit=10
/plugins/openclawbrain/search?query=pnpm&limit=10
/plugins/openclawbrain/graph?limit=50
/plugins/openclawbrain/graph/health
/plugins/openclawbrain/graph/dry-run
/plugins/openclawbrain/graph/proposals
/plugins/openclawbrain/route-policy
/plugins/openclawbrain/explain-last
/plugins/openclawbrain/codex/status
/plugins/openclawbrain/codex/threads
/plugins/openclawbrain/codex/handoff
```

You want to see:

- plugin loaded
- enabled and activated
- SQLite and FTS healthy
- hooks registered
- routes registered
- no active memory plugin warning
- proof rows appearing
- search returning scoped memories when they exist
- route-policy-v3 surfaced as the active route brain
- authority events explaining why relevant memories were injected, weakened,
  verified, confirmed, suppressed, or withheld

## Current Public Truth

As of `0.2.29`:

- The latest package is `openclawbrain@0.2.29`.
- The source tag is `v0.2.29`.
- The production route brain is route-policy-v3.
- Memory Authority now separates relevance from authority before injection.
- Memory Graph Maintenance keeps the graph healthier through dry-run proposals, canonical lineage, edge observations, tombstone recapture checks, and proofed deterministic repairs.
- v2 and heuristics are fallback/rollback paths.
- SQLite stores the graph and evidence locally.
- Prompt injection is bounded and proofed.
- Local LLM paths are used for semantic distillation and learning when available.
- Runtime should keep working if learning gets quieter.
- The Codex continuity bridge is owned by OpenClawBrain and does not require OpenClaw core patches.
- ClawHub source linking is present; scan status may still show pending while review catches up.

## The Big Lesson

The project started with a graph.

It became a memory system only when it learned to ask:

```text
Should memory be here at all?
```

That is the core of OpenClawBrain. The graph remembers. The search finds. The selector formats. The proof store records. But the learned route function is what makes memory behave like a product instead of a pile of facts.

The final shape is therefore:

```text
local evidence graph
  + learned route_fn
  + calibrated abstention
  + bounded prompt injection
  + memory authority
  + graph maintenance
  + proof and rollback
```

That is why OpenClawBrain works the way it works.
