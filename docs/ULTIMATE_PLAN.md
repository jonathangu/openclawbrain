# OpenClawBrain Ultimate Plan

**Status date:** 2026-05-01  
**Repo:** [`jonathangu/openclawbrain`](https://github.com/jonathangu/openclawbrain)  
**Current release lane:** `openclawbrain@0.2.4` published; ClawHub scan still suspicious; live Mac mini install is `0.2.3`.

---

## 0. One-sentence truth

OpenClawBrain is no longer just an idea or a flat-file prototype: the v0.2 local memory-graph runtime exists, but the final ultimate vision still needs a clean public release, live dogfooding on the newest build, stronger semantic capture/learning loops, richer proof/evaluation surfaces, and real-world reliability until it demonstrably makes OpenClaw better over time without getting noisy, unsafe, or slow.

---

## 0.5 Strategic sequencing

The end state is not reached by adding cleverer semantic memory first. It is reached by making the system **boringly reliable**, then **trustworthy**, then **adaptive**, then **provable**.

Execution order:

1. **Reliability first** — live install works, native SQLite works, memory search/graph/proof routes work under the exact Gateway Node runtime.
2. **Trust layer second** — weak evidence cannot become durable memory; candidates have explicit promotion reasons and contradiction/supersession proof.
3. **Adaptive routing third** — the learned route function optimizes for helpful injection, missed recall, noisy injection, and correct silence.
4. **Workflow memory fourth** — only successful, repeated, or explicitly approved tool workflows become procedural memory.
5. **Proof and evidence last** — public claims come from dogfood evidence, not vibes or broader retrieval.

The strategic discipline is: every feature must make memory more reliable, more selective, more inspectable, or more measurably useful. Anything else is a distraction.

---

## 1. The final ultimate vision

The target state is not “better memory.” The target state is:

> OpenClawBrain is OpenClaw’s local, inspectable, self-regulating decision memory layer. It learns the small pieces of past context, corrections, workflows, and non-interventions that actually improve important future turns, then injects only the few memories that matter.

The final system should satisfy all of these:

1. **Corrections stick automatically**  
   Jon says “use pnpm, not npm” once; future relevant turns use pnpm without a manual memory edit.

2. **Preferences and workflows become durable only when earned**  
   The system captures stable preferences, project conventions, and successful tool workflows, but does not overfit every casual sentence.

3. **Memory is a graph, not a log**  
   Memories have typed nodes, typed edges, supersession, confidence, importance, freshness, outcome history, and graph expansion.

4. **LLMs decide meaning; code enforces trust boundaries**  
   LLMs may distill feedback, classify route needs, and select candidate memories. They never write directly to storage. Code validates, redacts, scopes, budgets, dedupes, persists, audits, and prunes.

5. **Most turns pay zero extra synchronous LLM cost**  
   The learned `route_fn` / QTsim-style path is central. Cache, policy snapshots, SQLite retrieval, and graph scoring handle ordinary turns. One bounded planner call is allowed only for ambiguous/high-signal turns.

6. **The system knows when to stay silent**  
   Final OpenClawBrain should improve important decision points, not spray context into every answer. `STOP_LOCAL` / deliberate non-intervention is a first-class success case.

7. **Everything is local, scoped, inspectable, and easy to disable**  
   No raw transcript storage/upload. Redacted summaries, hashes, and proof records only. All behavior visible through status/proof/graph/learn/search surfaces.

8. **It is a native OpenClaw plugin**  
   No separate installer, no root `openclawbrain` config key, no daemon-first architecture. Configuration lives under `plugins.entries.openclawbrain.config`.

9. **It is publicly usable**  
   ClawHub install works, friend copy-paste instructions work, website copy matches reality, and scan/security posture is clean.

10. **It proves product impact honestly**  
   Claims are bounded. Unique product wins are counted separately from ties. Retrieval widening or score movement does not substitute for real decision-quality gains.

---

## 2. Where we are now

### 2.1 Built and working in repo

The v0.2 implementation exists in [`packages/openclaw-plugin`](../packages/openclaw-plugin):

- SQLite memory graph, FTS, edges, route decisions, injection records, distillation audits, jobs, and proof events: [`src/memory-store.ts`](../packages/openclaw-plugin/src/memory-store.ts)
- LLM feedback distillation with structured JSON validation: [`src/feedback-distiller.ts`](../packages/openclaw-plugin/src/feedback-distiller.ts)
- Validated memory operation application: [`src/memory-operations.ts`](../packages/openclaw-plugin/src/memory-operations.ts)
- Learned/local route function: [`src/route-fn.ts`](../packages/openclaw-plugin/src/route-fn.ts)
- Latency gate for sync planner calls: [`src/latency-controller.ts`](../packages/openclaw-plugin/src/latency-controller.ts)
- Bounded one-call memory planner: [`src/memory-planner.ts`](../packages/openclaw-plugin/src/memory-planner.ts)
- Bounded context selection and prompt formatting: [`src/context-selector.ts`](../packages/openclaw-plugin/src/context-selector.ts)
- Outcome learning, freshness decay, and pruning: [`src/learning.ts`](../packages/openclaw-plugin/src/learning.ts)
- Route examples and policy snapshots: [`src/route-learning.ts`](../packages/openclaw-plugin/src/route-learning.ts)
- Additive OpenClaw memory prompt/corpus supplements plus graph/search/learn payloads: [`src/search.ts`](../packages/openclaw-plugin/src/search.ts)
- Plugin registration, hooks, routes, service wiring: [`src/index.ts`](../packages/openclaw-plugin/src/index.ts)
- Runtime config and local/remote LLM gates: [`src/config.ts`](../packages/openclaw-plugin/src/config.ts)
- Plugin manifest and OpenClaw capability metadata: [`openclaw.plugin.json`](../packages/openclaw-plugin/openclaw.plugin.json)

### 2.2 Verified gates at handoff

Current handoff evidence:

```bash
pnpm --dir packages/openclaw-plugin test
# 53/53 pass

npm_config_cache=/tmp/openclawbrain-npm-cache npm pack --dry-run --workspace packages/openclaw-plugin
# openclawbrain@0.2.4, 46 files, 45.7 kB package, 215.5 kB unpacked
```

### 2.3 Published state

- Repo HEAD/tag at handoff: `a59a43f4` / `openclawbrain-v0.2.4`
- ClawHub latest: `openclawbrain@0.2.4`
- Current blocker: ClawHub scan is still `suspicious`
- Live Mac mini install: `openclawbrain@0.2.3`, enabled and loaded, with service `openclawbrain` and 5 HTTP routes
- `0.2.4` has not been installed live because latest scan is not clean yet

### 2.4 Important current defect

The installed live extension has a native dependency mismatch on Node `25.5.0`: `better-sqlite3` lacks the right native binding in `~/.openclaw/extensions/openclawbrain`. Repo tests pass because workspace dependencies are intact, but the installed extension path needs robust native-dependency packaging/rebuild handling.

---

## 3. What is already real vs. what is still missing

| Area | Real now | Missing for ultimate state |
|---|---|---|
| Native plugin | Package, manifest, hooks, service, routes exist | Clean scan, latest live install, robust public install/rebuild path |
| Memory graph | SQLite nodes, edges, FTS, injections, route decisions, audits | More mature contradiction handling, candidate promotion, graph explanations |
| LLM semantics | Feedback distiller and planner paths exist | Better default local model discovery, stronger semantic evals, real-world calibration |
| Latency safety | Route cache, policy snapshots, latency controller, bounded planner | More production telemetry showing most turns avoid sync calls |
| Injection | Bounded context selector exists | More proof that injected context is consistently useful and quiet |
| Learning | Outcome resolution, decay, pruning, route learning exist | Better success/failure signals, workflow capture, non-intervention learning |
| Proof | `/status`, `/proof`, `/graph`, `/learn`, `/search` surfaces exist | More operator-friendly explanations and end-to-end proof bundles |
| Public use | ClawHub releases and website updates exist | Clean ClawHub scan, copy-paste install that works on a friend’s machine |
| Safety | Redaction, no raw transcript storage, fail-closed gates | Broader adversarial tests and scanner-clean metadata |
| Evaluation | Historical trace/eval scaffolding and plugin unit tests exist | Fresh v0.2 dogfood eval showing product wins, restraint, and no regressions |

---

## 4. The gap list, in priority order

### P0 — Restore live memory and make the public release clean

This is the immediate blocker. Live memory search must work before higher-level semantic recall matters.

**Problem:** the installed extension can fail before recall starts if `better-sqlite3` cannot import, open an in-memory database, and create an FTS5 table under the exact Gateway Node runtime. Separately, `0.2.4` is published, but ClawHub still marks it suspicious. Static scan flags the localhost Ollama URL example as an untrusted install source. LLM scan also dislikes the package/skill text presentation.

**Required work:**

1. Repair and verify the live installed extension:
   - rebuild or reinstall `better-sqlite3` inside `/Users/guclaw/.openclaw/extensions/openclawbrain`
   - run a native SQLite smoke test: import, `:memory:` DB, `select 1`, FTS5 create/query
   - restart/reload Gateway
   - verify `/status`, `/doctor`, `/search`, `/graph`, and `/proof` routes
2. Cut `0.2.5` as a reliability release with scanner-safe metadata:
   - remove or reword concrete loopback URL examples from scanner-sensitive fields
   - describe localhost Ollama as a config concept, not as an install/download source
   - make manifest/package copy clearly describe runnable plugin code
   - remove any SKILL/package wording that looks like prompt injection
3. Re-run tests/build/pack.
4. Publish to ClawHub.
5. Confirm scan is clean.
6. Verify temp-HOME install outside repo-local `node_modules`.
7. Only then install latest on the live Mac mini, unless Jon explicitly accepts manual-review/danger path.

**Code/doc touchpoints:**

- [`packages/openclaw-plugin/openclaw.plugin.json`](../packages/openclaw-plugin/openclaw.plugin.json)
- [`packages/openclaw-plugin/package.json`](../packages/openclaw-plugin/package.json)
- [`README.md`](../README.md)
- [`docs/GETTING_STARTED.md`](GETTING_STARTED.md)

**Done means:**

```bash
pnpm --dir packages/openclaw-plugin test
npm_config_cache=/tmp/openclawbrain-npm-cache npm pack --dry-run --workspace packages/openclaw-plugin
clawhub package inspect openclawbrain --version 0.2.5 --json
openclaw plugins install clawhub:openclawbrain@0.2.5   # temp HOME first
openclaw plugins inspect openclawbrain --json
openclaw plugins doctor
curl /plugins/openclawbrain/doctor   # with Gateway auth; verifies native SQLite + FTS5
```

Scan clean, temp-HOME install works, live install works, routes respond, docs match.

---

### P1 — Fix live dependency robustness

**Problem:** live install can load a stale/native-mismatched dependency state. The installed extension’s `better-sqlite3` binding can break when Node changes.

**Required work:**

1. Decide the packaging path:
   - ship with install-time dependency rebuild support, or
   - remove native dependency risk by moving to a compatible SQLite layer, or
   - make OpenClaw plugin installer explicitly run/rebuild package dependencies for native modules.
2. Add a test that simulates a clean plugin install outside the repo workspace.
3. Make `openclaw plugins doctor` catch missing native bindings before runtime memory search fails.

**Code touchpoints:**

- [`packages/openclaw-plugin/package.json`](../packages/openclaw-plugin/package.json)
- [`src/memory-store.ts`](../packages/openclaw-plugin/src/memory-store.ts)
- [`test/memory-store.test.mjs`](../packages/openclaw-plugin/test/memory-store.test.mjs)

**Done means:** a clean temporary HOME install on the same Node version works without relying on repo-local `node_modules`.

---

### P2 — Make capture promotion more trustworthy

**Problem:** The runtime can distill feedback, but final-state OpenClawBrain needs a more careful candidate-to-memory lifecycle.

**Required work:**

1. Add explicit capture candidate state, or make current distillation audits serve that role more clearly.
2. Promote immediately only for high-confidence explicit corrections, explicit preferences, explicit standing instructions, or high-confidence workflow outcomes.
3. Treat assistant statements as weak evidence, never authoritative user preference.
4. Track evidence kind, promotion reason, review requirement, source evidence hash, superseded memory ID, and contradicted memory ID.
5. Add contradiction/supersession explanations that show exactly why an older memory lost authority.

**Code touchpoints:**

- [`src/feedback-distiller.ts`](../packages/openclaw-plugin/src/feedback-distiller.ts)
- [`src/memory-operations.ts`](../packages/openclaw-plugin/src/memory-operations.ts)
- [`src/memory-store.ts`](../packages/openclaw-plugin/src/memory-store.ts)
- [`src/search.ts`](../packages/openclaw-plugin/src/search.ts)

**Done means:** proof can answer: “what was observed, what was proposed, why did this become durable, what did it supersede?”

---

### P3 — Strengthen learned route function / QTsim path

**Problem:** `route_fn` exists, but the ultimate state requires it to become the center of product quality: when to retrieve, what to retrieve, and when to stay silent.

**Required work:**

1. Add richer route examples from real dogfood turns.
2. Store route decisions with outcome labels that distinguish:
   - helpful injection
   - harmless tie
   - noisy injection
   - missed recall
   - correct silence / `STOP_LOCAL`
   - wrong route
   - correct capture-only
   - harmful memory
3. Learn policy snapshots from actual outcomes, not just heuristics.
4. Keep a clear zero-extra-LLM default for routine turns.

**Code touchpoints:**

- [`src/route-fn.ts`](../packages/openclaw-plugin/src/route-fn.ts)
- [`src/latency-controller.ts`](../packages/openclaw-plugin/src/latency-controller.ts)
- [`src/route-learning.ts`](../packages/openclaw-plugin/src/route-learning.ts)
- [`src/learning.ts`](../packages/openclaw-plugin/src/learning.ts)
- [`src/memory-planner.ts`](../packages/openclaw-plugin/src/memory-planner.ts)

**Done means:** dashboard/proof can show that most turns use Tier 0/Tier 1, high-signal turns get better memory decisions, and correct silence is rewarded rather than treated as absence of behavior.

---

### P4 — Make workflow learning real

**Problem:** Corrections/preferences are the obvious memory wins. The ultimate product also needs to learn successful tool workflows.

**Required work:**

1. Detect successful tool chains from `after_tool_call` + `agent_end`.
2. Learn workflows only from completed, successful, repeated, or explicitly approved behavior.
3. Store repo scope, tools used, success signal, failure signal, last verified time, `applies_when`, and `do_not_apply_when`.
4. Summarize workflows as durable procedural memory without raw tool output.
5. Connect workflows to projects/repos/tools through graph edges.
6. Inject workflows only when task context matches strongly.

**Code touchpoints:**

- [`src/capture.ts`](../packages/openclaw-plugin/src/capture.ts)
- [`src/learning.ts`](../packages/openclaw-plugin/src/learning.ts)
- [`src/memory-operations.ts`](../packages/openclaw-plugin/src/memory-operations.ts)
- [`src/context-selector.ts`](../packages/openclaw-plugin/src/context-selector.ts)

**Done means:** OpenClawBrain can remember “for this repo, release means test → pack → temp-HOME install → ClawHub inspect → live install” and use it only when relevant.

---

### P5 — Upgrade proof surfaces from data dumps to operator explanations

**Problem:** The surfaces exist. The ultimate state needs them to be something Jon can actually trust quickly.

**Required work:**

1. Add story-shaped explanation fields:
   - what did you learn?
   - why did you learn it?
   - where did the evidence come from?
   - why did you inject or not inject it here?
   - what happened after injection?
2. Add route-tier counters and latency distribution.
3. Add “last 20 important decisions” and “recent misses” views.
4. Make `/search` show source, confidence, and supersession state clearly.

**Code touchpoints:**

- [`src/search.ts`](../packages/openclaw-plugin/src/search.ts)
- [`src/status.ts`](../packages/openclaw-plugin/src/status.ts)
- [`src/proof-store.ts`](../packages/openclaw-plugin/src/proof-store.ts)
- [`src/index.ts`](../packages/openclaw-plugin/src/index.ts)

**Done means:** Jon can inspect a bad answer and immediately see whether OpenClawBrain stayed silent, retrieved wrong memory, over-injected, or missed a memory.

---

### P6 — Build a fresh v0.2 dogfood eval

**Problem:** Unit tests prove mechanics. The ultimate vision needs product evidence.

**Required work:**

1. Capture a fresh privacy-scrubbed dogfood set from real v0.2 usage.
2. Score outcomes by product categories:
   - correction sticks
   - missed correction
   - workflow reuse
   - helpful project context
   - correct silence
   - harmful/noisy injection
3. Count unique wins separately from ties.
4. Publish an evidence doc with bounded claims.

**Code/doc touchpoints:**

- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
- [`docs/IMPLEMENTATION_FEEDBACK.md`](IMPLEMENTATION_FEEDBACK.md)
- new `docs/EVIDENCE_V02.md`

**Done means:** OpenClawBrain can honestly say what it improves, where it stays neutral, and where it still fails.

---

### P7 — Make the public story match the runtime

**Problem:** Public copy has been updated several times, but the final state needs stable, friend-proof instructions.

**Required work:**

1. Keep `README.md`, `docs/GETTING_STARTED.md`, `openclawbrain.ai`, and ClawHub metadata synchronized.
2. Add a “verify it works” path that does not require internal knowledge.
3. Make local Ollama the friendly default path, while remote LLM remains explicit/optional/disabled by default.
4. Keep safety language precise: local memory graph, redacted persistence, no raw transcript upload.

**Done means:** a friend can install, enable, verify routes, understand what it does, and uninstall/disable it without help.

---

## 5. Final-state milestone plan

### Milestone A — Restore live memory, clean release, and dogfood latest

- Repair live installed `better-sqlite3` and verify SQLite + FTS5 under Gateway Node.
- Ship `0.2.5` scanner-clean.
- Verify temp-HOME install.
- Install latest live on Mac mini.
- Fix native dependency robustness.
- Confirm status/proof/graph/search routes.

**Exit:** latest public release is clean and the live agent is actually running it.

### Milestone B — Trustworthy capture and proof

- Candidate/promotion lifecycle is explicit.
- Supersession/contradiction explanations are inspectable.
- Proof surfaces answer “why captured” and “why injected.”
- Assistant output is weak evidence only.

**Exit:** bad memories can be audited and corrected without mystery.

### Milestone C — Learned route function becomes product center

- Route examples accumulate from real usage.
- Policy snapshots update from outcomes.
- Correct silence is measured and rewarded.
- Sync LLM planner remains rare and bounded.

**Exit:** route decisions improve over time without making ordinary turns slower.

### Milestone D — Workflow learning

- Successful tool chains become procedural memories.
- Workflows are graph-linked to projects/repos/tools.
- Workflow injection is rare, concrete, and high precision.

**Exit:** OpenClawBrain remembers how to do recurring work, not just what Jon prefers.

### Milestone E — Product evidence

- Fresh v0.2 dogfood eval exists.
- Claims are bounded and public-safe.
- Unique wins, ties, misses, and noisy injections are separated.

**Exit:** we can say what OpenClawBrain improves with evidence, not vibes.

### Milestone F — Final public polish

- README/site/ClawHub all match.
- Install and verify path is friend-proof.
- Disable/uninstall is clear.
- Security scan stays clean.

**Exit:** OpenClawBrain is something Jon can confidently point people at.

---

## 6. The “ultimate complete” acceptance checklist

OpenClawBrain reaches the final vision when all of this is true:

- [ ] Live installed extension passes native SQLite + FTS5 smoke test under Gateway Node.
- [ ] Latest ClawHub release scans clean.
- [ ] Fresh temp-HOME install works.
- [ ] Live Mac mini runs latest release.
- [ ] Native dependency handling is robust across Node version changes.
- [ ] Corrections are captured automatically and survive sessions.
- [ ] Preferences are captured conservatively.
- [ ] Workflows are captured from successful tool use.
- [ ] Contradictions supersede old memory with clear proof.
- [ ] Route function learns when to retrieve and when to stay silent.
- [ ] Most normal turns use no extra synchronous LLM call.
- [ ] Prompt injection remains bounded and rare.
- [ ] Proof surfaces explain capture, retrieval, injection, outcome, and score changes.
- [ ] Search/graph surfaces are useful to an operator.
- [ ] No raw transcript/user text is stored or uploaded.
- [ ] Local Ollama path is standard and works.
- [ ] Remote LLM path is explicit, optional, redacted, and disabled by default.
- [ ] README, docs, ClawHub metadata, and website agree.
- [ ] v0.2 dogfood eval shows real unique wins, measured ties, and low/no harm.

---

## 7. North star

The final product should feel like this:

> Jon corrects OpenClaw once. OpenClawBrain quietly learns the correction, proves what it learned, remembers only the safe distilled form, retrieves it only when relevant, and then gets out of the way. Over weeks, the agent becomes measurably better at Jon’s actual work without becoming slower, noisier, creepier, or harder to debug.

That is the ultimate state.
