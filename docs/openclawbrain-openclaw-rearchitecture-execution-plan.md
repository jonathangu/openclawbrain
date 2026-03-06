# OpenClaw + OpenClawBrain Rearchitecture Execution Plan

Status: Canonical execution plan

This document operationalizes [docs/openclawbrain-openclaw-rearchitecture-plan.md](/Users/cormorantai/openclawbrain/docs/openclawbrain-openclaw-rearchitecture-plan.md) into the implementation program we will execute. Where this document conflicts with legacy docs, examples, or runtime assumptions, this document wins.

This is not architecture prose. It is the migration and delivery plan.

## Mission

Build the production OpenClawBrain architecture as a separate, TypeScript-first repo that:

- serves brain context through a narrow `runtime_compile.v1` contract
- treats learned runtime `route_fn` as the top serving invariant
- treats structural graph operations, Hebbian co-firing, and decay as first-class learner/pack concerns
- serves only from immutable promoted packs
- removes legacy runtime-overlap code aggressively instead of preserving dual architectures

## Fixed Assumptions

These assumptions are locked unless explicitly changed by a new canonical plan:

- OpenClaw owns runtime orchestration, sessions, diagnostics, prompt assembly, fail-open behavior, and active pack pointers.
- OpenClawBrain lives in a separate repo and is not a second runtime owner.
- The production implementation is TypeScript-first for contracts, compiler, pack loader, learner orchestration, evaluation, and promotion tooling.
- Python is allowed only behind offline artifact boundaries when justified by clear ML or research value.
- Learned `route_fn` is not optional when a pack declares learned routing; no silent heuristic fallback is allowed.
- Structural graph operations (`split`, `merge`, `prune`, `connect`), Hebbian co-firing reinforcement, and decay are part of the product, not background nice-to-haves.
- Legacy daemon/socket/hook/session-store parsing paths are migration bridges at most and should be deleted once replacement capability exists.

## Program Success Criteria

The program is complete only when all of the following are true:

1. OpenClaw can compile turn context against a promoted pack using the canonical `runtime_compile.v1` contract.
2. The production serving path uses the learned `route_fn` whenever the pack requires it, and activation rejects packs that cannot satisfy that requirement.
3. OpenClawBrain learner builds reproducible, immutable packs from normalized OpenClaw events plus declared workspace snapshots.
4. Structural graph maintenance, Hebbian reinforcement, and decay are implemented in the TypeScript learner/compiler stack and reflected in pack metadata/evaluation.
5. Promotion and rollback are pointer flips in OpenClaw, not rebuilds or daemon restarts.
6. Legacy runtime-overlap code and docs are removed or archived so operators cannot accidentally deploy the old shape.

## Delivery Principles

- Deliver the new serving contract before widening learner scope.
- Ship the minimum pack format that can enforce learned routing honestly.
- Port serving-critical logic before porting convenience CLIs.
- Delete runtime-overlap code as soon as the equivalent OpenClaw-owned path exists.
- Preserve only offline research value from the Python repo; do not preserve Python because of historical inertia.
- Every phase ends with an operator-visible gate: either the new path is active, or the old path remains the only supported one.

## Workstreams

| Workstream | Scope | Primary Outputs |
| --- | --- | --- |
| WS1. Repo and packaging | Separate TypeScript repo bootstrap, package boundaries, CI, release model | New repo skeleton, package layout, CI gates |
| WS2. Contracts | Canonical schemas, validators, fixtures, compatibility tests | `runtime_compile`, `interaction_events`, `feedback_events`, `artifact_manifest` packages |
| WS3. Pack format | Immutable pack layout, checksums, model/runtime requirements, provenance | `manifest.json`, chunk/graph/vector/router payload spec, validators |
| WS4. Compiler/serve path | TypeScript pack loader, traversal engine, learned `route_fn`, deterministic compile | compile library, worker entrypoint, golden compile tests |
| WS5. Learner pipeline | Event ingestion, workspace snapshots, Hebbian updates, decay, structural ops, pack assembly | learner orchestrator, pack builder, learner metrics |
| WS6. Eval and promotion | Candidate-vs-active comparison, activation validation, rollback rules | eval harness, promotion controller contract, activation checks |
| WS7. OpenClaw integration | OpenClaw-owned compile invocation, prompt injection, pack pointer management, observability | runtime integration PRs in OpenClaw, rollout plan |
| WS8. Legacy deletion | Remove daemon/socket/hook/runtime-overlap code, archive stale docs/examples/tests | deletion PRs, archived docs, simplified operator story |

## Sequencing

Critical path:

1. Lock contracts and pack schema.
2. Bootstrap separate TypeScript repo and move canonical contracts there.
3. Implement TypeScript pack loader and deterministic compile core.
4. Make learned `route_fn` enforcement real on the production serving path.
5. Implement learner build pipeline with first-class Hebbian/decay/structural operations.
6. Land OpenClaw promotion, activation, and fail-open integration.
7. Delete legacy runtime-overlap surfaces.

Parallelizable after Phase 1:

- evaluation harness work can start as soon as contract fixtures exist
- OpenClaw integration scaffolding can start as soon as compile request/response fixtures are stable
- learner event normalization can start before full pack builder completion

Non-parallelizable gates:

- no OpenClaw runtime cutover before learned-routing activation validation exists
- no deletion of legacy runtime paths before OpenClaw can compile against promoted packs
- no pack promotion before manifest validation and rollback pointer switching are complete

## Phase Plan

## Phase 0: Program Lock and Repo Split

Objective: establish the new execution boundary and prevent more work from landing into the wrong architecture.

### Scope

- Declare the separate TypeScript repo as the target production repo.
- Freeze new feature work in legacy Python runtime-overlap surfaces.
- Define ownership split between OpenClaw runtime work and OpenClawBrain compiler/learner work.
- Mark this document and the rearchitecture plan as canonical.

### Deliverables

- New TypeScript repo created with workspace/package layout:
  - `packages/contracts`
  - `packages/pack-format`
  - `packages/compiler`
  - `packages/learner`
  - `packages/eval`
  - `packages/cli`
- ADR or README in the new repo documenting:
  - repo purpose
  - production language stance
  - Python boundary policy
  - rollout rules
- Current repo banner docs updated to state Python runtime paths are transitional and not the long-term production architecture.

### Acceptance Criteria

- Separate repo exists and builds in CI.
- New work on runtime serving is directed only to the TypeScript repo except for short-term bridge fixes.
- Team agreement recorded on top invariants:
  - learned `route_fn`
  - immutable packs
  - OpenClaw-owned runtime
  - aggressive legacy deletion

### Deletion/Archive Targets

Archive or explicitly de-canonicalize as legacy guidance:

- [docs/openclaw-integration.md](/Users/cormorantai/openclawbrain/docs/openclaw-integration.md)
- [docs/openclawbrain-openclaw-hooks.md](/Users/cormorantai/openclawbrain/docs/openclawbrain-openclaw-hooks.md)
- [docs/new-agent-sop.md](/Users/cormorantai/openclawbrain/docs/new-agent-sop.md)
- [docs/architecture.md](/Users/cormorantai/openclawbrain/docs/architecture.md)
- [docs/FULL_REBUILD.md](/Users/cormorantai/openclawbrain/docs/FULL_REBUILD.md)

## Phase 1: Canonical Contracts and Pack Schema

Objective: make the cross-repo boundary executable and testable before compiler or learner implementation expands.

### Scope

- Finalize TypeScript-first schemas and validators for:
  - `runtime_compile.v1`
  - `interaction_events.v1`
  - `feedback_events.v1`
  - `artifact_manifest.v1`
- Define the first production pack layout.
- Write golden fixtures and compatibility tests shared between repos.

### Deliverables

- TypeScript schema/validator package as the source of truth.
- Generated JSON Schema artifacts for cross-language consumers.
- Fixture set covering:
  - learned-required compile requests/responses
  - heuristic-allowed compile requests/responses
  - manifest activation rejection cases
  - event stream examples for correction/teaching/approval/operator override
- Pack manifest spec that includes:
  - route policy
  - serve requirements
  - graph payload checksums
  - vector payload checksums
  - model fingerprints
  - workspace snapshot provenance
  - event-range provenance

### Acceptance Criteria

- OpenClaw and OpenClawBrain can both validate the same fixtures with identical outcomes.
- The contract package rejects responses that claim learned routing without explicit `used_learned_route_fn=true` and explicit learned mode.
- The manifest validator rejects packs missing required route model/runtime assets.
- The fixture suite is wired into CI in both repos.

### Deletion/Archive Targets

Delete or deprecate local schema authority outside the new contract package once parity is reached:

- [openclawbrain/contracts/v1.py](/Users/cormorantai/openclawbrain/openclawbrain/contracts/v1.py) as canonical source of truth
- [docs/contracts-v1.md](/Users/cormorantai/openclawbrain/docs/contracts-v1.md) as a landing doc only; replace with pointers to the new repo once moved

## Phase 2: TypeScript Compiler Core and Honest Serving Path

Objective: stand up a deterministic compile engine that serves from packs and enforces learned-routing truthfully.

### Scope

- Implement pack loading from promoted artifacts.
- Port serving-critical graph structures and traversal behavior to TypeScript.
- Implement production `route_fn` execution using pack-provided learned router data.
- Produce compile responses with deterministic diagnostics.

### Detailed Work

- Port or reimplement core data structures:
  - nodes
  - edges
  - vector index access
  - traversal config
  - suppression overlays
  - provenance lookups
- Define runtime compile algorithm:
  - load active pack snapshot
  - seed candidates from user message and runtime hints
  - traverse graph with learned `route_fn` controlling habitual candidate expansion/ranking
  - apply correction/suppression overlays
  - rank and trim context blocks to budget
  - return diagnostics and routing metadata
- Implement deterministic routing behavior:
  - same request + same pack => same selected candidates and context order
  - stable tie-break rules
  - explicit route mode in output

### Deliverables

- `packages/compiler` TypeScript library
- optional worker entrypoint managed by OpenClaw
- compile golden tests using frozen pack fixtures
- benchmark suite for p50/p95 compile latency and pack load latency

### Acceptance Criteria

- Compiler can answer `runtime_compile.v1` requests against a static pack fixture.
- Learned-required packs fail activation or compile initialization if learned router assets are unavailable.
- No compile path mutation occurs: no fired-log writes, no learner writes, no stateful per-turn training side effects.
- Compile outputs are deterministic under repeated runs.
- Diagnostics clearly expose `mode_requested`, `mode_effective`, `used_learned_route_fn`, and router identity.

### Deletion/Archive Targets

Once OpenClaw can call the new compiler:

- [openclawbrain/daemon.py](/Users/cormorantai/openclawbrain/openclawbrain/daemon.py)
- [openclawbrain/socket_server.py](/Users/cormorantai/openclawbrain/openclawbrain/socket_server.py)
- [openclawbrain/socket_client.py](/Users/cormorantai/openclawbrain/openclawbrain/socket_client.py)
- [openclawbrain/protocol.py](/Users/cormorantai/openclawbrain/openclawbrain/protocol.py)
- [tests/test_daemon.py](/Users/cormorantai/openclawbrain/tests/test_daemon.py)
- [tests/test_socket.py](/Users/cormorantai/openclawbrain/tests/test_socket.py)
- [tests/test_socket_client_cli.py](/Users/cormorantai/openclawbrain/tests/test_socket_client_cli.py)
- [tests/test_socket_server_args.py](/Users/cormorantai/openclawbrain/tests/test_socket_server_args.py)

## Phase 3: Learner Pipeline With First-Class Graph Dynamics

Objective: rebuild the learning system around normalized events, workspace snapshots, and pack production instead of runtime mutation and raw session parsing.

### Scope

- Ingest normalized OpenClaw interaction and feedback events.
- Build learning records for corrections, teachings, suppressions, summaries, and route labels.
- Reimplement graph-learning primitives in the TypeScript learner.
- Assemble candidate packs from learner outputs plus workspace snapshots.

### Required First-Class Learner Features

These are mandatory for Phase 3 exit. They are not optional follow-up work:

- Hebbian co-firing reinforcement
  - strengthen edges between co-activated nodes from successful turns
  - support source attribution and learning-rate controls by signal type
- Decay
  - time-aware edge weakening
  - manifest-recorded decay parameters
  - deterministic application during builds
- Structural graph operations
  - split overloaded nodes
  - merge redundant neighborhoods
  - prune weak or invalid edges/nodes
  - connect new learning nodes into workspace neighborhoods
- Correction and suppression overlays
  - explicit negative knowledge and suppression references in the pack
- Route label production
  - training/eval-ready decision points tied to pack/router provenance

### Detailed Work

- Define learner input model:
  - normalized event stream
  - workspace snapshot manifest
  - active pack baseline
  - optional offline-derived artifacts from Python jobs
- Port or reimplement current useful algorithms from Python:
  - decay logic from [openclawbrain/decay.py](/Users/cormorantai/openclawbrain/openclawbrain/decay.py)
  - maintenance orchestration from [openclawbrain/maintain.py](/Users/cormorantai/openclawbrain/openclawbrain/maintain.py)
  - route labeling/training inputs from [openclawbrain/ops/async_route_pg.py](/Users/cormorantai/openclawbrain/openclawbrain/ops/async_route_pg.py)
- Replace raw session-store ingestion as the primary interface with event ingestion.
- Keep raw replay/history import only as one-time or backfill tooling, never as the steady-state learner contract.

### Deliverables

- `packages/learner` orchestrator
- candidate-pack builder
- learner state store for checkpoints and input provenance
- learner metrics/events
- deterministic rebuild test from frozen input fixtures

### Acceptance Criteria

- Given the same event range, workspace snapshot, and offline artifacts, the learner produces the same pack checksum.
- Pack manifests record graph-dynamics provenance:
  - Hebbian parameters
  - decay parameters
  - structural-op counts/results
  - route training provenance
- Raw OpenClaw/Codex session parsing is no longer required for normal operation.
- Learner lag does not affect runtime serving.

### Deletion/Archive Targets

Delete or downgrade to explicit migration-only tooling:

- [openclawbrain/replay.py](/Users/cormorantai/openclawbrain/openclawbrain/replay.py) as steady-state pipeline
- [openclawbrain/full_learning.py](/Users/cormorantai/openclawbrain/openclawbrain/full_learning.py) as steady-state pipeline
- [openclawbrain/session_sources.py](/Users/cormorantai/openclawbrain/openclawbrain/session_sources.py) as primary ingestion
- [docs/FULL_REBUILD.md](/Users/cormorantai/openclawbrain/docs/FULL_REBUILD.md)
- replay/harvest examples in [examples/ops](/Users/cormorantai/openclawbrain/examples/ops) that describe the old operational model

## Phase 4: Learned Router Training, Evaluation, and Activation Gates

Objective: make learned routing an enforced production contract instead of a best-effort mode.

### Scope

- Define router artifact formats that are TypeScript-loadable in production.
- Build router training and evaluation pipeline.
- Add activation gates that reject packs whose learned routing cannot be served honestly.

### Deliverables

- route-model artifact format supported by Node/TypeScript runtime
- training pipeline that emits model fingerprints and evaluation summary
- evaluation suites:
  - candidate expansion/ranking accuracy
  - compile-output regression versus active pack
  - latency budget compliance
  - learned-routing enforcement tests
- activation validator used by OpenClaw before pointer flips

### Acceptance Criteria

- A pack marked `requires_learned_routing=true` cannot activate unless the runtime can load and use the router artifact.
- Evaluation output compares candidate pack vs active pack on held-out traces and policy checks.
- Runtime compile on a learned-required pack shows `mode_effective=learned` and `used_learned_route_fn=true` in production integration tests.
- There is no code path that silently falls back to heuristic routing for learned-required packs.

### Deletion/Archive Targets

Delete the old operator story around “learned if available, heuristic otherwise” from:

- [docs/learned-routing-audit.md](/Users/cormorantai/openclawbrain/docs/learned-routing-audit.md) once superseded by the new OpenClaw/OpenClawBrain operator checks
- legacy route-mode CLI flows in [openclawbrain/daemon.py](/Users/cormorantai/openclawbrain/openclawbrain/daemon.py)
- legacy route-model training CLIs that only feed the Python daemon path

## Phase 5: OpenClaw Runtime Integration and Promotion Control

Objective: move the runtime boundary into OpenClaw completely.

### Scope

- Add OpenClaw-owned pack pointer resolution per agent.
- Add OpenClaw-owned compile invocation, timeout policy, and fail-open behavior.
- Add OpenClaw-owned prompt assembly and memory diagnostics.
- Add OpenClaw-owned promotion and rollback control.

### Detailed Work

- OpenClaw runtime changes:
  - resolve active `pack_id`
  - issue `runtime_compile.v1` request
  - apply compile timeout policy
  - inject selected context blocks into final prompt
  - emit `memory_compiled` interaction events
  - surface compile diagnostics in OpenClaw tracing/metrics
- Promotion controller changes:
  - validate manifest
  - validate checksums
  - validate runtime compatibility and serve requirements
  - atomically flip active pack pointer
  - preserve previous pointer for rollback

### Deliverables

- OpenClaw integration PRs
- active-pack pointer storage design
- rollout config for fail-open and pack pinning
- production dashboards for compile success, fail-open, pack distribution, and learned-route usage

### Acceptance Criteria

- OpenClaw can serve requests end-to-end using promoted packs without any OpenClawBrain daemon/socket/hook dependency.
- OpenClaw can pin a specific `pack_id` for debugging and rollback.
- OpenClaw runtime traces include compile timing, route mode, pack id, and fail-open reason when applicable.
- Rollback to the prior pack is an atomic pointer change and completes without pack rebuild.

### Deletion/Archive Targets

Delete integration surfaces that give OpenClawBrain runtime ownership:

- [integrations/openclaw/hooks/openclawbrain-context-injector/HOOK.md](/Users/cormorantai/openclawbrain/integrations/openclaw/hooks/openclawbrain-context-injector/HOOK.md)
- [integrations/openclaw/hooks/openclawbrain-context-injector/handler.ts](/Users/cormorantai/openclawbrain/integrations/openclaw/hooks/openclawbrain-context-injector/handler.ts)
- [openclawbrain/openclaw_adapter/query_brain.py](/Users/cormorantai/openclawbrain/openclawbrain/openclaw_adapter/query_brain.py)
- any OpenClawBrain CLI path whose job is prompt injection instead of pack build/eval/publish

Archive migration-only feedback helpers only if OpenClaw event export supersedes them:

- [openclawbrain/openclaw_adapter/capture_feedback.py](/Users/cormorantai/openclawbrain/openclawbrain/openclaw_adapter/capture_feedback.py)
- [openclawbrain/openclaw_adapter/learn_by_chat_id.py](/Users/cormorantai/openclawbrain/openclawbrain/openclaw_adapter/learn_by_chat_id.py)
- [openclawbrain/openclaw_adapter/learn_correction.py](/Users/cormorantai/openclawbrain/openclawbrain/openclaw_adapter/learn_correction.py)

## Phase 6: Legacy Deletion and Operator Story Reset

Objective: end the dual-architecture period and make the new path the only supported production story.

### Scope

- Remove dead runtime-overlap code.
- Remove docs that teach the old daemon/hook/harvest workflow as the primary operating model.
- Publish the new operator runbooks around packs, compile, learner, promotion, and rollback.

### Deliverables

- deletion PRs merged
- docs refresh completed
- examples refreshed to show pack-based build/promote/rollback flow
- archived legacy docs moved under `docs/archive/` or removed

### Acceptance Criteria

- The repo no longer advertises `serve`, socket, or hook injection as the recommended production setup.
- Operators have one canonical setup path and one canonical rollback path.
- CI no longer exercises deleted runtime-overlap surfaces.
- Remaining Python code is explicitly categorized as:
  - offline research
  - migration/backfill tooling
  - temporary parity bridge with an owner and removal date

### Deletion/Archive Targets

- [examples/openclaw_quickstart.sh](/Users/cormorantai/openclawbrain/examples/openclaw_quickstart.sh)
- [examples/openclaw_adapter/README.md](/Users/cormorantai/openclawbrain/examples/openclaw_adapter/README.md)
- [examples/openclaw_adapter/query_brain.py](/Users/cormorantai/openclawbrain/examples/openclaw_adapter/query_brain.py)
- launchd/systemd daemon templates that exist only for runtime serving
- obsolete tests tied only to deleted surfaces

## Cross-Phase Deliverable Map

| Deliverable | Phase | Blocking Dependencies |
| --- | --- | --- |
| Separate TypeScript repo scaffold | 0 | none |
| Canonical contracts package | 1 | repo scaffold |
| Pack manifest and layout | 1 | contracts package |
| TypeScript compile core | 2 | contracts, pack layout |
| Learned-route enforcement | 2-4 | compile core, route artifact format |
| Learner orchestrator | 3 | contracts, pack layout |
| Structural-op + Hebbian + decay parity | 3 | learner orchestrator |
| Router training/eval pipeline | 4 | learner outputs, eval harness |
| OpenClaw integration | 5 | compile core, activation validator |
| Promotion/rollback controller | 5 | manifest validator |
| Legacy deletion | 6 | OpenClaw cutover |

## Testing and Verification Gates

Each phase must land with automated coverage. Minimum required gates:

- Contract fixtures and schema validation
- Deterministic compile golden tests
- Pack checksum and manifest validation tests
- Learned-routing activation rejection tests
- Learner reproducibility tests from frozen event/workspace inputs
- Regression tests for suppression/correction overlays
- Structural-op regression tests:
  - split
  - merge
  - prune
  - connect
- Hebbian reinforcement tests
- Decay parameter and output tests
- OpenClaw integration tests for:
  - fail-open on compile timeout
  - pack pinning
  - rollback
  - routing metadata emission

## Migration Rules for Existing Python Code

- Keep only what is needed to derive parity fixtures or offline artifacts.
- Do not deepen the Python daemon/socket/runtime integration.
- Do not add new OpenClaw hooks or adapter CLIs for prompt injection.
- Any Python module retained during migration must be labeled:
  - `bridge`
  - `offline`
  - or `delete_after_phase_X`

Recommended retention during migration:

- keep graph-learning logic as reference material until TypeScript parity exists
- keep route-label/training code only if it produces language-neutral artifacts consumed by the TypeScript pipeline
- keep replay/backfill only for one-time history imports

Recommended early deletions:

- runtime daemon lifecycle management
- socket transport
- hot in-memory serving claims in operator docs
- hook-based prompt injection as the recommended architecture

## Immediate Next Actions

These are the next concrete actions, in order:

1. Create the separate TypeScript repo and commit the package skeleton plus CI.
2. Move canonical contract source-of-truth into the new repo and wire shared golden fixtures.
3. Define `artifact_manifest.v1` and the first pack directory layout, including graph/router/vector payload boundaries and checksums.
4. Implement a minimal TypeScript compile core that can load a frozen pack fixture and answer `runtime_compile.v1`.
5. Add learned-routing activation validation before any runtime integration work proceeds.
6. Draft the OpenClaw runtime integration PR that resolves `pack_id`, invokes compile, records `memory_compiled`, and fails open on timeout.
7. Mark `daemon.py`, `socket_server.py`, hook injection docs, and adapter query paths as deprecated in this repo immediately.
8. Start a parity matrix mapping Python graph-learning behavior to TypeScript implementations:
   - Hebbian co-firing
   - decay
   - split
   - merge
   - prune
   - connect
   - suppression overlays
9. Define the event export path from OpenClaw so learner work can proceed without raw session parsing.
10. Open deletion trackers for every legacy runtime-overlap surface listed in this plan, each with an owner and removal phase.

## Explicit Non-Deliverables

The following are not acceptable substitutes for this plan:

- keeping the Python daemon as the production compile service indefinitely
- treating hook injection as the long-term OpenClaw integration
- logging learned-routing diagnostics without actually using the learned `route_fn`
- preserving raw session-store parsing as the steady-state learner interface
- postponing deletion until “after everything else is stable”

## Canonical Outcome

The end state is simple:

- OpenClaw owns runtime.
- OpenClawBrain ships packs, compiler logic, learner outputs, and evaluations from a separate TypeScript-first repo.
- Learned `route_fn` is a hard serving invariant.
- Structural graph operations, Hebbian co-firing, and decay are first-class pack-building behavior.
- Legacy runtime-overlap code is gone.
