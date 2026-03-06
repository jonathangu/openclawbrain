# OpenClaw + OpenClawBrain Rearchitecture Plan

Status: Proposed canonical plan

This is the canonical north-star architecture for integrating OpenClawBrain with regular OpenClaw. It supersedes the target-shape assumptions spread across [docs/architecture.md](/Users/cormorantai/openclawbrain/docs/architecture.md), [docs/openclaw-integration.md](/Users/cormorantai/openclawbrain/docs/openclaw-integration.md), [docs/shadow-routing-upg-architecture.md](/Users/cormorantai/openclawbrain/docs/shadow-routing-upg-architecture.md), and [docs/FULL_REBUILD.md](/Users/cormorantai/openclawbrain/docs/FULL_REBUILD.md) where they conflict with this plan.

This plan does not optimize for backward compatibility. If a current path blurs runtime ownership, duplicates responsibilities already held by OpenClaw, or depends on legacy session-store parsing as a steady-state interface, it should be deleted or rebuilt.

## Why This Rearchitecture

Today the repo still describes and ships multiple overlapping shapes:

- OpenClawBrain as a runtime daemon/socket service.
- OpenClawBrain as a hook-driven prompt injector.
- OpenClawBrain as a replay/harvest/teacher loop system.
- OpenClawBrain as a direct parser of raw OpenClaw and Codex session stores.

That creates the wrong long-term boundary. OpenClaw should own runtime orchestration. OpenClawBrain should own memory compilation and learning. The runtime must not depend on OpenClawBrain acting like a second agent platform.

Recent repo findings already show the cost of weak boundaries: placeholder training exports, brittle raw-session ingestion paths, and multiple partially overlapping learning loops. The fix is not more glue. The fix is a cleaner architecture.

## North Star

OpenClaw remains the sole runtime owner for:

- channels
- sessions and turn state
- diagnostics, tracing, and runtime budgets
- tool execution and completion calls
- fail-open/fail-closed runtime policy
- promotion and rollback of active memory artifacts

OpenClawBrain remains a separate repo and becomes:

- the memory/context compiler
- the learner system
- the offline and nearline artifact builder
- the scorer/router pack producer
- the evaluator of whether a new memory artifact is better than the current one

The runtime relationship is simple:

1. OpenClaw produces normalized runtime inputs.
2. OpenClaw invokes OpenClawBrain through a narrow compile contract.
3. OpenClawBrain returns context blocks plus structured metadata.
4. OpenClaw decides what to inject, how much to inject, how to log it, and what to do on failure.

The learner relationship is also simple:

1. OpenClaw emits normalized interaction and feedback events.
2. OpenClawBrain consumes those events and workspace snapshots.
3. OpenClawBrain builds versioned memory artifacts.
4. OpenClaw validates and promotes an artifact atomically.

## Anti-Goals

- Do not keep OpenClawBrain as an independent runtime owner with its own daemon, socket lifecycle, hook lifecycle, or session truth.
- Do not keep raw OpenClaw/Codex session parsing as the primary steady-state ingestion interface.
- Do not preserve CLI compatibility if the command exists only to maintain the old runtime split.
- Do not put teacher-model calls, feedback scanning, or learning extraction in the user-request critical path.
- Do not let OpenClawBrain own user-visible diagnostics, request IDs, or runtime latency policy.
- Do not couple runtime correctness to a separate learner plane being healthy.

## Ownership Boundaries

| Domain | Owner | Notes |
| --- | --- | --- |
| Channels, agents, sessions, turn state | OpenClaw | Source of truth. |
| Prompt assembly and final injected context | OpenClaw | OpenClaw decides budget and ordering. |
| Runtime request IDs, tracing, diagnostics | OpenClaw | OpenClawBrain can return metadata, not own the trace. |
| Workspace snapshot discovery | OpenClaw | OpenClaw tells OpenClawBrain what to compile. |
| Memory compilation from source inputs | OpenClawBrain | Produces versioned artifacts. |
| Learning from feedback and history | OpenClawBrain | Uses normalized exports from OpenClaw. |
| Artifact validation and promotion gates | Shared | OpenClawBrain evaluates; OpenClaw promotes. |
| Artifact activation and rollback | OpenClaw | Runtime owner must own rollback. |

Rules:

- OpenClawBrain may run a learner plane, but it is never a competing runtime owner.
- Any OpenClawBrain helper process on the runtime path is OpenClaw-started, OpenClaw-supervised, OpenClaw-timed out, and OpenClaw-instrumented.
- OpenClawBrain state is derived state. OpenClaw sessions and events remain canonical.

## Language and Implementation Stance

The default production target is TypeScript-native architecture everywhere it is practical:

- TypeScript is the canonical implementation language for runtime contracts, pack loading, compile serving, learner orchestration, promotion control, and operational tooling.
- OpenClawBrain remains a separate repo, but its production-facing surfaces should be published as TypeScript packages, worker entrypoints, and language-neutral pack artifacts that OpenClaw can consume directly.
- Python is optional and justified only for offline research, experimentation, or ML/data-processing steps that materially benefit from the Python ecosystem.
- Python is not the default compiler substrate, not the default learner orchestrator, and not part of the default production request path.

Implications:

- Separate repo does not imply separate runtime ownership or a Python daemon boundary.
- The canonical artifacts and contracts must be loadable, validated, and served from TypeScript without requiring Python in production.
- If a Python component is introduced, it must terminate at a versioned artifact boundary such as model weights, labels, summaries, or candidate-pack inputs. It must not become the source of truth for runtime behavior.

## Target Components and Planes

### 1. OpenClaw Runtime Plane

Responsibilities:

- receive user and channel events
- maintain session state and short-term transcript state
- call tools and models
- request memory compilation for a turn
- inject selected memory blocks into the prompt
- emit runtime metrics and traces
- emit normalized interaction events for learning

Artifacts owned by OpenClaw:

- session store
- runtime traces
- prompt budgets and policy
- active artifact pointer per agent

### 2. OpenClawBrain Compiler Plane

Responsibilities:

- compile workspace and learned memory into a versioned artifact pack
- answer turn-scoped compile requests against an active pack
- expose deterministic pack validation
- expose deterministic scoring metadata to OpenClaw

Default implementation target:

- a TypeScript compiler core library shared by tests, pack tooling, and the OpenClaw runtime integration
- an optional TypeScript worker entrypoint for isolation or preload benefits
- no Python requirement on the production compile path unless there is a narrow, explicit exception with the same contract and observability guarantees

This is not a second runtime. It is a compiler engine. In local development OpenClaw should be able to call it in-process. In production OpenClaw may call a local worker or bounded sidecar, but OpenClaw still owns lifecycle and observability.

### 3. OpenClawBrain Learner Plane

Responsibilities:

- ingest normalized OpenClaw event streams
- derive corrections, teachings, suppressions, summaries, and route updates
- build candidate artifact packs
- run offline evaluation and regression checks
- publish artifact metadata for OpenClaw promotion

Default implementation target:

- TypeScript for event ingestion, orchestration, evaluation harnesses, pack assembly, and promotion handoff
- Python only for clearly justified offline ML or research jobs that emit versioned outputs consumed by the TypeScript learner pipeline

The learner plane may be scheduled, queue-backed, or batch-oriented. It must never be required for an interactive request to succeed.

### 4. Artifact Plane

The integration pivots around immutable, versioned artifact packs.

Each pack should contain:

- `manifest.json`
- compiled content chunks
- embedding metadata and vector data
- route/scoring model data
- route policy metadata and serving requirements
- suppression and correction overlays
- provenance map back to workspace and event sources
- build metrics and evaluation summary
- checksums for every payload

Pack format rules:

- Pack contents must be language-neutral and TypeScript-loadable in production.
- No Python-specific runtime serialization formats such as pickle may appear in promoted serving packs.
- Any learned model artifact included in a pack must be usable from the TypeScript production path, whether through native JS, ONNX, WASM, or another explicitly supported runtime.

OpenClaw only serves from promoted packs. OpenClawBrain only writes new candidate packs.

## Non-Negotiable Serving Invariants

- Runtime compile is read-only and side-effect free. No fired-log writes, no per-turn state mutation, and no hidden learner writes on the request path.
- A promoted pack is self-contained for serving. Runtime compile may read only the request payload, the promoted pack, and OpenClaw-owned runtime/session state.
- One request uses one resolved promoted pack snapshot end-to-end. No mid-turn pack switching, partial pack reads, or cross-pack mixing.
- If a promoted pack declares learned routing as required, the production compile path must actually use the learned `route_fn` to influence candidate expansion or ranking. Learned routing cannot be diagnostic-only.
- Silent fallback from a learned-required pack to heuristic routing is not allowed. The only acceptable degradations are: reject activation before serving, or OpenClaw-owned fail-open that skips brain context for that turn.

## Canonical Contracts and APIs

All cross-repo boundaries should be versioned and live under a small contracts surface. Suggested location:

- `contracts/runtime_compile/v1`
- `contracts/interaction_events/v1`
- `contracts/feedback_events/v1`
- `contracts/artifact_manifest/v1`

Contract rules:

- The source of truth should be TypeScript-first schemas and validators, with generated JSON Schema or equivalent artifacts for cross-language consumers.
- Example payloads, golden fixtures, and compatibility tests should be authored against the TypeScript contract package first.
- Python consumers may bind to these contracts, but they should not define the canonical production schema behavior.

### Contract 1: Runtime Compile Request

Owned by OpenClaw. Sent per turn.

```json
{
  "contract_version": "runtime_compile.v1",
  "request_id": "req_123",
  "agent_id": "main",
  "session_id": "sess_456",
  "turn_id": "turn_789",
  "pack_id": "brainpack_2026_03_06_001",
  "user_message": "what failed last deploy?",
  "recent_turns": [],
  "tool_context": [],
  "budget": {
    "max_chars": 16000,
    "max_blocks": 8
  },
  "hints": {
    "recall": true,
    "correction_sensitive": true
  }
}
```

### Contract 2: Runtime Compile Response

Produced by OpenClawBrain. Interpreted and enforced by OpenClaw.

```json
{
  "contract_version": "runtime_compile.v1",
  "request_id": "req_123",
  "pack_id": "brainpack_2026_03_06_001",
  "context_blocks": [
    {
      "id": "ctx_1",
      "kind": "memory",
      "title": "Deploy rollback rule",
      "content": "If deploy verification fails, rollback before restarting workers.",
      "score": 0.93,
      "sources": [
        {
          "kind": "workspace",
          "ref": "docs/deploy.md#rollback"
        }
      ]
    }
  ],
  "suppressed_sources": [],
  "routing": {
    "mode_requested": "learned",
    "mode_effective": "learned",
    "used_learned_route_fn": true,
    "route_model_id": "local-route-model-v3",
    "decision_trace_ref": "route_decision_req_123"
  },
  "diagnostics": {
    "candidate_count": 42,
    "selected_count": 4,
    "compile_ms": 18,
    "router": "local-route-model-v3"
  }
}
```

Rules:

- OpenClawBrain returns candidates and metadata, not final prompt ownership.
- OpenClaw can truncate, reorder, discard, or augment returned blocks.
- OpenClaw logs runtime diagnostics under its own trace/span.
- `routing.mode_effective` must be explicit, never implied.
- `used_learned_route_fn=true` means the learned router changed traversal or ranking on the production serving path, not just that a learned model was loaded.
- If a pack requires learned routing, activation validation should prevent heuristic fallback long before runtime. A runtime response must not silently degrade to heuristic mode.

### Contract 3: Interaction Event Stream

Owned by OpenClaw. Exported continuously or in batches to the learner plane.

Required event kinds:

- `turn_started`
- `memory_compiled`
- `assistant_completed`
- `tool_called`
- `tool_result`
- `feedback_recorded`
- `session_closed`

Required fields across all events:

- `event_id`
- `event_ts`
- `agent_id`
- `session_id`
- `turn_id`
- `request_id`
- `pack_id`
- `schema_version`

This replaces raw parsing of OpenClaw internal stores as the primary contract.

Required fields for `memory_compiled` events:

- `selected_context_ids`
- `suppressed_source_refs`
- `routing.mode_requested`
- `routing.mode_effective`
- `routing.used_learned_route_fn`
- `routing.decision_trace_ref`
- `compile_latency_ms`

### Contract 4: Feedback Event

OpenClaw records feedback directly when a user correction, teaching, approval, or operator action occurs. OpenClawBrain learns from that exported record.

Required fields:

- `feedback_kind`: `correction`, `teaching`, `preference`, `approval`, `operator_override`
- `source_kind`: `human`, `system`, `evaluation`
- `message_ref`
- `affected_turn_id`
- `affected_context_ids`
- `dedup_key`

### Contract 5: Artifact Manifest

Produced by OpenClawBrain. Promoted by OpenClaw.

Required fields:

- `pack_id`
- `agent_id`
- `build_id`
- `parent_pack_id`
- `contract_versions`
- `pack_checksum`
- `compiler_fingerprint`
- `model_fingerprint`
- `runtime_compat`
- `route_policy`
- `serve_requirements`
- `workspace_snapshot_id`
- `event_range`
- `checksums`
- `eval_summary`
- `created_at`

Where:

- `runtime_compat` declares the minimum OpenClaw/OpenClawBrain compiler versions that may activate the pack.
- `route_policy` declares the intended runtime route mode and the artifacts required to serve it.
- `serve_requirements` declares non-negotiable activation checks such as `requires_learned_routing`, embedding dimension/model compatibility, and required local model assets.

OpenClaw must reject manifests that fail checksum, schema, model-compatibility, or serve-requirement validation.

## Runtime Path

Target runtime path:

1. OpenClaw receives a channel event and loads session state.
2. OpenClaw resolves the active `pack_id` for the agent.
3. OpenClaw issues a `runtime_compile.v1` request to the OpenClawBrain compiler engine.
4. OpenClawBrain returns ranked context blocks and diagnostics.
5. OpenClaw decides final prompt assembly and sends the model request.
6. OpenClaw records memory-use diagnostics in its own trace.
7. OpenClaw emits normalized interaction events for later learning.

Runtime invariants:

- no teacher LLM calls
- no replay jobs
- no direct parsing of on-disk session logs
- deterministic response given the same request and pack
- same resolved pack snapshot for the whole request
- read-only compile path with no runtime-owned brain writes
- strict timeout with fail-open behavior owned by OpenClaw
- learned-required packs must compile with the learned `route_fn` on the production path
- no silent heuristic fallback for learned-required packs
- ability to pin a specific `pack_id` during investigation or rollback

## Learner Path

Target learner path:

1. OpenClaw exports normalized interaction and feedback events.
2. OpenClawBrain ingests events plus a declared workspace snapshot.
3. OpenClawBrain derives learning records:
   - corrections
   - teachings
   - suppressions
   - route labels
   - summaries and compactions
4. OpenClawBrain compiles a candidate pack.
5. OpenClawBrain evaluates the candidate against held-out traces and policy checks.
6. OpenClawBrain publishes the candidate pack and manifest.
7. OpenClaw validates the manifest and atomically promotes or rejects the pack.

Learner invariants:

- learner lag never blocks runtime
- pack builds are immutable and reproducible from the same input set
- pack contents are content-addressed or checksum-validated before promotion
- promotion is atomic
- rollback is a pointer change, not a rebuild
- candidate promotion requires route-specific regression checks when the pack declares learned routing

## Deployment Topology

### Local Development

- OpenClaw runtime on the developer machine
- OpenClaw-owned active pack pointer on local disk
- OpenClawBrain TypeScript compiler called in-process or via a short-lived local worker
- OpenClawBrain learner orchestration runs as a TypeScript CLI or scheduled local task
- optional Python research/ML jobs run offline and publish artifacts back into the pack pipeline

### Single-Host Production

- OpenClaw runtime service
- local read-only promoted pack directory
- OpenClaw-owned TypeScript runtime worker for compile requests
- separate OpenClawBrain TypeScript learner service or cron job writing candidate packs to staging
- optional Python offline jobs triggered by the learner plane only when justified by model-training needs

### Multi-Host Production

- OpenClaw runtime fleet serves from replicated promoted packs
- pack artifacts stored in object storage or artifact registry
- OpenClawBrain TypeScript learner workers consume normalized event streams
- OpenClaw promotion controller flips agent pack pointers after validation
- optional Python batch workers publish derived artifacts back to the learner pipeline, never directly to runtime

Topology rule:

- OpenClawBrain learner infrastructure may scale independently.
- OpenClawBrain compile helpers may exist.
- Neither becomes the owner of runtime sessions, channels, or diagnostics.

## Failure Modes and Expected Behavior

| Failure | Expected Behavior |
| --- | --- |
| No active pack for agent | OpenClaw serves without brain context and logs `pack_missing`. |
| Compile timeout | OpenClaw fail-open, records timeout metric, keeps serving. |
| Pack checksum or schema mismatch | OpenClaw rejects candidate pack and keeps current pack. |
| Compiler crash | OpenClaw retries within bounded policy or fail-opens. |
| Learner backlog | Runtime unaffected; freshness degrades only. |
| Bad promoted pack | OpenClaw rolls back to previous pack pointer immediately. |
| Model mismatch between pack and runtime | OpenClaw rejects pack activation before serving it. |
| Learned-required pack missing usable route model/assets | OpenClaw rejects activation before serving it; no heuristic fallback. |
| Event export outage | Runtime unaffected; learner catches up later from durable export. |
| Workspace snapshot drift during build | Build marked invalid; no promotion. |

## Observability

OpenClaw runtime metrics:

- compile request count
- compile success/failure/timeout rate
- compile latency p50/p95/p99
- injected chars and blocks per turn
- suppressions applied per turn
- fail-open rate
- requested route mode versus effective route mode
- learned-route usage rate
- learned-route enforcement failures
- active pack distribution by agent

OpenClawBrain learner metrics:

- event ingestion lag
- pack build duration
- pack build failure rate
- candidate vs promoted pack count
- evaluation delta versus active pack
- correction and teaching extraction yield
- artifact size growth

Shared correlation fields:

- `request_id`
- `session_id`
- `turn_id`
- `pack_id`
- `build_id`

Observability rule:

- OpenClaw owns the runtime trace.
- OpenClawBrain returns structured diagnostics only.
- OpenClawBrain does not become a second source of truth for runtime health.

## Language, Model, and Runtime Guidance

Default production stack:

- TypeScript/Node for contracts, validators, compiler core, pack loader, learner orchestration, eval harnesses, and promotion tooling
- language-neutral artifact packs that OpenClaw can validate and serve without Python
- optional Python only behind offline artifact-producing boundaries when TypeScript-native tooling is not a reasonable substitute

Runtime guidance:

- Prefer local embeddings and a small deterministic scorer/router.
- Runtime compile must require zero remote completion-model calls.
- Preload any runtime model in an OpenClaw-owned TypeScript worker, not an independently managed OpenClawBrain daemon.
- CPU-first is preferred unless a GPU meaningfully reduces p95 latency without adding operational fragility.
- Prefer model runtimes that integrate cleanly with TypeScript production services.

Learner guidance:

- Local-first defaults are preferred for embeddings, extraction, summarization, and route-model training.
- TypeScript should own learner orchestration, data plumbing, and pack assembly even when a Python training job is used offline.
- Remote teacher models are acceptable only in the learner plane.
- Every pack manifest must record exact model fingerprints and dimensions.
- Pack promotion must validate model compatibility before activation.

Practical implication:

- Node-native, WASM, ONNX, llama.cpp bindings, fastembed bindings, or similar tooling are good fits for OpenClawBrain production compiler and learner internals.
- Python libraries remain valid for offline experimentation or training when they produce stable artifacts consumed by the TypeScript pipeline.
- A runtime design that requires an always-on separate OCB daemon or Python service to hold model state is not the target architecture.

## Migration Plan

### Phase 0: Freeze the Target Boundary

- Approve this document as the canonical architecture target.
- Declare OpenClaw the only runtime owner.
- Declare TypeScript-first production architecture as the default implementation stance.
- Stop adding new features to the OCB daemon/socket/hook runtime path except for migration support.

Exit criteria:

- architecture approved
- TypeScript-first production direction approved
- no new runtime-owned features added to legacy OCB paths

### Phase 1: Define and Land Contracts

- Add versioned contracts for runtime compile, interaction events, feedback events, and artifact manifests as TypeScript-first schemas plus generated portable schemas.
- Add OpenClaw-side normalized export of interaction and feedback events.
- Add contract test fixtures in both repos.
- Add golden fixtures that prove learned-routing fields are present and interpreted identically in both repos.
- Make the TypeScript contract package the canonical source for validators, examples, and compatibility tests.

Exit criteria:

- both repos validate the same schemas
- OpenClaw can emit normalized events without OCB parsing raw session stores
- both repos agree on how `mode_effective` and `used_learned_route_fn` are encoded
- TypeScript validators are used on the production path for request, response, and manifest validation

### Phase 2: Extract a Real Compiler Core

- Refactor OpenClawBrain runtime logic into a pure TypeScript compiler interface that accepts `runtime_compile.v1`.
- Make the compiler deterministic and pack-driven.
- Remove assumptions that the compiler owns session history, sockets, or chat-level fired logs.
- Move production compile logic behind a TypeScript library boundary that OpenClaw can call in-process first and via a worker second.
- Emit route decision metadata sufficient to prove whether learned routing actually drove the result.

Exit criteria:

- same pack + same request => same response
- compiler can run in-process under OpenClaw ownership with no Python dependency
- compiler tests fail if a learned-required pack can compile without `used_learned_route_fn=true`
- pack loading, scoring, and response assembly are exercised by TypeScript integration tests

### Phase 3: Build the Pack Pipeline

- Replace mutable runtime `state.json` assumptions with versioned pack builds.
- Add a TypeScript-first pack loader and build pipeline for manifest parsing, checksum verification, provenance lookup, and atomic staging layout.
- Keep a translation layer only as long as needed to build first-generation packs from current state.
- Add activation validation for `runtime_compat`, `route_policy`, and `serve_requirements`.
- Normalize promoted-pack contents around formats the TypeScript runtime can load directly.

Exit criteria:

- candidate packs can be built and validated from normalized inputs
- OpenClaw can load a promoted pack without legacy state mutation or Python runtime helpers
- OpenClaw rejects learned-required packs that cannot be served in learned mode
- promoted packs contain no Python-only serving artifacts

### Phase 4: Cut Runtime Over to OpenClaw

- Implement native OpenClaw integration for runtime compile requests using the TypeScript compiler core or worker.
- Remove hook-first runtime injection as the primary path.
- Make OpenClaw own timeout, budget, and logging around compile requests.
- Dual-run the native path in shadow mode on real traffic before cutover and compare context/routing outputs against the incumbent path.

Exit criteria:

- production runtime no longer depends on OCB-managed hook install or socket service
- production runtime no longer depends on Python services for normal compile serving
- compile failures appear in OpenClaw diagnostics, not only OCB logs
- pilot agents show `routing.mode_effective=learned` and `used_learned_route_fn=true` on real production traces for learned-required packs

### Phase 5: Move Learning to the New Plane

- Point learner ingestion to normalized OpenClaw event exports.
- Rebuild correction/teaching/route-learning logic on top of event contracts with TypeScript owning orchestration and pack assembly.
- Make pack promotion the only way learner outputs affect runtime.
- Gate promotion on offline eval, route-specific regression tests, and shadow/canary acceptance where available.
- Keep Python only for explicitly justified offline ML jobs that emit versioned outputs consumed by the TypeScript learner pipeline.

Exit criteria:

- learner can build and promote packs without reading raw OpenClaw session internals
- runtime consumes only promoted packs
- route-model changes reach production only through evaluated pack promotion
- learner scheduling, promotion, and validation do not depend on Python by default

### Phase 6: Delete Legacy Runtime Overlap

- Delete OCB runtime daemon/socket lifecycle.
- Delete hook-pack-centric runtime ownership.
- Delete direct raw-session parsing as the primary architecture.
- Archive or remove docs that describe the old shape as canonical.

Exit criteria:

- one runtime owner
- one learner pipeline
- one promoted-pack consumption path

## Deletion and Cleanup Strategy

Delete aggressively once the new path is proven. Likely deletions or hard deprecations:

- [openclawbrain/daemon.py](/Users/cormorantai/openclawbrain/openclawbrain/daemon.py)
- [openclawbrain/socket_server.py](/Users/cormorantai/openclawbrain/openclawbrain/socket_server.py)
- [openclawbrain/socket_client.py](/Users/cormorantai/openclawbrain/openclawbrain/socket_client.py)
- [integrations/openclaw/hooks/openclawbrain-context-injector/HOOK.md](/Users/cormorantai/openclawbrain/integrations/openclaw/hooks/openclawbrain-context-injector/HOOK.md) and the rest of that hook pack
- runtime-facing adapter CLIs under [openclawbrain/openclaw_adapter](/Users/cormorantai/openclawbrain/openclawbrain/openclaw_adapter)
- runtime install/status orchestration in [openclawbrain/cli.py](/Users/cormorantai/openclawbrain/openclawbrain/cli.py) for `serve`, `daemon`, `openclaw install`, and hook/loop lifecycle commands
- legacy runtime-owned artifacts described in [docs/state-directory.md](/Users/cormorantai/openclawbrain/docs/state-directory.md): `daemon.sock`, `daemon.pid`, `fired_log.jsonl`

Demote to migration-only and then delete:

- [openclawbrain/replay.py](/Users/cormorantai/openclawbrain/openclawbrain/replay.py)
- [openclawbrain/session_sources.py](/Users/cormorantai/openclawbrain/openclawbrain/session_sources.py)
- raw OpenClaw/Codex session parsers used as steady-state ingestion

Keep and reshape:

- learning logic
- feedback contract logic
- route-model training
- provenance and evaluation
- workspace compilation logic

Cleanup rule:

- if a component exists mainly because OpenClawBrain currently compensates for missing OpenClaw-native runtime integration, delete it after cutover

## Concrete Execution Plan

### Phase A: Architecture and Contracts

- ratify this plan
- create TypeScript-first schemas and golden fixtures
- add repo-level architecture tests that enforce owner boundaries

### Phase B: Pack Compiler MVP

- introduce pack build directory layout
- add a TypeScript compiler API that reads only pack artifacts plus request payload
- prove deterministic compile behavior

### Phase C: OpenClaw Native Runtime Integration

- wire OpenClaw runtime to the compiler API
- ship behind a feature flag
- shadow both paths on the same turns
- compare diagnostics, latency, and route-decision actuation before any serving cutover

### Phase D: Learner Rebase

- switch learner ingestion to normalized OpenClaw exports
- preserve current learning ideas, but rebuild the ingestion boundary around TypeScript orchestration
- add evaluation gates before promotion
- isolate any Python training or research step behind artifact handoff boundaries

### Phase E: Cutover

- promote a pilot agent to native runtime compile
- remove hook/socket dependencies for that agent
- verify rollback by pack pointer swap
- verify from production traces that learned-required packs are served with `used_learned_route_fn=true`

### Phase F: Deletion

- delete old daemon/socket/hook paths
- delete docs that present them as canonical
- simplify the CLI around TypeScript-native compile, learn, build-pack, eval-pack, and migrate tools

## Success Criteria

The rearchitecture is complete when:

- OpenClaw is the only runtime owner.
- OpenClawBrain is the compiler and learner, not a second runtime.
- TypeScript is the default production implementation language across runtime-serving and learner orchestration surfaces.
- Runtime memory use flows through one versioned compile contract.
- Learning flows through one normalized event contract.
- Runtime serves exclusively from promoted packs.
- Production serving does not require Python by default.
- Learned `route_fn` from the promoted pack materially drives production compile behavior and is observable in runtime traces.
- Rollback is atomic and operationally boring.
- Legacy daemon, socket, hook-first, and raw-session-primary paths are gone.
