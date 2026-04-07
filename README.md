# OpenClawBrain

**Better agent performance. Lower cost pressure.**

OpenClawBrain is a memory layer for [OpenClaw](https://github.com/anthropics/openclaw) agents. It remembers what worked, learns from corrections, and injects bounded, useful context before every prompt — so your agent stops repeating the same mistakes and starts getting better over time.

The mechanism: a background pipeline watches agent interactions, binds feedback to past decisions, and builds compact memory packs. Only promoted packs reach the live path. The agent gets continuity without unbounded context growth, and latency stays predictable because the hot path never calls a live LLM. If the memory layer goes down, the agent keeps running.

Current version: **0.4.38** · [Changelog](CHANGELOG.md) · [Claims boundary](CLAIMS.md)

## Install

Prerequisites: a working OpenClaw installation, Node.js 20+, npm.

```bash
openclawbrain install --openclaw-home ./openclaw-cormorantai
openclaw gateway restart
openclawbrain status --openclaw-home ./openclaw-cormorantai --detailed
openclawbrain proof --openclaw-home ./openclaw-cormorantai
```

`install` writes the hook for your chosen OpenClaw home. That can be the default `~/.openclaw`, a profile home like `~/.openclaw-Tern`, or an explicit nonstandard path like `./openclaw-cormorantai`. `status --detailed` verifies the wiring for that same home. `proof` captures a durable evidence bundle you can inspect or keep.

Fresh homes default to the cold-start prior. Existing homes rerun the same install lane and keep their saved preferences on top while OpenClawBrain rebuilds the stronger generic prior underneath them instead of wiping them.

A healthy install reports the profile as attached. After the first promoted pack is available, detailed status reports `serveState=serving_active_pack`.

Next: [Quick start](docs/getting-started/quick-start.md) · [Troubleshooting](docs/operating/troubleshooting.md) · [Lifecycle](docs/lifecycle.md)

## How it works

OpenClawBrain is not a bigger transcript buffer. It is a graph memory system with a learned routing policy.

The short version:

1. **Store the raw material.** Transcripts, corrections, tool traces, status facts, and examples become durable memory.
2. **Turn memory into a graph.** Nodes hold useful facts and artifacts. Edges start simple, then become meaningful: override, evidence, belongs-with, likely-next-step.
3. **Traverse with a learned route function.** At runtime, a small local `route_fn` decides which branches to expand, which memories to pull forward, and when to stop.
4. **Learn off the hot path.** Replay, human feedback, self-learning, harvester signals, and an async teacher label route decisions in the background.
5. **Promote, then serve.** Candidate packs have to survive replay and health checks. Only promoted packs reach the live runtime.

### What makes it different

Most retrieval systems ask, “what past text looks similar?” OpenClawBrain asks a harder question: “what path through memory will help this answer?”

That is the difference.

- **Graph, not flat recall.** Memory has structure, not just similarity scores.
- **Learned traversal, not static top-k.** The runtime can branch, stop, and choose better next hops.
- **Bounded runtime, not prompt sprawl.** The live path serves a small useful slice, not the whole past.
- **Background learning, not hot-path latency.** The async teacher and dreaming loop improve the system without slowing the current turn.
- **Promoted packs, not half-written state.** Learning stays off to the side until the result passes replay.
- **Fails open.** If memory is unavailable, the agent still runs.

### Mental model

Think of OpenClawBrain as four layers working together:

- **Graph memory** stores corrections, examples, traces, docs, and evidence as nodes and edges.
- **`route_fn`** is the fast runtime policy that decides what to retrieve now.
- **Teacher + labels** provide dense background supervision. Human correction stays the highest-trust signal.
- **Replay + promotion** decide what is safe enough to serve live.

The cool part is the loop:

1. The agent does work.
2. The system records the result.
3. Feedback gets attached to the earlier route decisions.
4. Replay and the async teacher produce better labels.
5. The route policy improves.
6. A promoted pack serves a better small slice next time.

If you want the deeper version — graph traversal, route selection, labels, async teacher, dreaming, and promotion — read the full page:

**[How OpenClawBrain works →](https://openclawbrain.ai/how-it-works/)**

## The new Graphify bridge

OpenClawBrain ships a real Graphify bridge.

The important boundary is simple:

- **Graphify is in the product now**
- **Graphify is not the live serve path**

The shipped Graphify lane is an **off-path compiler / diagnostics surface**.

It helps in two places:

1. **cold start** — stronger initial graph structure before a home has much learned personal history
2. **maintenance diagnostics** — bounded operator surfaces for drift, provenance gaps, and graph-vs-OCB review

What shipped includes:

- source-bundle export
- Graphify projection export
- managed Graphify runs
- compiled-artifact bridge
- deterministic lints
- conservative EXTRACTED-only import slice
- candidate-pack input bridge
- maintenance diff lane
- replay/eval proof lane

What we proved:

- `graphify_artifacts_only` won cold start in the proof packet
- `graphify_import + learned_route` beat learned-route without Graphify import
- deterministic lints and maintenance diff are useful, but **diagnostic-only**

So the honest product story is:

> Graphify makes OpenClawBrain better as an **offline graph compiler and maintenance lens**, while the live runtime still serves promoted OpenClawBrain packs.

If you want the focused explanation, read **[docs/graphify.md](docs/graphify.md)**.

## Continuous learning is now part of the shipped product

OpenClawBrain `0.4.38` keeps the first bounded continuous-learning loop on top of the existing memory and Graphify foundations and fixes the public CLI install seam that affected `0.4.37`.

What is now part of the public product:

- route rows that connect traced decisions to reviewable training data
- direct online supervision updates inside the same-family live `route_fn`
- Graphify delta/reorg scheduler registry that stays off the live serve path but makes the background loop inspectable
- periodic same-family retrain with replay-gated promotion
- operator status/control surfaces for Graphify cadence, retrain state, queue visibility, and pause controls

The install lane does **not** change. The same four commands still install, restart, status-check, and prove the system. What changed is that the ongoing improvement loop and its operator surfaces are now a shipped part of the product instead of only repo substrate.

## Scope

OpenClawBrain is the memory layer. It does not own the gateway.

**Does:**
- Install, repair, and manage the memory hook for a selected OpenClaw home
- Store sessions and corrections as durable memory
- Build and promote memory packs in the background
- Keep the live path bounded and latency predictable
- Expose status, rollback, detach, uninstall, and learning inspection
- Fail open when memory compilation cannot safely add context

**Does not:**
- Start, stop, or reconfigure the OpenClaw gateway
- Edit LaunchAgent or gateway environment files
- Claim multi-profile attachment or shared-root concurrent writes as proven

## Documentation

Start at [docs/README.md](docs/README.md).

| Topic | Link |
| --- | --- |
| Quick start | [docs/getting-started/quick-start.md](docs/getting-started/quick-start.md) |
| Troubleshooting | [docs/operating/troubleshooting.md](docs/operating/troubleshooting.md) |
| Architecture | [docs/architecture/overview.md](docs/architecture/overview.md) |
| Learning pipeline | [docs/architecture/learning-pipeline.md](docs/architecture/learning-pipeline.md) |
| Fail-open design | [docs/architecture/fail-open.md](docs/architecture/fail-open.md) |
| Deep dive | [docs/architecture/deep-dive.md](docs/architecture/deep-dive.md) |

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md), then read the [architecture overview](docs/architecture/overview.md) and [claims boundary](CLAIMS.md).

```bash
npm install
npm test
npm run release:verify
```

If you change the public story, update the README, docs index, changelog, and claims boundary in the same pass.
