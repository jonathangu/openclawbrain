# OpenClawBrain

**Better agent performance. Lower cost pressure.**

OpenClawBrain is a memory layer for [OpenClaw](https://github.com/anthropics/openclaw) agents. It remembers what worked, learns from corrections, and injects bounded, useful context before every prompt — so your agent stops repeating the same mistakes and starts getting better over time.

The mechanism: a background pipeline watches agent interactions, binds feedback to past decisions, and builds compact memory packs. Only promoted packs reach the live path. The agent gets continuity without unbounded context growth, and latency stays predictable because the hot path never calls a live LLM. If the memory layer goes down, the agent keeps running.

Current version: **0.4.26** · [Changelog](CHANGELOG.md) · [Claims boundary](CLAIMS.md)

## Install

Prerequisites: a working OpenClaw installation, Node.js 20+, npm.

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

`install` writes the hook for your OpenClaw home. `status --detailed` verifies the wiring. `proof` captures a durable evidence bundle you can inspect or keep.

A healthy install reports the profile as attached. After the first promoted pack is available, detailed status reports `serveState=serving_active_pack`.

Next: [Quick start](docs/getting-started/quick-start.md) · [Troubleshooting](docs/operating/troubleshooting.md) · [Lifecycle](docs/lifecycle.md)

## How it works

1. **Store.** Conversations and explicit user corrections become durable memory.
2. **Retrieve.** Before each prompt, the runtime injects a bounded slice of context from the current promoted pack.
3. **Learn.** After the response completes, a background pipeline exports turns, builds a candidate pack, and promotes it when ready.
4. **Trace.** Operator surfaces record which pack was served and why.

### What makes it different

Most retrieval systems find similar past text. OpenClawBrain goes further:

- **Bounded context, not growing context.** The live path serves a compact summary spine plus a protected tail of recent messages. Context size stays predictable.
- **Immutable promoted packs.** Learning happens in the background. The runtime only serves packs that passed promotion — never partially written state.
- **Corrections outrank stale recaps.** When a user corrects the agent, that correction takes priority over older summarized material.
- **Fails open.** If the memory layer is unavailable, the agent answers without it. No hard dependency.

### Mental model

- **Learner** — the background pipeline. It watches events, binds feedback to decisions, builds candidate packs, and updates the routing policy.
- **Teacher** — an optional local model that produces supervision artifacts off the hot path.
- **`route_fn`** — the learned policy the runtime uses to pick which graph blocks to inject before prompt build.

One pass through the system:

1. Agent turns produce interactions and feedback (corrections, approvals, suppressions).
2. OpenClawBrain normalizes these into event exports.
3. The learner builds a candidate pack: graph blocks, embeddings, metadata, and a learned `route_fn`.
4. Background learning attaches supervision from human feedback and teacher artifacts, then updates routing.
5. Only promoted packs serve. The runtime injects a small useful slice, and the hot path stays bounded.

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
