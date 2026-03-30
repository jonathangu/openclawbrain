# OpenClawBrain

A local memory layer for OpenClaw agents that keeps useful context bounded.

OpenClawBrain stores conversation history and explicit corrections, retrieves a bounded slice of the context most likely to matter, learns from outcomes in the background, and records why a retrieval won. The live path serves promoted packs so latency stays predictable. If the memory layer is unavailable, the agent keeps running.

Status: actively developed. See [CHANGELOG.md](CHANGELOG.md) for current package versions and release history.

[Documentation](docs/README.md) · [Claims boundary](CLAIMS.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [npm: plugin](https://www.npmjs.com/package/@openclawbrain/openclaw) · [npm: CLI](https://www.npmjs.com/package/@openclawbrain/cli)

## Start Here

| Audience | Start here | What you get |
| --- | --- | --- |
| Evaluator | [How it works](#how-it-works) | A fast read on what OpenClawBrain is and how it behaves |
| Operator | [Install and verify](#install-and-verify) | The one-command front door and the next docs to read |
| Contributor | [For contributors](#for-contributors) | Architecture docs, setup, and contribution boundaries |

## How it works

1. **Store.** OpenClawBrain keeps conversation history and explicit user corrections as durable memory.
2. **Retrieve.** Before OpenClaw builds a prompt, the installed extension compiles a bounded slice of context from the currently promoted pack.
3. **Learn.** After the response path completes, the learning pipeline exports turns, builds a candidate pack, and only serves it after promotion.
4. **Trace.** Operator surfaces record why the serve path chose the current pack and whether learned routing is active.

What makes it different:

- Useful context stays bounded on the live path, and learning stays off that path so latency stays predictable.
- The runtime serves promoted packs, not partially written state.
- Explicit user corrections can outrank stale recap material when they conflict.
- The extension fails open. When the memory layer is unavailable, the agent still answers.

### Mental model: learner, teacher, and `route_fn`

If you only remember one explanation, use this one:

- **Learner** = the background OpenClawBrain pipeline. It watches exported events, binds feedback to prior decisions, builds candidate packs, and updates the learned routing policy.
- **Teacher** = the optional local model that produces extra supervision artifacts off the hot path.
- **`route_fn`** = the learned policy artifact the live runtime uses to decide which bounded graph blocks to inject before prompt build.

In one pass:

1. OpenClaw turns produce interactions plus explicit feedback such as corrections, teachings, approvals, and suppressions.
2. OpenClawBrain normalizes those into event exports and serve-time route traces.
3. The learner builds a candidate pack with graph blocks, embeddings, structural metadata, and a learned `route_fn`.
4. Background learning attaches supervision from human feedback, harvested labels, and teacher artifacts, then updates the routing policy.
5. Only promoted packs serve on the live path. The runtime injects a small useful slice of context, and fails open if nothing safe or useful is available.

That is why OpenClawBrain is more than retrieval: it does not just find similar past text. It learns which context helps, keeps that learning off the hot path, and only serves immutable promoted packs.

## Install and verify

Prerequisites:

- A working OpenClaw installation
- Node.js 20+
- npm

The public operator front door is one command pinned to one OpenClaw home:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

`install` is the public front door for the selected home. It writes or repairs the hook for that home and pins the activation root the runtime serves from. `status --detailed` is the quick verification surface.

Activation and teacher wiring are separate checks. Seeing `BRAIN LOADED` or an attached home means the runtime hook is wired correctly for that OpenClaw home. It does **not** by itself prove that an optional teacher model is configured. Teacher wiring lives on its own config path (`brainTeacherEnabled`, `brainTeacherProvider`, `brainTeacherModel`) and should be verified through status fields such as `teacherConfigured`, `teacherProvider`, `teacherModel`, and `teacherConfigError`.

When you need durable operator evidence today, run the proof surface for the same home:

```bash
openclawbrain proof --openclaw-home ~/.openclaw
```

The intended canonical lane is the same install command with optional `--proof`. Until that lands cleanly across every operator surface, proof stays a separate follow-up command. `proof` writes `summary.md`, `steps.json`, `verdict.json`, raw step logs, and proof pointers under one bundle directory.

Manual native-package lane (not the public default):

```bash
openclaw plugins install @openclawbrain/openclaw
openclawbrain install --openclaw-home ~/.openclaw

openclaw plugins update openclawbrain
openclawbrain install --openclaw-home ~/.openclaw
```

Use that manual lane only for explicit compatibility or maintainer work on the native package layer. The public operator story stays on `openclawbrain install`.

A healthy install or repair should report the profile as attached. After the first promoted pack is available, detailed status should also report `serveState=serving_active_pack`. If you are using an optional teacher model, the same detailed status should also show `teacherConfigured=true`, the expected provider/model, and `teacherConfigError=null`.

Next docs:

- [Quick start](docs/getting-started/quick-start.md)
- [Troubleshooting](docs/operating/troubleshooting.md)
- [Lifecycle](docs/lifecycle.md)

## Scope and boundaries

OpenClawBrain is a memory layer for OpenClaw. It does not own the gateway.

It does:

- Provide one operator front door for install and repair on a selected OpenClaw home
- Store sessions and explicit corrections as durable memory
- Build candidate packs in the background and promote them when ready
- Keep the live path on promoted packs so useful context stays bounded and latency stays predictable
- Expose status, rollback, detach, uninstall, and learning inspection commands
- Fail open when memory compilation cannot safely add context

It does not:

- Start, stop, or reconfigure the OpenClaw gateway for you
- Edit LaunchAgent files or gateway environment files
- Claim same-gateway multi-profile attachment as a proven public lane
- Claim shared-root concurrent write safety

## Documentation

Start with the index at [docs/README.md](docs/README.md).

Key docs:

- [Quick start](docs/getting-started/quick-start.md) for the one-command install path
- [Troubleshooting](docs/operating/troubleshooting.md) for the most common operator issues
- [Architecture overview](docs/architecture/overview.md) for the high-level system design
- [Learning pipeline](docs/architecture/learning-pipeline.md) for export, candidate packs, promotion, and rollback
- [Fail-open design](docs/architecture/fail-open.md) for fallback behavior
- [Architecture deep dive](docs/architecture/deep-dive.md) for the existing architecture notes

## For contributors

If you want to work on the repo, start here:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/architecture/overview.md](docs/architecture/overview.md)
- [docs/architecture/deep-dive.md](docs/architecture/deep-dive.md)
- [CLAIMS.md](CLAIMS.md)

Repo setup and validation:

```bash
npm install
npm test
npm run release:verify
```

If you change the public story, update the README, docs index, changelog, and claims boundary in the same pass.
