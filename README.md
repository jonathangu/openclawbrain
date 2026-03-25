# OpenClawBrain

A local memory layer for OpenClaw agents that learns which memories help.

OpenClawBrain stores conversation history and explicit corrections, retrieves the context most likely to matter, learns from outcomes in the background, and records why a retrieval won. If the memory layer is unavailable, the agent keeps running.

Status: actively developed. See [CHANGELOG.md](CHANGELOG.md) for current package versions and release history.

[Documentation](docs/README.md) · [Claims boundary](CLAIMS.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [npm: plugin](https://www.npmjs.com/package/@openclawbrain/openclaw) · [npm: CLI](https://www.npmjs.com/package/@openclawbrain/cli)

## Start Here

| Audience | Start here | What you get |
| --- | --- | --- |
| Evaluator | [How it works](#how-it-works) | A fast read on what OpenClawBrain is and how it behaves |
| Operator | [Install and verify](#install-and-verify) | The canonical install lane and the next docs to read |
| Contributor | [For contributors](#for-contributors) | Architecture docs, setup, and contribution boundaries |

## How it works

1. **Store.** OpenClawBrain keeps conversation history and explicit user corrections as durable memory.
2. **Retrieve.** Before OpenClaw builds a prompt, the installed extension compiles context from the currently promoted pack.
3. **Learn.** After the response path completes, the learning pipeline exports turns, builds a candidate pack, and only serves it after promotion.
4. **Trace.** Operator surfaces record why the serve path chose the current pack and whether learned routing is active.

What makes it different:

- Learning stays off the agent's response path.
- The runtime serves promoted packs, not partially written state.
- Explicit user corrections can outrank stale recap material when they conflict.
- The extension fails open. When the memory layer is unavailable, the agent still answers.

## Install and verify

Prerequisites:

- A working OpenClaw installation
- Node.js 20+
- npm

The public operator story now has two explicit entry points: **fresh install** for a host that does not have OpenClawBrain yet, and **update** for a host that already has it installed.

### Fresh install

```bash
openclaw plugins install @openclawbrain/openclaw
npx -y @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx -y @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
npx -y @openclawbrain/cli proof --openclaw-home ~/.openclaw --skip-install --skip-restart
```

### Update an existing host

```bash
openclaw plugins update openclawbrain
npx -y @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx -y @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
npx -y @openclawbrain/cli proof --openclaw-home ~/.openclaw --skip-install --skip-restart
```

For an already-installed host, the plugin update is only step 1. You still need to rerun the CLI `install` command so the activation root and native package plugin wiring stay truthful for that OpenClaw home. `status --detailed` is the quick verify surface. `proof` writes `summary.md`, `steps.json`, `verdict.json`, raw step logs, and proof pointers under one bundle directory.

A healthy install or update should report the profile as attached. After the first promoted pack is available, detailed status should also report `serveState=serving_active_pack`.

Next docs:

- [Quick start](docs/getting-started/quick-start.md)
- [Troubleshooting](docs/operating/troubleshooting.md)
- [Lifecycle](docs/lifecycle.md)

## Scope and boundaries

OpenClawBrain is a memory layer for OpenClaw. It does not own the gateway.

It does:

- Install as an OpenClaw plugin plus operator CLI
- Store sessions and explicit corrections as durable memory
- Build candidate packs in the background and promote them when ready
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

- [Quick start](docs/getting-started/quick-start.md) for the minimal install path
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
