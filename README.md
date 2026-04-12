# OpenClawBrain

OpenClawBrain gives OpenClaw a useful memory.

It helps your agent carry forward corrections, preferences, and successful past work without stuffing the whole transcript back into every prompt.

Current version: **0.4.43** · [Changelog](CHANGELOG.md)

## What it does

- remembers what worked
- carries forward explicit corrections
- keeps prompts smaller and more focused
- keeps running even if the memory layer is unavailable
- lets you inspect whether it is actually loaded and working

## First-time install

If you already have OpenClaw and Node.js 20+, this is the simplest path:

```bash
npx @openclawbrain/cli@0.4.43 openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.43 openclawbrain status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.43 openclawbrain proof --openclaw-home ~/.openclaw
```

Use the same four commands later for upgrades and repairs.

## How it works

In plain English:

1. OpenClawBrain records useful past work, corrections, and outcomes.
2. It learns in the background which memories actually help.
3. It prepares a small memory pack that is safe to serve live.
4. At runtime, OpenClaw gets only the small slice that is likely to help now.

That means the agent gets continuity without turning every prompt into a giant history dump.

## What is different from simple retrieval

A basic archive can store the past.
A basic search system can find similar text.

OpenClawBrain tries to answer a harder question:

> What small piece of past context will actually help with this run?

That is the whole point.

## Start here

- [Quick start](docs/getting-started/quick-start.md)
- [Docs index](docs/README.md)
- [Install / lifecycle](docs/lifecycle.md)
- [Troubleshooting](docs/operating/troubleshooting.md)
- [How it works](https://openclawbrain.ai/how-it-works/)

## For maintainers

The deep architecture and release docs still exist, but they are not the first stop for a newcomer.

- [Architecture overview](docs/architecture/overview.md)
- [Claims boundary](CLAIMS.md)
- [Release contract](docs/RELEASE_CONTRACT.md)

## Contributing

```bash
npm install
npm test
npm run release:verify
```

If you change the public story, update the README, docs index, and site pages in the same pass.
