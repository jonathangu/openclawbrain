# OpenClawBrain

Stop reteaching your agent the same things.

OpenClawBrain gives OpenClaw a useful memory. It carries forward corrections, preferences, and successful past work so your agent can improve over time without turning every prompt into a giant transcript dump.

Current version: **0.4.44** · [Changelog](CHANGELOG.md)

## Why people use it

- fixes and preferences stick
- the agent stops starting from zero every session
- prompts stay smaller and more focused
- you can check whether it is really loaded and working
- if the memory layer is unavailable, the agent still runs

## Start here

If you already have OpenClaw and Node.js 20+, this is the simplest path:

```bash
npx @openclawbrain/cli@0.4.44 openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.44 openclawbrain status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.44 openclawbrain proof --openclaw-home ~/.openclaw
```

Use the same four commands later for upgrades and repairs.

## How it works

In plain English:

1. OpenClawBrain records useful past work, corrections, and outcomes.
2. It learns in the background which memories actually help.
3. It prepares a small memory pack that is safe to serve live.
4. At runtime, OpenClaw gets only the small slice that is likely to help now.

That means the agent gets continuity without turning every prompt into a giant history dump.

## Latest proof tranche

The newest published proof update is intentionally bounded.

- On the reviewed frozen binary-gate v2 cohort, lowering `activationThreshold` from `0.38` to `0.21` moved merged `must_fire_binary_gate_v2` from `0/10` to `10/10`.
- The checked restraint bundle stayed clean at the same time: unnecessary activations `0/65`, must-not-fire failures `0/69`, broad-live regressions `0/403`.
- Proof surfaces now expose route-decision summaries and compact runtime-mode tables with the familiar `no_brain`, `vector_only`, `graph_prior_only`, and `learned_route` vocabulary.
- The new teacher/compiler proposal lane is report-only and reviewable. It adds evidence refs, replay hooks, proof linkage, and rollback linkage without claiming live teacher-driven mutation.

Honest boundary: this is stronger proof packaging and promotion hardening on checked bundles, not a broad online-proof claim.

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
