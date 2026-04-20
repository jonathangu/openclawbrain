# OpenClawBrain

Stop reteaching your agent the same things.

OpenClawBrain gives OpenClaw a useful memory. It carries forward corrections, preferences, and successful past work so your agent can improve over time without turning every prompt into a giant transcript dump.

OpenClawBrain is now shipping measured improvements, not just architecture. In the latest public update, it went from missing every reviewed case where memory should have been used to catching all `10/10`, without adding extra false positives. And the proof is published, so you can inspect the claim instead of taking it on faith.

Current version: **0.4.45** · [Changelog](CHANGELOG.md)

## Why people use it

- fixes and preferences stick
- the agent stops starting from zero every session
- prompts stay smaller and more focused
- you can check whether it is really loaded and working
- if the memory layer is unavailable, the agent still runs

## Start here

If you already have OpenClaw and Node.js 20+, this is the simplest path:

```bash
npx @openclawbrain/cli@0.4.45 install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.45 status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.45 proof --openclaw-home ~/.openclaw
```

Use the same four commands later for upgrades and repairs.

## How it works

In plain English:

1. OpenClawBrain records useful past work, corrections, and outcomes.
2. It learns in the background which memories actually help.
3. It prepares a small memory pack that is safe to serve live.
4. At runtime, OpenClaw gets only the small slice that is likely to help now.

That means the agent gets continuity without turning every prompt into a giant history dump.

## Latest shipped win

The newest published update is intentionally bounded, but it is a real shipped improvement, not a vague benchmark story.

- On a reviewed frozen test set, OpenClawBrain went from catching `0/10` of the cases where memory should have helped to catching all `10/10`.
- At the same time, it stayed disciplined: `0/65` unnecessary activations, `0/69` must-not-fire failures, and `0/403` broad-live regressions on the checked bundle.
- The product now makes it easier to inspect what happened and why, instead of hiding the result behind internal-only artifacts.
- We also tightened the internal safety and review path so improvements are easier to verify and harder to ship by accident.

Honest boundary: this is a real measured improvement on checked bundles. It is not a claim that every live task is already solved.

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
