# OpenClawBrain

Selective intervention for OpenClaw.

OpenClawBrain is the selective intervention layer behind OpenClaw. The current achievable agenda is narrow on purpose: make current choices stick when the brain should help, stay out of the way when it should not, and give operators honest proof surfaces for what happened.

Current version: **0.4.45** · [Changelog](CHANGELOG.md)

## Why people use it

- current corrections and preferences can stick without dumping giant transcripts back into the prompt
- restraint is part of the product: the brain can stay off when it should
- the live path serves small promoted packs and fails open if the brain cannot load safely
- `status --detailed` and `proof` let operators inspect what is real on one OpenClaw home

## Start here

If you already have OpenClaw and Node.js 20+, this is the simplest path:

```bash
npx @openclawbrain/cli@0.4.45 install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.45 status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.45 proof --openclaw-home ~/.openclaw
```

Use the same four commands later for upgrades and repairs.

## What It Does Today

In plain English:

1. OpenClawBrain records corrections, outcomes, and route evidence.
2. It learns off the response path which small intervention helps at similar decision points.
3. It only serves promoted packs.
4. At runtime, it either injects a small current-relevant slice or stays out of the way.

This is not a claim of broad agent memory. It is a bounded selective-intervention system.

## Current Bounded Proof

The current checked proof surfaces are intentionally narrower than the end-state product story.

- The operator install / attach / `status --detailed` / `proof` lane is real on the exercised host surface.
- The latest checked activation-first bundle separates unique wins from ties instead of flattening them together: `18` better, `7` tied, `0` worse on the reviewed `felt_resume_25` traces.
- The same bundle kept restraint clean: `0/65` unnecessary activations, `0/69` must-not-fire failures, and `0/403` broad-live replay regressions on the checked guardrail bundle.
- Broad-live replay ties are guardrail evidence, not product wins.

Honest boundary: this is proof that bounded selective-intervention lanes and operator truth surfaces exist. It is not a claim that OpenClawBrain already solves broad live answer quality.

## What Is Next

The near-term agenda is:

- current-choice fidelity
- restraint / specificity
- honest proof surfaces
- operator-story quality
- tool-capability choice only after the first two lanes are real

## What Is Different From Simple Retrieval

A basic archive can store the past.
A basic search system can find similar text.

OpenClawBrain is trying to answer a narrower and more useful question:

> What small intervention helps this run right now, if any?

That is the current product job.

## Docs

- [Quick start](docs/getting-started/quick-start.md)
- [Docs index](docs/README.md)
- [Install / lifecycle](docs/lifecycle.md)
- [Troubleshooting](docs/operating/troubleshooting.md)
- [Proof map](docs/proof/README.md)
- [How it works](https://openclawbrain.ai/how-it-works/)

## For Maintainers

- [Claims boundary](CLAIMS.md)
- [Release contract](docs/RELEASE_CONTRACT.md)
- [Evidence ladder](docs/EVIDENCE.md)
- [End-state guide](docs/END_STATE.md)
- [Architecture overview](docs/architecture/overview.md)

## Contributing

```bash
npm install
npm test
npm run release:verify
```

If you change the public story, update the README, docs index, and the proof / claims surfaces in the same pass.
