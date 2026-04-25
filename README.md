# OpenClawBrain

Selective intervention for OpenClaw.

OpenClawBrain is the selective intervention layer behind OpenClaw. The current achievable agenda is narrow on purpose: make current choices stick when the brain should help, stay out of the way when it should not, and give operators honest proof surfaces for what happened.

Current version: **0.4.48** · [Changelog](CHANGELOG.md)

## Why people use it

- current corrections and preferences can stick without dumping giant transcripts back into the prompt
- restraint is part of the product: the brain can stay off when it should
- the live path serves small promoted packs and fails open if the brain cannot load safely
- `status --detailed` and `proof` let operators inspect what is real on one OpenClaw home

## Start here

If you already have OpenClaw and Node.js 20+, this is the simplest path:

```bash
npx @openclawbrain/cli@0.4.48 install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.48 status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.48 proof --openclaw-home ~/.openclaw
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

- The operator install / attach / `status --detailed` / `proof` lane is real on the exercised host surface: latest bundle verdict `success_and_proven`, severity `none`, warnings `0`.
- The protected current-choice lane remains clean: `full-ocb 5/5`, `regret=0`, `harm=0`.
- Newer explicit preferences now supersede older preferences in the same durable slot, including versioned tool/model choices such as `Codex GPT-5.4` → `Codex GPT-5.5`.
- The broader specificity/restraint cohort now passes `full-ocb 12/12`, with `regret=0` and `harm=0`.
- The first bounded tool-capability choice proof passes for `weather.current_conditions`: must-fire current weather/rain chooses `tool_capability`; must-not-fire weather definition chooses `stop_local`.
- Broad-live replay ties and route-level capability choice are guardrail evidence, not product wins.

Honest boundary: this is proof that bounded selective-intervention, operator truth, and one route-level capability-choice lane exist. It is not a claim that OpenClawBrain already solves broad live answer quality or executes live weather tools end to end.

## What Is Next

The near-term agenda is:

- generalizing capability-choice beyond the first weather lane
- keeping current-choice fidelity protected
- keeping restraint / specificity honest
- improving proof surfaces without broad-memory claims
- making the operator story boring across more homes

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
