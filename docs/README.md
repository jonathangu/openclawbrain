# OpenClawBrain documentation

Use this index to find the shortest path for your role.

Public operator front door: `openclawbrain install --openclaw-home <path>` for one OpenClaw home. That path can be the default `~/.openclaw`, a profile-specific home, or an explicit nonstandard directory like `./openclaw-cormorantai`. Fresh homes default to the cold-start prior. Existing homes rerun the same lane and keep their learned preferences on top while the stronger generic prior gets rebuilt underneath. `status --detailed` is the quick check, and it now reports the daemon-side CLI surface separately from the installed hook/runtime-guard surface so skew is explicit. Durable proof remains a separate command today.

## Getting started

- [Quick start](getting-started/quick-start.md) for the one-command install, verify, and optional proof flow
- [Lifecycle](lifecycle.md) for install, proof, rollback, detach, and uninstall
- [Configuration guide](configuration.md) for embeddings, worker mode, operator controls, and the canonical context-management model

## Operating

- [Troubleshooting](operating/troubleshooting.md) for common install and serve-path issues
- [Lifecycle](lifecycle.md) for removal and rollback
- [Configuration guide](configuration.md) for advanced operator commands and context-management truth

## Architecture

- [Graphify bridge](graphify.md) for the shipped Graphify story: what it helps, what stayed off-path, and how to explain the boundary clearly
- [Graphify scheduler](architecture/graphify-scheduler.md) for the delta/reorg cadences, registry links, and retention rules that keep Graphify inspectable and replayable
- [Overview](architecture/overview.md) for the high-level system design
- [Learning pipeline](architecture/learning-pipeline.md) for export, candidate packs, promotion, and rollback
- [Lint families](architecture/teacher-v3-lints.md) for deterministic CI-first vs teacher-assisted audits
- [Fail-open design](architecture/fail-open.md) for fallback behavior and recovery
- [Deep dive](architecture/deep-dive.md) for the existing architecture notes
- [Routing prior](architecture/routing-prior.md) for summary-aware retrieval
- [Corrections](architecture/corrections.md) for the explicit user-correction path
- [Teacher v3 proof surfaces](architecture/teacher-v3-proof.md) for the proposal-reporting / proof-bundle design and its shipped-vs-target-state mapping
- [Proof packaging](proof/README.md) for the shipped operator proof lane, the target-state Teacher v3 bundle, and the worked examples that keep the boundary honest

## Release history

- [Current release notes (0.4.38)](release-notes-0.4.38.md)
- [Full changelog](../CHANGELOG.md)

## Project notes

These files are useful for maintainers, not for a first-time operator:

- [Claims boundary](../CLAIMS.md)
- [Release contract](RELEASE_CONTRACT.md)
- [Evidence notes](EVIDENCE.md)
- [End state notes](END_STATE.md)
