# OpenClawBrain documentation

Use this index to find the shortest path for your role.

Public operator front door: `openclawbrain install --openclaw-home <path>` for one OpenClaw home. `status --detailed` is the quick check. Durable proof is still a separate follow-up surface today.

## Getting started

- [Quick start](getting-started/quick-start.md) for the one-command install, verify, and optional proof flow
- [Lifecycle](lifecycle.md) for install, proof, rollback, detach, and uninstall
- [Configuration guide](configuration.md) for embeddings, worker mode, operator controls, and the canonical context-management model

## Operating

- [Troubleshooting](operating/troubleshooting.md) for common install and serve-path issues
- [Lifecycle](lifecycle.md) for removal and rollback
- [Configuration guide](configuration.md) for advanced operator commands and context-management truth

## Architecture

- [Overview](architecture/overview.md) for the high-level system design
- [Learning pipeline](architecture/learning-pipeline.md) for export, candidate packs, promotion, and rollback
- [Fail-open design](architecture/fail-open.md) for fallback behavior and recovery
- [Deep dive](architecture/deep-dive.md) for the existing architecture notes
- [Routing prior](architecture/routing-prior.md) for summary-aware retrieval
- [Corrections](architecture/corrections.md) for the explicit user-correction path

## Release history

- [Current release notes (0.4.22)](release-notes-0.4.22.md)
- [Full changelog](../CHANGELOG.md)

## Project notes

These files are useful for maintainers, not for a first-time operator:

- [Claims boundary](../CLAIMS.md)
- [Release contract](RELEASE_CONTRACT.md)
- [Evidence notes](EVIDENCE.md)
- [End state notes](END_STATE.md)
