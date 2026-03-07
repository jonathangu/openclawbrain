# Contracts

This directory holds the versioned contract artifacts and golden fixtures for the TypeScript-first OpenClawBrain workspace.

## Source of truth

- `@openclawbrain/contracts` is the public library surface for these payloads.
- `contracts/` keeps synchronized schemas and golden JSON fixtures for docs, parity checks, and cross-repo validation.
- The canonical boundary is the TypeScript package surface plus these versioned fixtures.

## Current contract families

- `runtime_compile/v1`
- `interaction_events/v1`
- `feedback_events/v1`
- `artifact_manifest/v1`

Each version directory contains the schema for that payload family plus one or more deterministic fixtures.

## Highlights in this pass

- `runtime_compile/v1` documents larger-context budgets, token-aware context blocks, and native structural compaction.
- `artifact_manifest/v1` documents the immutable pack boundary that activation and compilation read from disk.
- interaction and feedback fixtures now match the normalized TS-first event shapes used by the published packages.
