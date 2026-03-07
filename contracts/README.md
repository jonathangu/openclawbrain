# Contracts

This directory holds the versioned contract artifacts and golden fixtures for the public OpenClawBrain package surface.

## Source of truth

- `@openclawbrain/contracts` is the public library surface for these payloads.
- `contracts/` keeps synchronized schemas and golden JSON fixtures for docs, parity checks, and cross-repo validation.
- the canonical supported contract boundary is the versioned package surface plus these versioned fixtures.

## Current contract families

- `runtime_compile/v1`
- `interaction_events/v1`
- `feedback_events/v1`
- `artifact_manifest/v1`

Each version directory contains the schema for that payload family plus one or more deterministic fixtures.

## Highlights in this pass

- `runtime_compile/v1` documents compile budgets, token-aware context blocks, learned-route evidence, and native structural compaction.
- `artifact_manifest/v1` documents the immutable pack boundary, route policy, and router identity that activation and compilation read from disk.
- interaction and feedback fixtures match the normalized event shapes used by the versioned package set shipped from this repo.
