# Contracts

This directory holds the versioned contract artifacts and golden fixtures for the TypeScript-first OpenClawBrain workspace.

## Source of truth

- `@openclawbrain/contracts` is the public library surface for these payloads.
- `contracts/` keeps the schema documents and golden JSON fixtures that support docs, parity checks, and cross-repo validation.
- The canonical boundary is the TypeScript package surface plus these versioned fixtures.

## Current contract families

- `runtime_compile/v1`
- `interaction_events/v1`
- `feedback_events/v1`
- `artifact_manifest/v1`

Each version directory contains the schema for that payload family plus one or more deterministic fixtures.

## Usage

- Import types, builders, and validators from `@openclawbrain/contracts` for application code.
- Use the files in this directory when you need cross-repo fixture parity or human-readable contract references.

The fixtures here describe the public artifact and runtime payload boundary only.
