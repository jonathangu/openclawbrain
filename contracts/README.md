# Canonical Contracts

This directory holds the canonical JSON schemas and golden fixtures for the public OpenClawBrain contract surface.

Versioned artifacts currently landed:

- `runtime_compile/v1`
- `interaction_events/v1`
- `feedback_events/v1`
- `artifact_manifest/v1`

Each version directory contains:

- a schema document for the contract shape
- one or more golden JSON fixtures for cross-package parity tests

The TypeScript validators and builders for these files live in `packages/contracts/src/index.ts`.
