# Canonical Contracts

This directory holds the Phase 1 canonical contract scaffolding from
`docs/openclawbrain-openclaw-rearchitecture-plan.md`.

Versioned artifacts currently landed:

- `runtime_compile/v1`
- `interaction_events/v1`
- `feedback_events/v1`
- `artifact_manifest/v1`

Each version directory contains:

- a schema document for the contract shape
- one or more golden JSON fixtures for cross-repo parity tests

OpenClawBrain's local validator/normalizer for these files lives in
`openclawbrain/contracts/v1.py`. The current runtime and daemon paths are
intentionally unchanged in this phase.
