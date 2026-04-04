# OpenClawBrain 0.4.28

Canonical install lane:

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

Internal published packages:

- `@openclawbrain/openclaw@0.4.28`
- `@openclawbrain/cli@0.4.28`

## Why this release exists

`0.4.28` is the public Teacher v3 release.

Before this release, the Teacher v3 substrate and discipline story was real, but the runtime/proof adoption path was not fully shipped as one public versioned release. This release closes that gap: proposal persistence is durable, proof bundles are emitted from real runtime/proof inputs, replay runs against real candidate state, bounded canary discipline exists and is rollback-bound, and worked examples/public proof packaging now show the lifecycle honestly.

This keeps the public contract simple: **one OpenClawBrain version, one install lane**.

## What changed

### Teacher v3 runtime/proof adoption

- durable Teacher proposals now persist stable ids, lineage, rollback keys, replay suites, and evidence refs
- operator-proof capture now emits the bounded Teacher v3 proof bundle (`summary.md`, `status.json`, `surface-map.json`, `proposal-report.json`, `verdict.json`)
- replay now runs against real candidate state for compiler/lint and shadow-only mutation/forgetting classes

### Bounded canary discipline

- canary plans are explicit, target-state objects
- canary remains off by default and rollback-bound
- activation is blocked unless replay summary, proof bundle, and rollback binding are present
- canary state is operator-visible in proof/report surfaces

### Worked examples and proof packaging

- repo artifacts now include promotable and shadow-only Teacher v3 worked examples
- repo docs now have a dedicated proof-packaging front door
- the public site proof/how-it-works pages now separate shipped truth from target-state clearly

## Verification

- `npm exec vitest run test/release-docs-drift.test.ts test/brain-store/migrations.test.ts test/brain-store/teacher-v3-proposals.test.ts test/capture-openclawbrain-operator-proof.test.ts test/teacher-v3-proof-bundle.test.ts`
- `npm exec vitest run test/teacher-v3-replay-outcomes.test.ts test/brain-core/teacher-v3-replay.test.ts test/brain-core/teacher-v3-shadow-replay.test.ts test/teacher-v3-proof-bundle.test.ts test/brain-core/teacher-v3-contracts.test.ts test/brain-store/teacher-v3-proposals.test.ts test/brain-core/teacher-v3-promotion-gates.test.ts`
- `npm exec vitest run test/teacher-v3-promotable-examples.test.ts test/teacher-v3-shadow-worked-examples.test.ts test/teacher-v3-proof-bundle.test.ts test/brain-core/teacher-v3-shadow-replay.test.ts test/brain-store/teacher-v3-proposals.test.ts`
- canonical local install + detailed status/proof verification on `~/.openclaw`

## Upgrade

```bash
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain proof --openclaw-home ~/.openclaw
```

If you are already on the canonical install lane, rerun the same lane. Do not use the retired compatibility package path.
