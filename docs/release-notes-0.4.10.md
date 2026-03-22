# OpenClawBrain CLI 0.4.10

## What shipped

- completed the native V2 router metadata truthfulness milestone
- repo-side patches (`63ea1e6`, `fe3c247`) make native PG V2 metadata in promoted packs truthful: `path.pg=v2`, `path.method=policy_gradient_v2`, `path.target=trajectory_reconstruction`
- published `@openclawbrain/contracts@0.3.5` and `@openclawbrain/cli@0.4.10` carry the metadata upgrade
- the validator path now reads accurate V2 metadata (real trajectory counts, supervision counts, update counts) from promoted packs instead of stale or placeholder fields
- live host verification confirms `STATUS ok` with truthful active-pack metadata: `traced.routes=344`, `traced.supervision=32`, `traced.updates=68`

## Operator lane

```bash
npx @openclawbrain/cli@0.4.10 install --openclaw-home ~/.openclaw
npx @openclawbrain/cli@0.4.10 status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli@0.4.10 learn --activation-root ~/.openclawbrain/activation --json
```

## Context in the native V2 sequence

This release is the capstone of the native V2 policy-gradient work:

| Release | What it fixed |
|---------|--------------|
| 0.4.5 | Native V2 PG route-update seam (serve-time vs candidate-pack block ids) |
| 0.4.6 | Full-replay tolerance for missing optional block-id arrays |
| 0.4.7 | V2 observability fields (`method`, `updateVersion`, `objective`) in router artifacts |
| 0.4.8 | Validator-compatible router artifact metadata after broken 0.4.7 publish |
| **0.4.10** | **Repo-side truthful V2 metadata + published proven packages** |

## Verification

```bash
npx @openclawbrain/cli@0.4.10 learn --activation-root /Users/guclaw/.openclawbrain/activation --json materialized/promoted pack-d099f991
```

Live detailed status output confirms:
- `STATUS ok`
- `path.pg=v2`
- `path.method=policy_gradient_v2`
- `path.target=trajectory_reconstruction`
- `path.trajectories=344`
- `traced.routes=344`
- `traced.supervision=32`
- `traced.updates=68`
