# OpenClawBrain CLI 0.4.8

## What shipped

- restores validator-compatible router artifact metadata after the broken 0.4.7 publish
- keeps the native V2 replay remap fix and full-replay hardening from 0.4.6
- preserves the real native-V2 proof path: zero V1 traces plus nonzero learned updates/materialization on full replay

## Operator lane

```bash
npx @openclawbrain/cli@0.4.8 install --openclaw-home ~/.openclaw
npx @openclawbrain/cli@0.4.8 status --openclaw-home ~/.openclaw --detailed
```
