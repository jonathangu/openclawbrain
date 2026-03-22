# OpenClawBrain CLI 0.4.6

## What shipped

- fixed a strict native-PG full-replay bug in the published CLI bundle: the serve-time decision remapper now tolerates missing optional block-id arrays during replay
- added regression coverage proving native V2 updates still materialize when replayed decisions omit optional context-id arrays or score compaction metadata

## Operator lane

```bash
npx @openclawbrain/cli@0.4.6 install --openclaw-home ~/.openclaw
npx @openclawbrain/cli@0.4.6 status --openclaw-home ~/.openclaw --detailed
```
