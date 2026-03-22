# OpenClawBrain CLI 0.4.7

## What shipped

- made native V2 policy-gradient observability truthful in the router artifact itself
- V2 artifacts now report `method=policy_gradient_v2`, `updateVersion=route_pg_update_v2`, `objective=supervised_route_pg_v2`
- V2 artifacts now surface truthful reconstructed-trajectory and supervised-trajectory counts while still keeping trace payloads empty (no V1 fallback traces)

## Operator lane

```bash
npx @openclawbrain/cli@0.4.7 install --openclaw-home ~/.openclaw
npx @openclawbrain/cli@0.4.7 status --openclaw-home ~/.openclaw --detailed
```
