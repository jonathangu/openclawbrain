---
name: openclawbrain
description: TypeScript-first OpenClawBrain workspace for contracts, event flows, pack artifacts, activation, compiler logic, and learner-side pack assembly.
metadata:
  openclaw:
    requires:
      node: ">=20"
---

# OpenClawBrain

Use the TypeScript workspace under [`packages/`](packages):

- `@openclawbrain/contracts`
- `@openclawbrain/events`
- `@openclawbrain/event-export`
- `@openclawbrain/workspace-metadata`
- `@openclawbrain/provenance`
- `@openclawbrain/pack-format`
- `@openclawbrain/activation`
- `@openclawbrain/compiler`
- `@openclawbrain/learner`

Workspace commands:

```bash
pnpm install
pnpm check
pnpm release:pack
pnpm release:check
```

The public repo surface is package-first and artifact-first.
