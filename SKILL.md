---
name: openclawbrain
description: TypeScript-first OpenClawBrain workspace for contracts, pack artifacts, compiler logic, and learner-side pack assembly.
metadata:
  openclaw:
    requires:
      node: ">=20"
---

# OpenClawBrain

Use the TypeScript workspace under [`packages/`](packages):

- `@openclawbrain/contracts`
- `@openclawbrain/pack-format`
- `@openclawbrain/compiler`
- `@openclawbrain/learner`

Workspace commands:

```bash
pnpm install
pnpm check
```

The public repo surface is package-first and artifact-first.
