# Setup Guide

For the current public OpenClawBrain surface, start here:

1. [openclaw-attach-quickstart.md](openclaw-attach-quickstart.md)
2. [operator-observability.md](operator-observability.md)
3. [openclaw-integration.md](openclaw-integration.md)
4. [reproduce-eval.md](reproduce-eval.md)
5. [contracts-v1.md](contracts-v1.md)

Minimal bootstrap:

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm release:pack
```
