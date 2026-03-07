# Setup Guide

For the current public OpenClawBrain package surface, start here:

1. [openclaw-attach-quickstart.md](openclaw-attach-quickstart.md)
2. [openclaw-integration.md](openclaw-integration.md)
3. [operator-observability.md](operator-observability.md)
4. [reproduce-eval.md](reproduce-eval.md)

Minimal bootstrap:

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm check
pnpm release:pack
```
