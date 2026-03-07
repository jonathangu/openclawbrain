# Brains Dashboard

This route is intentionally a documentation placeholder for the operator-facing dashboard story. It reserves the canonical docs path, but it is not a shipped browser dashboard.

For the current public repo, the operator-facing proof surface is:
- [operator-observability.md](../operator-observability.md)
- [openclaw-integration.md](../openclaw-integration.md)
- [reproduce-eval.md](../reproduce-eval.md)

## What exists today

There is no standalone UI under this route today. The public operator-facing proof surface in this repo is the docs plus `pnpm observability:smoke`.

Use `pnpm observability:smoke` to prove the current diagnostics contract for:
- activation health
- promotion readiness
- freshness inspection
- deterministic priority fallback

## Not claimed here

This repo does not ship a standalone browser dashboard application under this route.
The route exists so site/docs navigation has a canonical operator landing page while the operator proof surface remains documentation- and API-first.
