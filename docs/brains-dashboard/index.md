# Brains Dashboard

This route is a placeholder entrypoint for the operator-facing dashboard story.

For the current public repo, the operator-facing proof surface is:
- [operator-observability.md](../operator-observability.md)
- [openclaw-integration.md](../openclaw-integration.md)
- [reproduce-eval.md](../reproduce-eval.md)

## What exists today

Use `pnpm observability:smoke` to prove the current diagnostics contract for:
- activation health
- promotion readiness
- freshness inspection
- deterministic priority fallback

## Not claimed here

This repo does not yet ship a standalone browser dashboard application under this route.
The route exists so site/docs navigation has a canonical operator landing page instead of a dead link.
