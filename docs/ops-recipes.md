# Ops Recipes

## Check shipping surface truth

```bash
pnpm release:status
```

Use this first.
If it reports `shipSurface: "repo-tip"`, treat the repo commit plus optional `.release/` tarballs as the only truthful distribution surface for the wave.

## Rebuild and verify

```bash
pnpm install --frozen-lockfile
pnpm check
pnpm release:pack
```

## Lifecycle proof

```bash
pnpm lifecycle:smoke
```

## Observability proof

```bash
pnpm observability:smoke
```

## Publish dry run

```bash
pnpm release:publish:dry-run
```

Run this only after `pnpm release:status` shows a tagged release candidate and you are intentionally preparing package publication.

## Related docs
- [openclaw-integration.md](openclaw-integration.md)
- [operator-observability.md](operator-observability.md)
- [reproduce-eval.md](reproduce-eval.md)
