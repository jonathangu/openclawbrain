# Ops Recipes

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

## Related docs
- [openclaw-integration.md](openclaw-integration.md)
- [operator-observability.md](operator-observability.md)
- [reproduce-eval.md](reproduce-eval.md)
