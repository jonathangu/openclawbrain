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

## Rollback operator confidence

```bash
pnpm observability:report
```

Use that JSON proof when you need one quick local check for:

- active/candidate/previous inspection across staged, promoted, and rolled-back states
- active vs candidate freshness targets before promotion and after rollback
- explicit async-teacher no-op detection via `noOpDetection.duplicateExport`
- rollback readiness before the pointer move and the restored lineage after it

## Publish dry run

```bash
pnpm release:publish:dry-run
```

## Related docs
- [openclaw-integration.md](openclaw-integration.md)
- [operator-observability.md](operator-observability.md)
- [reproduce-eval.md](reproduce-eval.md)
