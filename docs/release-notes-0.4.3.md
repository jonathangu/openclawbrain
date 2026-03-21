# OpenClawBrain 0.4.3 release notes

`0.4.3` is a focused operator CLI patch.

## What changed

- operator CLI advances to `@openclawbrain/cli@0.4.3`
- embedding status is more truthful when active packs store numeric vectors in alternate shapes instead of only `entry.embedding`
- serve-time decision matching is more tolerant of event-id drift and small timestamp drift when harvesting supervision candidates

## Public lane

```bash
openclaw plugins install @openclawbrain/openclaw@0.4.0
npx @openclawbrain/cli@0.4.3 install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli@0.4.3 status --openclaw-home ~/.openclaw --detailed
```

## Scope

This patch improves operator truth surfaces and repo-side supervision matching. It does **not** claim to fix the separate live activation-root stale-candidate defect or the known plugin id mismatch warning (`openclawbrain` manifest id vs `openclaw` package/entry hint).
