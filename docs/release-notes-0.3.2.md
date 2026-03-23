# OpenClawBrain 0.3.2

OpenClawBrain 0.3.2 is the release where the new correction/routing shape becomes public package truth.

The headline change is architectural, not cosmetic:

> **LCM summaries are now treated as a routing prior over history, while explicit typed correction memories act as the durable current-truth layer.**

That gives the runtime a cleaner split between:
- where to look
- when to expand back toward source
- what should currently win when history conflicts

## Highlights

### Summary-aware routing guidance
The runtime can now distinguish between a few important retrieval situations:
- broad recap questions where summary-level context is a reasonable starting point
- precision/conflict-sensitive questions where the system should expand toward source before asserting specifics
- current-truth questions where typed correction memory should beat stale summary recap

This shipped through the new summary-aware assembly/runtime policy surface, including:
- `summary-routing-policy.ts`
- assembled `summaryMetadata`
- runtime guidance in `BrainAssemblerExtension`

### Explicit user-correction commit path
OpenClawBrain now has a real `BrainService.teachUserCorrection()` API and an ingest-time observation path for explicit user corrections.

That shipped with two complementary lanes:
- a **fast deterministic lane** for obvious explicit corrections
- an **off-path async proposal lane** for richer semantic interpretation without blocking the hot path

This makes the product much better at the behavior users actually feel:
- if the user clearly corrects something, the newer truth can win immediately
- older summaries can help locate the region of history, but they no longer need to be the winning truth object

### Repo-wide type surface cleanup
0.3.2 also catches the package up to current repo truth by reconciling the stale type/test surface drift that had accumulated around the SDK boundary and test fixtures.

That includes:
- repo-wide `tsc --noEmit` passing again
- reconciled session-binding tests
- reconciled engine/lcm integration test typing
- update-test fixes around edge vs seed update assertions

### Publish-path stability fix
Release verification exposed a flaky SQLite write race (`database is locked`) on the Brain DB connection path during worker-parent interaction.

0.3.2 fixes that by applying:

```sql
PRAGMA busy_timeout = 5000
```

at the Brain DB connection entry points used by runtime, worker, CLI, and shared connection setup.

That makes the release path itself more trustworthy instead of depending on timing luck.

## Docs added in this release

To make the GitHub/docs story match the code more clearly, 0.3.2 adds:
- `docs/architecture/routing-prior.md`
- `docs/architecture/corrections.md`

These are meant to explain the real architecture more plainly than a changelog bullet list can.

## Validation

Release verification passed with:

```bash
npm -C /Users/cormorantai/openclawbrain run release:verify
```

That includes:
- full test suite passing
- package dry-run succeeding

## In one sentence

> **0.3.2 is the release where OpenClawBrain publicly ships the summary-aware routing prior, the explicit correction commit path, the matching docs, and the cleanup needed to make that package boundary actually stable.**
