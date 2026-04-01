# T-20260331-077 Lane P1: Canonical Package Release Seam

## Executive Summary

The split-package release seam is well-defined and functioning. The four post-release
improvements on main (provenance routing, bounded-anytime serving, routing quality,
operator feedback) are all **root engine-level changes** that do not require direct
porting into `packages/openclaw` or `packages/cli`. The packages consume brain
capabilities through published `@openclawbrain/*` dependency packages, not through
direct source imports from root `src/`.

No code changes to the packages are required for the current main state.

---

## 1. Package Build Architecture

### Source Ownership Model

The packages use a **committed-dist, directly-maintained JavaScript** model:

| Layer | Location | Format | Compilation |
|-------|----------|--------|-------------|
| Root engine | `src/brain-core/`, `src/brain-runtime/`, etc. | TypeScript | Runs via `tsx`, root `dist/` gitignored |
| Package extension | `packages/*/extension/*.ts` | TypeScript | `tsc -p tsconfig.extension.json` -> `dist/extension/` |
| Package operator surface | `packages/*/dist/src/*.js` | **JavaScript (directly maintained)** | None. Hand-edited and committed as JS. |

Key finding: the `dist/src/` files in both packages are the **canonical source of truth**.
Source maps exist as artifacts from an original compilation, but the TypeScript source
tree (`packages/*/src/*.ts`) was never committed to git. All subsequent edits go directly
into the `.js` files.

### Dependency Boundary

Packages import exclusively from published `@openclawbrain/*` npm packages:

| Dependency | Used By | Pinned Version |
|------------|---------|----------------|
| `@openclawbrain/compiler` | openclaw, cli | `0.3.5` |
| `@openclawbrain/contracts` | openclaw, cli | `^0.3.5` |
| `@openclawbrain/learner` | openclaw, cli | `^0.3.4` |
| `@openclawbrain/pack-format` | openclaw, cli | `^0.3.4` |
| `@openclawbrain/provenance` | cli only | `^0.3.4` |
| `@openclawbrain/events` | cli only | `^0.3.4` |
| `@openclawbrain/event-export` | cli only | `^0.3.4` |
| `@openclawbrain/workspace-metadata` | cli only | `^0.3.4` |

Packages do **NOT** import from root `src/brain-core/`, `src/brain-runtime/`,
`src/brain-store/`, or `src/brain-worker/`. The root engine is a separate layer.

---

## 2. What Landed on Main Since Last Release

Last release: `split-openclaw-v0.4.5-cli-v0.4.16` at commit `41c6f49`.

| Commit | Description | Files Changed | Touches packages/ |
|--------|-------------|--------------|-------------------|
| `01295d0` | InterruptionAccounting for bounded-anytime serving | `src/brain-core/traverse.ts`, `src/brain-core/types.ts`, `src/brain-runtime/service.ts`, tests | **No** |
| `e0fa1a4` | Provenance routing and operator improvements | `src/brain-core/policy.ts`, `src/brain-core/update.ts`, `src/brain-store/store.ts`, `src/brain-worker/worker.ts`, + 15 others | **No** |
| `b800978` | Proof cron health and aggregate surfaces | `scripts/proof-cron.mjs`, `package.json`, tests | **No** |
| `0e865a3` | Docs: learner teacher route_fn mental model | `README.md` | **No** |

All 25 changed files are in root directories. Zero functional changes to
`packages/openclaw/` or `packages/cli/` since the release cut.

---

## 3. Port Assessment: Which Package(s) Need Code Changes

### Answer: Neither package needs code changes for current main state.

**Reasoning:**

1. **InterruptionAccounting** (`01295d0`): New `INTERRUPTED` marker in traverse footer,
   `InterruptionAccounting` type in brain-core/types.ts. Surfaces through `TraverseResult`
   and trace `selectionMetadata`. The packages' `runtime-core.js` calls
   `compileRuntimeFromActivation` from `@openclawbrain/compiler` which is an opaque
   boundary. No interruption references exist in package code. This is a new truth surface
   that the operator CLI could eventually expose, but it doesn't break anything.

2. **Provenance routing** (`e0fa1a4`): Internal engine changes to policy evaluation,
   trace handling, store operations, worker behavior. The packages don't import directly
   from these modules. The `@openclawbrain/contracts` and `@openclawbrain/provenance`
   dependencies would need updating first if these changes need to surface through packages.

3. **Proof cron** (`b800978`): Root infrastructure script. Package-irrelevant.

4. **Docs** (`0e865a3`): Documentation only.

### When packages WILL need changes

The packages need updates when:
- `@openclawbrain/*` dependency packages ship new APIs that the operator surface should expose
- New plugin config options need to be added to `openclaw.plugin.json`
- The operator CLI needs to display new truth surfaces (e.g., interruption accounting)
- Bug fixes are needed in the operator runtime/CLI

---

## 4. Package Dist: Generated or Source-Owned

**Source-owned.** The `dist/src/*.js` files are directly maintained JavaScript.

Evidence:
- No TypeScript source tree exists in packages (`packages/*/src/` is empty/nonexistent)
- Source maps reference `../../src/<file>.ts` but those files are not committed
- Git history shows direct edits to `.js` files (e.g., commit `d395c95` adds a
  `summarizeBridgeSource` function directly to `traced-learning-bridge.js`)
- `packages/cli` has no build script at all
- `packages/openclaw` only builds `extension/` via TypeScript; `dist/src/` is not rebuilt

The `extension/` layer (TypeScript) is the only compiled artifact. Everything in
`dist/src/` is hand-maintained.

---

## 5. Release Commands

When a future release is warranted:

```bash
# 1. Create changeset for releasable changes
npm run changeset

# 2. After changeset merges, let "Version Packages" workflow open PR
#    Review generated version bumps and CHANGELOG.md

# 3. After merging version PR, validate release plan
npm run release:plan

# 4. Full pre-publish verification
npm run release:verify

# 5. Trigger publish workflow on the merged release commit
#    (manual dispatch via GitHub Actions UI)
#    Workflow: .github/workflows/publish.yml

# Publish order (enforced by workflow):
#   1. npm publish ./packages/openclaw
#   2. npm publish ./packages/cli
# Tag: split-openclaw-v{openclaw_version}-cli-v{cli_version}
```

For local verification during development:
```bash
# Hydrate package dependencies
npm run release:prepare:packages

# Run package tests
npm --prefix packages/openclaw test
npm --prefix packages/cli test

# Verify tarballs
npm run release:verify:openclaw
npm run release:verify:cli
```

---

## 6. Blockers and Gotchas

### No blockers for current state.

### Gotchas for future release work:

1. **Dist JS is the source**: Edit `packages/*/dist/src/*.js` directly. Do not look
   for a TypeScript source tree to compile from.

2. **Selective `files` field**: `packages/openclaw/package.json` explicitly lists
   published files. New operator surface files must be added to the `files` array AND
   to the tarball verification script (`scripts/verify-openclaw-package-tarball.mjs`).
   The CLI package ships all of `dist/src` and `dist/extension`.

3. **Plugin manifest version sync**: `openclaw.plugin.json` version must match
   `packages/openclaw/package.json` version. The tarball verifier enforces this.

4. **Plugin must NOT have bin entries**: Verified by `verify-openclaw-package-tarball.mjs`.

5. **CLI must NOT have plugin manifest**: Verified by `verify-openclaw-cli-package-tarball.mjs`.

6. **Publish order**: openclaw first, then cli. This prevents the CLI from resolving
   against an older plugin payload during the release window.

7. **Dependency version bumps**: If `@openclawbrain/*` packages ship new versions,
   both package.json files need updated dependency versions before release.

8. **No pending changesets**: Currently none. A changeset must be created before the
   next release to drive the version PR workflow.

9. **Extension compilation**: Only `packages/openclaw/extension/` needs TypeScript
   compilation (`npm run build` in that package). The CLI extension is also TypeScript
   but the CLI has no build script -- its `dist/extension/` is also committed directly.

---

## 7. Commit SHA

No code changes were made. This is a research/analysis lane only.
