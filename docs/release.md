# Release Checklist

This repo releases the full public TypeScript package lane, including the OpenClaw runtime bridge package.

The GitHub repo is public. The root workspace package stays `private` only so the monorepo itself is not published to npm; the supported release surface is the `@openclawbrain/*` package set below.

The active public release line is `0.1.x` for the workspace marker and every published `@openclawbrain/*` package.
Historical repository tags such as `v12.x` are legacy milestones only; they are not the current npm package lane and should not be reused for public TypeScript releases.

## 1) Prepare package versions

Bump the version for each public package that will be published:

- `packages/contracts/package.json`
- `packages/events/package.json`
- `packages/event-export/package.json`
- `packages/workspace-metadata/package.json`
- `packages/provenance/package.json`
- `packages/pack-format/package.json`
- `packages/activation/package.json`
- `packages/compiler/package.json`
- `packages/learner/package.json`
- `packages/openclaw/package.json`

Keep the root `package.json` version aligned with the workspace release candidate if you want a single workspace-level marker.

For the current public lane, use a matching `v0.1.x` git tag for the release.

## 2) Create and verify the release candidate

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm release:check
ls -lh .release/
```

Expected artifacts:

- `openclawbrain-contracts-<version>.tgz`
- `openclawbrain-events-<version>.tgz`
- `openclawbrain-event-export-<version>.tgz`
- `openclawbrain-workspace-metadata-<version>.tgz`
- `openclawbrain-provenance-<version>.tgz`
- `openclawbrain-pack-format-<version>.tgz`
- `openclawbrain-activation-<version>.tgz`
- `openclawbrain-compiler-<version>.tgz`
- `openclawbrain-learner-<version>.tgz`
- `openclawbrain-openclaw-<version>.tgz`

## 3) Tag the release

```bash
VERSION=0.1.1
git checkout <release-branch>
git tag -a "v${VERSION}" -m "OpenClawBrain TS packages v${VERSION}"
git push origin <release-branch>
git push origin "v${VERSION}"
```

`.github/workflows/publish.yml` listens only for `v0.1.*` tags so the public automation matches the current TypeScript package lane.
If the project opens a new public minor line later, widen that workflow trigger in the same change that bumps package versions and release docs.

## 4) Publish packages

Preferred: GitHub Actions + npm trusted publishing.

Pushing a `v0.1.*` tag triggers `.github/workflows/publish.yml`, which verifies the workspace and then publishes every public `@openclawbrain/*` package from the monorepo.

Before relying on the workflow, configure npm trusted publishing for each package.

Optional manual publish:

```bash
pnpm release:publish:dry-run
pnpm release:publish
```

The manual publish commands use the same recursive `@openclawbrain/*` package selection as the workflow so the package set stays in sync.

Use `pnpm release:publish:provenance` when you want the same recursive publish lane with provenance enabled from a local trusted environment.

## 5) Post-publish sanity checks

```bash
npm view @openclawbrain/contracts version
npm view @openclawbrain/events version
npm view @openclawbrain/event-export version
npm view @openclawbrain/workspace-metadata version
npm view @openclawbrain/provenance version
npm view @openclawbrain/pack-format version
npm view @openclawbrain/activation version
npm view @openclawbrain/compiler version
npm view @openclawbrain/learner version
npm view @openclawbrain/openclaw version
```

As a final smoke check, install one or more packages from the registry in a clean directory and run the example snippets from the package READMEs.

For broader proof and integration routes, also verify:
- `docs/openclaw-integration.md`
- `docs/reproduce-eval.md`
