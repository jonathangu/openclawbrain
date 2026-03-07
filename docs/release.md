# Release Checklist

This branch releases the public TypeScript packages, not the legacy Python wheel.

## 1) Prepare package versions

Bump the version for each public package that will be published:

- `packages/contracts/package.json`
- `packages/pack-format/package.json`
- `packages/compiler/package.json`
- `packages/learner/package.json`

Keep the root `package.json` version aligned with the workspace release candidate if you want a single branch-level marker.

## 2) Create and verify the release candidate

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm release:check
ls -lh .release/
```

Expected artifacts:

- `openclawbrain-contracts-<version>.tgz`
- `openclawbrain-pack-format-<version>.tgz`
- `openclawbrain-compiler-<version>.tgz`
- `openclawbrain-learner-<version>.tgz`

## 3) Tag the release

```bash
git checkout <release-branch>
git tag -a v0.1.0 -m "OpenClawBrain TS workspace v0.1.0"
git push origin <release-branch>
git push origin v0.1.0
```

## 4) Publish packages

Preferred: GitHub Actions + npm trusted publishing.

Pushing a `v*` tag triggers `.github/workflows/publish.yml`, which verifies the workspace and then publishes the four `@openclawbrain/*` packages.

Before relying on the workflow, configure npm trusted publishing for each package.

Optional manual publish:

```bash
(cd packages/contracts && npm publish --access public)
(cd packages/pack-format && npm publish --access public)
(cd packages/compiler && npm publish --access public)
(cd packages/learner && npm publish --access public)
```

## 5) Post-publish sanity checks

```bash
npm view @openclawbrain/contracts version
npm view @openclawbrain/pack-format version
npm view @openclawbrain/compiler version
npm view @openclawbrain/learner version
```

As a final smoke check, install one or more packages from the registry in a clean directory and run the example snippets from the package READMEs.
