# Release Checklist

This repo publishes TypeScript packages from the `pnpm` workspace.

## 1) Prepare the release

```bash
pnpm install --frozen-lockfile
pnpm check
```

Update the affected package versions in `packages/*/package.json` and refresh `CHANGELOG.md`.

## 2) Create the tag

```bash
git tag -a release-vX.Y.Z -m "OpenClawBrain release vX.Y.Z"
git push origin release-vX.Y.Z
```

## 3) Publish

Pushing a `release-v*` tag triggers `.github/workflows/publish.yml`.

That workflow:

- installs Node 20 and `pnpm`
- runs `pnpm install --frozen-lockfile`
- runs `pnpm check`
- publishes the public workspace packages to npm

## 4) Post-publish verification

Install the published packages into a clean project and verify the expected entrypoints resolve.
