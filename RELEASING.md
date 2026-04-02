# Releasing

This repo uses Changesets to make npm releases reviewable across the split OpenClawBrain packages.

## Normal development

For any pull request that changes user-facing behavior, a changeset should be
added before the work is considered ready to release:

```bash
npm run changeset
```

Choose the smallest appropriate bump:

- `patch`: fixes, docs-visible behavior changes, small compatibility work
- `minor`: new features or notable new behavior
- `major`: breaking changes

The generated markdown file in `.changeset/` should explain the release impact in a sentence or two.

PRs that only touch internal tooling or CI can skip a changeset when they do not need an npm release note.

## Who adds the changeset

Maintainers own release metadata.

- For internal PRs, the author can add the changeset directly.
- For external PRs, do not expect the contributor to know or run the Changesets
  workflow. The reviewer or merge maintainer should add the changeset before
  merge, or immediately afterward in a small follow-up PR.
- If a releasable PR lands without a changeset, create a catch-up changeset PR
  before running the release flow.

The practical rule is simple: if the change should appear in npm release notes,
make sure a maintainer gets a `.changeset/*.md` file onto `main`.

## Release flow

1. Merge releasable PRs to `main`
2. Let the `Version Packages` workflow open or update the release PR
3. Review the generated version bumps and `CHANGELOG.md`
   - for the current public lane, confirm the split surfaces stay aligned:
     - `packages/openclaw`
     - `packages/cli`
     - `README.md`
     - `docs/lifecycle.md`
     - `docs/configuration.md`
     - `packages/openclaw/README.md`
     - `packages/cli/README.md`
   - if hosts still emit the known plugin id mismatch warning (`openclawbrain` manifest id vs `openclaw` package/entry hint), document it rather than implying it is fixed
4. Merge the release PR to `main`
5. On the merged release commit, run:

```bash
npm run release:plan
```

   It should print the single public OpenClawBrain version plus the internal publish order, tag, title, and release-notes file that publish will use. It also fails closed when:
   - the selected ref is not reachable from `main`
   - pending `.changeset/*.md` files are still present on the publish ref
   - `CHANGELOG.md` or `docs/release-notes-<public-version>.md` no longer matches the public release contract
6. Manually trigger the `Publish Package` workflow on that exact merged release commit
   - paste the exact release commit SHA into the workflow input when the selected UI ref is ambiguous
7. Approve the workflow if a protected GitHub Environment is configured
8. Let the workflow:
   - print the same publish plan into the GitHub step summary
   - install dependencies
   - hydrate package-local dependencies for the canonical split-package tests and tarball checks
   - run tests
   - run the checked-in proof smoke gate so release claims fail closed when the frozen proof bundle is missing, incomplete, or stale
   - verify both canonical package tarballs
   - publish `packages/openclaw` to npm
   - publish `packages/cli` to npm
   - create tag `openclawbrain-vX.Y.Z`
   - create the GitHub release from `docs/release-notes-<public-version>.md`

The internal publish order should stay deliberate even though the public story is one product version:

1. publish `@openclawbrain/openclaw` first
2. publish `@openclawbrain/cli` second

That keeps the public CLI front door from converging against an older plugin payload during the release window.

## External setup required

The repo-side files are not enough by themselves. A maintainer still needs npm publish credentials for this GitHub repository/workflow pair.

Recommended external setup:

1. Add an npm automation token as `NPM_TOKEN`, or migrate the workflow to trusted publishing before removing that secret
2. Optionally create a GitHub Environment named `npm-publish` and add required reviewers
3. Optionally keep `.github/release.yml` aligned if you still use GitHub-generated release notes outside the publish workflow

When configuring npm trusted publishing, register the GitHub workflow using the exact workflow filename in this repo: `.github/workflows/publish.yml`.

The publish workflow is intentionally manual. Release issuance should stay deliberate even after trusted publishing is enabled.
