# Changelog

## Unreleased

- Consolidated the repo around the public `@openclawbrain/*` TypeScript package surface and its release artifacts.
- Updated root docs and release guidance to the Node 20 + `pnpm` workspace lifecycle used by this repo.
- Removed stale repo pointers so the docs and scripts describe the current package/runtime boundary only.
- Restored the full TypeScript package lane to workspace metadata, release artifacts, publish automation, and top-level docs.
- Normalized npm metadata and prepack behavior across activation, events, event-export, provenance, and workspace-metadata packages.
- Dropped accidentally tracked generated JavaScript declaration artifacts from TypeScript source directories.
