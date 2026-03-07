# Changelog

This changelog tracks the public TypeScript package lane for `@openclawbrain/*`.
Historical repository tags such as `v12.x` are retained for legacy repo history and do not describe the current npm package line.

## Unreleased

- No unreleased changes yet.

## 0.1.1 - 2026-03-07

- Removed the remaining Python-facing runtime, benchmark, simulation, example, and compatibility-shim trees from the public repo surface.
- Replaced stale PyPI-era docs and release plumbing with Node 20 + `pnpm` workspace guidance.
- Removed hardcoded legacy repo pointers so the repo documents only the TypeScript package lane.
- Restored the full TypeScript package lane to workspace metadata, release artifacts, publish automation, and top-level docs.
- Normalized npm metadata and prepack behavior across activation, events, event-export, provenance, and workspace-metadata packages.
- Dropped accidentally tracked generated JavaScript declaration artifacts from TypeScript source directories.
