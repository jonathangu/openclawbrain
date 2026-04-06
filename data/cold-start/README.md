# Cold-start source intake bootstrap

This directory is the first governed substrate for the OpenClawBrain cold-start prior.

## Layout
- `registry.bootstrap.json` — top-level intake registry and approval policy
- `scripts/validate-cold-start-source-intake.ts` — validation / summary helper

## Current state
- No source bytes have been downloaded here yet.
- Every dataset row is an explicit intake placeholder with provenance, license, and approval state.
- Destructive ingest is disabled by policy until a human approves a concrete snapshot.
