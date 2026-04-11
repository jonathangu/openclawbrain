# Teacher v3 lint families

This document splits the Teacher v3 lint lane into two deterministic classes:

- **CI-first deterministic lints**: cheap, mechanical checks that should fail closed in CI when they detect obvious drift or broken structure.
- **Teacher-assisted lints**: semantic audits that benefit from synthesis and should usually be reported as findings / proposals rather than hard CI blockers.

The goal is to keep obvious repo and docs drift out of the teacher lane, while still letting the teacher help with judgment-heavy structural audits.

## 1) CI-first deterministic lint family

These checks should be reproducible without model judgment and should run before any teacher-assisted audit.

### Typical checks

- release/version drift across repo, docs, and proof surfaces
- broken backlinks and broken relative refs
- missing files / missing generated artifacts
- schema mismatches in machine-readable metadata
- docs/site string mismatches
- stale generated artifact pointers
- versioned index pages that point at an older release than the current release note set

### Expected behavior

- fail closed when the check is objective and the repo state is clearly wrong
- produce a narrow, human-readable diff or path list
- avoid invoking the teacher when the mismatch is already mechanically provable
- keep the output stable enough for CI and release automation

## 2) Teacher-assisted lint family

These checks are appropriate when the question is not just “is this broken?” but “what does this structure mean?”

### Typical checks

- contradictions across compiled artifacts and raw authority
- stale facts that require semantic interpretation
- orphan nodes or weak neighborhoods
- overloaded nodes that should split
- near-duplicate nodes that should merge
- missing provenance or thin evidence coverage
- suspicious neighborhood shape or unexpected cluster drift
- attribution coverage patterns that are technically valid but suspiciously low

### Expected behavior

- report findings with evidence and rationale
- keep the teacher off the hot path
- do not let semantic linting mask deterministic drift that CI could have caught first
- treat teacher output as auditable guidance, not current-truth authority

## 3) Release-drift motivating case

The repo now lines up on the current release surface:

- the root `README.md` says `Current version: 0.4.41`
- `docs/README.md` points the release-history index at `Current release notes (0.4.41)`
- `docs/END_STATE.md` keeps its current split-package truth on `0.4.41`
- `docs/release-notes-0.4.41.md` exists and describes `0.4.41`

That is the deterministic release-surface state this lint family is meant to keep enforced, not a semantic judgment call.

A CI-first lint should catch this class of drift before the teacher-assisted lane ever runs. The teacher lane can then focus on the harder question: whether the surrounding narrative still matches the actual shipped behavior and public claims boundary.

## 4) Placeholder checklist for an eventual runner

This is the safe TODO surface for a future implementation.

- [ ] scan public version strings across `README.md`, `docs/README.md`, `docs/END_STATE.md`, and release notes
- [ ] scan docs indexes for stale release-note links
- [ ] scan architecture docs for versioned claim drift
- [ ] scan backlinks and relative references for breakage
- [ ] scan generated artifact pointers for stale paths
- [ ] route objective mismatches to CI failure
- [ ] route semantic contradictions to the teacher-assisted report lane
- [ ] attach evidence paths to every reported finding

## 5) Implementation boundary

This document defines the target split. It does not claim the full runner is shipped yet.

The intended contract is:

1. deterministic checks run first
2. teacher-assisted audits run second
3. release-drift and other obvious structural mismatches never depend on model judgment
4. reports stay bounded, auditable, and easy to wire into CI later
