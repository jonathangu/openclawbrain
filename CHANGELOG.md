# Changelog

Release history for the published OpenClawBrain packages. The README and operator docs keep the install lane unpinned; release notes carry version-specific detail.

## 0.3.6

`0.3.6` is a compatibility-package patch release that carries the unified learned local branching update for older combined-package installs.

**Packages**

- `@jonathangu/openclawbrain@0.3.6`

**Changes**

- makes `STOP_LOCAL` a real learned action in the same local policy surface as `traverse(edge_i)`
- persists learned stop-local weights through SQLite state, pack snapshots, runtime reloads, and CLI load/promote hydration
- preserves truthful no-op behavior for forced `STOP_LOCAL` actions with probability `1.0`
- adds focused proof coverage for policy scoring, REINFORCE stop updates, snapshot/runtime round-trip, and worker/promotion surfaces

**Verification**

- `npm run release:verify`
  - passed
  - root Vitest: `43` files / `384` tests
  - `@openclawbrain/openclaw@0.4.1` tarball verification passed
  - `@openclawbrain/cli@0.4.12` tarball verification passed

**Full release note**

- [docs/release-notes-0.3.6.md](docs/release-notes-0.3.6.md)

## 0.4.12

`0.4.12` adds a first-class operator proof bundle and hardens daemon launch paths away from ephemeral `_npx` cache state.

**Packages**

- `@openclawbrain/cli@0.4.12`

**Changes**

- added `openclawbrain proof --openclaw-home <path>` to capture one durable operator proof bundle
- proof bundles now write `summary.md`, `steps.json`, `verdict.json`, raw step logs, startup breadcrumbs, and runtime-load-proof snapshots
- install/status guidance now points operators at the proof capture flow when durable evidence is needed
- daemon launch/status now avoid `_npx` cache paths and surface the configured runtime command and arguments explicitly
- README, quick-start, lifecycle, configuration, and troubleshooting docs now align around one canonical install / verify / proof lane

**Operator commands**

```bash
openclaw plugins install @openclawbrain/openclaw
npx @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli proof --openclaw-home ~/.openclaw --skip-install --skip-restart
```

**Full release note**

- [docs/release-notes-0.4.12.md](docs/release-notes-0.4.12.md)

## 0.4.10

Native V2 router metadata is now accurate end to end, so status and validator surfaces report real promoted-pack fields.

**Packages**

- `@openclawbrain/contracts@0.3.5`
- `@openclawbrain/cli@0.4.10`

**Changes**

- emitted active-pack metadata now reports the real V2 route path, method, and target
- validator reads promoted-pack metadata instead of stale placeholder fields
- repo-side landing commits for this release were `63ea1e6` and `fe3c247`

**Operator commands**

```bash
openclaw plugins install @openclawbrain/openclaw
npx @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
```

**Verification**

- live host verification reports `STATUS ok`
- detailed status shows V2 fields such as `path.pg=v2`, `path.method=policy_gradient_v2`, and `path.target=trajectory_reconstruction`

**Full release note**

- [docs/release-notes-0.4.10.md](docs/release-notes-0.4.10.md)

## 0.4.8

`0.4.8` restores validator-compatible router artifact metadata after the broken `0.4.7` publish.

**Packages**

- `@openclawbrain/cli@0.4.8`

**Changes**

- restored validator-compatible metadata for native V2 artifacts
- kept the replay hardening from `0.4.6`
- preserved the native V2 proof path with zero V1 traces and nonzero learned updates

**Full release note**

- [docs/release-notes-0.4.8.md](docs/release-notes-0.4.8.md)

## 0.4.7

`0.4.7` exposes native V2 policy-gradient observability directly in the router artifact.

**Packages**

- `@openclawbrain/cli@0.4.7`

**Changes**

- surfaced `method=policy_gradient_v2`, `updateVersion=route_pg_update_v2`, and `objective=supervised_route_pg_v2`
- reported reconstructed-trajectory and supervised-trajectory counts in V2 artifacts

**Full release note**

- [docs/release-notes-0.4.7.md](docs/release-notes-0.4.7.md)

## 0.4.6

`0.4.6` fixes a strict native V2 full-replay bug in the published CLI bundle.

**Packages**

- `@openclawbrain/cli@0.4.6`

**Changes**

- replay remapper now tolerates missing optional block-id arrays
- regression coverage proves V2 updates still materialize when replay metadata is incomplete

**Full release note**

- [docs/release-notes-0.4.6.md](docs/release-notes-0.4.6.md)

## 0.4.5

`0.4.5` ships the native V2 route-update fix in the published CLI bundle.

**Packages**

- `@openclawbrain/cli@0.4.5`

**Changes**

- vendored the learner bundle into `@openclawbrain/cli`
- fixed serve-time versus candidate-pack block-id reconstruction for native V2 updates
- added focused regression coverage for nonzero native V2 updates

**Full release note**

- [docs/release-notes-0.4.5.md](docs/release-notes-0.4.5.md)

## 0.4.4

`0.4.4` fixes reinstall and status reporting for the split-package operator flow.

**Packages**

- `@openclawbrain/cli@0.4.4`

**Changes**

- reinstall now normalizes `plugins.allow` and `plugins.entries.openclawbrain`
- status reports the canonical `openclawbrain` install identity

**Full release note**

- [docs/release-notes-0.4.4.md](docs/release-notes-0.4.4.md)

## 0.4.3

`0.4.3` improves operator status output and supervision matching.

**Packages**

- `@openclawbrain/cli@0.4.3`

**Changes**

- embedding status handles alternate numeric vector shapes more accurately
- serve-time decision matching is more tolerant of event-id and timestamp drift

**Full release note**

- [docs/release-notes-0.4.3.md](docs/release-notes-0.4.3.md)

## 0.4.2

`0.4.2` closes the remaining high-signal status and tarball seams left after `0.4.1`.

**Packages**

- `@openclawbrain/openclaw@0.4.0`
- `@openclawbrain/cli@0.4.2`

**Changes**

- shared attachment policy now persists under `activation-root/attachment-truth/policy-declaration.json`
- `status` reads the declared policy instead of drifting back to `policy=null` or `undeclared`
- traced-learning bridge and operator modules are frozen into the published CLI tarball

**Full release note**

- [docs/release-notes-0.4.2.md](docs/release-notes-0.4.2.md)

## 0.4.1

`0.4.1` makes the shared-home attach declaration idempotent.

**Packages**

- `@openclawbrain/openclaw@0.4.0`
- `@openclawbrain/cli@0.4.1`

**Changes**

- rerunning `install --shared` against an already pinned native package plugin now succeeds as a no-op
- the installer still fails when the installed loader entry does not expose a patchable `ACTIVATION_ROOT` constant

**Full release note**

- [docs/release-notes-0.4.1.md](docs/release-notes-0.4.1.md)

## 0.4.0

`0.4.0` is the split-package public release.

**Packages**

- `@openclawbrain/openclaw@0.4.0`
- `@openclawbrain/cli@0.4.0`
- `@jonathangu/openclawbrain@0.3.5` remains available as a compatibility holdover for older installs

**Changes**

- published the plugin/runtime payload and operator CLI as separate packages
- proved the public-registry install flow on a real host
- updated repo and package docs to lead with the split-package story

**Full release note**

- [docs/release-notes-0.4.0.md](docs/release-notes-0.4.0.md)

## 0.3.5

`0.3.5` was the last combined-package release before the split.

**Packages**

- `@jonathangu/openclawbrain@0.3.5`

**Changes**

- hardened prompt assembly when `before_prompt_build` carries empty or partial text envelopes
- improved teacher-status reporting for fresh no-op cycles
- restored the `packages/openclaw` front-door tree in the public repo

**Full release note**

- [docs/release-notes-0.3.5.md](docs/release-notes-0.3.5.md)

## 0.3.4

`0.3.4` removes fake supervision caused by runtime scaffolding.

**Packages**

- `@jonathangu/openclawbrain@0.3.4`

**Changes**

- filters heartbeat prompts, startup/reset scaffolding, and metadata wrappers out of human supervision evidence
- adds focused exclusion and inclusion tests around teacher-pollution cases

**Full release note**

- [docs/release-notes-0.3.4.md](docs/release-notes-0.3.4.md)

## 0.3.3

`0.3.3` fixes the child-worker boot path for launchd-style installs.

**Packages**

- `@jonathangu/openclawbrain@0.3.3`

**Changes**

- resolves the child worker loader to an absolute `file://` import
- launches the child from the plugin root instead of relying on cwd-sensitive module resolution
- adds a focused regression test for `cwd=/` worker launch

**Full release note**

- [docs/release-notes-0.3.3.md](docs/release-notes-0.3.3.md)

## 0.3.2

`0.3.2` introduced the summary-aware routing prior and explicit correction commit path.

**Packages**

- `@jonathangu/openclawbrain@0.3.2`

**Changes**

- summaries now act as a routing prior over history instead of the durable correction layer
- explicit user corrections can commit through a dedicated runtime path
- architecture notes were added for [docs/architecture/routing-prior.md](docs/architecture/routing-prior.md) and [docs/architecture/corrections.md](docs/architecture/corrections.md)

**Full release note**

- [docs/release-notes-0.3.2.md](docs/release-notes-0.3.2.md)

## 0.3.0

`0.3.0` captured the combined-package release state before the later architecture and packaging work.

**Packages**

- `@jonathangu/openclawbrain@0.3.0`

**Changes**

- added Anthropic OAuth setup-token support in the TUI
- resolved SecretRef-backed auth-profile credentials during summarization
- switched LCM tool timestamps to the local timezone

**Full release note**

- `0.3.0` predates the later release-note archive; see the historical notes in this repo if you need more detail
