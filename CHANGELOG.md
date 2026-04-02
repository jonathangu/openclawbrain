# Changelog

Release history for the published OpenClawBrain releases. The README and operator docs lead with the current public front door; release notes carry version-specific detail. Internal split-package versions are maintainer detail and appear under each entry's **Internal published packages** label.

## Unreleased

## 0.4.20

`0.4.20` is the public OpenClawBrain release for the post-swarm operator follow-up: the front door now carries cheap deterministic savings proxies, a versioned estimated-cost path, tighter public proof truth, and compatibility hardening for legacy `custom_message` session records in the local tail path.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.6`
- `@openclawbrain/cli@0.4.20`

**Changes**

- proof surfaces now expose deterministic prompt-side savings proxies (selected context chars, blocks, estimated prompt tokens)
- proof surfaces now expose deterministic hop/correction proxy metrics from replay and trace truth
- a versioned pricing table now drives estimated prompt / completion / total USD rollups when the required signals exist
- public truth surfaces now explicitly keep replay small/mixed, reject direct spend-savings claims, and preserve the bounded hot-path / no-live-LLM-per-hop framing
- session-tail parsing now accepts legacy `custom_message` records instead of treating them as unknown and skipping the watch path

**Full release note**

- [docs/release-notes-0.4.20.md](docs/release-notes-0.4.20.md)

## 0.4.19

`0.4.19` publishes the operator-surface follow-up after the merged OCB tranche: the CLI now keeps large-log status/proof reads bounded **and** the proof health snapshot stops declaring false outages when the live runtime is actually serving a pack.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.6`
- `@openclawbrain/cli@0.4.19`

**Changes**

- keeps status/proof learning-spine reads bounded for oversized logs
- hardens traced-learning status surfaces so repeated persisted status payloads stay JSON-serializable
- fixes proof-cron health snapshots to read operator/runtime truth instead of stale legacy worker-only status
- reports live serve-state facts in health snapshots (`runtime healthy`, `serve state`, active pack, learned-route state, load proof)

**Full release note**

- [docs/release-notes-0.4.19.md](docs/release-notes-0.4.19.md)

## 0.4.18

`0.4.18` publishes the operator-surface status fix so the split-package lane no longer whole-reads oversized learning-spine logs during status/proof inspection.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.6`
- `@openclawbrain/cli@0.4.18`

**Changes**

- `openclawbrain status` now uses bounded/tail reads for oversized learning-spine logs instead of unconditional whole-file reads
- route-freshness and last-learning-update status/proof surfaces stop whole-reading large learning-spine logs
- regression coverage now locks the large-log status path to the bounded read behavior

**Full release note**

- [docs/release-notes-0.4.18.md](docs/release-notes-0.4.18.md)

## 0.4.17

`0.4.17` publishes the bounded-anytime interruption-truth follow-up so the packaged runtime and CLI now surface interruption state and accounting instead of leaving that truth stranded in the repo-root runtime path.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.6`
- `@openclawbrain/cli@0.4.17`

**Changes**

- the packaged `@openclawbrain/openclaw` runtime now forwards interruption summaries into runtime context and serve-time decision records
- the packaged `@openclawbrain/cli` traced-learning bridge now surfaces a compact last-interruption summary derived from `last_assembly_decision_json`
- persisted package status surfaces no longer hide newer bounded-serving truth
- split-package verification stays green after package-local dependency hydration and tarball checks

**Full release note**

- [docs/release-notes-0.4.17.md](docs/release-notes-0.4.17.md)

## 0.4.16

`0.4.16` publishes the harvested post-0.4.15 fixes: the public CLI proof/reinstall surfaces now classify fresh-state runtime truth correctly, learned-route replay no longer duplicates non-seed carry-forward evidence into the held-out eval turn, teacher no-op status separates benign idle cycles from likely missed teachable material, and the release contract is enforced from one canonical plan helper.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.5`
- `@openclawbrain/cli@0.4.16`

**Changes**

- `proof` now supports explicit gateway probe overrides and downgrades fresh-state `STATUS warn` to a warning when stronger runtime proofs already show live load
- reinstall/repair no longer returns `manual_action_required` when runtime is already proven
- the former `real-trace-live-proof-story` tie is closed in the published CLI lane: `learned_route` now reruns at `100` versus `70` for `graph_prior_only` and `vector_only`
- teacher status now distinguishes benign no-artifact idle cycles from likely missed teachable-material cycles
- publish preflight now derives the split tag/title/release-notes contract from one checked-in helper and verifies package-local dependencies before canonical tarball checks

**Full release note**

- [docs/release-notes-0.4.16.md](docs/release-notes-0.4.16.md)

## 0.4.15

`0.4.15` ships the runtime-fix release: learned-route proof scoring is sharper, embedder-backed promoted packs carry live numeric embeddings through the canonical learner path, teacher/runtime truth surfaces are cleaner, and live `learn` no longer recurses the traced-learning bridge until persistence fails.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.4`
- `@openclawbrain/cli@0.4.15`

**Changes**

- learned-route replay proof now preserves aggregate phrase coverage instead of flattening truthful wins into ties
- first-promotion route selection now carries seed cue blocks forward so the learned route can surface the right proof context earlier
- candidate-pack embedder reindexing now reuses the canonical learner path, so promoted packs retain live numeric embeddings truthfully
- traced-learning bridge persistence now flattens bridged source metadata instead of recursively nesting prior bridge state during live `learn`
- teacher status truth now distinguishes fresh no-artifact no-op cycles from stale/broken teacher state

**Full release note**

- [docs/release-notes-0.4.15.md](docs/release-notes-0.4.15.md)

- repo docs now lead with `openclawbrain install --openclaw-home <path>` as the public front door for one OpenClaw home
- release docs keep `proof` framed as the current follow-up surface, with `--proof` still documented as planned rather than shipped
- release truth now reflects that detailed status and proof expose runtime-guard/load-proof state for the selected home
- release contract now treats bundle evaluation as a real replay gate and keeps the remaining public gaps narrowed to attribution, broader host/profile proof, and live-gain evidence
- public positioning now emphasizes bounded useful context on promoted packs and predictable live-path latency

## 0.4.14

`0.4.14` carries the canonical install lane forward after the payload-sync work landed in the publishable package payloads.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.3`
- `@openclawbrain/cli@0.4.14`

**Changes**

- `openclawbrain install --openclaw-home <path>` is now the explicit public front door for one OpenClaw home, with converge logic that installs or refreshes `@openclawbrain/openclaw`, repairs hook wiring, and only restarts when runtime-affecting state changed
- operator docs and CLI help now describe `proof` as the current follow-up surface, while keeping `--proof` framed as the intended future add-on to `install`
- proof and status surfaces stay aligned with the selected `--openclaw-home`, so install, restart, verify, and durable evidence read as one operator lane
- the published plugin manifest now stays version-locked to `packages/openclaw/package.json`, and the runtime payload carries the bounded-runtime config and status/proof truth the repo already exercises

**Full release note**

- [docs/release-notes-0.4.14.md](docs/release-notes-0.4.14.md)

## 0.3.7

`0.3.7` is a compatibility-package patch release that carries the bounded-runtime hardening lane for older combined-package installs.

**Internal published packages**

- `@jonathangu/openclawbrain@0.3.7`

**Changes**

- adds soft compile-deadline support through `brainMaxCompileMs`
- adds structured compile/deadline/drop metadata on the compatibility live path
- decouples retrieval/query budget from the final `maxContextChars` injection cap
- preserves truthful deadline/clip attribution across trace, observation, teacher materialization, and status surfaces
- adds focused proof coverage for bounded-runtime fail-open outcomes

**Verification**

- `npm run release:verify`
  - passed
  - root Vitest: `43` files / `395` tests
  - `@openclawbrain/openclaw@0.4.2` tarball verification passed
  - `@openclawbrain/cli@0.4.13` tarball verification passed

**Full release note**

- [docs/release-notes-0.3.7.md](docs/release-notes-0.3.7.md)

## 0.4.13

`0.4.13` completes the split-package STOP_LOCAL release lane: the split public install path now carries learned source-specific STOP_LOCAL updates through the CLI learner and the runtime graph-walk compiler.

**Internal published packages**

- `@openclawbrain/compiler@0.3.5`
- `@openclawbrain/openclaw@0.4.2`
- `@openclawbrain/cli@0.4.13`

**Changes**

- split-package learner now emits source-specific learned STOP_LOCAL policy updates instead of treating STOP as a virtual no-op action
- graph-walk compilation now respects learned STOP_LOCAL updates when deciding whether to keep expanding from a source block
- canonical split package dependencies now pin the STOP-aware compiler patch so install/upgrade behavior is truthful
- adds focused proof coverage for source-specific STOP updates and graph-walk halt behavior

**Full release note**

- [docs/release-notes-0.4.13.md](docs/release-notes-0.4.13.md)

## 0.3.6

`0.3.6` is a compatibility-package patch release that carries the unified learned local branching update for older combined-package installs.

**Internal published packages**

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

**Internal published packages**

- `@openclawbrain/cli@0.4.12`

**Changes**

- added `openclawbrain proof --openclaw-home <path>` to capture one durable operator proof bundle
- proof bundles now write `summary.md`, `steps.json`, `verdict.json`, raw step logs, startup breadcrumbs, and runtime-load-proof snapshots
- install/status guidance now points operators at the proof capture flow when durable evidence is needed
- daemon launch/status now avoid `_npx` cache paths and surface the configured runtime command and arguments explicitly
- README, quick-start, lifecycle, configuration, and troubleshooting docs now align around one canonical install / verify / proof lane

**Historical operator commands (pre-front-door; maintainer reference only)**

```bash
openclaw plugins install @openclawbrain/openclaw
npx @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
npx @openclawbrain/cli proof --openclaw-home ~/.openclaw --skip-install --skip-restart
```

The public front door is now `openclawbrain install --openclaw-home ~/.openclaw`.

**Full release note**

- [docs/release-notes-0.4.12.md](docs/release-notes-0.4.12.md)

## 0.4.10

Native V2 router metadata is now accurate end to end, so status and validator surfaces report real promoted-pack fields.

**Internal published packages**

- `@openclawbrain/contracts@0.3.5`
- `@openclawbrain/cli@0.4.10`

**Changes**

- emitted active-pack metadata now reports the real V2 route path, method, and target
- validator reads promoted-pack metadata instead of stale placeholder fields
- repo-side landing commits for this release were `63ea1e6` and `fe3c247`

**Historical operator commands (pre-front-door; maintainer reference only)**

```bash
openclaw plugins install @openclawbrain/openclaw
npx @openclawbrain/cli install --openclaw-home ~/.openclaw
openclaw gateway restart
npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed
```

The public front door is now `openclawbrain install --openclaw-home ~/.openclaw`.

**Verification**

- live host verification reports `STATUS ok`
- detailed status shows V2 fields such as `path.pg=v2`, `path.method=policy_gradient_v2`, and `path.target=trajectory_reconstruction`

**Full release note**

- [docs/release-notes-0.4.10.md](docs/release-notes-0.4.10.md)

## 0.4.8

`0.4.8` restores validator-compatible router artifact metadata after the broken `0.4.7` publish.

**Internal published packages**

- `@openclawbrain/cli@0.4.8`

**Changes**

- restored validator-compatible metadata for native V2 artifacts
- kept the replay hardening from `0.4.6`
- preserved the native V2 proof path with zero V1 traces and nonzero learned updates

**Full release note**

- [docs/release-notes-0.4.8.md](docs/release-notes-0.4.8.md)

## 0.4.7

`0.4.7` exposes native V2 policy-gradient observability directly in the router artifact.

**Internal published packages**

- `@openclawbrain/cli@0.4.7`

**Changes**

- surfaced `method=policy_gradient_v2`, `updateVersion=route_pg_update_v2`, and `objective=supervised_route_pg_v2`
- reported reconstructed-trajectory and supervised-trajectory counts in V2 artifacts

**Full release note**

- [docs/release-notes-0.4.7.md](docs/release-notes-0.4.7.md)

## 0.4.6

`0.4.6` fixes a strict native V2 full-replay bug in the published CLI bundle.

**Internal published packages**

- `@openclawbrain/cli@0.4.6`

**Changes**

- replay remapper now tolerates missing optional block-id arrays
- regression coverage proves V2 updates still materialize when replay metadata is incomplete

**Full release note**

- [docs/release-notes-0.4.6.md](docs/release-notes-0.4.6.md)

## 0.4.5

`0.4.5` ships the native V2 route-update fix in the published CLI bundle.

**Internal published packages**

- `@openclawbrain/cli@0.4.5`

**Changes**

- vendored the learner bundle into `@openclawbrain/cli`
- fixed serve-time versus candidate-pack block-id reconstruction for native V2 updates
- added focused regression coverage for nonzero native V2 updates

**Full release note**

- [docs/release-notes-0.4.5.md](docs/release-notes-0.4.5.md)

## 0.4.4

`0.4.4` fixes reinstall and status reporting for the split-package operator flow.

**Internal published packages**

- `@openclawbrain/cli@0.4.4`

**Changes**

- reinstall now normalizes `plugins.allow` and `plugins.entries.openclawbrain`
- status reports the canonical `openclawbrain` install identity

**Full release note**

- [docs/release-notes-0.4.4.md](docs/release-notes-0.4.4.md)

## 0.4.3

`0.4.3` improves operator status output and supervision matching.

**Internal published packages**

- `@openclawbrain/cli@0.4.3`

**Changes**

- embedding status handles alternate numeric vector shapes more accurately
- serve-time decision matching is more tolerant of event-id and timestamp drift

**Full release note**

- [docs/release-notes-0.4.3.md](docs/release-notes-0.4.3.md)

## 0.4.2

`0.4.2` closes the remaining high-signal status and tarball seams left after `0.4.1`.

**Internal published packages**

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

**Internal published packages**

- `@openclawbrain/openclaw@0.4.0`
- `@openclawbrain/cli@0.4.1`

**Changes**

- rerunning `install --shared` against an already pinned native package plugin now succeeds as a no-op
- the installer still fails when the installed loader entry does not expose a patchable `ACTIVATION_ROOT` constant

**Full release note**

- [docs/release-notes-0.4.1.md](docs/release-notes-0.4.1.md)

## 0.4.0

`0.4.0` is the split-package public release.

**Internal published packages**

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

**Internal published packages**

- `@jonathangu/openclawbrain@0.3.5`

**Changes**

- hardened prompt assembly when `before_prompt_build` carries empty or partial text envelopes
- improved teacher-status reporting for fresh no-op cycles
- restored the `packages/openclaw` front-door tree in the public repo

**Full release note**

- [docs/release-notes-0.3.5.md](docs/release-notes-0.3.5.md)

## 0.3.4

`0.3.4` removes fake supervision caused by runtime scaffolding.

**Internal published packages**

- `@jonathangu/openclawbrain@0.3.4`

**Changes**

- filters heartbeat prompts, startup/reset scaffolding, and metadata wrappers out of human supervision evidence
- adds focused exclusion and inclusion tests around teacher-pollution cases

**Full release note**

- [docs/release-notes-0.3.4.md](docs/release-notes-0.3.4.md)

## 0.3.3

`0.3.3` fixes the child-worker boot path for launchd-style installs.

**Internal published packages**

- `@jonathangu/openclawbrain@0.3.3`

**Changes**

- resolves the child worker loader to an absolute `file://` import
- launches the child from the plugin root instead of relying on cwd-sensitive module resolution
- adds a focused regression test for `cwd=/` worker launch

**Full release note**

- [docs/release-notes-0.3.3.md](docs/release-notes-0.3.3.md)

## 0.3.2

`0.3.2` introduced the summary-aware routing prior and explicit correction commit path.

**Internal published packages**

- `@jonathangu/openclawbrain@0.3.2`

**Changes**

- summaries now act as a routing prior over history instead of the durable correction layer
- explicit user corrections can commit through a dedicated runtime path
- architecture notes were added for [docs/architecture/routing-prior.md](docs/architecture/routing-prior.md) and [docs/architecture/corrections.md](docs/architecture/corrections.md)

**Full release note**

- [docs/release-notes-0.3.2.md](docs/release-notes-0.3.2.md)

## 0.3.0

`0.3.0` captured the combined-package release state before the later architecture and packaging work.

**Internal published packages**

- `@jonathangu/openclawbrain@0.3.0`

**Changes**

- added Anthropic OAuth setup-token support in the TUI
- resolved SecretRef-backed auth-profile credentials during summarization
- switched LCM tool timestamps to the local timezone

**Full release note**

- `0.3.0` predates the later release-note archive; see the historical notes in this repo if you need more detail
