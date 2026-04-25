# Changelog

Release history for the published OpenClawBrain releases. The README and operator docs lead with the current public front door; release notes carry version-specific detail. Internal split-package versions are maintainer detail and appear under each entry's **Internal published packages** label.

## Unreleased

## 0.4.48

Release notes: [docs/release-notes-0.4.48.md](docs/release-notes-0.4.48.md)

- Strengthen cold-start candidate-artifact replay selection with bounded multi-select and a deterministic cold-start scorecard.
- Add learned-route activation-usefulness accounting for beneficial wins, harmful activations, neutral ties, missed opportunities, and proxy cost deltas.
- Add a narrow shadow-only Teacher v3 graph-maintenance proposal lifecycle with durable replay and rollback evidence.

## 0.4.47

`0.4.47` is the explicit-preference precedence release: it packages the post-`0.4.46` fix that keeps the newest durable preference current when an older versioned tool/model preference would otherwise still retrieve.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.47`
- `@openclawbrain/cli@0.4.47`

**Changes**

- treats newer explicit preferences for the same durable subject as replacements instead of sibling current memories
- recognizes explicit rejection of an older preference value as deterministic supersession, including versioned model/tool choices such as `Codex GPT-5.4` → `Codex GPT-5.5`
- serves only the current explicit preference for normal current-truth queries after supersession
- documents the correction/routing precedence rule that newer explicit preferences supersede older values in the same durable slot
- adds regression coverage for the Codex model preference handoff and superseded-node exclusion
- aligns README, docs, release notes, and split-package version surfaces to `0.4.47`

**Full release note**

- [docs/release-notes-0.4.47.md](docs/release-notes-0.4.47.md)

## 0.4.46

`0.4.46` is the bounded selective-intervention proof release: it keeps current-choice fidelity protected, broadens specificity/restraint proof, cleans the exercised one-home operator proof, and adds the first route-level tool-capability choice proof without claiming broad memory or live weather execution.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.46`
- `@openclawbrain/cli@0.4.46`

**Changes**

- keeps the later-preference/current-choice lane at `full-ocb 5/5`, with `regret=0` and `harm=0`
- expands the specificity/restraint cohort to `full-ocb 12/12`, with `regret=0` and `harm=0`
- accepts current gateway-health wording so the one-home operator proof is clean: `success_and_proven`, severity `none`, warnings `0`
- adds the bounded `weather.current_conditions` capability-choice lane with paired must-fire and must-not-fire route-level proof
- restores the tiny checked cold-start router sample fixture required by the release verification suite
- aligns README, docs, release notes, and split-package version surfaces to `0.4.46`

**Full release note**

- [docs/release-notes-0.4.46.md](docs/release-notes-0.4.46.md)

## 0.4.45

`0.4.45` is the status hot-path release: it keeps the shipped proof and hardening story from `0.4.44`, but makes plain human `openclawbrain status` cheaper, calmer, and more trustworthy on real hosts.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.45`
- `@openclawbrain/cli@0.4.45`

**Changes**

- reduces status read amplification so plain summary status stops eagerly loading detailed-only teacher snapshot and event-export surfaces
- keeps summary status on cheap local truth by skipping active-pack embedding inspection and synchronous Ollama probing unless the operator asks for `--detailed`
- preserves the public operator contract by keeping the lighter summary-path detail selection internal instead of widening the API
- aligns README, docs, release notes, and split-package version surfaces to `0.4.45`

**Full release note**

- [docs/release-notes-0.4.45.md](docs/release-notes-0.4.45.md)

## 0.4.44

`0.4.44` is the post-win proof + hardening release: it packages the binary-gate promotion hardening, runtime-mode proof surfacing, and safe report-only teacher proposal lane into one public OpenClawBrain release.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.44`
- `@openclawbrain/cli@0.4.44`

**Changes**

- hardens binary-gate promotion flow with automatic threshold selection and cleaned merged trap truth
- exposes route-decision summaries and compact runtime-mode proof tables on the shipped proof surfaces
- adds a report-only teacher proposal artifact lane with evidence refs, replay hooks, proof linkage, rollback linkage, and markdown rendering
- aligns README, docs, release notes, and split-package version surfaces to `0.4.44`

**Full release note**

- [docs/release-notes-0.4.44.md](docs/release-notes-0.4.44.md)

## 0.4.43

`0.4.43` is the OCB continuation + compaction release: it ships the cold-start continuation and explainable eval tranche, then layers in budgeted routing, compact-health metrics, and retry-visible identity handoff for compaction-safe downstream behavior.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.43`
- `@openclawbrain/cli@0.4.43`

**Changes**

- ships the cold-start continuation runtime truth: same-family warm-start retrain, clearer base-vs-delta status/proof surfaces, and the converged explainable eval/reporting lane
- adds budgeted routing fit metrics plus compact-health scorecard metrics so compaction pressure and retrieval quality become visible instead of implicit
- hardens retry-visible identity handoff through trace, route-row, and observation surfaces so downstream OpenClaw-side dedupe has stable material to bind against
- aligns README, docs, release notes, and split-package version surfaces to `0.4.43`

**Full release note**

- [docs/release-notes-0.4.43.md](docs/release-notes-0.4.43.md)

## 0.4.42

`0.4.42` is the install self-heal release: it keeps the `0.4.41` cold-start lineage visibility and operator/proof lane, and it fixes the remaining stale nested-duplicate install seam where a real home could fail converge because a second extension copy lived under a nested OpenClaw home.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.42`
- `@openclawbrain/cli@0.4.42`

**Changes**

- auto-detects and quarantines stale nested duplicate OpenClawBrain installs before plugin-manager converge
- lets the canonical `install -> gateway restart -> status --detailed -> proof` lane repair that duplicate-surface class instead of failing halfway through
- keeps the `0.4.41` cold-start lineage visibility and operator/proof story intact
- aligns README, docs, release notes, and split-package version surfaces to `0.4.42`

**Full release note**

- [docs/release-notes-0.4.42.md](docs/release-notes-0.4.42.md)

## 0.4.41

`0.4.41` is the cold-start lineage visibility release: it keeps the `0.4.40` install-convergence and operator/proof lane, and it makes the learned-route inheritance from the approved cold-start prior visible again on the shipped status/proof surfaces.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.41`
- `@openclawbrain/cli@0.4.41`

**Changes**

- makes the packaged traced-learning bridge derive `retrainLineage` from durable promotion truth when the persisted status surface is thin
- makes `status --detailed` and `proof` report visible learned-route lineage for the active pack instead of falling back to `retrain_lineage_not_visible`
- keeps fresh homes on the stronger cold-start prior and keeps retrains rooted in the same live `route_fn` family while making that inheritance legible to operators
- aligns README, docs, release notes, and split-package version surfaces to `0.4.41`

**Full release note**

- [docs/release-notes-0.4.41.md](docs/release-notes-0.4.41.md)

## 0.4.40

`0.4.40` is the unified operator-truth / proof release: it keeps the `0.4.39` install-convergence lane, and it packages the just-landed bounded-anytime, economics, route-quality, and provenance surfaces into the public split-package release.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.40`
- `@openclawbrain/cli@0.4.40`

**Changes**

- makes `status --detailed` surface the bounded-anytime summary and route-quality summary as first-class operator truth
- makes proof-cron health/nightly outputs publish bounded economics scorecards with explicit measured / derived / proxy labels
- makes `openclawbrain proof --openclaw-home ...` capture the provenance audit chain alongside the existing proof bundle
- aligns README, docs, release notes, and split-package version surfaces to `0.4.40`

**Full release note**

- [docs/release-notes-0.4.40.md](docs/release-notes-0.4.40.md)

## 0.4.39

`0.4.39` is the install-convergence release: it keeps the shipped continuous-learning/operator story from `0.4.38`, but it fixes the real upgrade seam where `install` could preserve a stale installed hook/plugin record even though the daemon/runtime side had already moved ahead.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.39`
- `@openclawbrain/cli@0.4.39`

**Changes**

- makes `openclawbrain install --openclaw-home ...` refresh the authoritative native plugin state when the installed hook package version lags the daemon/runtime surface for that same home
- keeps half-converged daemon-vs-installed-hook states as explicit blocking truth instead of a fake-success install story
- updates README/docs/site install language so the public repair story says plainly that the same four-command lane now repairs stale-hook skew too
- aligns repo docs and public version surfaces to `0.4.39`

**Full release note**

- [docs/release-notes-0.4.39.md](docs/release-notes-0.4.39.md)

## 0.4.38

`0.4.38` is the install-hotfix follow-up to the ongoing-learning release: it keeps the shipped continuous-learning/operator story from `0.4.37` and fixes the published CLI packaging seam so the public install lane works on a real host.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.38`
- `@openclawbrain/cli@0.4.38`

**Changes**

- fixes the published `@openclawbrain/cli` package so Graphify/import-export surfaces no longer reach outside the installed package for `openclawbrain-contracts.js`
- hardens the CLI tarball verify step by importing the packed `dist/src/import-export.js` surface directly from the tarball extraction path
- keeps the `0.4.37` continuous-learning loop, operator controls, replay/eval hardening, and public product story intact while restoring the real host install lane
- aligns repo docs and public version surfaces to `0.4.38`

**Full release note**

- [docs/release-notes-0.4.38.md](docs/release-notes-0.4.38.md)

## 0.4.37

`0.4.37` is the continuous ongoing learning release: the first bounded ongoing-learning loop, its operator controls, and the final replay/eval verification hardening now ship as one public release.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.37`
- `@openclawbrain/cli@0.4.37`

**Changes**

- ships the first bounded continuous-learning loop: route rows, direct online supervision, Graphify delta/reorg scheduler registry, and periodic same-family retrain
- adds operator-facing continuous-learning status/control surfaces for Graphify cadence, retrain/promotion visibility, queue visibility, and pause controls
- hardens the release path so repo-wide `tsc` passes and the recorded-session replay/eval acceptance bundle is green on the shipped path
- aligns repo docs, release notes, public site, and Jon badge to `0.4.37`

**Full release note**

- [docs/release-notes-0.4.37.md](docs/release-notes-0.4.37.md)

## 0.4.36

`0.4.36` is the Graphify + hardening follow-up release: Graphify stays off-path, exact pinning stays locked, and the release truth now reflects the final follow-up publish.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.36`
- `@openclawbrain/cli@0.4.36`

**Changes**

- keeps Graphify as an artifact-first compiler / diagnostic lane instead of a hot-path dependency
- preserves the hardened dependency-policy guard and exact pinning posture for the split packages
- aligns repo docs, release notes, public site, and jon badge to `0.4.36`
- repins the OpenClaw peer to `2026.4.5` in the release follow-up lane

**Full release note**

- [docs/release-notes-0.4.36.md](docs/release-notes-0.4.36.md)

## 0.4.35

`0.4.35` is the tool-action symmetry release: trace, learning, and runtime scoring now distinguish generic tool capabilities from concrete tool-instance bindings so the router can learn and serve the actual tool action it took.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.35`
- `@openclawbrain/cli@0.4.35`

**Changes**

- splits tool capability vs bound tool-instance decision traces so replay/debug surfaces can tell the abstract action family from the concrete tool that was actually chosen
- updates seed-phase learning to write tool-action priors from chosen toolcard traversals instead of leaving concrete tool-action reinforcement lagging behind the traced decision
- scores explicit tool-instance bindings above generic capability matches when both are present so runtime retrieval prefers the real bound tool action
- keeps the route-function action family more coherent across trace, learn, and serve surfaces without changing the public install lane

**Full release note**

- [docs/release-notes-0.4.35.md](docs/release-notes-0.4.35.md)

## 0.4.34

`0.4.34` is the install-hardening release: it fixes the `0.4.33` activation-root self-heal / repin seam that could rewrite the installed hook into invalid JavaScript and break load on the real gateway surface.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.34`
- `@openclawbrain/cli@0.4.34`

**Changes**

- rewrites only the real `ACTIVATION_ROOT` constant line during activation-root repin / self-heal instead of risking corruption of the helper regex line
- hardens generated shadow-extension templating so the install lane fails closed if the patchable activation-root line is missing
- keeps repeated register/self-heal calls from corrupting the installed hook source in one process
- pins plugin-manager converge actions to the selected `OPENCLAW_HOME` target so install/update operations stop drifting onto the ambient default home
- closes the real install bug tracked in issue `#19`

**Full release note**

- [docs/release-notes-0.4.34.md](docs/release-notes-0.4.34.md)

## 0.4.33

`0.4.33` is the cold-start prior default-on release. New homes now start from the learned prior by default, existing homes rebuild that stronger generic prior underneath the user layer they already earned, and the public install / upgrade / proof story now says that plainly.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.33`
- `@openclawbrain/cli@0.4.33`

**Changes**

- makes the learned serve-time `v2` prior the default path instead of falling back to the older heuristic/v1 seam on fresh installs
- preserves existing-user baseline state on upgrade so learned preferences, corrections, and recent overlay stay on top of the new base prior
- ships the broader governed cold-prior tranche and its repaired `STOP_LOCAL` behavior as the default story for new installs
- aligns the repo docs, site copy, and upgrade language around one clear story: new users start on the cold-start prior; existing users keep their earned preferences on top
- removes the old compatibility-note detour from the active public lane

**Full release note**

- [docs/release-notes-0.4.33.md](docs/release-notes-0.4.33.md)

## 0.4.32

`0.4.32` is the trust-surfaces hardening release: it removes avoidable no-op churn from converge and traced-learning persistence, and it stops status surfaces from underreporting a clearly serving promoted pack.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.32`
- `@openclawbrain/cli@0.4.32`

**Changes**

- treats an already-authoritative native OpenClawBrain plugin install as a converge no-op instead of blindly refreshing plugin-manager state
- upgrades traced-learning/status pack truth from active-pack plus watch-snapshot evidence when stale bridge state says `materializedPackId=null` / `promoted=false`
- avoids rewriting traced-learning persisted status summaries when the normalized payload is unchanged
- keeps the release repo-only and trust-surface focused so operator truth improves without reintroducing live-host churn

**Full release note**

- [docs/release-notes-0.4.32.md](docs/release-notes-0.4.32.md)

## 0.4.31

`0.4.31` is the proof-surface closure release: the canonical operator lane now preserves the installed-hook package version through current-profile status/proof reporting, so a healthy shared host can prove concrete daemon-vs-hook same-version convergence instead of stopping at `split_path_version_unverified`.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.31`
- `@openclawbrain/cli@0.4.31`

**Changes**

- preserves installed-hook `packageVersion` through the current-profile status/report surface instead of dropping it before hotfix-boundary formatting
- makes detailed status and proof bundles report the installed hook as `@openclawbrain/openclaw@<version>` when the on-disk package/manifest already proves it
- closes the final host-side daemon-vs-hook proof seam from issue `#16`, turning the shared-host lane from `success_but_proof_incomplete` into fully proven convergence
- keeps restart/profile-token inference warnings honest without letting them blur actual brain-health truth

**Full release note**

- [docs/release-notes-0.4.31.md](docs/release-notes-0.4.31.md)

## 0.4.30

`0.4.30` is the runtime-surface convergence release: the canonical operator lane now makes daemon-vs-hook skew explicit, blocks half-converged install/runtime states more honestly, and treats explicit custom `--openclaw-home` paths as first-class in docs/help/examples.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.30`
- `@openclawbrain/cli@0.4.30`

**Changes**

- surfaces daemon-vs-installed-hook/runtime-guard split-runtime skew more explicitly in status/proof truth
- blocks half-converged daemon vs installed-hook states in converge/proof verification instead of reading like success
- makes explicit custom OpenClaw homes like `./openclaw-cormorantai` first-class in docs/help/examples through the canonical `--openclaw-home` path
- preserves the one-version / one-install-lane public contract

**Full release note**

- [docs/release-notes-0.4.30.md](docs/release-notes-0.4.30.md)

## 0.4.29

`0.4.29` is the compatibility release that makes **Gemma 4 31B** a first-class compatible OpenClawBrain teacher on the canonical local Ollama install lane, while preserving the single public version / single install path story.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.29`
- `@openclawbrain/cli@0.4.29`

**Changes**

- adds Gemma 4 teacher compatibility to the local-teacher autodetect/install path
- keeps the latest Teacher v3 runtime/proof/replay/canary/worked-example surfaces from `0.4.28`
- closes the host seam where Gemma could be configured manually but not selected as a first-class compatible local teacher

**Full release note**

- [docs/release-notes-0.4.29.md](docs/release-notes-0.4.29.md)

## 0.4.28

`0.4.28` is the Teacher v3 public release: the substrate, persistence, proof-bundle emission, replay over real candidate state, bounded canary discipline, and honest worked-example/public-proof surfaces are now shipped together under one public version and one install lane.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.28`
- `@openclawbrain/cli@0.4.28`

**Changes**

- lands durable Teacher v3 proposal persistence and runtime-emitted proof bundles
- lands replay over real candidate state for promotable and shadow-only proposal classes
- lands bounded canary discipline that is rollback-bound, operator-visible, and off by default
- publishes honest worked examples plus repo/site proof packaging that keep shipped vs target-state boundaries explicit
- preserves the one-version / one-install-lane public contract

**Full release note**

- [docs/release-notes-0.4.28.md](docs/release-notes-0.4.28.md)

## 0.4.27

`0.4.27` is the closed-loop repair release: the historical serve-decision recovery path is repaired, learner/materialization inputs are compacted so the loop can progress without hauling oversized raw payloads, operator status surfaces are more honest about learning progress and runtime truth, and the live host now proves nonzero supervision/router updates with a promoted pack.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.27`
- `@openclawbrain/cli@0.4.27`

**Changes**

- repairs the closed learning loop so historical serve decisions can still be recovered for supervision even after they fall out of the bounded tail
- admits `message_delivered` interactions into the teacher labeler so real user feedback is no longer starved out by one interaction-shape seam
- compacts serve-decision / learner-materialization inputs and clarifies operator health/runtime-truth surfaces instead of hiding split-surface uncertainty
- proves the repair live on host with nonzero supervision/router updates and promoted pack `pack-213597b7`
- preserves the one-version / one-install-lane public contract

**Full release note**

- [docs/release-notes-0.4.27.md](docs/release-notes-0.4.27.md)

## 0.4.26

`0.4.26` is the post-0.4.25 hardening release: proof capture defaults now avoid repo dirt, teacher status reuses one live operator snapshot instead of double-sampling watch state, and canonical `feedback` / `attrCover` surfaces now expose real on-disk supervision/queue truth on split-package hosts instead of falsely reading all-zero.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.26`
- `@openclawbrain/cli@0.4.26`

**Changes**

- routes default operator-proof artifacts to the shared workspace sibling instead of dirtying repo-root `artifacts/operator-proof-*`
- tightens ignore coverage for generated proof/runtime scratch paths
- fixes the detailed-status teacher/watch seam by reusing one shared operator snapshot per command
- surfaces historical active-pack supervision and live sparse-feedback queue truth when legacy Brain tables are empty on split-package hosts
- preserves the one-version / one-install-lane public contract

**Full release note**

- [docs/release-notes-0.4.26.md](docs/release-notes-0.4.26.md)

## 0.4.25

`0.4.25` is the public OpenClawBrain proof/operator follow-up release: the default proof lane is hardened so the canonical `openclawbrain proof --openclaw-home ...` path completes cleanly again, and canonical status/proof surfaces now expose thin feedback and attribution-coverage truth directly.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.25`
- `@openclawbrain/cli@0.4.25`

**Changes**

- hardens the default proof lane by skipping redundant restart choreography when install already handled restart or explicitly reported restart not required
- keeps proof closed over real runtime failures while letting healthy runtime truth complete to full proof artifacts without the earlier skip-flag ritual
- adds canonical `feedback` and `attrCover` lines to `openclawbrain status --detailed`
- adds a thin operator readout plus thin proof-truth surfaces for feedback, attribution coverage, and replay freshness
- preserves the one-version / one-install-lane public contract

**Full release note**

- [docs/release-notes-0.4.25.md](docs/release-notes-0.4.25.md)

## 0.4.24

`0.4.24` is the public repair release that restores the one-version OpenClawBrain contract after the accidental mixed `0.4.23` split-package publish. It keeps the single-source-of-truth operator fixes and realigns both public package surfaces back onto the same visible version.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.24`
- `@openclawbrain/cli@0.4.24`

**Changes**

- preserves the watcher freshness/operator-health/context-management fixes from the single-source-of-truth tranche
- repairs the publish contract so users again see one visible OpenClawBrain version instead of a mixed CLI/runtime pair
- keeps the canonical install lane unchanged: `openclawbrain install --openclaw-home ...`

**Full release note**

- [docs/release-notes-0.4.24.md](docs/release-notes-0.4.24.md)

## 0.4.23

`0.4.23` is the public OpenClawBrain follow-up release for the single-source-of-truth tranche: the CLI now reports watcher freshness, operator health, and context-management truth from one cleaner operator model instead of leaving those surfaces fragmented across status, proof, and stale docs.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.22`
- `@openclawbrain/cli@0.4.23`

**Changes**

- watcher freshness now reports a structured `lagging` state so near-threshold heartbeat jitter does not flip straight to `stale_snapshot`
- proof health and nightly aggregates now consume a shared `operatorHealth` contract with explicit `partial`, `unknown`, `stale`, and `unhealthy` semantics
- `openclawbrain status` now exposes a canonical `contextManagement` model covering summary spine + protected fresh tail, freshness states, prefetch lifecycle, expand-to-source behavior, and budget controls
- stale docs/operator seams are cleaned up, including the phantom `openclawbrain context` command claim

**Full release note**

- [docs/release-notes-0.4.23.md](docs/release-notes-0.4.23.md)

## 0.4.22

`0.4.22` is the public OpenClawBrain release for the post-issue-#7 async watch/teacher reliability fixes. It closes the wedge where 0.4.21 could install cleanly, attach cleanly, and still prove green while the passive watch/teacher loop was unhealthy. It also realigns the split packages onto the same published version number so the public product story stays one version, not two competing live numbers.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.22`
- `@openclawbrain/cli@0.4.22`

**Changes**

- startup replay now skips the historical export rewalk when restored teacher state already knows the seen export digests
- no-op always-on learner cycles stop rebuilding the runtime graph when `selectedSlices=0`
- session-tail cursor state now self-heals: missing path/file sessions stabilize and stale cursor entries are pruned instead of churning forever
- session parsing now accepts top-level `compaction` records and string-valued `message.content` payloads from current real session shapes
- install-time local teacher autodetect now prefers larger compatible Qwen lanes instead of biasing toward smaller models
- internal split-package publish numbers are realigned to the same release number so the public OpenClawBrain version stays singular

**Full release note**

- [docs/release-notes-0.4.22.md](docs/release-notes-0.4.22.md)

## 0.4.21

`0.4.21` is the public OpenClawBrain release that makes the single-product story truer in both behavior and operator proof: legacy compatibility seams now fail closed or self-correct, compatibility migration onto the canonical lane is more coherent, generated shadow extension deps are correct, and proof stops degrading the repaired target profile for the wrong reason.

**Internal published packages**

- `@openclawbrain/openclaw@0.4.6`
- `@openclawbrain/cli@0.4.21`

**Changes**

- retired compatibility binary now fails closed and points operators back to the canonical install path
- install/daemon guardrails now detect and refresh stale legacy compatibility runtime seams instead of treating them as healthy
- compatibility migration onto the canonical plugin lane now replaces the wrong install seam instead of drifting into it
- generated shadow extension package metadata now resolves the real runtime dependency correctly
- proof accepts generated shadow hook sources and no longer degrades the target repaired profile just because unrelated attached profiles are only partially covered
- first-read public docs now keep the product story on one install path and treat manual/plugin surgery as maintainer-only background detail

**Full release note**

- [docs/release-notes-0.4.21.md](docs/release-notes-0.4.21.md)

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
