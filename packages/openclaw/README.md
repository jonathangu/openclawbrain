# @openclawbrain/openclaw

OpenClaw integration helpers for promoted-pack compile consumption and normalized event emission.

This is the default front door for first-time OpenClawBrain users.

- Start here when you want one package to import from instead of guessing across the `@openclawbrain/*` split.
- The stable front-door names are `@openclawbrain/openclaw` and `openclawbrain`.
- If you are validating this repo-tip checkout, prepare the checkout with `pnpm install --frozen-lockfile`, run `pnpm release:status`, then `pnpm release:pack`, then install the local `.release/*.tgz` tarballs.
- For the current published wave, install it with `npm install -g @openclawbrain/openclaw@0.3.5`.
- Use the `openclawbrain` CLI for status, rollback, and narrow scanner scans; `openclawbrain-ops` stays available as a compatibility alias.
- `npm exec openclawbrain -- --help` works after either install path, but the exact quickstart commands remain the clearest first attach path.

The compatibility package `@jonathangu/openclawbrain@0.3.5` remains published for older plugin/wrapper installs, but it is not the main operator path.

Decision and migration note: [`docs/lifecycle.md`](../../docs/lifecycle.md)

## Operator quickstart

Install or upgrade through the same lane:

```bash
npm install -g @openclawbrain/openclaw@0.3.5
openclawbrain install --openclaw-home ~/.openclaw
openclaw gateway restart
openclawbrain status --openclaw-home ~/.openclaw --detailed
```

Verify the installed target:

```bash
openclawbrain status --openclaw-home ~/.openclaw --detailed
openclawbrain status --openclaw-home ~/.openclaw --json
```

Remove only the OpenClaw profile hook and keep brain data:

```bash
openclawbrain detach --openclaw-home ~/.openclaw
openclaw gateway restart
```

Remove the hook and purge the OpenClawBrain data for that install:

```bash
openclawbrain uninstall --openclaw-home ~/.openclaw --purge-data
openclaw gateway restart
npm uninstall -g @openclawbrain/openclaw
```

If you want explicit uninstall semantics without purging, use `openclawbrain uninstall --openclaw-home ~/.openclaw --keep-data`. `detach` remains the simpler keep-data path.

Use this package when OpenClaw needs the narrow supported operator bridge captured by `OPERATOR_API_CONTRACT_V1`:

- bootstrap an initial attach pack from a normalized export, or truthfully self-boot from zero events, and activate it
- inspect attach status and canonical current-profile operator truth
- emit deterministic normalized event-export bundles for learner handoff
- refresh learner state into candidate packs through the learner boundary
- stage/promote/rollback activation pointers explicitly
- prove local activation, freshness, learned-route, and kernel-vs-brain boundaries through observability surfaces

The supported operator contract is intentionally decomposed into bootstrap/attach, status, export, refresh, promote, rollback, and proof/observability. The post-attach loop stays off the hot path: it exports turns, builds candidate packs, and promotes them later rather than mutating the current active pack in place.

Use `docs/lifecycle.md` for the copy-paste install/upgrade/verify/remove story. The rest of this README is the family-by-family contract map and the explicitly quarantined surfaces.

## Operator boundary

This package keeps the runtime boundary narrow:

- OpenClaw remains the runtime owner
- OpenClawBrain owns the attach / export / proof / activation helpers behind that boundary
- the supported status read is the canonical current-profile answer rather than a machine-global brain report

For first install, the boring path is intentionally simple: pass `interactionEvents: []` / `feedbackEvents: []` when you have no live turns yet, or hand `bootstrapRuntimeAttach()` a zero-event `normalizedEventExport` payload and let it canonicalize the derived range/provenance fields from the embedded event arrays. The resulting attach stays truthful with `awaiting_first_export` until the first live export lands; truly invalid non-empty exports fail with attach-specific remediation instead of a raw provenance validator surprise.

```ts
import {
  OPERATOR_API_CONTRACT_V1,
  bootstrapRuntimeAttach,
  describeAttachStatus,
  describeCurrentProfileBrainStatus,
  formatBootstrapRuntimeAttachReport,
  rollbackRuntimeAttach
} from "@openclawbrain/openclaw";

console.log(OPERATOR_API_CONTRACT_V1.families);

const attach = bootstrapRuntimeAttach({
  profileSelector: "current_profile",
  activationRoot: "/var/openclawbrain/activation",
  packRoot: "/var/openclawbrain/packs/bootstrap",
  packLabel: "customer-launch-attach",
  workspace: {
    workspaceId: "workspace-1",
    snapshotId: "workspace-1@snapshot-1",
    capturedAt: "2026-03-07T16:59:00.000Z",
    rootDir: "/workspace/openclawbrain",
    revision: "attach-rev-1"
  },
  interactionEvents: [],
  feedbackEvents: []
});

console.log(formatBootstrapRuntimeAttachReport(attach));
console.log(attach.currentProfile.profile.selector);
console.log(attach.nextSteps[0]?.command);

const attachStatus = describeAttachStatus({
  activationRoot: attach.activationRoot
});

console.log(attachStatus.inspection.active?.packId);
console.log(attachStatus.compile?.handoffState);
console.log(attachStatus.landingBoundaries.compileBoundary.activationSlot);
console.log(attachStatus.landingBoundaries.failOpenSemantics.eventExportWriteFailurePreservesCompile);

const currentProfile = describeCurrentProfileBrainStatus({
  activationRoot: attach.activationRoot,
  brainAttachmentPolicy: "dedicated"
});

console.log(currentProfile.profile.selector);
console.log(currentProfile.attachment.policyMode);
console.log(currentProfile.brainStatus.serveState);

const rollbackPreview = rollbackRuntimeAttach({
  activationRoot: "/var/openclawbrain/activation",
  dryRun: true
});

console.log(rollbackPreview.allowed);
console.log(rollbackPreview.before.activePackId);
console.log(rollbackPreview.after?.activePackId);

```

This package stays fail-open for non-learned-required compile misses, and event-export write failures do not erase successful compile output.

## CLI lifecycle quickstart

Keep `install` as the front-door lifecycle command:

- `openclawbrain install --openclaw-home <path>` is the default attach path. On many-profile hosts, pass `--openclaw-home <path>` or set `OPENCLAW_HOME` so the target profile is explicit instead of guessed.
- `openclawbrain install` now writes only the real profile hook and activation state. It no longer creates `BRAIN.md` or rewrites `AGENTS.md`.
- `openclawbrain detach --openclaw-home <path>` removes only the OpenClaw profile hook. It preserves OpenClawBrain activation data and now says so plainly in both human output and JSON.
- `openclawbrain uninstall --openclaw-home <path> --keep-data|--purge-data` removes the profile hook and forces the operator to pick the data outcome explicitly.
- `--restart <never|safe|external>` is guidance only for detach/uninstall. The default `safe` guidance stays conservative: restart the running OpenClaw profile before expecting hook-state changes to take effect; otherwise the next launch picks them up.
- plain `status` stays the human answer to “How’s the brain?” and now keeps compact lifecycle truth visible without dumping teacher/no-op or other proof chatter by default; use `--openclaw-home <path> --detailed` when you want the dense operator report for one installed target.

## Narrow operator contract

Treat these as the supported operator families for Wave 1:

- bootstrap / attach — `bootstrapRuntimeAttach()`, `formatBootstrapRuntimeAttachReport()`, `describeAttachStatus()`
- status / rollback — `openclawbrain ...`, `describeCurrentProfileBrainStatus()`, `formatOperatorRollbackReport()`
- export — `buildNormalizedRuntimeEventExport()`, `writeRuntimeEventExportBundle()`, `loadRuntimeEventExportBundle()`
- proof / observability — `describeAttachStatus()`, `describeKernelBrainBoundary()`, `describeActivationObservability()`

The refresh and promote families stay explicit in their owning packages instead of being hidden behind a catch-all convenience wrapper:

- refresh — `@openclawbrain/learner.createAlwaysOnLearningRuntimeState()`, `advanceAlwaysOnLearningRuntime()`, `materializeAlwaysOnLearningCandidatePack()`
- promote — `@openclawbrain/pack-format.stageCandidatePack()`, `promoteCandidatePack()`

Wrong-shaped surfaces are intentionally quarantined from this contract even if they remain exported elsewhere for proof or integration work:

- `openclawbrain doctor`
- `openclawbrain-ops doctor`
- `buildOperatorSurfaceReport()`
- `formatOperatorStatusReport()`
- `formatOperatorDoctorReport()`
- `runContinuousProductLoopTurn()`
- `runRecordedSessionReplay()`
- `runRuntimeTurn()`
- `createAsyncTeacherLiveLoop()`

For one repo-local end-to-end proof of the full attach -> status -> run turn -> export -> refresh -> promote -> status story, run `pnpm current-profile-lifecycle:smoke` from the workspace root. That harness prints the before/after canonical current-profile answer, the matching compact `status` summary plus `status --json` proof built from those same snapshots, and the active graph/router/objective/weights/freshness checksum changes that became true only after promotion.

For the richer single-profile learning story on that same boundary, run `pnpm current-profile-learning-proof`. That checked-in proof shows one current profile serving an initial turn, changing later served context only after promotion, then improving again after a correction export -> refresh -> promote cycle without pretending that the active pack mutated in place.

## Boundary truth

`describeAttachStatus()` now returns `landingBoundaries`, a compact runtime-facing statement of the actual OpenClaw handoff:

- `compileBoundary`: OpenClaw compiles only from activation's `active` slot through `compileRuntimeContext()` / `runtime_compile.v1`
- `eventExportBoundary`: `runRuntimeTurn()` emits normalized interaction/feedback exports for later learning handoff, and bundle-write failure does not erase successful compile output
- `activePackBoundary`: `active` is the only serve-visible slot; `candidate` and `previous` stay inspectable until promotion or rollback changes pointers
- `promotionBoundary`: learner refresh lands in `candidate`, activation promotion flips pointers, and only later compiles can see the fresher pack
- `failOpenSemantics`: missing active packs fail open, but learned-required route-artifact drift hard-fails instead of silently serving static context
- `runtimeResponsibilities`: OpenClaw remains responsible for runtime orchestration, prompt assembly, response delivery, and guarded serve-path decisions

Operator-facing CLI:

```bash
pnpm exec openclawbrain status --activation-root /var/openclawbrain/activation --event-export /var/openclawbrain/exports/latest --teacher-snapshot /var/openclawbrain/runtime/teacher-snapshot.json
pnpm exec openclawbrain rollback --activation-root /var/openclawbrain/activation --dry-run
pnpm exec openclawbrain scan --session ./fixtures/recorded-session-replay/sanitized-support-escalation.trace.json --root /tmp/openclawbrain-scan-session
pnpm exec openclawbrain scan --live /var/openclawbrain/exports/latest --workspace /var/openclawbrain/runtime/workspace.json --snapshot-out /var/openclawbrain/runtime/scan-live-snapshot.json
```

- `openclawbrain-ops` remains available as the same CLI entrypoint if you already scripted against the older name
- `bootstrapRuntimeAttach()` resolves the activation root, stamps the attach as a `current_profile` action, returns the canonical current-profile answer immediately, and includes copy-paste-ready next steps for the attached root that prefer the primary `openclawbrain` CLI while matching the current `npm`/`pnpm` invocation style when it is detectable; those hints now cover `status`, canonical `status --json`, and `rollback --dry-run`
- `resolveActivationRoot()` keeps the existing explicit-path and global fallback chain, and same-gateway hosts with multiple `~/.openclaw-*` homes can now pin resolution to one installed OpenClaw profile home via `openclawHome` or `OPENCLAW_HOME`
- `formatBootstrapRuntimeAttachReport()` prints the shipped attach handoff as a current-profile action instead of dropping callers straight into lower-level proof fields
- plain `status` is the human answer to “How's the brain?” for the current profile on that activation root; it now keeps the principal backlog frontier (`learnedThrough -> newestPending`), live-vs-backfill passive-learning progress, explicit backlog warning states, structural-decision origin, and active structural-graph evolution visible in the compact text summary, while `status --json` still emits the canonical `current_profile_brain_status.v1` answer from that same boundary
- with `--teacher-snapshot`, that same human summary also keeps the learner `mode`, next lane, and exact scheduler `priority` bucket visible so operators can see live-first intake, principal-priority work, and passive backfill without pretending the canonical JSON object widened
- `describeCurrentProfileBrainStatus()` freezes the current-profile `Host / Profile / Brain / Attachment` answer shape instead of widening to a second report payload
- `rollback --dry-run` previews the exact `active <- previous` and `active -> candidate` move before you apply rollback for real
- `--event-export` accepts either a runtime event-export bundle root or a normalized-export JSON payload
- `--teacher-snapshot` accepts JSON from `teacherLoop.snapshot()` or `await teacherLoop.flush()` when you want the last duplicate/no-op reason plus learner bootstrapped/mode/next-lane/scheduler-priority truth kept visible
- `scan --session` is a checked-in proof-facing replay over a sanitized recorded-session trace; it reports the no-brain vs seed-pack vs learned-replay score spread and does not pretend to be the runtime hot path
- `scan --live` is a one-shot live-export scan over a real event-export payload plus explicit workspace metadata; it truthfully reports teacher/learner state for that export and does not pretend a background daemon is running

Programmatic rollback is available too:

- `rollbackRuntimeAttach({ activationRoot, dryRun: true })` previews the move without changing activation pointers
- `rollbackRuntimeAttach({ activationRoot })` applies the rollback and reports which pack was restored to `active`

Wave 1 also freezes shared-vs-dedicated brain policy semantics:

- `brainAttachmentPolicy: "dedicated"` means one current profile is intentionally attached to one Brain activation root until operators reattach it elsewhere
- `brainAttachmentPolicy: "shared"` means multiple profiles may intentionally attach to the same Brain activation root, so attribution must stay current-profile explicit and operators must not read served context as profile-exclusive
- `brainAttachmentPolicy: "undeclared"` is the truthful fallback when the host has not declared which policy it is using

Per-turn normalized interaction and feedback events can carry the same current-profile attribution fields so later learner/export analysis can see `brainStatus`, `brainAttachmentPolicy`, pack/router identity, and selection evidence turn by turn.

The learned-required serve path is stricter:

- `compileRuntimeContext()` now forwards the active-slot diagnostics from `@openclawbrain/compiler.compileRuntimeFromActivation()`
- learned-required active packs return `hardRequirementViolated=true` and `fallbackToStaticContext=false` when route artifacts drift or disappear
- `runRuntimeTurn()` throws instead of silently serving through those learned-required failures
- event-export write failures still do not erase successful compile output
