import assert from "node:assert/strict";
import { rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  FIXTURE_FEEDBACK_EVENTS,
  FIXTURE_INTERACTION_EVENTS,
  FIXTURE_NORMALIZED_EVENT_EXPORT,
  createFeedbackEvent,
  createInteractionEvent
} from "@openclawbrain/contracts";
import {
  advanceAlwaysOnLearningRuntime,
  buildTeacherSupervisionArtifactsFromNormalizedEventExport,
  buildCandidatePack,
  buildCandidatePackBundleFromNormalizedEventExportBridge,
  buildCandidatePackFromNormalizedEventExport,
  buildCandidatePackFromNormalizedEventExportSlice,
  createAlwaysOnLearningRuntimeState,
  drainAlwaysOnLearningRuntime,
  materializeAlwaysOnLearningCandidatePack,
  materializeCandidatePack,
  materializeCandidatePackBundleFromNormalizedEventExportBridge,
  materializeCandidatePackFromNormalizedEventExportSlice,
  materializeCandidatePackFromNormalizedEventExport
} from "@openclawbrain/learner";
import { buildNormalizedEventExportBridge } from "@openclawbrain/event-export";

function buildRuntimeInteractionEvents(start: number, end: number) {
  return Array.from({ length: end - start + 1 }, (_, offset) => {
    const sequence = start + offset;
    const createdAt = new Date(Date.parse("2026-03-07T09:00:00.000Z") + offset * 60_000).toISOString();

    return createInteractionEvent({
      eventId: `evt-runtime-int-${sequence}`,
      agentId: "agent-runtime-batch",
      sessionId: "session-runtime-batch",
      channel: "whatsapp",
      sequence,
      kind: "message_delivered",
      createdAt,
      source: {
        runtimeOwner: "openclaw",
        stream: "openclaw/runtime/whatsapp"
      },
      messageId: `msg-runtime-${sequence}`
    });
  });
}

test("learner emits deterministic immutable pack manifests for always-on learning", (t) => {
  const input = {
    packLabel: "demo",
    workspace: {
      workspaceId: "workspace-1",
      snapshotId: "workspace-1@snapshot-1",
      capturedAt: "2026-03-06T00:00:00.000Z",
      rootDir: "/workspace/demo",
      branch: "main",
      revision: "demo-rev-1",
      labels: ["demo", "typescript"]
    },
    eventRange: {
      start: 101,
      end: 104
    },
    eventExports: {
      interactionEvents: FIXTURE_INTERACTION_EVENTS,
      feedbackEvents: FIXTURE_FEEDBACK_EVENTS
    },
    learnedRouting: true,
    structuralOps: {
      connect: 2,
      prune: 1
    }
  } as const;

  const first = buildCandidatePack(input);
  const second = buildCandidatePack(input);
  const rootDir = path.join(tmpdir(), `openclawbrain-ts-learn-${Date.now()}`);
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  const descriptor = materializeCandidatePack(rootDir, input);

  assert.equal(first.summary.packId, second.summary.packId);
  assert.equal(first.manifest.immutable, true);
  assert.equal(first.manifest.routePolicy, "requires_learned_routing");
  assert.equal(descriptor.manifest.packId, first.summary.packId);
  assert.equal(descriptor.router?.routerIdentity, `${first.summary.packId}:route_fn`);
  assert.equal(descriptor.graph.blocks.length, 12);
  assert.equal(first.summary.eventRange.start, 101);
  assert.equal(first.summary.eventRange.end, 104);
  assert.equal(first.summary.eventRange.count, 4);
  assert.equal(first.summary.workspaceSnapshot, "workspace-1@snapshot-1");
  assert.equal(first.summary.bootstrapping.fastBootDefaults, true);
  assert.equal(first.summary.bootstrapping.passiveBackgroundLearning, true);
  assert.equal(first.summary.learningSurface.bootProfile, "fast_boot_defaults");
  assert.equal(first.summary.learningSurface.labelHarvest.humanLabels, FIXTURE_FEEDBACK_EVENTS.length);
  assert.equal(first.manifest.provenance.workspace.snapshotId, "workspace-1@snapshot-1");
  assert.equal(first.summary.eventExportDigest, first.manifest.provenance.eventExports?.exportDigest ?? null);
  assert.equal(first.manifest.provenance.eventExports?.interactionCount, FIXTURE_INTERACTION_EVENTS.length);
  assert.equal(first.manifest.provenance.eventExports?.feedbackCount, FIXTURE_FEEDBACK_EVENTS.length);
  assert.match(descriptor.graph.blocks.map((block) => block.text).join("\n"), /Fast boot defaults stay live at startup/);
  assert.match(descriptor.graph.blocks.map((block) => block.text).join("\n"), /Human label harvest is first-class/);
  assert.equal(descriptor.graph.blocks.some((block) => block.source === "docs/openclaw-attach-quickstart.md"), true);
  assert.equal(descriptor.graph.blocks.some((block) => block.source === "docs/learning-first-convergence.md"), true);
  assert.equal(
    descriptor.graph.blocks.some((block) => block.source.includes("openclawbrain-openclaw-rearchitecture")),
    false
  );
  assert.equal(
    descriptor.graph.blocks.some((block) => block.source === "memory/2026-03-05-openclawbrain-vnext-roadmap.md"),
    false
  );
  assert.equal(descriptor.graph.blocks.some((block) => block.learning.role === "boot_default"), true);
  assert.equal(descriptor.graph.blocks.some((block) => block.learning.humanLabels > 0), true);
  assert.equal(descriptor.vectors.entries.some((entry) => entry.keywords.includes("fast_boot")), true);
});

test("learner rejects mismatched explicit event ranges when event exports are supplied", () => {
  assert.throws(
    () =>
      buildCandidatePack({
        packLabel: "bad-range",
        workspace: {
          workspaceId: "workspace-1",
          snapshotId: "workspace-1@snapshot-bad-range",
          capturedAt: "2026-03-06T00:00:00.000Z",
          rootDir: "/workspace/demo"
        },
        eventRange: {
          start: 100,
          end: 104
        },
        eventExports: {
          interactionEvents: FIXTURE_INTERACTION_EVENTS,
          feedbackEvents: FIXTURE_FEEDBACK_EVENTS
        },
        learnedRouting: false
      }),
    /does not match requested range/
  );
});

test("learner can materialize a candidate pack directly from a normalized event export", (t) => {
  const rootDir = path.join(tmpdir(), `openclawbrain-ts-export-${Date.now()}`);
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  const result = buildCandidatePackFromNormalizedEventExport({
    packLabel: "from-export",
    workspace: {
      workspaceId: "workspace-export",
      snapshotId: "workspace-export@snapshot-1",
      capturedAt: "2026-03-06T01:00:00.000Z",
      rootDir: "/workspace/export",
      revision: "workspace-export-rev"
    },
    normalizedEventExport: FIXTURE_NORMALIZED_EVENT_EXPORT,
    learnedRouting: true,
    structuralOps: {
      split: 1,
      connect: 1
    }
  });
  const descriptor = materializeCandidatePackFromNormalizedEventExport(rootDir, {
    packLabel: "from-export",
    workspace: {
      workspaceId: "workspace-export",
      snapshotId: "workspace-export@snapshot-1",
      capturedAt: "2026-03-06T01:00:00.000Z",
      rootDir: "/workspace/export",
      revision: "workspace-export-rev"
    },
    normalizedEventExport: FIXTURE_NORMALIZED_EVENT_EXPORT,
    learnedRouting: true,
    structuralOps: {
      split: 1,
      connect: 1
    }
  });

  assert.equal(result.summary.eventExportDigest, FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.exportDigest);
  assert.equal(result.summary.workspaceSnapshot, "workspace-export@snapshot-1");
  assert.equal(result.summary.learningSurface.labelHarvest.selfLabels, 1);
  assert.equal(descriptor.manifest.provenance.eventRange.firstEventId, "evt-interaction-fixture-1");
  assert.equal(descriptor.manifest.provenance.eventRange.lastEventId, "evt-feedback-fixture-2");
  assert.equal(descriptor.graph.blocks.some((block) => block.id.endsWith("evt-feedback-fixture-1")), true);
  assert.equal(descriptor.graph.blocks.some((block) => block.learning.role === "label_surface"), true);
  assert.match(descriptor.graph.blocks.map((block) => block.source).join("\n"), /openclaw\/runtime\/whatsapp:teaching/);
});

test("teacher supervision artifacts dedupe repeated exports and land in future candidate packs", () => {
  const teacherArtifacts = buildTeacherSupervisionArtifactsFromNormalizedEventExport({
    normalizedEventExport: FIXTURE_NORMALIZED_EVENT_EXPORT,
    observedAt: "2026-03-06T00:00:15.000Z"
  });
  const repeatedTeacherArtifacts = buildTeacherSupervisionArtifactsFromNormalizedEventExport({
    normalizedEventExport: FIXTURE_NORMALIZED_EVENT_EXPORT,
    observedAt: "2026-03-06T00:04:00.000Z"
  });

  assert.equal(teacherArtifacts.length, 2);
  assert.deepEqual(
    teacherArtifacts.map((artifact) => artifact.kind).sort(),
    ["approval", "teaching"]
  );
  assert.equal(repeatedTeacherArtifacts[0]?.freshness.status, "fresh");
  assert.equal(repeatedTeacherArtifacts[1]?.freshness.status, "fresh");

  const result = buildCandidatePackFromNormalizedEventExport({
    packLabel: "from-export-with-teacher",
    workspace: {
      workspaceId: "workspace-export",
      snapshotId: "workspace-export@snapshot-2",
      capturedAt: "2026-03-06T01:05:00.000Z",
      rootDir: "/workspace/export",
      revision: "workspace-export-teacher-rev"
    },
    normalizedEventExport: FIXTURE_NORMALIZED_EVENT_EXPORT,
    teacherSupervisionArtifacts: [...teacherArtifacts, ...repeatedTeacherArtifacts],
    learnedRouting: false
  });

  assert.equal(result.payloads.graph.blocks.some((block) => block.id.endsWith(":teacher-supervision-summary")), true);
  assert.equal(result.payloads.graph.blocks.some((block) => block.learning.role === "teacher_supervision"), true);
  assert.match(result.payloads.graph.blocks.map((block) => block.text).join("\n"), /Teacher teaching/);
  assert.match(
    result.payloads.graph.blocks.map((block) => block.text).join("\n"),
    /deduplicated records \(fresh=2, stale=0\) flowing into future candidate packs/
  );
});

test("always-on learner boots from live slices without blocking on passive backfill", () => {
  const result = advanceAlwaysOnLearningRuntime({
    packLabel: "attach-runtime",
    workspace: {
      workspaceId: "workspace-runtime",
      snapshotId: "workspace-runtime@snapshot-attach",
      capturedAt: "2026-03-07T08:00:00.000Z",
      rootDir: "/workspace/runtime",
      revision: "runtime-attach-rev"
    },
    interactionEvents: FIXTURE_INTERACTION_EVENTS,
    feedbackEvents: FIXTURE_FEEDBACK_EVENTS,
    learnedRouting: false,
    builtAt: "2026-03-07T08:00:30.000Z",
    liveSliceSize: 2,
    backfillSliceSize: 2,
    state: createAlwaysOnLearningRuntimeState()
  });

  assert.equal(result.runtimeOwner, "openclaw");
  assert.equal(result.hotPathLearning, false);
  assert.equal(result.attachBlocksOnFullReplay, false);
  assert.deepEqual(result.bridge.slices.map((slice) => slice.lane), ["live", "backfill"]);
  assert.deepEqual(result.selectedSlices.map((slice) => slice.lane), ["live"]);
  assert.equal(result.materialization?.reason, "attach_bootstrap");
  assert.equal(result.materialization?.lane, "live");
  assert.equal(result.materialization?.selectedEventRange.start, 103);
  assert.equal(result.materialization?.selectedEventRange.end, 104);
  assert.equal(result.state.learnedEventExport?.range.start, 103);
  assert.equal(result.state.learnedEventExport?.range.end, 104);
  assert.equal(result.deferred.live, 0);
  assert.equal(result.deferred.backfill, 1);
});

test("always-on learner prioritizes fresh live events while passive history catch-up continues behind them", () => {
  const attach = advanceAlwaysOnLearningRuntime({
    packLabel: "runtime-priority",
    workspace: {
      workspaceId: "workspace-runtime",
      snapshotId: "workspace-runtime@snapshot-attach",
      capturedAt: "2026-03-07T08:00:00.000Z",
      rootDir: "/workspace/runtime",
      revision: "runtime-attach-rev"
    },
    interactionEvents: FIXTURE_INTERACTION_EVENTS,
    feedbackEvents: FIXTURE_FEEDBACK_EVENTS,
    learnedRouting: true,
    builtAt: "2026-03-07T08:01:00.000Z",
    liveSliceSize: 2,
    backfillSliceSize: 2,
    state: createAlwaysOnLearningRuntimeState()
  });
  const next = advanceAlwaysOnLearningRuntime({
    packLabel: "runtime-priority",
    workspace: {
      workspaceId: "workspace-runtime",
      snapshotId: "workspace-runtime@snapshot-live",
      capturedAt: "2026-03-07T08:02:00.000Z",
      rootDir: "/workspace/runtime",
      revision: "runtime-live-rev"
    },
    interactionEvents: [
      ...FIXTURE_INTERACTION_EVENTS,
      createInteractionEvent({
        eventId: "evt-interaction-runtime-live-105",
        agentId: "agent-runtime",
        sessionId: "session-runtime",
        channel: "whatsapp",
        sequence: 105,
        kind: "message_delivered",
        createdAt: "2026-03-07T08:01:30.000Z",
        source: {
          runtimeOwner: "openclaw",
          stream: "openclaw/runtime/whatsapp"
        },
        messageId: "msg-runtime-105"
      })
    ],
    feedbackEvents: [
      ...FIXTURE_FEEDBACK_EVENTS,
      createFeedbackEvent({
        eventId: "evt-feedback-runtime-live-106",
        agentId: "agent-runtime",
        sessionId: "session-runtime",
        channel: "whatsapp",
        sequence: 106,
        kind: "teaching",
        createdAt: "2026-03-07T08:01:45.000Z",
        source: {
          runtimeOwner: "openclaw",
          stream: "openclaw/runtime/whatsapp"
        },
        content: "Newest live learning should land before history catch-up finishes.",
        relatedInteractionId: "evt-interaction-runtime-live-105"
      })
    ],
    learnedRouting: true,
    builtAt: "2026-03-07T08:02:30.000Z",
    liveSliceSize: 2,
    backfillSliceSize: 2,
    state: attach.state
  });

  assert.deepEqual(next.selectedSlices.map((slice) => slice.lane), ["live", "backfill"]);
  assert.equal(next.materialization?.reason, "fresh_live_events");
  assert.equal(next.materialization?.lane, "live");
  assert.equal(next.materialization?.candidate.summary.eventRange.start, 101);
  assert.equal(next.materialization?.candidate.summary.eventRange.end, 106);
  assert.equal(next.state.learnedEventExport?.range.start, 101);
  assert.equal(next.state.learnedEventExport?.range.end, 106);
  assert.equal(next.state.learnedEventExport?.range.count, 6);
  assert.equal(next.deferred.backfill, 0);
});

test("always-on learner picks the freshest pending live slice before older live backlog", () => {
  const attach = advanceAlwaysOnLearningRuntime({
    packLabel: "runtime-live-queue",
    workspace: {
      workspaceId: "workspace-runtime-live-queue",
      snapshotId: "workspace-runtime-live-queue@snapshot-attach",
      capturedAt: "2026-03-07T09:00:00.000Z",
      rootDir: "/workspace/runtime-live-queue",
      revision: "runtime-live-queue-attach"
    },
    interactionEvents: buildRuntimeInteractionEvents(100, 104),
    feedbackEvents: [],
    learnedRouting: false,
    builtAt: "2026-03-07T09:00:30.000Z",
    liveSliceSize: 1,
    backfillSliceSize: 1,
    state: createAlwaysOnLearningRuntimeState()
  });
  const next = advanceAlwaysOnLearningRuntime({
    packLabel: "runtime-live-queue",
    workspace: {
      workspaceId: "workspace-runtime-live-queue",
      snapshotId: "workspace-runtime-live-queue@snapshot-live",
      capturedAt: "2026-03-07T09:01:00.000Z",
      rootDir: "/workspace/runtime-live-queue",
      revision: "runtime-live-queue-live"
    },
    interactionEvents: buildRuntimeInteractionEvents(100, 108),
    feedbackEvents: [],
    learnedRouting: false,
    builtAt: "2026-03-07T09:01:30.000Z",
    liveSliceSize: 1,
    backfillSliceSize: 1,
    cadence: {
      liveSlicesPerCycle: 1,
      backfillSlicesPerCycle: 1
    },
    state: attach.state
  });

  assert.equal(next.selectedSlices[0]?.lane, "live");
  assert.equal(next.selectedSlices[0]?.export.range.start, 108);
  assert.equal(next.selectedSlices[0]?.export.range.end, 108);
  assert.deepEqual(
    next.selectedSlices.map((slice) => `${slice.lane}:${slice.export.range.end}`),
    ["live:108", "backfill:103"]
  );
  assert.equal(next.materialization?.reason, "fresh_live_events");
  assert.deepEqual(
    next.state.pending.live.map((slice) => slice.export.range.end),
    [107, 106, 105]
  );
  assert.equal(next.state.pending.backfill[0]?.export.range.end, 102);
});

test("always-on learner drain helper runs live-first then background backfill to idle", () => {
  const attach = advanceAlwaysOnLearningRuntime({
    packLabel: "runtime-drain",
    workspace: {
      workspaceId: "workspace-runtime-drain",
      snapshotId: "workspace-runtime-drain@snapshot-attach",
      capturedAt: "2026-03-07T09:10:00.000Z",
      rootDir: "/workspace/runtime-drain",
      revision: "runtime-drain-attach"
    },
    interactionEvents: buildRuntimeInteractionEvents(100, 106),
    feedbackEvents: [],
    learnedRouting: true,
    builtAt: "2026-03-07T09:10:30.000Z",
    liveSliceSize: 2,
    backfillSliceSize: 2,
    state: createAlwaysOnLearningRuntimeState()
  });
  const drained = drainAlwaysOnLearningRuntime({
    packLabel: "runtime-drain",
    workspace: {
      workspaceId: "workspace-runtime-drain",
      snapshotId: "workspace-runtime-drain@snapshot-drain",
      capturedAt: "2026-03-07T09:11:00.000Z",
      rootDir: "/workspace/runtime-drain",
      revision: "runtime-drain-run"
    },
    interactionEvents: buildRuntimeInteractionEvents(100, 108),
    feedbackEvents: [],
    learnedRouting: true,
    builtAt: "2026-03-07T09:11:30.000Z",
    liveSliceSize: 2,
    backfillSliceSize: 2,
    cadence: {
      liveSlicesPerCycle: 1,
      backfillSlicesPerCycle: 1
    },
    state: attach.state,
    maxCycles: 8
  });

  assert.equal(drained.drained, true);
  assert.equal(drained.stopReason, "idle");
  assert.deepEqual(
    drained.materializations.map((job) => job.reason),
    ["fresh_live_events", "passive_history_catchup", "passive_history_catchup"]
  );
  assert.deepEqual(
    drained.materializations.map((job) => job.lane),
    ["live", "backfill", "backfill"]
  );
  assert.equal(drained.materializations[0]?.selectedEventRange.start, 103);
  assert.equal(drained.materializations[0]?.selectedEventRange.end, 108);
  assert.equal(drained.materializations[1]?.selectedEventRange.start, 101);
  assert.equal(drained.materializations[2]?.selectedEventRange.start, 100);
  assert.equal(drained.cycles.length, 4);
  assert.equal(drained.state.pending.live.length, 0);
  assert.equal(drained.state.pending.backfill.length, 0);
  assert.equal(drained.state.materializationCount, 4);
});

test("always-on learner materialization hook emits a real candidate pack from the background job", (t) => {
  const rootDir = path.join(tmpdir(), `openclawbrain-ts-runtime-job-${Date.now()}`);
  t.after(() => rmSync(rootDir, { recursive: true, force: true }));

  const result = advanceAlwaysOnLearningRuntime({
    packLabel: "runtime-job",
    workspace: {
      workspaceId: "workspace-runtime-job",
      snapshotId: "workspace-runtime-job@snapshot-1",
      capturedAt: "2026-03-07T08:10:00.000Z",
      rootDir: "/workspace/runtime-job",
      revision: "runtime-job-rev"
    },
    interactionEvents: FIXTURE_INTERACTION_EVENTS,
    feedbackEvents: FIXTURE_FEEDBACK_EVENTS,
    learnedRouting: true,
    builtAt: "2026-03-07T08:10:30.000Z",
    liveSliceSize: 2,
    backfillSliceSize: 2,
    state: createAlwaysOnLearningRuntimeState()
  });

  assert.notEqual(result.materialization, null);
  const descriptor = materializeAlwaysOnLearningCandidatePack(rootDir, result.materialization!);

  assert.equal(descriptor.manifest.provenance.eventRange.start, 103);
  assert.equal(descriptor.manifest.provenance.eventRange.end, 104);
  assert.equal(
    descriptor.manifest.provenance.eventExports?.exportDigest,
    result.materialization?.candidate.summary.eventExportDigest ?? null
  );
  assert.equal(descriptor.manifest.routePolicy, "requires_learned_routing");
});

test("learner builds deterministic slice packs and bridge bundle materializations", (t) => {
  const workspace = {
    workspaceId: "workspace-bridge",
    snapshotId: "workspace-bridge@snapshot-1",
    capturedAt: "2026-03-07T08:20:00.000Z",
    rootDir: "/workspace/bridge",
    revision: "workspace-bridge-rev"
  } as const;
  const bridge = buildNormalizedEventExportBridge({
    interactionEvents: FIXTURE_INTERACTION_EVENTS,
    feedbackEvents: FIXTURE_FEEDBACK_EVENTS,
    liveSliceSize: 2,
    backfillSliceSize: 2
  });
  const firstSlice = bridge.slices[0] as NonNullable<(typeof bridge.slices)[number]>;
  const sliceRootDir = path.join(tmpdir(), `openclawbrain-ts-bridge-slice-${Date.now()}`);
  const bundleRootDir = path.join(tmpdir(), `openclawbrain-ts-bridge-bundle-${Date.now()}`);

  t.after(() => rmSync(sliceRootDir, { recursive: true, force: true }));
  t.after(() => rmSync(bundleRootDir, { recursive: true, force: true }));

  const sliceCandidate = buildCandidatePackFromNormalizedEventExportSlice({
    packLabel: "bridge-slice",
    workspace,
    normalizedEventExportSlice: firstSlice,
    learnedRouting: true,
    structuralOps: {
      split: 1
    }
  });
  const sliceDescriptor = materializeCandidatePackFromNormalizedEventExportSlice(sliceRootDir, {
    packLabel: "bridge-slice",
    workspace,
    normalizedEventExportSlice: firstSlice,
    learnedRouting: true,
    structuralOps: {
      split: 1
    }
  });
  const bundle = buildCandidatePackBundleFromNormalizedEventExportBridge({
    packLabel: "bridge-bundle",
    workspace,
    normalizedEventExportBridge: bridge,
    learnedRouting: false,
    structuralOps: {
      connect: 1
    }
  });
  const materializedBundle = materializeCandidatePackBundleFromNormalizedEventExportBridge(bundleRootDir, {
    packLabel: "bridge-bundle",
    workspace,
    normalizedEventExportBridge: bridge,
    learnedRouting: false,
    structuralOps: {
      connect: 1
    }
  });

  assert.equal(sliceCandidate.summary.eventExportDigest, firstSlice.export.provenance.exportDigest);
  assert.equal(sliceDescriptor.manifest.provenance.eventExports?.exportDigest, firstSlice.export.provenance.exportDigest);
  assert.equal(bundle.entries.length, bridge.slices.length);
  assert.equal(bundle.entries[0]?.packLabel, "bridge-bundle-01-live-103-104");
  assert.equal(bundle.entries[1]?.packLabel, "bridge-bundle-02-backfill-101-102");
  assert.deepEqual(
    bundle.entries.map((entry) => entry.normalizedEventExport.provenance.exportDigest),
    bridge.slices.map((slice) => slice.export.provenance.exportDigest)
  );
  assert.equal(
    bundle.bundleDigest,
    buildCandidatePackBundleFromNormalizedEventExportBridge({
      packLabel: "bridge-bundle",
      workspace,
      normalizedEventExportBridge: bridge,
      learnedRouting: false,
      structuralOps: {
        connect: 1
      }
    }).bundleDigest
  );
  assert.equal(materializedBundle.entries.length, bridge.slices.length);
  assert.equal(materializedBundle.entries[0]?.descriptor.manifest.provenance.eventExports?.exportDigest, bridge.slices[0]?.export.provenance.exportDigest);
  assert.equal(materializedBundle.entries[1]?.descriptor.manifest.provenance.eventExports?.exportDigest, bridge.slices[1]?.export.provenance.exportDigest);
  assert.equal(materializedBundle.entries[0]?.descriptor.manifest.routePolicy, "heuristic_allowed");
  assert.match(materializedBundle.entries[0]?.rootDir ?? "", /01-live-103-104-/);
});
