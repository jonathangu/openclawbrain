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
  assert.equal(descriptor.graph.blocks.length >= 11, true);
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
  assert.equal(descriptor.graph.blocks.some((block) => block.state !== undefined), true);
  assert.equal(descriptor.graph.blocks.some((block) => (block.edges?.length ?? 0) > 0), true);
  assert.equal(descriptor.graph.evolution?.decayApplied, true);
  assert.equal(descriptor.vectors.entries.some((entry) => entry.keywords.includes("fast_boot")), true);
  assert.equal(descriptor.vectors.entries.some((entry) => entry.keywords.includes("connected")), true);
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

test("learner turns supervision and elapsed time into real graph evolution", () => {
  const interaction = createInteractionEvent({
    eventId: "evt-graph-evolution-201",
    agentId: "agent-graph-evolution",
    sessionId: "session-graph-evolution",
    channel: "cli",
    sequence: 201,
    kind: "message_delivered",
    createdAt: "2026-03-07T10:00:00.000Z",
    source: {
      runtimeOwner: "openclaw",
      stream: "openclaw/runtime/cli"
    },
    messageId: "msg-graph-evolution-201"
  });
  const teaching = createFeedbackEvent({
    eventId: "evt-graph-evolution-202",
    agentId: interaction.agentId,
    sessionId: interaction.sessionId,
    channel: interaction.channel,
    sequence: 202,
    kind: "teaching",
    createdAt: "2026-03-07T10:01:00.000Z",
    source: interaction.source,
    content: "Prefer retry budget rollback guidance when quota alarms rise.",
    relatedInteractionId: interaction.eventId
  });
  const suppression = createFeedbackEvent({
    eventId: "evt-graph-evolution-203",
    agentId: interaction.agentId,
    sessionId: interaction.sessionId,
    channel: interaction.channel,
    sequence: 203,
    kind: "suppression",
    createdAt: "2026-03-07T10:02:00.000Z",
    source: interaction.source,
    content: "Suppress the stale quota fallback route after the retry-budget lesson lands.",
    relatedInteractionId: interaction.eventId
  });

  const base = buildCandidatePack({
    packLabel: "graph-evolution",
    workspace: {
      workspaceId: "workspace-graph-evolution",
      snapshotId: "workspace-graph-evolution@snapshot-1",
      capturedAt: "2026-03-07T09:55:00.000Z",
      rootDir: "/workspace/graph-evolution",
      revision: "graph-evolution-rev-1"
    },
    eventRange: {
      start: 201,
      end: 201
    },
    eventExports: {
      interactionEvents: [interaction],
      feedbackEvents: []
    },
    learnedRouting: false,
    builtAt: "2026-03-07T10:00:30.000Z",
    structuralOps: {
      split: 1,
      merge: 1,
      prune: 1,
      connect: 2
    }
  });
  const reinforced = buildCandidatePack({
    packLabel: "graph-evolution",
    workspace: {
      workspaceId: "workspace-graph-evolution",
      snapshotId: "workspace-graph-evolution@snapshot-2",
      capturedAt: "2026-03-07T10:03:00.000Z",
      rootDir: "/workspace/graph-evolution",
      revision: "graph-evolution-rev-2"
    },
    eventRange: {
      start: 201,
      end: 203
    },
    eventExports: {
      interactionEvents: [interaction],
      feedbackEvents: [teaching, suppression]
    },
    learnedRouting: false,
    builtAt: "2026-03-07T10:03:30.000Z",
    structuralOps: {
      split: 1,
      merge: 1,
      prune: 1,
      connect: 2
    }
  });
  const decayed = buildCandidatePack({
    packLabel: "graph-evolution",
    workspace: {
      workspaceId: "workspace-graph-evolution",
      snapshotId: "workspace-graph-evolution@snapshot-2",
      capturedAt: "2026-03-07T10:03:00.000Z",
      rootDir: "/workspace/graph-evolution",
      revision: "graph-evolution-rev-2"
    },
    eventRange: {
      start: 201,
      end: 203
    },
    eventExports: {
      interactionEvents: [interaction],
      feedbackEvents: [teaching, suppression]
    },
    learnedRouting: false,
    builtAt: "2026-06-07T10:03:30.000Z",
    structuralOps: {
      split: 1,
      merge: 1,
      prune: 1,
      connect: 2
    }
  });

  const reinforcedSplit = reinforced.payloads.graph.blocks.find((block) => block.source === "split:openclaw/runtime/cli:teaching");
  const decayedSplit = decayed.payloads.graph.blocks.find((block) => block.source === "split:openclaw/runtime/cli:teaching");
  const reinforcedSplitVector = reinforcedSplit === undefined ? undefined : reinforced.payloads.vectors.entries.find((entry) => entry.blockId === reinforcedSplit.id);
  const decayedSplitVector = decayedSplit === undefined ? undefined : decayed.payloads.vectors.entries.find((entry) => entry.blockId === decayedSplit.id);

  assert.equal(base.payloads.graph.evolution?.structuralOps.split, 0);
  assert.equal(reinforced.payloads.graph.evolution?.structuralOps.split, 1);
  assert.equal(reinforced.payloads.graph.evolution?.structuralOps.merge, 1);
  assert.equal(reinforced.payloads.graph.evolution?.structuralOps.prune, 1);
  assert.equal(reinforced.payloads.graph.blocks.some((block) => block.id.includes(":split:")), true);
  assert.equal(reinforced.payloads.graph.blocks.some((block) => (block.compactedFrom?.length ?? 0) > 1), true);
  assert.equal(reinforced.payloads.graph.blocks.some((block) => (block.edges?.length ?? 0) > 0), true);
  assert.equal(
    reinforced.payloads.graph.evolution?.prunedBlockIds.includes(`${reinforced.summary.packId}:event:${interaction.eventId}`),
    true
  );
  assert.notEqual(reinforced.summary.packId, decayed.summary.packId);
  assert.notEqual(reinforcedSplit, undefined);
  assert.notEqual(decayedSplit, undefined);
  assert.notEqual(reinforcedSplitVector, undefined);
  assert.notEqual(decayedSplitVector, undefined);
  assert.equal((decayedSplit?.state?.freshness ?? 1) < (reinforcedSplit?.state?.freshness ?? 1), true);
  assert.equal((decayedSplit?.state?.strength ?? 0) < (reinforcedSplit?.state?.strength ?? 0), true);
  assert.equal((decayedSplitVector?.boost ?? 0) < (reinforcedSplitVector?.boost ?? 0), true);
});

test("always-on learner keeps the evolving learned graph in runtime state across supervision cycles", () => {
  const initial = advanceAlwaysOnLearningRuntime({
    packLabel: "runtime-graph-state",
    workspace: {
      workspaceId: "workspace-runtime-graph-state",
      snapshotId: "workspace-runtime-graph-state@snapshot-1",
      capturedAt: "2026-03-07T11:00:00.000Z",
      rootDir: "/workspace/runtime-graph-state",
      revision: "runtime-graph-state-rev-1"
    },
    interactionEvents: buildRuntimeInteractionEvents(301, 302),
    feedbackEvents: [],
    learnedRouting: false,
    builtAt: "2026-03-07T11:00:30.000Z",
    liveSliceSize: 2,
    backfillSliceSize: 2,
    state: createAlwaysOnLearningRuntimeState(),
    structuralOps: {
      split: 1,
      merge: 1,
      prune: 1,
      connect: 2
    }
  });
  const supervised = advanceAlwaysOnLearningRuntime({
    packLabel: "runtime-graph-state",
    workspace: {
      workspaceId: "workspace-runtime-graph-state",
      snapshotId: "workspace-runtime-graph-state@snapshot-2",
      capturedAt: "2026-03-07T11:02:00.000Z",
      rootDir: "/workspace/runtime-graph-state",
      revision: "runtime-graph-state-rev-2"
    },
    interactionEvents: buildRuntimeInteractionEvents(301, 302),
    feedbackEvents: [
      createFeedbackEvent({
        eventId: "evt-runtime-graph-state-303",
        agentId: "agent-runtime-batch",
        sessionId: "session-runtime-batch",
        channel: "whatsapp",
        sequence: 303,
        kind: "teaching",
        createdAt: "2026-03-07T11:01:30.000Z",
        source: {
          runtimeOwner: "openclaw",
          stream: "openclaw/runtime/whatsapp"
        },
        content: "Prefer rollback guidance when retry-budget pressure appears.",
        relatedInteractionId: "evt-runtime-int-302"
      })
    ],
    learnedRouting: false,
    builtAt: "2026-03-07T11:02:30.000Z",
    liveSliceSize: 2,
    backfillSliceSize: 2,
    state: initial.state,
    structuralOps: {
      split: 1,
      merge: 1,
      prune: 1,
      connect: 2
    }
  });

  assert.notEqual(initial.state.learnedGraph, null);
  assert.notEqual(supervised.state.learnedGraph, null);
  assert.equal((supervised.state.learnedGraph?.blocks.length ?? 0) > (initial.state.learnedGraph?.blocks.length ?? 0), true);
  assert.equal((supervised.state.learnedGraph?.evolution?.structuralOps.split ?? 0) >= 1, true);
  assert.equal((supervised.state.learnedGraph?.blocks.some((block) => (block.edges?.length ?? 0) > 0) ?? false), true);
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

test("changed supervision changes learned router identity, checksum, weights, and visible delta", () => {
  const workspace = {
    workspaceId: "workspace-router-delta",
    snapshotId: "workspace-router-delta@snapshot-1",
    capturedAt: "2026-03-07T10:15:00.000Z",
    rootDir: "/workspace/router-delta",
    revision: "router-delta-rev"
  } as const;
  const interaction = createInteractionEvent({
    eventId: "evt-router-delta-interaction",
    agentId: "agent-router-delta",
    sessionId: "session-router-delta",
    channel: "cli",
    sequence: 200,
    kind: "message_delivered",
    createdAt: "2026-03-07T10:15:00.000Z",
    source: {
      runtimeOwner: "openclaw",
      stream: "openclaw/runtime/router-delta"
    },
    messageId: "msg-router-delta"
  });
  const scannerFeedback = createFeedbackEvent({
    eventId: "evt-router-delta-feedback-scanner",
    agentId: interaction.agentId,
    sessionId: interaction.sessionId,
    channel: interaction.channel,
    sequence: 201,
    kind: "teaching",
    createdAt: "2026-03-07T10:16:00.000Z",
    source: interaction.source,
    content: "Teach the feedback scanner and checkpoint resume path for this route refresh.",
    relatedInteractionId: interaction.eventId
  });
  const structuralFeedback = createFeedbackEvent({
    eventId: "evt-router-delta-feedback-structural",
    agentId: interaction.agentId,
    sessionId: interaction.sessionId,
    channel: interaction.channel,
    sequence: 201,
    kind: "teaching",
    createdAt: "2026-03-07T10:16:00.000Z",
    source: interaction.source,
    content: "Teach structural graph prune connect merge routing for this route refresh.",
    relatedInteractionId: interaction.eventId
  });

  const scannerPack = buildCandidatePack({
    packLabel: "router-delta",
    workspace,
    eventRange: {
      start: 200,
      end: 201
    },
    eventExports: {
      interactionEvents: [interaction],
      feedbackEvents: [scannerFeedback]
    },
    learnedRouting: true,
    builtAt: "2026-03-07T10:17:00.000Z"
  });
  const structuralPack = buildCandidatePack({
    packLabel: "router-delta",
    workspace,
    eventRange: {
      start: 200,
      end: 201
    },
    eventExports: {
      interactionEvents: [interaction],
      feedbackEvents: [structuralFeedback]
    },
    learnedRouting: true,
    builtAt: "2026-03-07T10:17:00.000Z"
  });

  const scannerRouter = scannerPack.payloads.router;
  const structuralRouter = structuralPack.payloads.router;
  const scannerStructuralDelta = scannerRouter?.policyUpdates.find((update) => update.blockId.endsWith(":structural-ops"))?.delta ?? 0;
  const structuralStructuralDelta = structuralRouter?.policyUpdates.find((update) => update.blockId.endsWith(":structural-ops"))?.delta ?? 0;

  assert.notEqual(scannerRouter, null);
  assert.notEqual(structuralRouter, null);
  assert.notEqual(scannerRouter?.routerIdentity, structuralRouter?.routerIdentity);
  assert.notEqual(scannerPack.manifest.payloadChecksums.router, structuralPack.manifest.payloadChecksums.router);
  assert.notEqual(scannerRouter?.training.weightsChecksum, structuralRouter?.training.weightsChecksum);
  assert.notDeepEqual(scannerPack.summary.learnedRouter.visibleDelta, structuralPack.summary.learnedRouter.visibleDelta);
  assert.equal(scannerPack.summary.learnedRouter.refreshStatus, "updated");
  assert.equal(structuralPack.summary.learnedRouter.refreshStatus, "updated");
  assert.equal(scannerPack.summary.learnedRouter.updateCount > 0, true);
  assert.equal(structuralPack.summary.learnedRouter.updateCount > 0, true);
  assert.equal(structuralStructuralDelta > scannerStructuralDelta, true);
});

test("learned routing without supervision emits loud no-op refresh diagnostics", () => {
  const pack = buildCandidatePack({
    packLabel: "router-noop",
    workspace: {
      workspaceId: "workspace-router-noop",
      snapshotId: "workspace-router-noop@snapshot-1",
      capturedAt: "2026-03-07T11:00:00.000Z",
      rootDir: "/workspace/router-noop"
    },
    eventRange: {
      start: 0,
      end: -1
    },
    learnedRouting: true,
    builtAt: "2026-03-07T11:00:30.000Z"
  });

  assert.equal(pack.payloads.router?.training.status, "no_supervision");
  assert.equal(pack.payloads.router?.training.updateCount, 0);
  assert.match(pack.payloads.router?.training.noOpReason ?? "", /no normalized event export supplied/);
  assert.deepEqual(pack.summary.learnedRouter.visibleDelta, []);
  assert.match(pack.summary.learnedRouter.noOpReason ?? "", /no normalized event export supplied/);
});
