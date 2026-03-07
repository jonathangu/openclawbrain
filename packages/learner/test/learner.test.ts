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
  buildCandidatePackFromNormalizedEventExport,
  createAlwaysOnLearningRuntimeState,
  materializeAlwaysOnLearningCandidatePack,
  materializeCandidatePack,
  materializeCandidatePackFromNormalizedEventExport
} from "@openclawbrain/learner";

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
