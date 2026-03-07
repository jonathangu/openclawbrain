import assert from "node:assert/strict";
import test from "node:test";

import {
  FIXTURE_FEEDBACK_EVENTS,
  FIXTURE_INTERACTION_EVENTS,
  createFeedbackEvent,
  createInteractionEvent
} from "@openclawbrain/contracts";
import {
  FIXTURE_NORMALIZED_EVENT_EXPORT,
  buildNormalizedEventExportBundleFromEvents,
  buildNormalizedEventExportBundle,
  buildNormalizedEventExport,
  buildNormalizedEventExportBridge,
  buildNormalizedEventDedupId,
  createEventExportCursor,
  describeNormalizedEventExportObservability,
  validateEventExportCursor,
  validateNormalizedEventExport,
  validateNormalizedEventExportBridge
} from "@openclawbrain/event-export";

function buildBridgeFixtureEvents() {
  return {
    interactionEvents: [
      createInteractionEvent({
        eventId: "evt-bridge-int-100",
        agentId: "agent-bridge",
        sessionId: "session-bridge",
        channel: "whatsapp",
        sequence: 100,
        kind: "message_delivered",
        createdAt: "2026-03-06T00:00:00.000Z",
        source: {
          runtimeOwner: "openclaw",
          stream: "openclaw/runtime/whatsapp"
        },
        messageId: "msg-100"
      }),
      createInteractionEvent({
        eventId: "evt-bridge-int-101",
        agentId: "agent-bridge",
        sessionId: "session-bridge",
        channel: "whatsapp",
        sequence: 101,
        kind: "operator_override",
        createdAt: "2026-03-06T00:01:00.000Z",
        source: {
          runtimeOwner: "openclaw",
          stream: "openclaw/runtime/whatsapp"
        },
        messageId: "msg-101"
      }),
      createInteractionEvent({
        eventId: "evt-bridge-int-104",
        agentId: "agent-bridge",
        sessionId: "session-bridge",
        channel: "whatsapp",
        sequence: 104,
        kind: "memory_compiled",
        createdAt: "2026-03-06T00:04:00.000Z",
        source: {
          runtimeOwner: "openclaw",
          stream: "openclaw/runtime/whatsapp"
        },
        packId: "pack-live-104"
      })
    ],
    feedbackEvents: [
      createFeedbackEvent({
        eventId: "evt-bridge-feed-102",
        agentId: "agent-bridge",
        sessionId: "session-bridge",
        channel: "whatsapp",
        sequence: 102,
        kind: "correction",
        createdAt: "2026-03-06T00:02:00.000Z",
        source: {
          runtimeOwner: "openclaw",
          stream: "openclaw/runtime/whatsapp"
        },
        content: "Prefer deterministic bridge slices for live export.",
        relatedInteractionId: "evt-bridge-int-101"
      }),
      createFeedbackEvent({
        eventId: "evt-bridge-feed-103",
        agentId: "agent-bridge",
        sessionId: "session-bridge",
        channel: "whatsapp",
        sequence: 103,
        kind: "approval",
        createdAt: "2026-03-06T00:03:00.000Z",
        source: {
          runtimeOwner: "openclaw",
          stream: "openclaw/runtime/whatsapp"
        },
        content: "Background backfill can stay passive.",
        relatedInteractionId: "evt-bridge-int-101"
      }),
      createFeedbackEvent({
        eventId: "evt-bridge-feed-105",
        agentId: "agent-bridge",
        sessionId: "session-bridge",
        channel: "whatsapp",
        sequence: 105,
        kind: "teaching",
        createdAt: "2026-03-06T00:05:00.000Z",
        source: {
          runtimeOwner: "openclaw",
          stream: "openclaw/runtime/whatsapp"
        },
        content: "Keep live updates ahead of history replay.",
        relatedInteractionId: "evt-bridge-int-104"
      })
    ]
  };
}

test("event-export package derives deterministic range, provenance, and learning surfaces from events", () => {
  const rebuilt = buildNormalizedEventExport({
    interactionEvents: FIXTURE_INTERACTION_EVENTS,
    feedbackEvents: FIXTURE_FEEDBACK_EVENTS
  });

  assert.deepEqual(validateNormalizedEventExport(rebuilt), []);
  assert.deepEqual(rebuilt, FIXTURE_NORMALIZED_EVENT_EXPORT);
  assert.equal(rebuilt.provenance.exportDigest, FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.exportDigest);
  assert.equal(rebuilt.provenance.learningSurface.bootProfile, "fast_boot_defaults");
  assert.equal(rebuilt.provenance.learningSurface.learningCadence, "passive_background");
  assert.equal(rebuilt.provenance.learningSurface.scanPolicy, "always_on");
  assert.equal(rebuilt.provenance.learningSurface.labelHarvest.humanLabels, FIXTURE_FEEDBACK_EVENTS.length);
  assert.equal(rebuilt.provenance.learningSurface.labelHarvest.selfLabels, 1);
  assert.match(rebuilt.provenance.learningSurface.scanSurfaces.join("\n"), /openclaw\/runtime\/whatsapp:teaching/);
});

test("event-export observability reports supervision freshness by source and latest teacher signal", () => {
  const report = describeNormalizedEventExportObservability(FIXTURE_NORMALIZED_EVENT_EXPORT);

  assert.equal(report.exportDigest, FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.exportDigest);
  assert.equal(report.range.end, 104);
  assert.deepEqual(report.sourceStreams, ["openclaw/runtime/whatsapp"]);
  assert.equal(report.supervisionFreshnessBySource.length, 1);
  assert.deepEqual(report.supervisionFreshnessBySource[0], {
    sourceStream: "openclaw/runtime/whatsapp",
    eventCount: 4,
    interactionCount: 2,
    feedbackCount: 2,
    humanLabelCount: 2,
    selfLabelCount: 1,
    freshestEventId: "evt-feedback-fixture-2",
    freshestSequence: 104,
    freshestCreatedAt: "2026-03-06T00:03:00.000Z",
    freshestKind: "approval"
  });
  assert.deepEqual(report.teacherFreshness, {
    freshestEventId: "evt-feedback-fixture-2",
    freshestSequence: 104,
    freshestCreatedAt: "2026-03-06T00:03:00.000Z",
    freshestKind: "approval",
    sourceStream: "openclaw/runtime/whatsapp",
    humanLabelCount: 2,
    sources: ["openclaw/runtime/whatsapp"]
  });
});

test("event-export bridge emits deterministic live-first slices and non-blocking backfill cursors", () => {
  const input = buildBridgeFixtureEvents();
  const bridge = buildNormalizedEventExportBridge({
    ...input,
    liveSliceSize: 2,
    backfillSliceSize: 2
  });

  assert.deepEqual(validateNormalizedEventExportBridge(bridge), []);
  assert.deepEqual(
    bridge.slices.map((slice) => slice.lane),
    ["live", "backfill"]
  );
  assert.equal(bridge.slices[0]?.export.range.start, 104);
  assert.equal(bridge.slices[0]?.export.range.end, 105);
  assert.equal(bridge.slices[0]?.watermark.first?.sequence, 104);
  assert.equal(bridge.slices[0]?.watermark.last?.sequence, 105);
  assert.equal(bridge.slices[1]?.export.range.start, 102);
  assert.equal(bridge.slices[1]?.export.range.end, 103);
  assert.equal(bridge.cursor.live.after?.sequence, 105);
  assert.equal(bridge.cursor.live.exhausted, true);
  assert.equal(bridge.cursor.backfill.before?.sequence, 102);
  assert.equal(bridge.cursor.backfill.exhausted, false);
  assert.equal(bridge.slices[0]?.provenance.bridgeDigest, bridge.bridgeDigest);
  assert.equal(bridge.slices[1]?.provenance.runtimeOwner, "openclaw");
  assert.deepEqual(
    buildNormalizedEventExportBridge({
      ...input,
      liveSliceSize: 2,
      backfillSliceSize: 2
    }),
    bridge
  );
});

test("event-export bridge drops replay duplicates by dedup-safe identity", () => {
  const input = buildBridgeFixtureEvents();
  const bridge = buildNormalizedEventExportBridge({
    interactionEvents: [...input.interactionEvents, input.interactionEvents[0] as typeof input.interactionEvents[number]],
    feedbackEvents: [...input.feedbackEvents, input.feedbackEvents[0] as typeof input.feedbackEvents[number]],
    liveSliceSize: 10,
    backfillSliceSize: 10
  });

  assert.deepEqual(validateNormalizedEventExportBridge(bridge), []);
  assert.equal(bridge.dedupedInputCount, 6);
  assert.equal(bridge.duplicateIdentityCount, 2);
  assert.equal(bridge.slices.length, 1);
  assert.equal(bridge.slices[0]?.export.range.count, 6);
  assert.deepEqual(
    bridge.slices[0]?.eventIdentities,
    [...new Set((bridge.slices[0]?.eventIdentities ?? []).map((identity) => identity))]
  );
  assert.equal(buildNormalizedEventDedupId(input.interactionEvents[0] as typeof input.interactionEvents[number]), bridge.slices[0]?.eventIdentities[0]);
});

test("event-export bridge advances live slices before passive history catch-up", () => {
  const input = buildBridgeFixtureEvents();
  const initial = buildNormalizedEventExportBridge({
    ...input,
    liveSliceSize: 2,
    backfillSliceSize: 2
  });
  const next = buildNormalizedEventExportBridge({
    interactionEvents: [
      ...input.interactionEvents,
      createInteractionEvent({
        eventId: "evt-bridge-int-106",
        agentId: "agent-bridge",
        sessionId: "session-bridge",
        channel: "whatsapp",
        sequence: 106,
        kind: "message_delivered",
        createdAt: "2026-03-06T00:06:00.000Z",
        source: {
          runtimeOwner: "openclaw",
          stream: "openclaw/runtime/whatsapp"
        },
        messageId: "msg-106"
      }),
      createInteractionEvent({
        eventId: "evt-bridge-int-108",
        agentId: "agent-bridge",
        sessionId: "session-bridge",
        channel: "whatsapp",
        sequence: 108,
        kind: "memory_compiled",
        createdAt: "2026-03-06T00:08:00.000Z",
        source: {
          runtimeOwner: "openclaw",
          stream: "openclaw/runtime/whatsapp"
        },
        packId: "pack-live-108"
      })
    ],
    feedbackEvents: [
      ...input.feedbackEvents,
      createFeedbackEvent({
        eventId: "evt-bridge-feed-107",
        agentId: "agent-bridge",
        sessionId: "session-bridge",
        channel: "whatsapp",
        sequence: 107,
        kind: "approval",
        createdAt: "2026-03-06T00:07:00.000Z",
        source: {
          runtimeOwner: "openclaw",
          stream: "openclaw/runtime/whatsapp"
        },
        content: "Newest live slice should land ahead of backfill.",
        relatedInteractionId: "evt-bridge-int-106"
      })
    ],
    cursor: initial.cursor,
    liveSliceSize: 2,
    backfillSliceSize: 2
  });

  assert.deepEqual(validateNormalizedEventExportBridge(next), []);
  assert.deepEqual(
    next.slices.map((slice) => slice.lane),
    ["live", "live", "backfill"]
  );
  assert.equal(next.slices[0]?.export.range.start, 106);
  assert.equal(next.slices[0]?.export.range.end, 107);
  assert.equal(next.slices[1]?.export.range.start, 108);
  assert.equal(next.slices[1]?.export.range.end, 108);
  assert.equal(next.slices[2]?.export.range.start, 100);
  assert.equal(next.slices[2]?.export.range.end, 101);
  assert.equal(next.cursor.live.after?.sequence, 108);
  assert.equal(next.cursor.live.exhausted, true);
  assert.equal(next.cursor.backfill.before?.sequence, 100);
  assert.equal(next.cursor.backfill.exhausted, true);
});

test("event-export bridge validation rejects incoherent observability metadata", () => {
  const input = buildBridgeFixtureEvents();
  const bridge = buildNormalizedEventExportBridge({
    ...input,
    liveSliceSize: 2,
    backfillSliceSize: 2
  });
  const liveSlice = bridge.slices[0];

  if (liveSlice === undefined) {
    throw new Error("live slice fixture is required");
  }

  const errors = validateNormalizedEventExportBridge({
    ...bridge,
    dedupedInputCount: 1,
    slices: [
      {
        ...liveSlice,
        duplicateIdentityCount: bridge.duplicateIdentityCount + 1,
        provenance: {
          ...liveSlice.provenance,
          dedupedEventCount: liveSlice.provenance.dedupedEventCount + 1,
          sourceStreams: ["openclaw/runtime/stale"],
          duplicateIdentityCount: liveSlice.provenance.duplicateIdentityCount
        }
      },
      ...bridge.slices.slice(1)
    ]
  });

  assert.match(errors.join(";"), /provenance dedupedEventCount must match slice dedupedEventCount/);
  assert.match(errors.join(";"), /provenance duplicateIdentityCount must match slice duplicateIdentityCount/);
  assert.match(errors.join(";"), /provenance sourceStreams must match export provenance sourceStreams/);
  assert.match(errors.join(";"), /duplicateIdentityCount must match bridge duplicateIdentityCount/);
  assert.match(errors.join(";"), /dedupedInputCount must be >= emitted slice event count/);
});

test("event-export bridge rejects inverted cursors and projects deterministic export bundles", () => {
  const input = buildBridgeFixtureEvents();
  const bridge = buildNormalizedEventExportBridge({
    ...input,
    liveSliceSize: 2,
    backfillSliceSize: 2
  });
  const invalidCursor = createEventExportCursor();

  invalidCursor.live.after = bridge.slices[0]?.watermark.first ?? null;
  invalidCursor.backfill.before = bridge.slices[0]?.watermark.last ?? null;

  assert.match(validateEventExportCursor(invalidCursor).join("\n"), /backfill\.before must not sort after live\.after/);
  assert.throws(
    () =>
      buildNormalizedEventExportBridge({
        ...input,
        cursor: invalidCursor,
        liveSliceSize: 2,
        backfillSliceSize: 2
      }),
    /backfill\.before must not sort after live\.after/
  );

  const bundle = buildNormalizedEventExportBundle(bridge);

  assert.equal(bundle.bridgeDigest, bridge.bridgeDigest);
  assert.equal(bundle.cursor.live.after?.eventId, bridge.cursor.live.after?.eventId);
  assert.equal(bundle.entries.length, bridge.slices.length);
  assert.deepEqual(
    bundle.entries.map((entry) => entry.export.provenance.exportDigest),
    bridge.slices.map((slice) => slice.export.provenance.exportDigest)
  );
  assert.deepEqual(buildNormalizedEventExportBundle(bridge), bundle);
});

test("event-export package can project a live-first bundle directly from raw events", () => {
  const input = buildBridgeFixtureEvents();
  const direct = buildNormalizedEventExportBundleFromEvents({
    ...input,
    liveSliceSize: 2,
    backfillSliceSize: 2
  });
  const bridged = buildNormalizedEventExportBridge({
    ...input,
    liveSliceSize: 2,
    backfillSliceSize: 2
  });

  assert.deepEqual(direct, buildNormalizedEventExportBundle(bridged));
  assert.deepEqual(
    direct.entries.map((entry) => entry.lane),
    ["live", "backfill"]
  );
  assert.equal(direct.bridgeDigest, bridged.bridgeDigest);
  assert.equal(direct.entries[0]?.export.range.start, 104);
  assert.equal(direct.entries[1]?.export.range.end, 103);
});
