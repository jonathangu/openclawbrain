import assert from "node:assert/strict";

import { CONTRACT_IDS, validateRuntimeCompileRequest } from "@openclawbrain/contracts";
import { buildNormalizedEventExport } from "@openclawbrain/event-export";
import { createFeedbackEvent, createInteractionEvent } from "@openclawbrain/events";

const request = {
  contract: CONTRACT_IDS.runtimeCompile,
  agentId: "agent-consumer",
  userMessage: "compile feedback context",
  maxContextBlocks: 3,
  maxContextChars: 1600,
  modeRequested: "heuristic",
  compactionMode: "native"
};

assert.deepEqual(validateRuntimeCompileRequest(request), []);

const normalizedEventExport = buildNormalizedEventExport({
  interactionEvents: [
    createInteractionEvent({
      eventId: "evt-consumer-1",
      agentId: "agent-consumer",
      sessionId: "session-consumer",
      channel: "cli",
      sequence: 10,
      kind: "memory_compiled",
      createdAt: "2026-03-06T00:10:00.000Z",
      source: { runtimeOwner: "openclaw", stream: "openclaw/runtime/demo" },
      packId: "pack-demo"
    })
  ],
  feedbackEvents: [
    createFeedbackEvent({
      eventId: "evt-consumer-2",
      agentId: "agent-consumer",
      sessionId: "session-consumer",
      channel: "cli",
      sequence: 11,
      kind: "teaching",
      createdAt: "2026-03-06T00:11:00.000Z",
      source: { runtimeOwner: "openclaw", stream: "openclaw/runtime/demo" },
      content: "Promote the pack only after validation.",
      relatedInteractionId: "evt-consumer-1"
    })
  ]
});

assert.equal(normalizedEventExport.range.start, 10);
assert.equal(normalizedEventExport.range.end, 11);
assert.equal(normalizedEventExport.range.count, 2);
assert.deepEqual(normalizedEventExport.provenance.contracts, [
  CONTRACT_IDS.interactionEvents,
  CONTRACT_IDS.feedbackEvents
]);

console.log(
  JSON.stringify(
    {
      ok: true,
      requestContract: request.contract,
      eventRange: normalizedEventExport.range,
      exportDigest: normalizedEventExport.provenance.exportDigest,
      sourceStreams: normalizedEventExport.provenance.sourceStreams
    },
    null,
    2
  )
);
