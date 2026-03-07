import assert from "node:assert/strict";
import test from "node:test";

import {
  createFeedbackEvent,
  createInteractionEvent,
  sortNormalizedEvents,
  validateFeedbackEvent,
  validateInteractionEvent
} from "@openclawbrain/events";

test("events package builds and sorts canonical interaction and feedback events", () => {
  const interaction = createInteractionEvent({
    eventId: "evt-interaction-events-1",
    agentId: "agent-events",
    sessionId: "session-events",
    channel: "cli",
    sequence: 11,
    kind: "message_delivered",
    createdAt: "2026-03-06T02:11:00.000Z",
    source: {
      runtimeOwner: "openclaw",
      stream: "openclaw/runtime/cli"
    },
    messageId: "msg-events-1"
  });
  const feedback = createFeedbackEvent({
    eventId: "evt-feedback-events-1",
    agentId: "agent-events",
    sessionId: "session-events",
    channel: "cli",
    sequence: 10,
    kind: "teaching",
    createdAt: "2026-03-06T02:10:00.000Z",
    source: {
      runtimeOwner: "openclaw",
      stream: "openclaw/runtime/cli"
    },
    content: "Use the package surface directly for event normalization.",
    relatedInteractionId: interaction.eventId
  });

  assert.deepEqual(validateInteractionEvent(interaction), []);
  assert.deepEqual(validateFeedbackEvent(feedback), []);
  assert.deepEqual(sortNormalizedEvents([interaction, feedback]).map((event) => event.eventId), [feedback.eventId, interaction.eventId]);
});
