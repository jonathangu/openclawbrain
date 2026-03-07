import assert from "node:assert/strict";
import test from "node:test";

import { FIXTURE_FEEDBACK_EVENTS, FIXTURE_INTERACTION_EVENTS } from "@openclawbrain/contracts";
import {
  FIXTURE_NORMALIZED_EVENT_EXPORT,
  buildNormalizedEventExport,
  validateNormalizedEventExport
} from "@openclawbrain/event-export";

test("event-export package derives deterministic range and provenance from events", () => {
  const rebuilt = buildNormalizedEventExport({
    interactionEvents: FIXTURE_INTERACTION_EVENTS,
    feedbackEvents: FIXTURE_FEEDBACK_EVENTS
  });

  assert.deepEqual(validateNormalizedEventExport(rebuilt), []);
  assert.deepEqual(rebuilt, FIXTURE_NORMALIZED_EVENT_EXPORT);
  assert.equal(rebuilt.provenance.exportDigest, FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.exportDigest);
});
