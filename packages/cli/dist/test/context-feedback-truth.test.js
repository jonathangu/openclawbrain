import test from "node:test";
import assert from "node:assert/strict";
import {
  CONTEXT_FEEDBACK_HELPFUL_MIN,
  CONTEXT_FEEDBACK_HARMFUL_MAX,
  classifyContextFeedbackVerdict,
  buildEmptyContextFeedbackSummary,
} from "../src/index.js";

test("classifyContextFeedbackVerdict returns helpful for scores at or above threshold", () => {
  assert.equal(classifyContextFeedbackVerdict(0.25), "helpful");
  assert.equal(classifyContextFeedbackVerdict(0.5), "helpful");
  assert.equal(classifyContextFeedbackVerdict(1.0), "helpful");
});

test("classifyContextFeedbackVerdict returns harmful for scores at or below threshold", () => {
  assert.equal(classifyContextFeedbackVerdict(-0.25), "harmful");
  assert.equal(classifyContextFeedbackVerdict(-0.5), "harmful");
  assert.equal(classifyContextFeedbackVerdict(-1.0), "harmful");
});

test("classifyContextFeedbackVerdict returns irrelevant for scores between thresholds", () => {
  assert.equal(classifyContextFeedbackVerdict(0), "irrelevant");
  assert.equal(classifyContextFeedbackVerdict(0.1), "irrelevant");
  assert.equal(classifyContextFeedbackVerdict(-0.1), "irrelevant");
  assert.equal(classifyContextFeedbackVerdict(0.24), "irrelevant");
  assert.equal(classifyContextFeedbackVerdict(-0.24), "irrelevant");
});

test("score band constants match the canonical thresholds", () => {
  assert.equal(CONTEXT_FEEDBACK_HELPFUL_MIN, 0.25);
  assert.equal(CONTEXT_FEEDBACK_HARMFUL_MAX, -0.25);
});

test("buildEmptyContextFeedbackSummary returns canonical empty shape", () => {
  const summary = buildEmptyContextFeedbackSummary();
  assert.deepEqual(summary.scoreBands, {
    helpfulMin: 0.25,
    harmfulMax: -0.25,
  });
  assert.deepEqual(summary.verdictCounts, {
    helpful: 0,
    irrelevant: 0,
    harmful: 0,
  });
  assert.equal(summary.coverage.routeTraceCount, 0);
  assert.equal(summary.coverage.identifiedRouteTraceCount, 0);
  assert.equal(summary.coverage.unidentifiedRouteTraceCount, 0);
  assert.equal(summary.coverage.agentIdentityCoverage, 0);
  assert.equal(summary.coverage.observationCount, 0);
  assert.equal(summary.coverage.completedObservationCount, 0);
  assert.equal(summary.coverage.supervisedTraceCount, 0);
  assert.equal(summary.coverage.unsupervisedTraceCount, 0);
  assert.equal(summary.coverage.observationCoverage, 0);
  assert.equal(summary.coverage.supervisionCoverage, 0);
  assert.equal(summary.coverage.pendingFollowupCount, 0);
  assert.equal(summary.coverage.pendingTeacherCount, 0);
  assert.deepEqual(summary.agents, []);
  assert.equal(summary.latest, null);
  assert.equal(summary.focus.action, "monitor");
  assert.equal(typeof summary.focus.detail, "string");
  assert.equal(typeof summary.detail, "string");
});

test("buildEmptyContextFeedbackSummary returns a fresh object each call", () => {
  const a = buildEmptyContextFeedbackSummary();
  const b = buildEmptyContextFeedbackSummary();
  assert.notEqual(a, b);
  assert.deepEqual(a, b);
  a.verdictCounts.helpful = 99;
  assert.equal(b.verdictCounts.helpful, 0, "mutation of one summary must not affect another");
});
