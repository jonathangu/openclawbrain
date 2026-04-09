import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { buildNormalizedEventExport, createFeedbackEvent, createInteractionEvent } from "@openclawbrain/contracts";
import {
  advanceAlwaysOnLearningRuntime,
  buildCandidatePackFromNormalizedEventExport,
  buildTeacherSupervisionArtifactsFromNormalizedEventExport,
  createAlwaysOnLearningRuntimeState,
  DEFAULT_SPARSE_FEEDBACK_POLICY,
} from "../src/local-learner.js";

function createWorkspace(t) {
  const rootDir = mkdtempSync(path.join(os.tmpdir(), "ocb-sparse-feedback-"));
  writeFileSync(path.join(rootDir, "README.md"), "# Sparse feedback test\n", "utf8");
  t.after(() => {
    rmSync(rootDir, { recursive: true, force: true });
  });
  return {
    workspaceId: "ws-sparse-feedback",
    snapshotId: "snapshot-sparse-feedback",
    capturedAt: "2026-04-09T14:20:00.000Z",
    rootDir,
    branch: "main",
    revision: "test",
    dirty: false,
    labels: ["test"],
    files: ["README.md"],
  };
}

function createFixtureExport() {
  const interaction = createInteractionEvent({
    eventId: "evt-message-1",
    agentId: "main",
    sessionId: "session-1",
    channel: "cli",
    sequence: 1,
    kind: "memory_compiled",
    createdAt: "2026-04-09T14:20:00.000Z",
    source: { runtimeOwner: "openclaw", stream: "session.tail" },
    messageId: "msg-1",
  });
  const feedbackEvents = [0, 1, 2, 3].map((index) => createFeedbackEvent({
    eventId: `evt-feedback-${index + 1}`,
    agentId: "main",
    sessionId: "session-1",
    channel: "cli",
    sequence: index + 2,
    kind: "correction",
    createdAt: `2026-04-09T14:20:0${index + 1}.000Z`,
    source: { runtimeOwner: "openclaw", stream: "session.tail" },
    relatedInteractionId: interaction.eventId,
    content: `feedback ${index + 1}`,
  }));
  return buildNormalizedEventExport({
    interactionEvents: [interaction],
    feedbackEvents,
  });
}

test("default sparse feedback budget is raised to drain the live backlog", () => {
  assert.equal(DEFAULT_SPARSE_FEEDBACK_POLICY.teacherBudget, 64);
});

test("sparse feedback rotates through unprocessed events and retains prior selections in routing", (t) => {
  const workspace = createWorkspace(t);
  const normalizedEventExport = createFixtureExport();
  const firstArtifacts = buildTeacherSupervisionArtifactsFromNormalizedEventExport({
    normalizedEventExport,
    observedAt: "2026-04-09T14:21:00.000Z",
    sparseFeedback: { teacherBudget: 2 },
  });
  const firstArtifactContents = new Set(firstArtifacts.map((artifact) => artifact.content));
  assert.equal(firstArtifacts.length, 2);

  const firstCycle = advanceAlwaysOnLearningRuntime({
    packLabel: "sparse-feedback-runtime",
    workspace,
    interactionEvents: normalizedEventExport.interactionEvents,
    feedbackEvents: normalizedEventExport.feedbackEvents,
    learnedRouting: true,
    sparseFeedback: { teacherBudget: 2 },
    state: createAlwaysOnLearningRuntimeState(),
  });

  assert.equal(firstCycle.state.sparseFeedback.budgetedOutFeedbackCount, 2);
  assert.equal(firstCycle.state.sparseFeedback.processedFeedbackEventIds.length, 2);

  const secondArtifacts = buildTeacherSupervisionArtifactsFromNormalizedEventExport({
    normalizedEventExport,
    observedAt: "2026-04-09T14:21:30.000Z",
    sparseFeedback: firstCycle.state.sparseFeedback,
  });
  assert.equal(secondArtifacts.length, 2);
  for (const artifact of secondArtifacts) {
    assert.equal(firstArtifactContents.has(artifact.content), false);
  }

  const secondPack = buildCandidatePackFromNormalizedEventExport({
    packLabel: "sparse-feedback-pack",
    workspace,
    normalizedEventExport,
    learnedRouting: true,
    sparseFeedback: firstCycle.state.sparseFeedback,
  });
  const feedbackEventIds = new Set(normalizedEventExport.feedbackEvents.map((event) => event.eventId));
  const routedFeedbackEventIds = new Set((secondPack.payloads.router?.traces ?? [])
    .map((trace) => trace.sourceEventId)
    .filter((eventId) => feedbackEventIds.has(eventId)));
  assert.deepEqual([...routedFeedbackEventIds].sort(), [...feedbackEventIds].sort());
});
