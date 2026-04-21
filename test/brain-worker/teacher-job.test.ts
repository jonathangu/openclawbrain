import { describe, expect, it } from "vitest";

import {
  buildTeacherBatchFlowLifecycleEvent,
  buildTeacherBatchLookupKey,
  createTeacherBatchFlowState,
} from "../../src/brain-worker/teacher-job.js";

describe("teacher batch flow helpers", () => {
  it("builds a deterministic lookup key regardless of input order", () => {
    const a = buildTeacherBatchLookupKey({
      observations: [
        { observationId: "bo_b", createdAt: 20 },
        { observationId: "bo_a", createdAt: 10 },
      ],
      batchBudget: 20,
    });
    const b = buildTeacherBatchLookupKey({
      observations: [
        { observationId: "bo_a", createdAt: 10 },
        { observationId: "bo_b", createdAt: 20 },
      ],
      batchBudget: 20,
    });

    expect(a).toBe(b);
    expect(a).toContain("ocb-teacher-batch:10:bo_a:2:");
  });

  it("materializes lifecycle events from the bounded flow state", () => {
    const state = createTeacherBatchFlowState({
      observations: [
        { observationId: "bo_a", episodeId: "ep_1", conversationId: 7, createdAt: 10 },
        { observationId: "bo_b", episodeId: "ep_2", conversationId: 7, createdAt: 20 },
      ],
      batchBudget: 20,
    });
    const event = buildTeacherBatchFlowLifecycleEvent({
      state: {
        ...state,
        labelIds: ["bl_1"],
        evaluatedCount: 1,
        skippedCount: 1,
        appliedLabelCount: 1,
      },
      step: "reward_signal_applied",
      status: "completed",
      emittedAt: 123,
      detail: "1 reward label applied",
    });

    expect(event).toMatchObject({
      contract: "openclawbrain_teacher_batch_flow_event.v1",
      version: 1,
      step: "reward_signal_applied",
      status: "completed",
      observationIds: ["bo_a", "bo_b"],
      episodeIds: ["ep_1", "ep_2"],
      conversationIds: [7],
      labelIds: ["bl_1"],
      selectedCount: 2,
      evaluatedCount: 1,
      skippedCount: 1,
      appliedLabelCount: 1,
      emittedAt: 123,
      detail: "1 reward label applied",
    });
  });
});
