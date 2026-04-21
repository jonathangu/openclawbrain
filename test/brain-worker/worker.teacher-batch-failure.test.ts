import { describe, expect, it, vi } from "vitest";

import { DEFAULT_BRAIN_CONFIG } from "../../src/brain-core/types.js";
import { BrainWorker } from "../../src/brain-worker/worker.js";

describe("BrainWorker teacher batch failure lifecycle", () => {
  it("emits a failed teacher-batch lifecycle event when teacher evaluation throws", async () => {
    const events: Array<{ step: string; status: string; detail: string | null }> = [];
    const store = {
      getTeacherQueueSummary: vi.fn(() => ({
        pendingCount: 1,
        readyCount: 1,
        delayedCount: 0,
        budgetDeferredCount: 0,
        sparseReadyCount: 0,
        richReadyCount: 1,
        sample: [],
      })),
      getPendingObservations: vi.fn(() => ([{
        id: "bo_1",
        episodeId: "ep_1",
        conversationId: 42,
        traceId: "bt_1",
        followUpText: "thanks",
        toolResults: [],
        createdAt: 10,
      }])),
      getEpisode: vi.fn(() => ({
        id: "ep_1",
        conversationId: 42,
        queryText: "How do I open a pull request?",
      })),
    };
    const teacher = {
      evaluateObservation: vi.fn(async () => {
        throw new Error("teacher provider timed out");
      }),
    };
    const worker = new BrainWorker(
      store as never,
      {} as never,
      teacher as never,
      {} as never,
      {} as never,
      {
        ...DEFAULT_BRAIN_CONFIG,
        teacherEnabled: true,
        trainerIntervalMs: 10_000,
      },
      {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
      },
      {
        onTeacherBatchLifecycle: async (event) => {
          events.push({
            step: event.step,
            status: event.status,
            detail: event.detail,
          });
        },
      },
    );

    await expect((worker as unknown as {
      evaluatePendingObservations: () => Promise<unknown>;
    }).evaluatePendingObservations()).rejects.toThrow("teacher provider timed out");

    expect(events).toEqual([
      expect.objectContaining({
        step: "observation_batch_bound",
        status: "running",
      }),
      expect.objectContaining({
        step: "teacher_invoked",
        status: "running",
      }),
      expect.objectContaining({
        step: "teacher_invoked",
        status: "failed",
        detail: "teacher provider timed out",
      }),
    ]);
  });
});
