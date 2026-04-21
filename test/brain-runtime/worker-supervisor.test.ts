import { describe, expect, it, vi } from "vitest";

import { WorkerSupervisor } from "../../src/brain-runtime/worker-supervisor.js";
import { buildTeacherBatchFlowLifecycleEvent, createTeacherBatchFlowState } from "../../src/brain-worker/teacher-job.js";
import type { WorkerTeacherBatchLifecycleMessage } from "../../src/brain-worker/protocol.js";
import { DEFAULT_BRAIN_CONFIG } from "../../src/brain-core/types.js";

function createStore() {
  const values = new Map<string, string>();
  return {
    values,
    getTrainingState: (key: string) => values.get(key) ?? null,
    setTrainingState: vi.fn((key: string, value: string | number) => {
      values.set(key, String(value));
    }),
    setTrainingStateJson: vi.fn((key: string, value: unknown | null) => {
      values.set(key, value === null ? "" : JSON.stringify(value));
    }),
  };
}

describe("WorkerSupervisor teacher batch lifecycle bridge", () => {
  it("records lifecycle breadcrumbs and forwards teacher batch events", async () => {
    const store = createStore();
    const onTeacherBatchLifecycle = vi.fn(async () => undefined);
    const supervisor = new WorkerSupervisor({
      config: {
        ...DEFAULT_BRAIN_CONFIG,
        root: "/tmp/openclawbrain-test",
        semanticThreshold: 0.1,
        trainerIntervalMs: 10_000,
        workerMode: "child",
        workerHeartbeatTimeoutMs: 5_000,
        workerRestartDelayMs: 100,
        teacherEnabled: false,
      },
      store: store as never,
      log: {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
      },
      teacherModel: null,
      isEnabled: () => true,
      onPackPromoted: vi.fn(),
      onTeacherComplete: vi.fn(async () => ({
        type: "teacher-complete-result",
        requestId: "req_1",
        ok: true,
        content: [],
      })),
      onTeacherBatchLifecycle,
    });

    const message: WorkerTeacherBatchLifecycleMessage = {
      type: "teacher-batch-lifecycle",
      pid: 123,
      at: 456,
      event: buildTeacherBatchFlowLifecycleEvent({
        state: createTeacherBatchFlowState({
          observations: [
            { observationId: "bo_1", episodeId: "ep_1", conversationId: 42, createdAt: 10 },
          ],
          batchBudget: 1,
        }),
        step: "teacher_invoked",
        emittedAt: 456,
      }),
    };

    await (supervisor as unknown as {
      handleMessage: (message: WorkerTeacherBatchLifecycleMessage, child: unknown) => Promise<void>;
    }).handleMessage(message, {});

    expect(store.setTrainingState).toHaveBeenCalledWith(
      "worker_last_teacher_batch_flow_lookup_key",
      message.event.lookupKey,
    );
    expect(store.setTrainingState).toHaveBeenCalledWith(
      "worker_last_teacher_batch_flow_step",
      "teacher_invoked",
    );
    expect(store.setTrainingState).toHaveBeenCalledWith(
      "worker_last_teacher_batch_flow_status",
      "running",
    );
    expect(store.setTrainingStateJson).toHaveBeenCalledWith(
      "last_teacher_batch_flow_event_json",
      message.event,
    );
    expect(onTeacherBatchLifecycle).toHaveBeenCalledWith(message);
  });
});
