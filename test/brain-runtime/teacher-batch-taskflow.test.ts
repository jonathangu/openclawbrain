import { describe, expect, it, vi } from "vitest";

import { TeacherBatchTaskFlowCoordinator } from "../../src/brain-runtime/teacher-batch-taskflow.js";
import {
  buildTeacherBatchFlowLifecycleEvent,
  createTeacherBatchFlowState,
} from "../../src/brain-worker/teacher-job.js";
import type {
  BoundManagedTaskFlowRuntimeLike,
  ManagedTaskFlowMutationResultLike,
  ManagedTaskFlowRecordLike,
} from "../../src/types.js";

type TrainingStateStore = {
  values: Map<string, string>;
  setTrainingState: (key: string, value: string | number) => void;
  getTrainingStateJson: <T>(key: string) => T | null;
  setTrainingStateJson: (key: string, value: unknown | null) => void;
};

function createTrainingStateStore(): TrainingStateStore {
  const values = new Map<string, string>();
  return {
    values,
    setTrainingState: (key, value) => {
      values.set(key, String(value));
    },
    getTrainingStateJson: (key) => {
      const raw = values.get(key)?.trim();
      if (!raw) {
        return null;
      }
      return JSON.parse(raw) as never;
    },
    setTrainingStateJson: (key, value) => {
      values.set(key, value === null ? "" : JSON.stringify(value));
    },
  };
}

function createManagedRuntime() {
  const flows = new Map<string, ManagedTaskFlowRecordLike>();
  let flowCounter = 0;

  const applyMutation = (
    flowId: string,
    expectedRevision: number,
    mutate: (flow: ManagedTaskFlowRecordLike) => ManagedTaskFlowRecordLike,
  ): ManagedTaskFlowMutationResultLike => {
    const current = flows.get(flowId);
    if (!current) {
      return { applied: false, code: "not_found" };
    }
    if (current.revision !== expectedRevision) {
      return { applied: false, code: "revision_conflict", current };
    }
    const next = mutate(current);
    flows.set(flowId, next);
    return { applied: true, flow: next };
  };

  const runtime: BoundManagedTaskFlowRuntimeLike = {
    createManaged: (params) => {
      const flow: ManagedTaskFlowRecordLike = {
        flowId: `flow_${++flowCounter}`,
        revision: 1,
        status: params.status ?? "queued",
        currentStep: params.currentStep ?? null,
        stateJson: params.stateJson ?? null,
        waitJson: params.waitJson ?? null,
        endedAt: params.endedAt ?? null,
      };
      flows.set(flow.flowId, flow);
      return flow;
    },
    get: (flowId) => flows.get(flowId),
    resume: (params) => applyMutation(params.flowId, params.expectedRevision, (flow) => ({
      ...flow,
      revision: flow.revision + 1,
      status: params.status ?? flow.status,
      currentStep: params.currentStep ?? flow.currentStep ?? null,
      stateJson: params.stateJson ?? flow.stateJson ?? null,
    })),
    finish: (params) => applyMutation(params.flowId, params.expectedRevision, (flow) => ({
      ...flow,
      revision: flow.revision + 1,
      status: "completed",
      stateJson: params.stateJson ?? flow.stateJson ?? null,
      endedAt: params.endedAt ?? flow.endedAt ?? null,
    })),
    fail: (params) => applyMutation(params.flowId, params.expectedRevision, (flow) => ({
      ...flow,
      revision: flow.revision + 1,
      status: "failed",
      stateJson: params.stateJson ?? flow.stateJson ?? null,
      endedAt: params.endedAt ?? flow.endedAt ?? null,
    })),
  };

  return { runtime, flows };
}

describe("TeacherBatchTaskFlowCoordinator", () => {
  it("binds lifecycle events into a managed Task Flow for the owner session", async () => {
    const store = createTrainingStateStore();
    const { runtime, flows } = createManagedRuntime();
    const bindManagedTaskFlowSession = vi.fn(() => runtime);
    const coordinator = new TeacherBatchTaskFlowCoordinator({
      store: store as never,
      bindManagedTaskFlowSession,
      log: {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        debug: vi.fn(),
      },
    });

    coordinator.rememberOwnerSession({
      conversationId: 42,
      sessionKey: "agent:main:telegram:8518484672",
    });

    const state = createTeacherBatchFlowState({
      observations: [
        { observationId: "bo_1", episodeId: "ep_1", conversationId: 42, createdAt: 10 },
        { observationId: "bo_2", episodeId: "ep_2", conversationId: 42, createdAt: 20 },
      ],
      batchBudget: 2,
    });

    await coordinator.handleLifecycleEvent(buildTeacherBatchFlowLifecycleEvent({
      state,
      step: "observation_batch_bound",
      emittedAt: 100,
    }));

    const afterCreateBinding = store.getTrainingStateJson<Record<string, { flowId: string; revision: number }>>(
      "teacher_batch_taskflow_bindings_json",
    );
    expect(bindManagedTaskFlowSession).toHaveBeenCalledWith({
      sessionKey: "agent:main:telegram:8518484672",
    });
    expect(afterCreateBinding?.[state.lookupKey]).toMatchObject({
      flowId: "flow_1",
      revision: 2,
    });
    expect(flows.get("flow_1")).toMatchObject({
      status: "running",
      currentStep: "observation_batch_bound",
    });

    await coordinator.handleLifecycleEvent(buildTeacherBatchFlowLifecycleEvent({
      state: {
        ...state,
        labelIds: ["bl_1"],
        evaluatedCount: 2,
        appliedLabelCount: 1,
      },
      step: "reward_signal_applied",
      status: "completed",
      emittedAt: 200,
      detail: "applied reward label",
    }));

    const finalFlow = flows.get("flow_1");
    const finalBinding = store.getTrainingStateJson<Record<string, { flowId: string; revision: number; status: string }>>(
      "teacher_batch_taskflow_bindings_json",
    );
    expect(finalFlow).toMatchObject({
      status: "completed",
      endedAt: 200,
    });
    expect(finalBinding?.[state.lookupKey]).toMatchObject({
      flowId: "flow_1",
      revision: 3,
      status: "completed",
    });
    expect(store.getTrainingStateJson("last_teacher_batch_flow_event_json")).toMatchObject({
      lookupKey: state.lookupKey,
      step: "reward_signal_applied",
      status: "completed",
      conversationIds: [42],
    });
  });

  it("skips Task Flow mutation when no owner session is known", async () => {
    const store = createTrainingStateStore();
    const bindManagedTaskFlowSession = vi.fn();
    const coordinator = new TeacherBatchTaskFlowCoordinator({
      store: store as never,
      bindManagedTaskFlowSession,
      log: {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        debug: vi.fn(),
      },
    });

    const state = createTeacherBatchFlowState({
      observations: [
        { observationId: "bo_1", episodeId: "ep_1", conversationId: 99, createdAt: 10 },
      ],
      batchBudget: 1,
    });

    await coordinator.handleLifecycleEvent(buildTeacherBatchFlowLifecycleEvent({
      state,
      step: "teacher_invoked",
      emittedAt: 50,
    }));

    expect(bindManagedTaskFlowSession).not.toHaveBeenCalled();
    expect(store.getTrainingStateJson("teacher_batch_taskflow_bindings_json")).toBeNull();
    expect(store.getTrainingStateJson("last_teacher_batch_flow_event_json")).toMatchObject({
      lookupKey: state.lookupKey,
      step: "teacher_invoked",
    });
  });

  it("fails the managed flow when a lifecycle event reports batch failure", async () => {
    const store = createTrainingStateStore();
    const { runtime, flows } = createManagedRuntime();
    const coordinator = new TeacherBatchTaskFlowCoordinator({
      store: store as never,
      bindManagedTaskFlowSession: vi.fn(() => runtime),
      log: {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        debug: vi.fn(),
      },
    });

    coordinator.rememberOwnerSession({
      conversationId: 42,
      sessionKey: "agent:main:telegram:8518484672",
    });

    const state = createTeacherBatchFlowState({
      observations: [
        { observationId: "bo_1", episodeId: "ep_1", conversationId: 42, createdAt: 10 },
      ],
      batchBudget: 1,
    });

    await coordinator.handleLifecycleEvent(buildTeacherBatchFlowLifecycleEvent({
      state,
      step: "teacher_invoked",
      emittedAt: 100,
    }));
    await coordinator.handleLifecycleEvent(buildTeacherBatchFlowLifecycleEvent({
      state,
      step: "teacher_invoked",
      status: "failed",
      emittedAt: 130,
      detail: "teacher provider timed out",
    }));

    expect(flows.get("flow_1")).toMatchObject({
      status: "failed",
      endedAt: 130,
      currentStep: "teacher_invoked",
    });
    expect(store.getTrainingStateJson<Record<string, { status: string; revision: number }>>(
      "teacher_batch_taskflow_bindings_json",
    )?.[state.lookupKey]).toMatchObject({
      status: "failed",
      revision: 3,
    });
    expect(store.getTrainingStateJson("last_teacher_batch_flow_event_json")).toMatchObject({
      step: "teacher_invoked",
      status: "failed",
      detail: "teacher provider timed out",
    });
  });
});
