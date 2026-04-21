import type { BrainStore } from "../brain-store/store.js";
import type { TeacherBatchFlowLifecycleEventV1 } from "../brain-worker/teacher-job.js";
import type {
  BindManagedTaskFlowSessionFn,
  BoundManagedTaskFlowRuntimeLike,
  ManagedTaskFlowMutationResultLike,
  ManagedTaskFlowRecordLike,
  TaskFlowJsonValue,
} from "../types.js";

const OWNER_SESSIONS_KEY = "teacher_batch_owner_sessions_json";
const FLOW_BINDINGS_KEY = "teacher_batch_taskflow_bindings_json";

type TeacherBatchFlowBindingRecord = {
  flowId: string;
  revision: number;
  ownerSessionKey: string;
  lastStep: TeacherBatchFlowLifecycleEventV1["step"];
  status: TeacherBatchFlowLifecycleEventV1["status"];
  updatedAt: number;
};

type TeacherBatchFlowBindingRegistry = Record<string, TeacherBatchFlowBindingRecord>;
type TeacherBatchOwnerSessionRegistry = Record<string, string>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function normalizeString(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : null;
}

function toBindingRegistry(value: unknown): TeacherBatchFlowBindingRegistry {
  if (!isRecord(value)) {
    return {};
  }
  const result: TeacherBatchFlowBindingRegistry = {};
  for (const [lookupKey, entry] of Object.entries(value)) {
    if (!isRecord(entry)) {
      continue;
    }
    const flowId = normalizeString(entry.flowId);
    const ownerSessionKey = normalizeString(entry.ownerSessionKey);
    const lastStep = normalizeString(entry.lastStep);
    const status = normalizeString(entry.status);
    const revision = typeof entry.revision === "number" && Number.isFinite(entry.revision) ? entry.revision : null;
    const updatedAt = typeof entry.updatedAt === "number" && Number.isFinite(entry.updatedAt) ? entry.updatedAt : Date.now();
    if (!flowId || !ownerSessionKey || !lastStep || !status || revision === null) {
      continue;
    }
    result[lookupKey] = {
      flowId,
      revision,
      ownerSessionKey,
      lastStep: lastStep as TeacherBatchFlowLifecycleEventV1["step"],
      status: status as TeacherBatchFlowLifecycleEventV1["status"],
      updatedAt,
    };
  }
  return result;
}

function toOwnerRegistry(value: unknown): TeacherBatchOwnerSessionRegistry {
  if (!isRecord(value)) {
    return {};
  }
  const result: TeacherBatchOwnerSessionRegistry = {};
  for (const [conversationId, sessionKey] of Object.entries(value)) {
    const normalized = normalizeString(sessionKey);
    if (normalized) {
      result[conversationId] = normalized;
    }
  }
  return result;
}

function buildFlowStateJson(event: TeacherBatchFlowLifecycleEventV1): TaskFlowJsonValue {
  return {
    contract: "openclawbrain_teacher_batch_flow_state.v1",
    lookupKey: event.lookupKey,
    batchId: event.batchId,
    step: event.step,
    status: event.status,
    observationIds: event.observationIds,
    episodeIds: event.episodeIds,
    conversationIds: event.conversationIds,
    labelIds: event.labelIds,
    batchBudget: event.batchBudget,
    selectedCount: event.selectedCount,
    evaluatedCount: event.evaluatedCount,
    skippedCount: event.skippedCount,
    appliedLabelCount: event.appliedLabelCount,
    batchCreatedAt: event.batchCreatedAt,
    emittedAt: event.emittedAt,
    detail: event.detail,
  };
}

function buildFlowGoal(event: TeacherBatchFlowLifecycleEventV1): string {
  return `process OpenClawBrain teacher batch ${event.batchId}`;
}

export class TeacherBatchTaskFlowCoordinator {
  constructor(
    private params: {
      store: BrainStore;
      bindManagedTaskFlowSession?: BindManagedTaskFlowSessionFn;
      log: { info: (msg: string) => void; warn: (msg: string) => void; error: (msg: string) => void; debug?: (msg: string) => void };
    },
  ) {}

  rememberOwnerSession(params: { conversationId: number; sessionKey: string }): void {
    const sessionKey = normalizeString(params.sessionKey);
    if (!sessionKey || !Number.isFinite(params.conversationId)) {
      return;
    }
    const registry = this.readOwnerRegistry();
    registry[String(params.conversationId)] = sessionKey;
    this.params.store.setTrainingStateJson(OWNER_SESSIONS_KEY, registry);
  }

  async handleLifecycleEvent(event: TeacherBatchFlowLifecycleEventV1): Promise<void> {
    this.recordLifecycleBreadcrumb(event);
    const bindManagedTaskFlowSession = this.params.bindManagedTaskFlowSession;
    if (!bindManagedTaskFlowSession) {
      return;
    }

    const bindings = this.readBindingRegistry();
    const existingBinding = bindings[event.lookupKey];
    const ownerSessionKey = existingBinding?.ownerSessionKey ?? this.resolveOwnerSessionKey(event);
    if (!ownerSessionKey) {
      this.params.log.debug?.(`[brain] no owner session found for teacher batch ${event.lookupKey}`);
      return;
    }

    const runtime = bindManagedTaskFlowSession({ sessionKey: ownerSessionKey });
    if (!runtime) {
      return;
    }

    const stateJson = buildFlowStateJson(event);
    let flow = existingBinding ? runtime.get(existingBinding.flowId) : undefined;
    if (!flow) {
      flow = runtime.createManaged({
        controllerId: "openclawbrain/teacher-batch",
        goal: buildFlowGoal(event),
        status: event.status === "failed" ? "failed" : "running",
        currentStep: event.step,
        stateJson,
        createdAt: event.batchCreatedAt,
        updatedAt: event.emittedAt,
        ...(event.status === "failed" ? { endedAt: event.emittedAt } : {}),
      });
    }

    const mutation = event.status === "failed"
      ? this.failFlow(runtime, flow, stateJson, event.emittedAt, event.detail)
      : event.step === "reward_signal_applied" && event.status === "completed"
        ? this.finishFlow(runtime, flow, stateJson, event.emittedAt)
        : this.resumeFlow(runtime, flow, event.step, stateJson, event.emittedAt);

    if (!mutation.applied) {
      this.params.log.warn(`[brain] teacher batch flow mutation failed (${mutation.code}) for ${event.lookupKey}`);
      return;
    }

    bindings[event.lookupKey] = {
      flowId: mutation.flow.flowId,
      revision: mutation.flow.revision,
      ownerSessionKey,
      lastStep: event.step,
      status: event.status,
      updatedAt: event.emittedAt,
    };
    this.params.store.setTrainingStateJson(FLOW_BINDINGS_KEY, bindings);
  }

  private recordLifecycleBreadcrumb(event: TeacherBatchFlowLifecycleEventV1): void {
    this.params.store.setTrainingState("worker_last_teacher_batch_flow_lookup_key", event.lookupKey);
    this.params.store.setTrainingState("worker_last_teacher_batch_flow_step", event.step);
    this.params.store.setTrainingState("worker_last_teacher_batch_flow_status", event.status);
    this.params.store.setTrainingStateJson("last_teacher_batch_flow_event_json", event);
  }

  private resolveOwnerSessionKey(event: TeacherBatchFlowLifecycleEventV1): string | null {
    const ownerRegistry = this.readOwnerRegistry();
    for (const conversationId of event.conversationIds) {
      const sessionKey = ownerRegistry[String(conversationId)];
      if (sessionKey) {
        return sessionKey;
      }
    }
    return null;
  }

  private readOwnerRegistry(): TeacherBatchOwnerSessionRegistry {
    return toOwnerRegistry(this.params.store.getTrainingStateJson(OWNER_SESSIONS_KEY));
  }

  private readBindingRegistry(): TeacherBatchFlowBindingRegistry {
    return toBindingRegistry(this.params.store.getTrainingStateJson(FLOW_BINDINGS_KEY));
  }

  private resumeFlow(
    runtime: BoundManagedTaskFlowRuntimeLike,
    flow: ManagedTaskFlowRecordLike,
    step: TeacherBatchFlowLifecycleEventV1["step"],
    stateJson: TaskFlowJsonValue,
    updatedAt: number,
  ): ManagedTaskFlowMutationResultLike {
    const firstAttempt = runtime.resume({
      flowId: flow.flowId,
      expectedRevision: flow.revision,
      status: "running",
      currentStep: step,
      stateJson,
      updatedAt,
    });
    if (firstAttempt.applied || !firstAttempt.current) {
      return firstAttempt;
    }
    return runtime.resume({
      flowId: flow.flowId,
      expectedRevision: firstAttempt.current.revision,
      status: "running",
      currentStep: step,
      stateJson,
      updatedAt,
    });
  }

  private finishFlow(
    runtime: BoundManagedTaskFlowRuntimeLike,
    flow: ManagedTaskFlowRecordLike,
    stateJson: TaskFlowJsonValue,
    endedAt: number,
  ): ManagedTaskFlowMutationResultLike {
    const firstAttempt = runtime.finish({
      flowId: flow.flowId,
      expectedRevision: flow.revision,
      stateJson,
      updatedAt: endedAt,
      endedAt,
    });
    if (firstAttempt.applied || !firstAttempt.current) {
      return firstAttempt;
    }
    return runtime.finish({
      flowId: flow.flowId,
      expectedRevision: firstAttempt.current.revision,
      stateJson,
      updatedAt: endedAt,
      endedAt,
    });
  }

  private failFlow(
    runtime: BoundManagedTaskFlowRuntimeLike,
    flow: ManagedTaskFlowRecordLike,
    stateJson: TaskFlowJsonValue,
    endedAt: number,
    detail: string | null,
  ): ManagedTaskFlowMutationResultLike {
    const firstAttempt = runtime.fail({
      flowId: flow.flowId,
      expectedRevision: flow.revision,
      stateJson,
      blockedSummary: detail,
      updatedAt: endedAt,
      endedAt,
    });
    if (firstAttempt.applied || !firstAttempt.current) {
      return firstAttempt;
    }
    return runtime.fail({
      flowId: flow.flowId,
      expectedRevision: firstAttempt.current.revision,
      stateJson,
      blockedSummary: detail,
      updatedAt: endedAt,
      endedAt,
    });
  }
}
