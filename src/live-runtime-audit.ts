import type {
  BrainObservationBindingMode,
  TeacherFeedbackRichness,
} from "./brain-core/types.js";

export {
  OPERATOR_HEALTH_CONTRACT,
  isOperatorHealthSummary,
  summarizeOperatorHealth,
  summarizeTeacherLoopTruth,
} from "../scripts/operator-health-contract.mjs";

export type TeacherLoopTruthInput = {
  failureMode: string | null;
  lastNoOpReason: string | null;
  latestFreshness: string | null;
  queueDepth: number | null;
  running: boolean | null;
  watch?: {
    state: string | null;
  } | null;
  watchState?: string | null;
};

export type TeacherLoopTruthSummary = {
  healthy: boolean | null;
  idle: boolean | null;
  stale: boolean | null;
};

export type OperatorHealthInput = {
  workerHealthy: boolean | null;
  workerMode: string | null;
  workerStatus: string | null;
  watchState: string | null;
  proofState: string | null;
  teacherArtifactCount: number | null;
  teacherLoopTruth?: TeacherLoopTruthSummary | null;
};

export type OperatorHealthStatus =
  | "healthy"
  | "partial"
  | "unknown"
  | "stale"
  | "unhealthy";

export type OperatorHealthSummary = {
  contract: "openclawbrain_operator_health.v1";
  status: OperatorHealthStatus;
  healthy: boolean | null;
  partial: boolean;
  unknown: boolean;
  stale: boolean;
  detail: string;
  workerHealthy: boolean | null;
  workerMode: string | null;
  workerStatus: string | null;
  watchState: string | null;
  proofState: string | null;
  teacherArtifactCount: number | null;
  backgroundLearning: TeacherLoopTruthSummary;
  reasons: string[];
};

function normalizeNullableNumber(value: unknown): number | null {
  return Number.isFinite(value) ? Number(value) : null;
}

function normalizeNullableString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function pushUnique<T>(values: T[], next: T | null): void {
  if (next === null || values.includes(next)) {
    return;
  }
  values.push(next);
}

export const ATTRIBUTION_TRUTH_CONTRACT = "openclawbrain_attribution_truth.v1";

export type AttributionTruthState =
  | "matched"
  | "ambiguous"
  | "unmatched"
  | "missing_attribution"
  | "ready"
  | "delayed"
  | "budget_deferred"
  | "sparse_ready";

export type AttributionTruthPrimaryState =
  | AttributionTruthState
  | "idle"
  | "missing"
  | "mixed";

export type AttributionTruthQueueSample = {
  observationId: string;
  episodeId: string;
  traceId: string | null;
  status: string;
  gate: "ready" | "delayed" | "budget_deferred";
  reason: string;
  feedbackRichness: TeacherFeedbackRichness;
  createdAt: number;
};

export type AttributionTruthLatestNonExact = {
  observationId: string;
  episodeId: string;
  traceId: string | null;
  bindingMode: BrainObservationBindingMode;
  attributionQuality: "fallback" | "unbound";
  feedbackRichness: TeacherFeedbackRichness;
  confidence: number | null;
  reason: string | null;
  evaluatedAt: number | null;
};

export type AttributionTruthInput = {
  observationAttribution?: {
    totalObservationCount?: number;
    teacherEvaluationCount?: number;
    completedWithoutEvaluationCount?: number;
    attributionQuality?: {
      exact?: number;
      fallback?: number;
      unbound?: number;
    };
    latestAmbiguous?: AttributionTruthLatestNonExact | null;
    latestUnmatched?: AttributionTruthLatestNonExact | null;
    latestNonExact?: AttributionTruthLatestNonExact | null;
  } | null;
  teacherTruth?: {
    queue?: {
      budgetPerTick?: number;
      delayMs?: number;
      pendingCount?: number;
      pendingFollowupCount?: number;
      pendingTeacherCount?: number;
      readyCount?: number;
      delayedCount?: number;
      budgetDeferredCount?: number;
      sparseReadyCount?: number;
      richReadyCount?: number;
      sample?: AttributionTruthQueueSample[];
      detail?: string;
    } | null;
  } | null;
};

export type AttributionTruthSummary = {
  contract: typeof ATTRIBUTION_TRUTH_CONTRACT;
  visible: boolean;
  primaryState: AttributionTruthPrimaryState;
  activeStates: AttributionTruthState[];
  detail: string;
  counts: {
    observationCount: number;
    evaluatedCount: number;
    completedWithoutEvaluationCount: number;
    matchedCount: number;
    ambiguousCount: number;
    unmatchedCount: number;
    pendingCount: number;
    pendingFollowupCount: number;
    pendingTeacherCount: number;
    readyCount: number;
    delayedCount: number;
    budgetDeferredCount: number;
    sparseReadyCount: number;
    richReadyCount: number;
  };
  states: {
    matched: {
      count: number;
      detail: string;
    };
    ambiguous: {
      count: number;
      detail: string;
    };
    unmatched: {
      count: number;
      detail: string;
    };
    missingAttribution: {
      count: number;
      detail: string;
    };
    ready: {
      count: number;
      detail: string;
    };
    delayed: {
      count: number;
      detail: string;
    };
    budgetDeferred: {
      count: number;
      detail: string;
    };
    sparseReady: {
      count: number;
      detail: string;
    };
  };
  gating: {
    budgetPerTick: number | null;
    delayMs: number | null;
    detail: string;
  };
  latest: {
    ambiguous: AttributionTruthLatestNonExact | null;
    unmatched: AttributionTruthLatestNonExact | null;
    followupPending: AttributionTruthQueueSample | null;
    delayed: AttributionTruthQueueSample | null;
    budgetDeferred: AttributionTruthQueueSample | null;
    sparseReady: AttributionTruthQueueSample | null;
  };
};

function summarizeStateCounts(counts: Array<{ label: string; count: number }>): string {
  return counts
    .filter((entry) => entry.count > 0)
    .map((entry) => `${entry.count} ${entry.label}`)
    .join(", ");
}

export function summarizeAttributionTruth(input: AttributionTruthInput | null | undefined): AttributionTruthSummary {
  const observationAttribution = input?.observationAttribution ?? null;
  const queue = input?.teacherTruth?.queue ?? null;
  const latestNonExact = observationAttribution?.latestNonExact ?? null;
  const latestAmbiguous = observationAttribution?.latestAmbiguous
    ?? (latestNonExact?.attributionQuality === "fallback" ? latestNonExact : null);
  const latestUnmatched = observationAttribution?.latestUnmatched
    ?? (latestNonExact?.attributionQuality === "unbound" ? latestNonExact : null);
  const sample = Array.isArray(queue?.sample) ? queue.sample : [];

  const counts = {
    observationCount: normalizeNullableNumber(observationAttribution?.totalObservationCount) ?? 0,
    evaluatedCount: normalizeNullableNumber(observationAttribution?.teacherEvaluationCount) ?? 0,
    completedWithoutEvaluationCount: normalizeNullableNumber(observationAttribution?.completedWithoutEvaluationCount) ?? 0,
    matchedCount: normalizeNullableNumber(observationAttribution?.attributionQuality?.exact) ?? 0,
    ambiguousCount: normalizeNullableNumber(observationAttribution?.attributionQuality?.fallback) ?? 0,
    unmatchedCount: normalizeNullableNumber(observationAttribution?.attributionQuality?.unbound) ?? 0,
    pendingCount: normalizeNullableNumber(queue?.pendingCount) ?? 0,
    pendingFollowupCount: normalizeNullableNumber(queue?.pendingFollowupCount) ?? 0,
    pendingTeacherCount: normalizeNullableNumber(queue?.pendingTeacherCount) ?? 0,
    readyCount: normalizeNullableNumber(queue?.readyCount) ?? 0,
    delayedCount: normalizeNullableNumber(queue?.delayedCount) ?? 0,
    budgetDeferredCount: normalizeNullableNumber(queue?.budgetDeferredCount) ?? 0,
    sparseReadyCount: normalizeNullableNumber(queue?.sparseReadyCount) ?? 0,
    richReadyCount: normalizeNullableNumber(queue?.richReadyCount) ?? 0,
  };

  const visible = observationAttribution !== null || queue !== null;
  const activeStates: AttributionTruthState[] = [];
  if (counts.unmatchedCount > 0) {
    pushUnique(activeStates, "unmatched");
  }
  if (counts.ambiguousCount > 0) {
    pushUnique(activeStates, "ambiguous");
  }
  if (counts.completedWithoutEvaluationCount > 0) {
    pushUnique(activeStates, "missing_attribution");
  }
  if (counts.delayedCount > 0) {
    pushUnique(activeStates, "delayed");
  }
  if (counts.budgetDeferredCount > 0) {
    pushUnique(activeStates, "budget_deferred");
  }
  if (counts.sparseReadyCount > 0) {
    pushUnique(activeStates, "sparse_ready");
  }
  if (counts.readyCount > 0) {
    pushUnique(activeStates, "ready");
  }
  if (counts.matchedCount > 0) {
    pushUnique(activeStates, "matched");
  }

  let primaryState: AttributionTruthPrimaryState;
  if (!visible) {
    primaryState = "missing";
  } else if (activeStates.length > 1) {
    primaryState = "mixed";
  } else if (activeStates.length === 1) {
    primaryState = activeStates[0];
  } else {
    primaryState = "idle";
  }

  const latest = {
    ambiguous: latestAmbiguous,
    unmatched: latestUnmatched,
    followupPending: sample.find((entry) => entry.status === "pending_followup") ?? null,
    delayed: sample.find((entry) => entry.gate === "delayed") ?? null,
    budgetDeferred: sample.find((entry) => entry.gate === "budget_deferred") ?? null,
    sparseReady: sample.find((entry) => entry.gate !== "delayed" && entry.feedbackRichness === "sparse") ?? null,
  };

  const gating = {
    budgetPerTick: normalizeNullableNumber(queue?.budgetPerTick),
    delayMs: normalizeNullableNumber(queue?.delayMs),
    detail:
      normalizeNullableString(queue?.detail)
      ?? "teacher gating truth is not visible in the current status surface",
  };

  const states = {
    matched: {
      count: counts.matchedCount,
      detail: "exact-bound teacher evaluations matched a concrete route/decision binding",
    },
    ambiguous: {
      count: counts.ambiguousCount,
      detail: "fallback-bound teacher evaluations are inferred rather than exact-bound",
    },
    unmatched: {
      count: counts.unmatchedCount,
      detail: "teacher evaluations have no surviving route binding",
    },
    missingAttribution: {
      count: counts.completedWithoutEvaluationCount,
      detail: "completed observations still have no recorded teacher attribution",
    },
    ready: {
      count: counts.readyCount,
      detail: "observations are ready for teacher scoring",
    },
    delayed: {
      count: counts.delayedCount,
      detail: "observations are still waiting for follow-up or the teacher delay window",
    },
    budgetDeferred: {
      count: counts.budgetDeferredCount,
      detail: "ready observations are deferred behind older work by the per-tick teacher budget",
    },
    sparseReady: {
      count: counts.sparseReadyCount,
      detail: "ready observations are sparse-feedback cases made eligible only by the teacher delay",
    },
  };

  let detail = "attribution truth is not visible in the current status surface";
  if (visible) {
    if (
      counts.observationCount === 0
      && counts.evaluatedCount === 0
      && counts.completedWithoutEvaluationCount === 0
      && counts.pendingCount === 0
    ) {
      detail = "no attribution observations or teacher gating signals are recorded yet";
    } else if (primaryState === "mixed") {
      detail = `attribution truth is mixed: ${
        summarizeStateCounts([
          { label: "unmatched", count: counts.unmatchedCount },
          { label: "ambiguous", count: counts.ambiguousCount },
          { label: "missing-attribution", count: counts.completedWithoutEvaluationCount },
          { label: "delayed", count: counts.delayedCount },
          { label: "pending-followup", count: counts.pendingFollowupCount },
          { label: "pending-teacher", count: counts.pendingTeacherCount },
          { label: "budget-deferred", count: counts.budgetDeferredCount },
          { label: "sparse-ready", count: counts.sparseReadyCount },
          { label: "ready", count: counts.readyCount },
          { label: "matched", count: counts.matchedCount },
        ])
      }`;
    } else if (primaryState === "unmatched") {
      detail = `${counts.unmatchedCount} teacher evaluation(s) are unmatched to a route binding`;
    } else if (primaryState === "ambiguous") {
      detail = `${counts.ambiguousCount} teacher evaluation(s) use fallback attribution rather than an exact match`;
    } else if (primaryState === "missing_attribution") {
      detail = `${counts.completedWithoutEvaluationCount} completed observation(s) still have no recorded teacher attribution`;
    } else if (primaryState === "delayed") {
      detail = `${counts.delayedCount} observation(s) are still delayed before teacher scoring; pending_followup=${counts.pendingFollowupCount}, pending_teacher=${counts.pendingTeacherCount}`;
    } else if (primaryState === "budget_deferred") {
      detail = `${counts.budgetDeferredCount} ready observation(s) are deferred by the teacher budget`;
    } else if (primaryState === "sparse_ready") {
      detail = `${counts.sparseReadyCount} ready observation(s) are sparse-feedback cases made eligible by delay`;
    } else if (primaryState === "ready") {
      detail = `${counts.readyCount} observation(s) are ready for teacher scoring; pending_followup=${counts.pendingFollowupCount}, pending_teacher=${counts.pendingTeacherCount}`;
    } else if (primaryState === "matched") {
      detail =
        counts.completedWithoutEvaluationCount > 0
          ? `${counts.matchedCount} teacher evaluation(s) are exact-bound, but ${counts.completedWithoutEvaluationCount} completed observation(s) still have no recorded teacher attribution`
          : `${counts.matchedCount} teacher evaluation(s) are exact-bound with no ambiguous, unmatched, delayed, or missing teacher attribution visible`;
    } else {
      detail = "no attribution observations or teacher gating signals are recorded yet";
    }
  }

  return {
    contract: ATTRIBUTION_TRUTH_CONTRACT,
    visible,
    primaryState,
    activeStates,
    detail,
    counts,
    states,
    gating,
    latest,
  };
}
