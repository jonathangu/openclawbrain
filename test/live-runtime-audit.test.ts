import { describe, expect, it } from "vitest";

import {
  summarizeAttributionTruth,
  summarizeTeacherLoopTruth,
} from "../src/live-runtime-audit.js";

describe("summarizeTeacherLoopTruth", () => {
  it("treats a fresh watch heartbeat with no teacher artifacts as healthy and not stale", () => {
    expect(
      summarizeTeacherLoopTruth({
        failureMode: "none",
        lastNoOpReason: "no_teacher_artifacts",
        latestFreshness: "stale",
        queueDepth: 0,
        running: false,
        watchState: "watching",
      }),
    ).toEqual({
      healthy: true,
      idle: true,
      stale: false,
    });
  });

  it("keeps stale snapshots stale even without an explicit failure", () => {
    expect(
      summarizeTeacherLoopTruth({
        failureMode: "none",
        lastNoOpReason: "none",
        latestFreshness: "fresh",
        queueDepth: 0,
        running: false,
        watchState: "stale_snapshot",
      }),
    ).toEqual({
      healthy: false,
      idle: true,
      stale: true,
    });
  });

  it("keeps stale snapshots unhealthy even when the last cycle was a no-op and prior artifacts stay fresh", () => {
    expect(
      summarizeTeacherLoopTruth({
        failureMode: "none",
        lastNoOpReason: "no_teacher_artifacts",
        latestFreshness: "fresh",
        queueDepth: 0,
        running: false,
        watchState: "stale_snapshot",
      }),
    ).toEqual({
      healthy: false,
      idle: true,
      stale: true,
    });
  });

  it("keeps explicit teacher failures unhealthy", () => {
    expect(
      summarizeTeacherLoopTruth({
        failureMode: "provider_error",
        lastNoOpReason: "none",
        latestFreshness: "fresh",
        queueDepth: 1,
        running: false,
        watchState: "watching",
      }),
    ).toEqual({
      healthy: false,
      idle: false,
      stale: false,
    });
  });

  it("treats lagging watch heartbeats as aging instead of healthy or stale", () => {
    expect(
      summarizeTeacherLoopTruth({
        failureMode: "none",
        lastNoOpReason: "no_teacher_artifacts",
        latestFreshness: "fresh",
        queueDepth: 0,
        running: false,
        watch: {
          state: "lagging",
        },
      }),
    ).toEqual({
      healthy: false,
      idle: true,
      stale: false,
    });
  });
});

describe("summarizeAttributionTruth", () => {
  it("marks attribution truth missing when neither evaluation nor queue surfaces are present", () => {
    expect(summarizeAttributionTruth({})).toEqual({
      contract: "openclawbrain_attribution_truth.v1",
      visible: false,
      primaryState: "missing",
      activeStates: [],
      detail: "attribution truth is not visible in the current status surface",
      counts: {
        observationCount: 0,
        evaluatedCount: 0,
        completedWithoutEvaluationCount: 0,
        matchedCount: 0,
        ambiguousCount: 0,
        unmatchedCount: 0,
        pendingCount: 0,
        pendingFollowupCount: 0,
        pendingTeacherCount: 0,
        readyCount: 0,
        delayedCount: 0,
        budgetDeferredCount: 0,
        sparseReadyCount: 0,
        richReadyCount: 0,
      },
      states: {
        matched: {
          count: 0,
          detail: "exact-bound teacher evaluations matched a concrete route/decision binding",
        },
        ambiguous: {
          count: 0,
          detail: "fallback-bound teacher evaluations are inferred rather than exact-bound",
        },
        unmatched: {
          count: 0,
          detail: "teacher evaluations have no surviving route binding",
        },
        missingAttribution: {
          count: 0,
          detail: "completed observations still have no recorded teacher attribution",
        },
        ready: {
          count: 0,
          detail: "observations are ready for teacher scoring",
        },
        delayed: {
          count: 0,
          detail: "observations are still waiting for follow-up or the teacher delay window",
        },
        budgetDeferred: {
          count: 0,
          detail: "ready observations are deferred behind older work by the per-tick teacher budget",
        },
        sparseReady: {
          count: 0,
          detail: "ready observations are sparse-feedback cases made eligible only by the teacher delay",
        },
      },
      gating: {
        budgetPerTick: null,
        delayMs: null,
        detail: "teacher gating truth is not visible in the current status surface",
      },
      latest: {
        ambiguous: null,
        unmatched: null,
        followupPending: null,
        delayed: null,
        budgetDeferred: null,
        sparseReady: null,
      },
    });
  });

  it("surfaces ambiguous, delayed, budget-deferred, and sparse-ready attribution truth together", () => {
    const summary = summarizeAttributionTruth({
      observationAttribution: {
        totalObservationCount: 4,
        teacherEvaluationCount: 3,
        latestAmbiguous: {
          observationId: "bo_ambiguous",
          episodeId: "ep_ambiguous",
          traceId: "bt_ambiguous",
          bindingMode: "trace_id",
          attributionQuality: "fallback",
          feedbackRichness: "followup_only",
          confidence: 0.55,
          reason: "follow-up stayed ambiguous",
          evaluatedAt: 123,
        },
        latestUnmatched: {
          observationId: "bo_unmatched",
          episodeId: "ep_unmatched",
          traceId: null,
          bindingMode: "unbound",
          attributionQuality: "unbound",
          feedbackRichness: "sparse",
          confidence: 0.31,
          reason: "teacher review could not recover a binding",
          evaluatedAt: 456,
        },
        attributionQuality: {
          exact: 1,
          fallback: 1,
          unbound: 1,
        },
      },
      teacherTruth: {
        queue: {
          budgetPerTick: 20,
          delayMs: 10_000,
          pendingCount: 2,
          pendingFollowupCount: 2,
          pendingTeacherCount: 0,
          readyCount: 1,
          delayedCount: 1,
          budgetDeferredCount: 1,
          sparseReadyCount: 1,
          richReadyCount: 0,
          sample: [
            {
              observationId: "bo_delayed",
              episodeId: "ep_delayed",
              traceId: "bt_delayed",
              status: "pending_followup",
              gate: "delayed",
              reason: "waiting for follow-up or the teacher delay window",
              feedbackRichness: "sparse",
              createdAt: 111,
            },
            {
              observationId: "bo_budget",
              episodeId: "ep_budget",
              traceId: "bt_budget",
              status: "pending_followup",
              gate: "budget_deferred",
              reason: "ready, but deferred behind older observations by the per-tick teacher budget",
              feedbackRichness: "sparse",
              createdAt: 222,
            },
          ],
          detail: "pending_followup=2, pending_teacher=0; ready=1, delayed=1, budget_deferred=1, sparse_ready=1, rich_ready=0",
        },
      },
    });

    expect(summary).toMatchObject({
      contract: "openclawbrain_attribution_truth.v1",
      visible: true,
      primaryState: "mixed",
      activeStates: ["unmatched", "ambiguous", "delayed", "budget_deferred", "sparse_ready", "ready", "matched"],
      counts: {
        observationCount: 4,
        evaluatedCount: 3,
        completedWithoutEvaluationCount: 0,
        matchedCount: 1,
        ambiguousCount: 1,
        unmatchedCount: 1,
        pendingCount: 2,
        pendingFollowupCount: 2,
        pendingTeacherCount: 0,
        readyCount: 1,
        delayedCount: 1,
        budgetDeferredCount: 1,
        sparseReadyCount: 1,
        richReadyCount: 0,
      },
      gating: {
        budgetPerTick: 20,
        delayMs: 10_000,
      },
      latest: {
        ambiguous: {
          observationId: "bo_ambiguous",
          attributionQuality: "fallback",
        },
        unmatched: {
          observationId: "bo_unmatched",
          attributionQuality: "unbound",
        },
        followupPending: {
          observationId: "bo_delayed",
          status: "pending_followup",
        },
        delayed: {
          observationId: "bo_delayed",
          gate: "delayed",
        },
        budgetDeferred: {
          observationId: "bo_budget",
          gate: "budget_deferred",
        },
        sparseReady: {
          observationId: "bo_budget",
          feedbackRichness: "sparse",
        },
      },
    });
    expect(summary.detail).toContain("attribution truth is mixed");
  });

  it("treats completed observations without teacher attribution as explicit missing attribution rather than matched truth", () => {
    expect(summarizeAttributionTruth({
      observationAttribution: {
        totalObservationCount: 1,
        teacherEvaluationCount: 0,
        completedWithoutEvaluationCount: 1,
        attributionQuality: {
          exact: 0,
          fallback: 0,
          unbound: 0,
        },
      },
      teacherTruth: {
        queue: {
          budgetPerTick: 20,
          delayMs: 10_000,
          pendingCount: 0,
          pendingFollowupCount: 0,
          pendingTeacherCount: 0,
          readyCount: 0,
          delayedCount: 0,
          budgetDeferredCount: 0,
          sparseReadyCount: 0,
          richReadyCount: 0,
          sample: [],
          detail: "no teacher observations are pending",
        },
      },
    })).toMatchObject({
      visible: true,
      primaryState: "missing_attribution",
      activeStates: ["missing_attribution"],
      counts: {
        completedWithoutEvaluationCount: 1,
      },
      states: {
        missingAttribution: {
          count: 1,
        },
      },
      detail: "1 completed observation(s) still have no recorded teacher attribution",
    });
  });
});
