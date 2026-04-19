import { describe, expect, it } from "vitest";
import type { DecisionTrace } from "../../src/brain-core/types.js";
import {
  buildRecentRouteDecisionSummaryV1,
  buildRouteDecisionEventV1,
  materializeRouteDecisionEventFromTraceV1,
  summarizeRouteDecisionEventV1,
  validateRouteDecisionEventV1,
} from "../../src/brain-core/route-decision-event.js";

function makeDecisionTraceFixture(overrides?: Partial<DecisionTrace>): DecisionTrace {
  const base = {
    id: "trace_route_decision_event_fixture_01",
    episodeId: "episode_route_decision_event_fixture_01",
    routeTrace: {
      routerIdentity: "brain-graph-traverse.v2",
      selectedNodeIds: ["doc_1"],
      selectionMetadata: {
        candidateCount: 2,
        budgetChars: 512,
        queryBudgetChars: 256,
        routeSelectionMs: 4,
        totalQueryMs: 11,
        decisionPointSnapshots: [
          {
            chosenActionProbability: 0.84,
            stopProbability: 0.16,
            stopReason: null,
            routeContext: {
              selectedNodeIds: ["doc_1"],
            },
          },
        ],
      },
    },
  } as unknown as DecisionTrace;
  return { ...base, ...overrides };
}

describe("route decision event", () => {
  it("builds and validates an activated route decision event", () => {
    const event = buildRouteDecisionEventV1({
      traceId: "trace_a",
      episodeId: "episode_a",
      routeFnVersion: "brain-graph-traverse.v2",
      activated: true,
      confidence: 0.93,
      selectedContextCount: 2,
      selectedTokenBudget: 256,
      decisionPointCount: 3,
      activationThreshold: 0.55,
      costEstimateMs: 14,
      timestamp: "2026-04-16T02:20:00.000Z",
    });

    expect(validateRouteDecisionEventV1(event)).toMatchObject({ valid: true });
    expect(summarizeRouteDecisionEventV1(event)).toMatchObject({
      traceId: "trace_a",
      activated: true,
      confidence: 0.93,
      selectedContextCount: 2,
      decisionPointCount: 3,
      stopReason: null,
    });
  });

  it("materializes an activated event from a decision trace", () => {
    const event = materializeRouteDecisionEventFromTraceV1({
      trace: makeDecisionTraceFixture(),
      activationThreshold: 0.6,
    });

    expect(validateRouteDecisionEventV1(event)).toMatchObject({ valid: true });
    expect(event).toMatchObject({
      trace_id: "trace_route_decision_event_fixture_01",
      episode_id: "episode_route_decision_event_fixture_01",
      route_fn_version: "brain-graph-traverse.v2",
      activated: true,
      confidence: 0.84,
      selected_context_count: 1,
      selected_token_budget: 256,
      stop_reason: null,
      decision_point_count: 1,
      activation_threshold: 0.6,
      cost_estimate_ms: 11,
    });
  });

  it("materializes a non-activated event with a no-candidates stop reason", () => {
    const event = materializeRouteDecisionEventFromTraceV1({
      trace: makeDecisionTraceFixture({
        routeTrace: {
          routerIdentity: "brain-graph-traverse.v2",
          selectedNodeIds: [],
          selectionMetadata: {
            candidateCount: 0,
            budgetChars: 512,
            queryBudgetChars: 256,
            routeSelectionMs: 4,
            totalQueryMs: 9,
            decisionPointSnapshots: [
              {
                chosenActionProbability: 0.12,
                stopProbability: 0.88,
                stopReason: "no_traversable_candidates",
                routeContext: {
                  selectedNodeIds: [],
                },
              },
            ],
          },
        },
      } as unknown as DecisionTrace),
    });

    expect(validateRouteDecisionEventV1(event)).toMatchObject({ valid: true });
    expect(event).toMatchObject({
      activated: false,
      confidence: 0.88,
      selected_context_count: 0,
      stop_reason: "no_candidates",
      decision_point_count: 1,
    });
  });

  it("builds a bounded recent route-decision summary from the latest valid events", () => {
    const summary = buildRecentRouteDecisionSummaryV1([
      buildRouteDecisionEventV1({
        traceId: "trace_old",
        episodeId: "episode_old",
        routeFnVersion: "brain-graph-traverse.v2",
        activated: false,
        confidence: 0.2,
        predictedRegretOfAbstaining: 0.01,
        selectedContextCount: 0,
        selectedTokenBudget: 128,
        stopReason: "activation_threshold_not_met",
        decisionPointCount: 1,
        activationThreshold: 0.6,
        costEstimateMs: 4,
        timestamp: "2026-04-16T02:19:00.000Z",
      }),
      buildRouteDecisionEventV1({
        traceId: "trace_a",
        episodeId: "episode_a",
        routeFnVersion: "brain-graph-traverse.v2",
        activated: true,
        confidence: 0.92,
        predictedUtility: 0.48,
        predictedRegretOfAbstaining: 0.12,
        selectedContextCount: 2,
        selectedTokenBudget: 256,
        decisionPointCount: 3,
        activationThreshold: 0.55,
        costEstimateMs: 12,
        timestamp: "2026-04-16T02:20:00.000Z",
      }),
      buildRouteDecisionEventV1({
        traceId: "trace_b",
        episodeId: "episode_b",
        routeFnVersion: "brain-graph-traverse.v2",
        activated: false,
        confidence: 0.41,
        predictedUtility: -0.08,
        predictedRegretOfAbstaining: 0.03,
        selectedContextCount: 0,
        selectedTokenBudget: 256,
        stopReason: "no_candidates",
        decisionPointCount: 2,
        activationThreshold: 0.55,
        costEstimateMs: 7,
        timestamp: "2026-04-16T02:21:00.000Z",
      }),
      {
        contract: "ocb.route_decision.v1",
        invalid: true,
      },
    ], 3);

    expect(summary).toMatchObject({
      contract: "openclawbrain_recent_route_decision_summary.v1",
      windowSize: 3,
      sampleSize: 2,
      activation: {
        activatedCount: 1,
        nonActivatedCount: 1,
        activationRate: 0.5,
      },
      coverage: {
        confidence: {
          observedCount: 2,
          observedRate: 1,
        },
        predictedUtility: {
          observedCount: 2,
          observedRate: 1,
        },
      },
      selectedContextCount: {
        observedCount: 2,
        total: 2,
        mean: 1,
        max: 2,
      },
      decisionPointCount: {
        observedCount: 2,
        total: 5,
        mean: 2.5,
        max: 3,
      },
      costEstimateMs: {
        observedCount: 2,
        total: 19,
        mean: 9.5,
        max: 12,
      },
      predictedUtility: {
        observedCount: 2,
        observedRate: 1,
        mean: 0.2,
        positiveCount: 1,
        nonPositiveCount: 1,
      },
      stopReasonCounts: {
        budget_exhausted: 0,
        stop_local: 0,
        stop_global: 0,
        no_candidates: 1,
        activation_threshold_not_met: 0,
      },
    });
    expect(summary.detail).toContain("1/2 activated");
    expect(summary.detail).toContain("topStop=no_candidates=1");
  });
});
