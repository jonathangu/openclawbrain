import { describe, expect, it } from "vitest";
import type { DecisionTrace } from "../../src/brain-core/types.js";
import {
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
});
