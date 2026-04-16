import { describe, expect, it } from "vitest";
import {
  buildEpisodeResolutionEventV1,
  buildRetryOrInterventionEventV1,
  buildRouteServedEventV1,
  buildTurnOutcomeEventV1,
  validateEpisodeResolutionEventV1,
  validateRetryOrInterventionEventV1,
  validateRouteServedEventV1,
  validateTurnOutcomeEventV1,
} from "../../src/brain-core/route-outcome-events.js";

const identity = {
  conversationId: 42,
  episodeId: "episode_live_outcome_fixture_01",
  traceId: "trace_live_outcome_fixture_01",
  selectionDigest: "sel_digest_fixture_01",
};

describe("route outcome events", () => {
  it("builds and validates a route served event", () => {
    const event = buildRouteServedEventV1({
      identity,
      modeRequested: "learned_route",
      modeEffective: "learned_route",
      usedLearnedRouteFn: true,
      activationKind: "learned_nontrivial",
      activePackId: "pack_live_outcome_fixture_01",
      routerIdentity: "router_live_outcome_fixture_01",
      candidateNodeIds: ["doc_1", "doc_2"],
      selectedNodeIds: ["doc_1"],
      selectedTraversalNodeIds: ["doc_1"],
      selectedPathNodeIds: ["root", "doc_1"],
      toolCount: 1,
      promptTokensEstimate: 123,
      latencyMs: 9,
      eventAt: "2026-04-16T03:16:00.000Z",
    });

    expect(validateRouteServedEventV1(event)).toMatchObject({ valid: true });
    expect(event).toMatchObject({
      contract: "ocb.route_served.v1",
      used_learned_route_fn: true,
      activation_kind: "learned_nontrivial",
      selected_node_ids: ["doc_1"],
    });
  });

  it("builds and validates a turn outcome event", () => {
    const event = buildTurnOutcomeEventV1({
      identity,
      outcomeClass: "correction",
      correctionRequired: true,
      source: "user_followup",
      followUpClass: "correction",
      routeIntegrityClass: "exact_serve",
      reason: "user corrected the factual claim",
      closedAt: "2026-04-16T03:17:00.000Z",
    });

    expect(validateTurnOutcomeEventV1(event)).toMatchObject({ valid: true });
    expect(event).toMatchObject({
      contract: "ocb.turn_outcome.v1",
      outcome_class: "correction",
      correction_required: true,
    });
  });

  it("builds and validates a retry/intervention event", () => {
    const event = buildRetryOrInterventionEventV1({
      identity,
      triggerKind: "assistant_recovery",
      triggeredBy: "assistant",
      reasonClass: "incomplete",
      retryCountDelta: 1,
      interventionCountDelta: 0,
      triggeredAt: "2026-04-16T03:18:00.000Z",
    });

    expect(validateRetryOrInterventionEventV1(event)).toMatchObject({ valid: true });
    expect(event).toMatchObject({
      contract: "ocb.retry_or_intervention.v1",
      trigger_kind: "assistant_recovery",
      retry_count_delta: 1,
    });
  });

  it("builds and validates an episode resolution event", () => {
    const event = buildEpisodeResolutionEventV1({
      identity,
      resolutionClass: "completed",
      resolved: true,
      resolutionUserTurnIndex: 3,
      resolutionAssistantTurnIndex: 4,
      totalRetryCount: 1,
      totalInterventionCount: 0,
      finalOutcomeQuality: "accepted",
      resolvedAt: "2026-04-16T03:19:00.000Z",
    });

    expect(validateEpisodeResolutionEventV1(event)).toMatchObject({ valid: true });
    expect(event).toMatchObject({
      contract: "ocb.episode_resolution.v1",
      resolution_class: "completed",
      resolved: true,
    });
  });

  it("rejects inconsistent correction and retry payloads", () => {
    const badOutcome = buildTurnOutcomeEventV1({
      identity,
      outcomeClass: "retry",
      correctionRequired: false,
      source: "runtime_recovery",
    });
    const badRetry = buildRetryOrInterventionEventV1({
      identity,
      triggerKind: "tool_rerun",
      triggeredBy: "runtime",
      retryCountDelta: 0,
      interventionCountDelta: 0,
    });

    expect(validateTurnOutcomeEventV1(badOutcome)).toMatchObject({ valid: false });
    expect(validateRetryOrInterventionEventV1(badRetry)).toMatchObject({ valid: false });
  });
});
