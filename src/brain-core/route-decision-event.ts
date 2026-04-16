import { createHash } from "node:crypto";
import { Type, type Static } from "@sinclair/typebox";
import { Value } from "@sinclair/typebox/value";
import type { DecisionTrace, TrajectoryStopReason } from "./types.js";

export const ROUTE_DECISION_EVENT_VERSION_V1 = 1 as const;
export const ROUTE_DECISION_EVENT_CONTRACT_V1 = "ocb.route_decision.v1";

const STOP_REASONS = [
  "budget_exhausted",
  "stop_local",
  "stop_global",
  "no_candidates",
  "activation_threshold_not_met",
] as const;

export type RouteDecisionStopReason = (typeof STOP_REASONS)[number];

export const RouteDecisionEventSchemaV1 = Type.Object(
  {
    schema_version: Type.Literal(ROUTE_DECISION_EVENT_VERSION_V1),
    contract: Type.Literal(ROUTE_DECISION_EVENT_CONTRACT_V1),
    event_id: Type.String({ minLength: 1 }),
    trace_id: Type.String({ minLength: 1 }),
    episode_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    timestamp: Type.String({ minLength: 1 }),
    route_fn_version: Type.String({ minLength: 1 }),
    activated: Type.Boolean(),
    confidence: Type.Union([Type.Number({ minimum: 0, maximum: 1 }), Type.Null()]),
    predicted_utility: Type.Union([Type.Number(), Type.Null()]),
    predicted_regret_of_abstaining: Type.Union([Type.Number(), Type.Null()]),
    selected_context_count: Type.Integer({ minimum: 0 }),
    selected_token_budget: Type.Union([Type.Integer({ minimum: 0 }), Type.Null()]),
    stop_reason: Type.Union([
      ...STOP_REASONS.map((reason) => Type.Literal(reason)),
      Type.Null(),
    ]),
    decision_point_count: Type.Integer({ minimum: 0 }),
    activation_threshold: Type.Union([Type.Number(), Type.Null()]),
    cost_estimate_ms: Type.Union([Type.Number({ minimum: 0 }), Type.Null()]),
  },
  { additionalProperties: false },
);

export type RouteDecisionEventV1 = Static<typeof RouteDecisionEventSchemaV1>;

export interface ContractValidationResultV1 {
  contract: string;
  valid: boolean;
  issues: string[];
}

export function validateRouteDecisionEventV1(value: unknown): ContractValidationResultV1 {
  const contract = ROUTE_DECISION_EVENT_CONTRACT_V1;
  const valid = Value.Check(RouteDecisionEventSchemaV1, value);
  if (!valid) {
    return { contract, valid: false, issues: [`${contract}/schema_validation_failed`] };
  }

  const event = value as RouteDecisionEventV1;
  const issues: string[] = [];

  if (event.activated && event.selected_context_count === 0) {
    issues.push(`${contract}/activated event must have selected_context_count > 0`);
  }

  if (!event.activated && event.stop_reason === null) {
    issues.push(`${contract}/non-activated event should have a stop_reason`);
  }

  if (event.confidence !== null && (event.confidence < 0 || event.confidence > 1)) {
    issues.push(`${contract}/confidence must be between 0 and 1`);
  }

  return issues.length === 0
    ? { contract, valid: true, issues: [] }
    : { contract, valid: false, issues };
}

function hashStableParts(parts: Array<string | number | null | undefined>): string {
  return createHash("sha256").update(parts.map((part) => String(part ?? "")).join("\u001f")).digest("hex").slice(0, 16);
}

export interface BuildRouteDecisionEventParams {
  traceId: string;
  episodeId?: string | null;
  routeFnVersion: string;
  activated: boolean;
  confidence?: number | null;
  predictedUtility?: number | null;
  predictedRegretOfAbstaining?: number | null;
  selectedContextCount: number;
  selectedTokenBudget?: number | null;
  stopReason?: RouteDecisionStopReason | null;
  decisionPointCount: number;
  activationThreshold?: number | null;
  costEstimateMs?: number | null;
  timestamp?: string;
}

export function buildRouteDecisionEventV1(params: BuildRouteDecisionEventParams): RouteDecisionEventV1 {
  const timestamp = params.timestamp ?? new Date().toISOString();
  const eventId = `rde_${hashStableParts([
    params.traceId,
    params.routeFnVersion,
    String(params.activated),
    timestamp,
  ])}`;

  return {
    schema_version: ROUTE_DECISION_EVENT_VERSION_V1,
    contract: ROUTE_DECISION_EVENT_CONTRACT_V1,
    event_id: eventId,
    trace_id: params.traceId,
    episode_id: params.episodeId ?? null,
    timestamp,
    route_fn_version: params.routeFnVersion,
    activated: params.activated,
    confidence: params.confidence ?? null,
    predicted_utility: params.predictedUtility ?? null,
    predicted_regret_of_abstaining: params.predictedRegretOfAbstaining ?? null,
    selected_context_count: params.selectedContextCount,
    selected_token_budget: params.selectedTokenBudget ?? null,
    stop_reason: params.stopReason ?? null,
    decision_point_count: params.decisionPointCount,
    activation_threshold: params.activationThreshold ?? null,
    cost_estimate_ms: params.costEstimateMs ?? null,
  };
}

export interface RouteDecisionEventSummaryV1 {
  eventId: string;
  traceId: string;
  activated: boolean;
  confidence: number | null;
  selectedContextCount: number;
  decisionPointCount: number;
  stopReason: RouteDecisionStopReason | null;
}

export function summarizeRouteDecisionEventV1(event: RouteDecisionEventV1): RouteDecisionEventSummaryV1 {
  return {
    eventId: event.event_id,
    traceId: event.trace_id,
    activated: event.activated,
    confidence: event.confidence,
    selectedContextCount: event.selected_context_count,
    decisionPointCount: event.decision_point_count,
    stopReason: event.stop_reason,
  };
}

function mapTrajectoryStopReasonToEventStopReason(stopReason: TrajectoryStopReason | null | undefined): RouteDecisionStopReason {
  switch (stopReason) {
    case "policy_stop":
      return "stop_local";
    case "selection_budget_exhausted":
    case "fanout_cap":
    case "frontier_cap":
      return "budget_exhausted";
    case "no_traversable_candidates":
    case "missing_target_node":
      return "no_candidates";
    case "interrupt":
    default:
      return "activation_threshold_not_met";
  }
}

export function materializeRouteDecisionEventFromTraceV1(params: {
  trace: DecisionTrace;
  routeFnVersion?: string | null;
  activationThreshold?: number | null;
  predictedUtility?: number | null;
  predictedRegretOfAbstaining?: number | null;
  timestamp?: string;
}): RouteDecisionEventV1 {
  const selectionMetadata = params.trace.routeTrace?.selectionMetadata ?? null;
  const snapshots = selectionMetadata?.decisionPointSnapshots ?? [];
  const lastSnapshot = snapshots.length > 0 ? snapshots[snapshots.length - 1] : null;
  const selectedNodeIds = params.trace.routeTrace?.selectedNodeIds ?? lastSnapshot?.routeContext.selectedNodeIds ?? [];
  const activated = selectedNodeIds.length > 0;

  const inferredStopReason = activated
    ? null
    : selectionMetadata?.candidateCount === 0
      ? "no_candidates"
      : mapTrajectoryStopReasonToEventStopReason(lastSnapshot?.stopReason ?? null);

  return buildRouteDecisionEventV1({
    traceId: params.trace.id,
    episodeId: params.trace.episodeId,
    routeFnVersion: params.routeFnVersion ?? params.trace.routeTrace?.routerIdentity ?? "unknown_route_fn",
    activated,
    confidence: activated
      ? (lastSnapshot?.chosenActionProbability ?? null)
      : (lastSnapshot?.stopProbability ?? null),
    predictedUtility: params.predictedUtility ?? null,
    predictedRegretOfAbstaining: params.predictedRegretOfAbstaining ?? null,
    selectedContextCount: selectedNodeIds.length,
    selectedTokenBudget: selectionMetadata?.queryBudgetChars ?? selectionMetadata?.budgetChars ?? null,
    stopReason: inferredStopReason,
    decisionPointCount: snapshots.length,
    activationThreshold: params.activationThreshold ?? null,
    costEstimateMs: selectionMetadata?.totalQueryMs ?? selectionMetadata?.routeSelectionMs ?? null,
    timestamp: params.timestamp,
  });
}
