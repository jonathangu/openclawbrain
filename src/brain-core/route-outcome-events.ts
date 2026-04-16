import { createHash } from "node:crypto";
import { Type, type Static } from "@sinclair/typebox";
import { Value } from "@sinclair/typebox/value";

export const ROUTE_OUTCOME_EVENT_VERSION_V1 = 1 as const;

export const ROUTE_SERVED_EVENT_CONTRACT_V1 = "ocb.route_served.v1";
export const TURN_OUTCOME_EVENT_CONTRACT_V1 = "ocb.turn_outcome.v1";
export const RETRY_OR_INTERVENTION_EVENT_CONTRACT_V1 = "ocb.retry_or_intervention.v1";
export const EPISODE_RESOLUTION_EVENT_CONTRACT_V1 = "ocb.episode_resolution.v1";

const SHARED_IDENTITY_SCHEMA_V1 = {
  event_id: Type.String({ minLength: 1 }),
  event_at: Type.String({ minLength: 1 }),
  conversation_id: Type.Union([Type.Integer(), Type.Null()]),
  episode_id: Type.String({ minLength: 1 }),
  trace_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  observation_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  serve_decision_record_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  selection_digest: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  turn_compile_event_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
} as const;

const ACTIVATION_KINDS = ["graph_prior_only", "learned_prior_like", "learned_nontrivial", "fail_open", "hard_fail"] as const;
const OUTCOME_CLASSES = ["resolved", "correction", "contradiction", "reask", "retry", "abandoned", "unknown"] as const;
const OUTCOME_SOURCES = ["user_followup", "teacher_eval", "operator_audit", "runtime_recovery"] as const;
const RETRY_TRIGGER_KINDS = ["user_retry", "assistant_recovery", "operator_override", "tool_rerun", "fallback_recovery"] as const;
const RETRY_TRIGGERED_BY = ["user", "assistant", "operator", "runtime"] as const;
const RETRY_REASON_CLASSES = ["wrong", "incomplete", "tool_failure", "latency", "unknown"] as const;
const RESOLUTION_CLASSES = ["completed", "handed_off", "abandoned", "unknown"] as const;

export type RouteServedActivationKindV1 = (typeof ACTIVATION_KINDS)[number];
export type TurnOutcomeClassV1 = (typeof OUTCOME_CLASSES)[number];
export type TurnOutcomeSourceV1 = (typeof OUTCOME_SOURCES)[number];
export type RetryOrInterventionTriggerKindV1 = (typeof RETRY_TRIGGER_KINDS)[number];
export type RetryOrInterventionTriggeredByV1 = (typeof RETRY_TRIGGERED_BY)[number];
export type RetryOrInterventionReasonClassV1 = (typeof RETRY_REASON_CLASSES)[number];
export type EpisodeResolutionClassV1 = (typeof RESOLUTION_CLASSES)[number];

export const RouteServedEventSchemaV1 = Type.Object({
  schema_version: Type.Literal(ROUTE_OUTCOME_EVENT_VERSION_V1),
  contract: Type.Literal(ROUTE_SERVED_EVENT_CONTRACT_V1),
  ...SHARED_IDENTITY_SCHEMA_V1,
  mode_requested: Type.String({ minLength: 1 }),
  mode_effective: Type.String({ minLength: 1 }),
  used_learned_route_fn: Type.Boolean(),
  activation_kind: Type.Union(ACTIVATION_KINDS.map((value) => Type.Literal(value))),
  request_digest: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  agent_identity: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  active_pack_id: Type.String({ minLength: 1 }),
  router_identity: Type.String({ minLength: 1 }),
  active_pack_event_export_digest: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  active_pack_graph_checksum: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  active_pack_router_checksum: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  binding_mode: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  candidate_node_ids: Type.Array(Type.String({ minLength: 1 })),
  selected_node_ids: Type.Array(Type.String({ minLength: 1 })),
  selected_traversal_node_ids: Type.Array(Type.String({ minLength: 1 })),
  selected_path_node_ids: Type.Array(Type.String({ minLength: 1 })),
  selected_seed_node_ids: Type.Array(Type.String({ minLength: 1 })),
  served_artifact: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  source_summary: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  tool_count: Type.Union([Type.Integer({ minimum: 0 }), Type.Null()]),
  prompt_tokens_estimate: Type.Union([Type.Integer({ minimum: 0 }), Type.Null()]),
  latency_ms: Type.Union([Type.Number({ minimum: 0 }), Type.Null()]),
  fail_open: Type.Boolean(),
  hard_requirement_violated: Type.Union([Type.Boolean(), Type.Null()]),
}, { additionalProperties: false });

export const TurnOutcomeEventSchemaV1 = Type.Object({
  schema_version: Type.Literal(ROUTE_OUTCOME_EVENT_VERSION_V1),
  contract: Type.Literal(TURN_OUTCOME_EVENT_CONTRACT_V1),
  ...SHARED_IDENTITY_SCHEMA_V1,
  outcome_class: Type.Union(OUTCOME_CLASSES.map((value) => Type.Literal(value))),
  correction_required: Type.Boolean(),
  source: Type.Union(OUTCOME_SOURCES.map((value) => Type.Literal(value))),
  follow_up_class: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  tool_outcome_class: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  route_integrity_class: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  reason: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  closed_at: Type.String({ minLength: 1 }),
}, { additionalProperties: false });

export const RetryOrInterventionEventSchemaV1 = Type.Object({
  schema_version: Type.Literal(ROUTE_OUTCOME_EVENT_VERSION_V1),
  contract: Type.Literal(RETRY_OR_INTERVENTION_EVENT_CONTRACT_V1),
  ...SHARED_IDENTITY_SCHEMA_V1,
  trigger_kind: Type.Union(RETRY_TRIGGER_KINDS.map((value) => Type.Literal(value))),
  triggered_by: Type.Union(RETRY_TRIGGERED_BY.map((value) => Type.Literal(value))),
  reason_class: Type.Union([...RETRY_REASON_CLASSES.map((value) => Type.Literal(value)), Type.Null()]),
  retry_count_delta: Type.Union([Type.Integer({ minimum: 0 }), Type.Null()]),
  intervention_count_delta: Type.Union([Type.Integer({ minimum: 0 }), Type.Null()]),
  triggered_at: Type.String({ minLength: 1 }),
}, { additionalProperties: false });

export const EpisodeResolutionEventSchemaV1 = Type.Object({
  schema_version: Type.Literal(ROUTE_OUTCOME_EVENT_VERSION_V1),
  contract: Type.Literal(EPISODE_RESOLUTION_EVENT_CONTRACT_V1),
  ...SHARED_IDENTITY_SCHEMA_V1,
  resolution_class: Type.Union(RESOLUTION_CLASSES.map((value) => Type.Literal(value))),
  resolved: Type.Boolean(),
  resolved_at: Type.String({ minLength: 1 }),
  resolution_user_turn_index: Type.Integer({ minimum: 0 }),
  resolution_assistant_turn_index: Type.Union([Type.Integer({ minimum: 0 }), Type.Null()]),
  total_retry_count: Type.Union([Type.Integer({ minimum: 0 }), Type.Null()]),
  total_intervention_count: Type.Union([Type.Integer({ minimum: 0 }), Type.Null()]),
  final_outcome_quality: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
}, { additionalProperties: false });

export type RouteServedEventV1 = Static<typeof RouteServedEventSchemaV1>;
export type TurnOutcomeEventV1 = Static<typeof TurnOutcomeEventSchemaV1>;
export type RetryOrInterventionEventV1 = Static<typeof RetryOrInterventionEventSchemaV1>;
export type EpisodeResolutionEventV1 = Static<typeof EpisodeResolutionEventSchemaV1>;

export interface ContractValidationResultV1 {
  contract: string;
  valid: boolean;
  issues: string[];
}

export interface RouteOutcomeEventIdentityV1 {
  conversationId?: number | null;
  episodeId: string;
  traceId?: string | null;
  observationId?: string | null;
  serveDecisionRecordId?: string | null;
  selectionDigest?: string | null;
  turnCompileEventId?: string | null;
}

function hashStableParts(parts: Array<string | number | null | undefined>): string {
  return createHash("sha256").update(parts.map((part) => String(part ?? "")).join("\u001f")).digest("hex").slice(0, 16);
}

function normalizeIdentity(identity: RouteOutcomeEventIdentityV1) {
  return {
    conversation_id: identity.conversationId ?? null,
    episode_id: identity.episodeId,
    trace_id: identity.traceId ?? null,
    observation_id: identity.observationId ?? null,
    serve_decision_record_id: identity.serveDecisionRecordId ?? null,
    selection_digest: identity.selectionDigest ?? null,
    turn_compile_event_id: identity.turnCompileEventId ?? null,
  };
}

function validateSharedIdentity(contract: string, identity: RouteOutcomeEventIdentityV1): string[] {
  const populated = [
    identity.traceId,
    identity.serveDecisionRecordId,
    identity.selectionDigest,
    identity.turnCompileEventId,
  ].filter((value) => typeof value === "string" && value.trim().length > 0);
  if (populated.length > 0) {
    return [];
  }
  return [`${contract}/at least one join key among traceId, serveDecisionRecordId, selectionDigest, or turnCompileEventId is required`];
}

function validateSchema(contract: string, schema: typeof RouteServedEventSchemaV1, value: unknown): ContractValidationResultV1;
function validateSchema(contract: string, schema: typeof TurnOutcomeEventSchemaV1, value: unknown): ContractValidationResultV1;
function validateSchema(contract: string, schema: typeof RetryOrInterventionEventSchemaV1, value: unknown): ContractValidationResultV1;
function validateSchema(contract: string, schema: typeof EpisodeResolutionEventSchemaV1, value: unknown): ContractValidationResultV1;
function validateSchema(contract: string, schema: any, value: unknown): ContractValidationResultV1 {
  const valid = Value.Check(schema, value);
  if (!valid) {
    return { contract, valid: false, issues: [`${contract}/schema_validation_failed`] };
  }
  return { contract, valid: true, issues: [] };
}

export function buildRouteServedEventV1(params: {
  identity: RouteOutcomeEventIdentityV1;
  modeRequested: string;
  modeEffective: string;
  usedLearnedRouteFn: boolean;
  activationKind: RouteServedActivationKindV1;
  activePackId: string;
  routerIdentity: string;
  requestDigest?: string | null;
  agentIdentity?: string | null;
  activePackEventExportDigest?: string | null;
  activePackGraphChecksum?: string | null;
  activePackRouterChecksum?: string | null;
  bindingMode?: string | null;
  candidateNodeIds?: string[];
  selectedNodeIds?: string[];
  selectedTraversalNodeIds?: string[];
  selectedPathNodeIds?: string[];
  selectedSeedNodeIds?: string[];
  servedArtifact?: string | null;
  sourceSummary?: string | null;
  toolCount?: number | null;
  promptTokensEstimate?: number | null;
  latencyMs?: number | null;
  failOpen?: boolean;
  hardRequirementViolated?: boolean | null;
  eventAt?: string;
}): RouteServedEventV1 {
  const eventAt = params.eventAt ?? new Date().toISOString();
  return {
    schema_version: ROUTE_OUTCOME_EVENT_VERSION_V1,
    contract: ROUTE_SERVED_EVENT_CONTRACT_V1,
    event_id: `rse_${hashStableParts([params.identity.episodeId, params.identity.traceId, params.modeEffective, eventAt])}`,
    event_at: eventAt,
    ...normalizeIdentity(params.identity),
    mode_requested: params.modeRequested,
    mode_effective: params.modeEffective,
    used_learned_route_fn: params.usedLearnedRouteFn,
    activation_kind: params.activationKind,
    request_digest: params.requestDigest ?? null,
    agent_identity: params.agentIdentity ?? null,
    active_pack_id: params.activePackId,
    router_identity: params.routerIdentity,
    active_pack_event_export_digest: params.activePackEventExportDigest ?? null,
    active_pack_graph_checksum: params.activePackGraphChecksum ?? null,
    active_pack_router_checksum: params.activePackRouterChecksum ?? null,
    binding_mode: params.bindingMode ?? null,
    candidate_node_ids: params.candidateNodeIds ?? [],
    selected_node_ids: params.selectedNodeIds ?? [],
    selected_traversal_node_ids: params.selectedTraversalNodeIds ?? [],
    selected_path_node_ids: params.selectedPathNodeIds ?? [],
    selected_seed_node_ids: params.selectedSeedNodeIds ?? [],
    served_artifact: params.servedArtifact ?? null,
    source_summary: params.sourceSummary ?? null,
    tool_count: params.toolCount ?? null,
    prompt_tokens_estimate: params.promptTokensEstimate ?? null,
    latency_ms: params.latencyMs ?? null,
    fail_open: params.failOpen ?? false,
    hard_requirement_violated: params.hardRequirementViolated ?? null,
  };
}

export function buildTurnOutcomeEventV1(params: {
  identity: RouteOutcomeEventIdentityV1;
  outcomeClass: TurnOutcomeClassV1;
  correctionRequired: boolean;
  source: TurnOutcomeSourceV1;
  followUpClass?: string | null;
  toolOutcomeClass?: string | null;
  routeIntegrityClass?: string | null;
  reason?: string | null;
  closedAt?: string;
  eventAt?: string;
}): TurnOutcomeEventV1 {
  const eventAt = params.eventAt ?? params.closedAt ?? new Date().toISOString();
  const closedAt = params.closedAt ?? eventAt;
  return {
    schema_version: ROUTE_OUTCOME_EVENT_VERSION_V1,
    contract: TURN_OUTCOME_EVENT_CONTRACT_V1,
    event_id: `toe_${hashStableParts([params.identity.episodeId, params.identity.traceId, params.outcomeClass, closedAt])}`,
    event_at: eventAt,
    ...normalizeIdentity(params.identity),
    outcome_class: params.outcomeClass,
    correction_required: params.correctionRequired,
    source: params.source,
    follow_up_class: params.followUpClass ?? null,
    tool_outcome_class: params.toolOutcomeClass ?? null,
    route_integrity_class: params.routeIntegrityClass ?? null,
    reason: params.reason ?? null,
    closed_at: closedAt,
  };
}

export function buildRetryOrInterventionEventV1(params: {
  identity: RouteOutcomeEventIdentityV1;
  triggerKind: RetryOrInterventionTriggerKindV1;
  triggeredBy: RetryOrInterventionTriggeredByV1;
  reasonClass?: RetryOrInterventionReasonClassV1 | null;
  retryCountDelta?: number | null;
  interventionCountDelta?: number | null;
  triggeredAt?: string;
  eventAt?: string;
}): RetryOrInterventionEventV1 {
  const triggeredAt = params.triggeredAt ?? params.eventAt ?? new Date().toISOString();
  return {
    schema_version: ROUTE_OUTCOME_EVENT_VERSION_V1,
    contract: RETRY_OR_INTERVENTION_EVENT_CONTRACT_V1,
    event_id: `rie_${hashStableParts([params.identity.episodeId, params.identity.traceId, params.triggerKind, triggeredAt])}`,
    event_at: params.eventAt ?? triggeredAt,
    ...normalizeIdentity(params.identity),
    trigger_kind: params.triggerKind,
    triggered_by: params.triggeredBy,
    reason_class: params.reasonClass ?? null,
    retry_count_delta: params.retryCountDelta ?? null,
    intervention_count_delta: params.interventionCountDelta ?? null,
    triggered_at: triggeredAt,
  };
}

export function buildEpisodeResolutionEventV1(params: {
  identity: RouteOutcomeEventIdentityV1;
  resolutionClass: EpisodeResolutionClassV1;
  resolved: boolean;
  resolutionUserTurnIndex: number;
  resolutionAssistantTurnIndex?: number | null;
  totalRetryCount?: number | null;
  totalInterventionCount?: number | null;
  finalOutcomeQuality?: string | null;
  resolvedAt?: string;
  eventAt?: string;
}): EpisodeResolutionEventV1 {
  const resolvedAt = params.resolvedAt ?? params.eventAt ?? new Date().toISOString();
  return {
    schema_version: ROUTE_OUTCOME_EVENT_VERSION_V1,
    contract: EPISODE_RESOLUTION_EVENT_CONTRACT_V1,
    event_id: `ere_${hashStableParts([params.identity.episodeId, params.identity.traceId, params.resolutionClass, resolvedAt])}`,
    event_at: params.eventAt ?? resolvedAt,
    ...normalizeIdentity(params.identity),
    resolution_class: params.resolutionClass,
    resolved: params.resolved,
    resolved_at: resolvedAt,
    resolution_user_turn_index: params.resolutionUserTurnIndex,
    resolution_assistant_turn_index: params.resolutionAssistantTurnIndex ?? null,
    total_retry_count: params.totalRetryCount ?? null,
    total_intervention_count: params.totalInterventionCount ?? null,
    final_outcome_quality: params.finalOutcomeQuality ?? null,
  };
}

export function validateRouteServedEventV1(value: unknown): ContractValidationResultV1 {
  const contract = ROUTE_SERVED_EVENT_CONTRACT_V1;
  const base = validateSchema(contract, RouteServedEventSchemaV1, value);
  if (!base.valid) {
    return base;
  }
  const event = value as RouteServedEventV1;
  const issues = validateSharedIdentity(contract, {
    conversationId: event.conversation_id,
    episodeId: event.episode_id,
    traceId: event.trace_id,
    observationId: event.observation_id,
    serveDecisionRecordId: event.serve_decision_record_id,
    selectionDigest: event.selection_digest,
    turnCompileEventId: event.turn_compile_event_id,
  });
  if (event.activation_kind === "learned_nontrivial" && !event.used_learned_route_fn) {
    issues.push(`${contract}/learned_nontrivial requires used_learned_route_fn=true`);
  }
  if (event.activation_kind === "hard_fail" && !event.fail_open) {
    issues.push(`${contract}/hard_fail should set fail_open=true`);
  }
  if (event.selected_node_ids.length > 0 && event.activation_kind === "graph_prior_only") {
    issues.push(`${contract}/graph_prior_only should not carry selected_node_ids`);
  }
  return issues.length === 0 ? { contract, valid: true, issues: [] } : { contract, valid: false, issues };
}

export function validateTurnOutcomeEventV1(value: unknown): ContractValidationResultV1 {
  const contract = TURN_OUTCOME_EVENT_CONTRACT_V1;
  const base = validateSchema(contract, TurnOutcomeEventSchemaV1, value);
  if (!base.valid) {
    return base;
  }
  const event = value as TurnOutcomeEventV1;
  const issues = validateSharedIdentity(contract, {
    conversationId: event.conversation_id,
    episodeId: event.episode_id,
    traceId: event.trace_id,
    observationId: event.observation_id,
    serveDecisionRecordId: event.serve_decision_record_id,
    selectionDigest: event.selection_digest,
    turnCompileEventId: event.turn_compile_event_id,
  });
  if (["correction", "contradiction", "reask", "retry"].includes(event.outcome_class) && !event.correction_required) {
    issues.push(`${contract}/${event.outcome_class} requires correction_required=true`);
  }
  return issues.length === 0 ? { contract, valid: true, issues: [] } : { contract, valid: false, issues };
}

export function validateRetryOrInterventionEventV1(value: unknown): ContractValidationResultV1 {
  const contract = RETRY_OR_INTERVENTION_EVENT_CONTRACT_V1;
  const base = validateSchema(contract, RetryOrInterventionEventSchemaV1, value);
  if (!base.valid) {
    return base;
  }
  const event = value as RetryOrInterventionEventV1;
  const issues = validateSharedIdentity(contract, {
    conversationId: event.conversation_id,
    episodeId: event.episode_id,
    traceId: event.trace_id,
    observationId: event.observation_id,
    serveDecisionRecordId: event.serve_decision_record_id,
    selectionDigest: event.selection_digest,
    turnCompileEventId: event.turn_compile_event_id,
  });
  if ((event.retry_count_delta ?? 0) === 0 && (event.intervention_count_delta ?? 0) === 0) {
    issues.push(`${contract}/at least one of retry_count_delta or intervention_count_delta must be > 0`);
  }
  return issues.length === 0 ? { contract, valid: true, issues: [] } : { contract, valid: false, issues };
}

export function validateEpisodeResolutionEventV1(value: unknown): ContractValidationResultV1 {
  const contract = EPISODE_RESOLUTION_EVENT_CONTRACT_V1;
  const base = validateSchema(contract, EpisodeResolutionEventSchemaV1, value);
  if (!base.valid) {
    return base;
  }
  const event = value as EpisodeResolutionEventV1;
  const issues = validateSharedIdentity(contract, {
    conversationId: event.conversation_id,
    episodeId: event.episode_id,
    traceId: event.trace_id,
    observationId: event.observation_id,
    serveDecisionRecordId: event.serve_decision_record_id,
    selectionDigest: event.selection_digest,
    turnCompileEventId: event.turn_compile_event_id,
  });
  if ((event.resolution_class === "completed" || event.resolution_class === "handed_off") && !event.resolved) {
    issues.push(`${contract}/${event.resolution_class} requires resolved=true`);
  }
  if (event.resolution_class === "unknown" && event.resolved) {
    issues.push(`${contract}/unknown resolution should not set resolved=true`);
  }
  return issues.length === 0 ? { contract, valid: true, issues: [] } : { contract, valid: false, issues };
}
