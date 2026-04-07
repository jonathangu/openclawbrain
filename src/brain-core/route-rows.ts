import { createHash } from "node:crypto";
import { Type, type Static } from "@sinclair/typebox";
import { Value } from "@sinclair/typebox/value";
import type {
  AttributionTruthMatchBasis,
  AttributionTruthRecord,
  AttributionTruthState,
  BrainEvidenceKind,
  BrainObservationBindingMode,
  DecisionPointActionCandidateV1,
  DecisionPointSnapshotV1,
  DecisionTrace,
  DecisionTraceInjectedNodeSummary,
  ObservationBindingQuality,
  TeacherFeedbackRichness,
} from "./types.js";

export const ROUTE_DECISION_ROW_VERSION_V1 = 1 as const;
export const LABEL_PROVENANCE_VERSION_V1 = 1 as const;

export type RouteDecisionActionKindV1 = DecisionPointActionCandidateV1["actionKind"];

export const ROUTE_DECISION_ACTION_KINDS_V1 = [
  "traverse",
  "tool_capability",
  "tool_instance",
  "stop_local",
  "stop",
] as const;

export type RouteDecisionActionKindTupleV1 = (typeof ROUTE_DECISION_ACTION_KINDS_V1)[number];

const STRING_RECORD_SCHEMA = Type.Record(
  Type.String({ minLength: 1 }),
  Type.Union([Type.Number(), Type.String(), Type.Boolean()]),
);

export const EvidenceSpanSchemaV1 = Type.Object(
  {
    source_ref: Type.String({ minLength: 1 }),
    start: Type.Integer({ minimum: 0 }),
    end: Type.Integer({ minimum: 0 }),
    excerpt: Type.Optional(Type.String()),
  },
  { additionalProperties: false },
);

export type EvidenceSpanV1 = Static<typeof EvidenceSpanSchemaV1>;

export const RouteDecisionActionCandidateSchemaV1 = Type.Object(
  {
    action_id: Type.String({ minLength: 1 }),
    action_kind: Type.Union(ROUTE_DECISION_ACTION_KINDS_V1.map((kind) => Type.Literal(kind))),
    node_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    tool_name: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    tool_capability_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    tool_instance_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    tool_args_shape: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    prior_score: Type.Union([Type.Number(), Type.Null()]),
    probability: Type.Union([Type.Number(), Type.Null()]),
    retrieval_features: Type.Union([STRING_RECORD_SCHEMA, Type.Null()]),
  },
  { additionalProperties: false },
);

export type RouteDecisionActionCandidateV1 = Static<typeof RouteDecisionActionCandidateSchemaV1>;

export const DecisionPointBudgetContextSchemaV1 = Type.Object(
  {
    budget_remaining: Type.Number(),
    initial_budget: Type.Number(),
    reserved_token_cost: Type.Number(),
    budget_used: Type.Number(),
    budget_used_fraction: Type.Number(),
    max_hops: Type.Number(),
    max_frontier_size: Type.Union([Type.Number(), Type.Null()]),
    frontier_size: Type.Number(),
    visited_count: Type.Number(),
    fired_count: Type.Number(),
    pending_selection_count: Type.Number(),
    pressure_level: Type.Union([Type.Number(), Type.Null()]),
    frontier_pressure: Type.Union([Type.Number(), Type.Null()]),
    budget_pressure: Type.Union([Type.Number(), Type.Null()]),
    budget_fraction: Type.Union([Type.Number(), Type.Null()]),
    query_budget_chars: Type.Union([Type.Number(), Type.Null()]),
    max_context_chars: Type.Union([Type.Number(), Type.Null()]),
    injected_chars: Type.Union([Type.Number(), Type.Null()]),
    dropped_chars: Type.Union([Type.Number(), Type.Null()]),
    context_clipped: Type.Union([Type.Boolean(), Type.Null()]),
    route_selection_ms: Type.Union([Type.Number(), Type.Null()]),
    total_query_ms: Type.Union([Type.Number(), Type.Null()]),
    compile_deadline_ms: Type.Union([Type.Number(), Type.Null()]),
    compile_deadline_hit: Type.Union([Type.Boolean(), Type.Null()]),
  },
  { additionalProperties: false },
);

export type DecisionPointBudgetContextSchemaTypeV1 = Static<typeof DecisionPointBudgetContextSchemaV1>;

export const RouteContextSchemaV1 = Type.Object(
  {
    request_digest: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    active_pack_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    router_identity: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    candidate_node_ids: Type.Array(Type.String({ minLength: 1 })),
    selected_node_ids: Type.Array(Type.String({ minLength: 1 })),
    selected_traversal_node_ids: Type.Array(Type.String({ minLength: 1 })),
    selected_path_node_ids: Type.Array(Type.String({ minLength: 1 })),
    selected_seed_node_ids: Type.Array(Type.String({ minLength: 1 })),
  },
  { additionalProperties: false },
);

export type RouteContextV1 = Static<typeof RouteContextSchemaV1>;

export const LabelProvenanceSchemaV1 = Type.Object(
  {
    provenance_id: Type.String({ minLength: 1 }),
    state: Type.Union([
      Type.Literal("matched"),
      Type.Literal("ambiguous"),
      Type.Literal("unmatched"),
      Type.Literal("delayed"),
    ]),
    basis: Type.Union([
      Type.Literal("decision_record_id"),
      Type.Literal("selection_digest"),
      Type.Literal("turn_compile_event_id"),
      Type.Literal("trace_id"),
      Type.Literal("episode_id"),
      Type.Literal("heuristic"),
      Type.Literal("manual"),
      Type.Literal("pending_observation"),
      Type.Literal("pending_update"),
      Type.Literal("missing"),
      Type.Literal("conflict"),
    ]),
    confidence: Type.Union([Type.Number({ minimum: 0, maximum: 1 }), Type.Null()]),
    detail: Type.Union([Type.String(), Type.Null()]),
    source: Type.Union([
      Type.Literal("human"),
      Type.Literal("self"),
      Type.Literal("scanner"),
      Type.Literal("teacher"),
      Type.Null(),
    ]),
    kind: Type.Union([
      Type.Literal("human_feedback"),
      Type.Literal("self_result"),
      Type.Literal("scanner_signal"),
      Type.Literal("teacher_review"),
      Type.Literal("teach_correction"),
      Type.Null(),
    ]),
    binding_mode: Type.Union([
      Type.Literal("exact_decision_id"),
      Type.Literal("exact_selection_digest"),
      Type.Literal("turn_compile_event_id"),
      Type.Literal("trace_id"),
      Type.Literal("legacy_heuristic"),
      Type.Literal("unbound"),
      Type.Null(),
    ]),
    attribution_quality: Type.Union([
      Type.Literal("exact"),
      Type.Literal("fallback"),
      Type.Literal("unbound"),
      Type.Literal("mixed"),
      Type.Null(),
    ]),
    feedback_richness: Type.Union([
      Type.Literal("followup_and_tool"),
      Type.Literal("followup_only"),
      Type.Literal("tool_only"),
      Type.Literal("sparse"),
      Type.Literal("mixed"),
      Type.Null(),
    ]),
    trace_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    episode_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    decision_point_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    observation_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    supervision_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    update_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    candidate_ids: Type.Array(Type.String({ minLength: 1 })),
    provenance_ref: Type.String({ minLength: 1 }),
    content_hash: Type.String({ minLength: 1 }),
    lineage_hash: Type.String({ minLength: 1 }),
    created_at: Type.Integer({ minimum: 0 }),
  },
  { additionalProperties: false },
);

export type LabelProvenanceV1 = Static<typeof LabelProvenanceSchemaV1>;

export const RouteDecisionRowSchemaV1 = Type.Object(
  {
    schema_version: Type.Literal(1),
    row_id: Type.String({ minLength: 1 }),
    trace_id: Type.String({ minLength: 1 }),
    episode_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    conversation_id: Type.Union([Type.Integer(), Type.Null()]),
    route_fn_version: Type.String({ minLength: 1 }),
    decision_point_id: Type.String({ minLength: 1 }),
    decision_point_kind: Type.Union([Type.Literal("seed"), Type.Literal("local")]),
    source_node_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    expansion_index: Type.Integer({ minimum: 0 }),
    selection_index: Type.Integer({ minimum: 0 }),
    query_text: Type.String({ minLength: 1 }),
    cursor_path: Type.Array(Type.String({ minLength: 1 })),
    local_action_set: Type.Array(RouteDecisionActionCandidateSchemaV1, { minItems: 1 }),
    chosen_action_id: Type.String({ minLength: 1 }),
    chosen_action_kind: Type.Union(ROUTE_DECISION_ACTION_KINDS_V1.map((kind) => Type.Literal(kind))),
    chosen_node_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    chosen_tool_name: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    chosen_tool_capability_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    chosen_tool_instance_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    chosen_action_probability: Type.Number(),
    stop_probability: Type.Number(),
    stop_label: Type.Union([Type.Literal("CONTINUE"), Type.Literal("STOP_LOCAL"), Type.Literal("STOP")]),
    budget_context: DecisionPointBudgetContextSchemaV1,
    route_context: RouteContextSchemaV1,
    label_provenance: LabelProvenanceSchemaV1,
    evidence_spans: Type.Array(EvidenceSpanSchemaV1, { minItems: 1 }),
    hard_negatives: Type.Array(Type.String({ minLength: 1 })),
    created_at: Type.String({ minLength: 1 }),
  },
  { additionalProperties: false },
);

export type RouteDecisionRowV1 = Static<typeof RouteDecisionRowSchemaV1>;

export interface ContractValidationResultV1 {
  contract: string;
  valid: boolean;
  issues: string[];
}

function hashValue(value: string): string {
  return createHash("sha256").update(value).digest("hex").slice(0, 16);
}

function hashStableParts(parts: Array<string | number | null | undefined>): string {
  return hashValue(parts.map((part) => String(part ?? "")).join("\u001f"));
}

function normalizeString(value: string | null | undefined): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim();
  return normalized.length > 0 ? normalized : null;
}

function normalizeStringArray(values: string[] | null | undefined): string[] {
  if (!Array.isArray(values)) {
    return [];
  }
  return [...new Set(values.map((value) => normalizeString(value)).filter((value): value is string => value !== null))].sort();
}

function normalizeConfidence(value: number | null | undefined): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return Math.max(0, Math.min(1, value));
}

function toSnakeCaseActionCandidate(candidate: DecisionPointActionCandidateV1): RouteDecisionActionCandidateV1 {
  return {
    action_id: candidate.actionId,
    action_kind: candidate.actionKind,
    node_id: candidate.nodeId,
    tool_name: candidate.toolName,
    tool_capability_id: candidate.toolCapabilityId,
    tool_instance_id: candidate.toolInstanceId,
    tool_args_shape: candidate.toolArgsShape,
    prior_score: candidate.priorScore,
    probability: candidate.probability,
    retrieval_features: candidate.retrievalFeatures,
  };
}

function toEvidenceSpan(summary: DecisionTraceInjectedNodeSummary, fallbackSourceRef: string): EvidenceSpanV1 {
  const sourceRef = normalizeString(summary.provenanceRef) ?? fallbackSourceRef;
  const excerpt = normalizeString(summary.contentPreview) ?? undefined;
  const excerptLength = excerpt ? Math.max(1, excerpt.length) : 1;
  return {
    source_ref: sourceRef,
    start: 0,
    end: excerptLength,
    ...(excerpt ? { excerpt } : {}),
  };
}

function summarizeEvidenceSpans(trace: DecisionTrace, snapshot: DecisionPointSnapshotV1): EvidenceSpanV1[] {
  const summaries = trace.routeTrace?.injectedNodeSummaries ?? [];
  const spans = summaries.length > 0
    ? summaries.map((summary, index) => toEvidenceSpan(summary, `${trace.id}:evidence:${index}`))
    : [
        {
          source_ref: trace.routeTrace?.requestDigest ?? trace.id,
          start: 0,
          end: Math.max(1, trace.queryText.length),
          excerpt: trace.queryText,
        },
      ];

  const seen = new Set<string>();
  const deduped: EvidenceSpanV1[] = [];
  for (const span of spans) {
    const key = `${span.source_ref}:${span.start}:${span.end}:${span.excerpt ?? ""}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    deduped.push(span);
  }

  if (deduped.length === 0) {
    return [
      {
        source_ref: snapshot.decisionPointId,
        start: 0,
        end: Math.max(1, trace.queryText.length),
        excerpt: trace.queryText,
      },
    ];
  }

  return deduped;
}

function deriveHardNegatives(snapshot: DecisionPointSnapshotV1): string[] {
  const chosenIds = new Set<string>();
  if (snapshot.chosenNodeId) {
    chosenIds.add(snapshot.chosenNodeId);
  }
  if (snapshot.chosenToolName) {
    chosenIds.add(snapshot.chosenToolName);
  }
  if (snapshot.chosenToolCapabilityId) {
    chosenIds.add(snapshot.chosenToolCapabilityId);
  }
  if (snapshot.chosenToolInstanceId) {
    chosenIds.add(snapshot.chosenToolInstanceId);
  }

  return normalizeStringArray(
    snapshot.localActionSet
      .filter((candidate) => candidate.nodeId !== null)
      .map((candidate) => candidate.nodeId as string)
      .filter((candidateId) => !chosenIds.has(candidateId)),
  );
}

function routeContextFromTrace(trace: DecisionTrace): RouteContextV1 {
  const routeTrace = trace.routeTrace;
  return {
    request_digest: routeTrace?.requestDigest ?? null,
    active_pack_id: routeTrace?.activePackId ?? null,
    router_identity: routeTrace?.routerIdentity ?? null,
    candidate_node_ids: [...(routeTrace?.candidateNodeIds ?? [])],
    selected_node_ids: [...(routeTrace?.selectedNodeIds ?? [])],
    selected_traversal_node_ids: [...(routeTrace?.selectedTraversalNodeIds ?? [])],
    selected_path_node_ids: [...(routeTrace?.selectedPathNodeIds ?? routeTrace?.selectedTraversalNodeIds ?? [])],
    selected_seed_node_ids: [...(routeTrace?.selectedSeedNodeIds ?? [])],
  };
}

function budgetContextFromSnapshot(snapshot: DecisionPointSnapshotV1): DecisionPointBudgetContextSchemaTypeV1 {
  return {
    budget_remaining: snapshot.budgetContext.budgetRemaining,
    initial_budget: snapshot.budgetContext.initialBudget,
    reserved_token_cost: snapshot.budgetContext.reservedTokenCost,
    budget_used: snapshot.budgetContext.budgetUsed,
    budget_used_fraction: snapshot.budgetContext.budgetUsedFraction,
    max_hops: snapshot.budgetContext.maxHops,
    max_frontier_size: snapshot.budgetContext.maxFrontierSize,
    frontier_size: snapshot.budgetContext.frontierSize,
    visited_count: snapshot.budgetContext.visitedCount,
    fired_count: snapshot.budgetContext.firedCount,
    pending_selection_count: snapshot.budgetContext.pendingSelectionCount,
    pressure_level: snapshot.budgetContext.pressureLevel,
    frontier_pressure: snapshot.budgetContext.frontierPressure,
    budget_pressure: snapshot.budgetContext.budgetPressure,
    budget_fraction: snapshot.budgetContext.budgetFraction,
    query_budget_chars: snapshot.budgetContext.queryBudgetChars,
    max_context_chars: snapshot.budgetContext.maxContextChars,
    injected_chars: snapshot.budgetContext.injectedChars,
    dropped_chars: snapshot.budgetContext.droppedChars,
    context_clipped: snapshot.budgetContext.contextClipped,
    route_selection_ms: snapshot.budgetContext.routeSelectionMs,
    total_query_ms: snapshot.budgetContext.totalQueryMs,
    compile_deadline_ms: snapshot.budgetContext.compileDeadlineMs,
    compile_deadline_hit: snapshot.budgetContext.compileDeadlineHit,
  };
}

function normalizeAttributionRecordCandidate(
  provenance: AttributionTruthRecord | LabelProvenanceV1,
): LabelProvenanceV1 {
  if ("provenance_id" in provenance) {
    return {
      ...provenance,
      candidate_ids: normalizeStringArray(provenance.candidate_ids),
      confidence: normalizeConfidence(provenance.confidence),
      detail: normalizeString(provenance.detail),
      trace_id: normalizeString(provenance.trace_id),
      episode_id: normalizeString(provenance.episode_id),
      decision_point_id: normalizeString(provenance.decision_point_id),
      observation_id: normalizeString(provenance.observation_id),
      supervision_id: normalizeString(provenance.supervision_id),
      update_id: normalizeString(provenance.update_id),
      provenance_ref: normalizeString(provenance.provenance_ref) ?? provenance.provenance_id,
      content_hash: normalizeString(provenance.content_hash) ?? provenance.provenance_id,
      lineage_hash: normalizeString(provenance.lineage_hash) ?? provenance.provenance_id,
      created_at: Number.isFinite(provenance.created_at) ? Math.floor(provenance.created_at) : Date.now(),
    };
  }

  const observation = provenance.observation;
  const supervision = provenance.supervision;
  const update = provenance.update;
  const observationToSupervision = provenance.linkage.observationToSupervision;
  const supervisionToUpdate = provenance.linkage.supervisionToUpdate;
  const candidateIds = normalizeStringArray([
    ...observationToSupervision.candidateIds,
    ...supervisionToUpdate.candidateIds,
  ]);

  return {
    provenance_id: provenance.attributionTruthId,
    state: provenance.state,
    basis: observationToSupervision.basis,
    confidence: normalizeConfidence(observationToSupervision.confidence ?? supervisionToUpdate.confidence),
    detail: normalizeString(observationToSupervision.detail) ?? normalizeString(supervisionToUpdate.detail),
    source: supervision?.source ?? null,
    kind: supervision?.kind ?? null,
    binding_mode: observation?.bindingMode ?? supervision?.bindingMode ?? null,
    attribution_quality: supervision?.attributionQuality ?? update?.attributionQuality ?? null,
    feedback_richness: supervision?.feedbackRichness ?? update?.feedbackRichness ?? null,
    trace_id: normalizeString(observation?.traceId) ?? normalizeString(supervision?.traceId) ?? normalizeString(update?.traceIds[0]),
    episode_id: normalizeString(provenance.episodeId) ?? normalizeString(observation?.episodeId) ?? normalizeString(supervision?.episodeId) ?? normalizeString(update?.episodeId),
    decision_point_id: normalizeString(supervision?.serveDecisionRecordId) ?? normalizeString(observation?.serveDecisionRecordId) ?? null,
    observation_id: normalizeString(observation?.observationId),
    supervision_id: normalizeString(supervision?.supervisionId),
    update_id: normalizeString(update?.updateId),
    candidate_ids: candidateIds,
    provenance_ref: provenance.provenanceRef,
    content_hash: provenance.contentHash,
    lineage_hash: provenance.lineageHash,
    created_at: provenance.createdAt,
  };
}

function syntheticLabelProvenanceFromSnapshot(params: {
  trace: DecisionTrace;
  snapshot: DecisionPointSnapshotV1;
  routeContext: RouteContextV1;
}): LabelProvenanceV1 {
  const candidateIds = normalizeStringArray(params.snapshot.localActionSet.map((candidate) => candidate.nodeId ?? candidate.toolName ?? candidate.actionId));
  const provenanceId = `lp_${hashStableParts([
    params.trace.id,
    params.snapshot.decisionPointId,
    params.snapshot.chosenActionId,
    params.snapshot.chosenActionKind,
  ])}`;
  return {
    provenance_id: provenanceId,
    state: "delayed",
    basis: "pending_observation",
    confidence: null,
    detail: "synthetic fallback provenance materialized from traced decision point",
    source: null,
    kind: null,
    binding_mode: null,
    attribution_quality: null,
    feedback_richness: null,
    trace_id: params.trace.id,
    episode_id: params.trace.episodeId,
    decision_point_id: params.snapshot.decisionPointId,
    observation_id: null,
    supervision_id: null,
    update_id: null,
    candidate_ids: candidateIds,
    provenance_ref: `prov_${provenanceId}`,
    content_hash: hashValue(JSON.stringify({ traceId: params.trace.id, decisionPointId: params.snapshot.decisionPointId, chosenActionId: params.snapshot.chosenActionId })),
    lineage_hash: hashValue(JSON.stringify({ traceId: params.trace.id, decisionPointId: params.snapshot.decisionPointId, routeContext: params.routeContext })),
    created_at: Date.now(),
  };
}

export function materializeLabelProvenanceV1(params: {
  trace: DecisionTrace;
  snapshot: DecisionPointSnapshotV1;
  provenance?: AttributionTruthRecord | LabelProvenanceV1 | null;
}): LabelProvenanceV1 {
  if (params.provenance) {
    return normalizeAttributionRecordCandidate(params.provenance);
  }
  return syntheticLabelProvenanceFromSnapshot({
    trace: params.trace,
    snapshot: params.snapshot,
    routeContext: routeContextFromTrace(params.trace),
  });
}

export function materializeRouteDecisionRowFromSnapshotV1(params: {
  trace: DecisionTrace;
  snapshot: DecisionPointSnapshotV1;
  routeFnVersion?: string | null;
  provenance?: AttributionTruthRecord | LabelProvenanceV1 | null;
}): RouteDecisionRowV1 {
  const routeContext = routeContextFromTrace(params.trace);
  const labelProvenance = materializeLabelProvenanceV1({
    trace: params.trace,
    snapshot: params.snapshot,
    provenance: params.provenance ?? null,
  });
  const localActionSet = params.snapshot.localActionSet.map(toSnakeCaseActionCandidate);
  const chosen = localActionSet.find((candidate) => candidate.action_id === params.snapshot.chosenActionId) ?? localActionSet[0]!;
  const stopLabel: RouteDecisionRowV1["stop_label"] = params.snapshot.chosenActionKind === "stop_local"
    ? "STOP_LOCAL"
    : params.snapshot.chosenActionKind === "stop"
      ? "STOP"
      : "CONTINUE";
  const hardNegatives = deriveHardNegatives(params.snapshot);
  const evidenceSpans = summarizeEvidenceSpans(params.trace, params.snapshot);

  return {
    schema_version: ROUTE_DECISION_ROW_VERSION_V1,
    row_id: `rr_${hashStableParts([
      params.trace.id,
      params.snapshot.decisionPointId,
      labelProvenance.provenance_id,
      chosen.action_id,
      params.snapshot.chosenActionKind,
    ])}`,
    trace_id: params.trace.id,
    episode_id: params.trace.episodeId,
    conversation_id: params.trace.routeTrace?.conversationId ?? null,
    route_fn_version: normalizeString(params.routeFnVersion) ?? normalizeString(params.trace.routeTrace?.routerIdentity) ?? "route_fn.v1",
    decision_point_id: params.snapshot.decisionPointId,
    decision_point_kind: params.snapshot.decisionPointKind,
    source_node_id: params.snapshot.sourceNodeId,
    expansion_index: params.snapshot.expansionIndex,
    selection_index: params.snapshot.selectionIndex,
    query_text: params.trace.queryText,
    cursor_path: [...routeContext.selected_path_node_ids],
    local_action_set: localActionSet,
    chosen_action_id: params.snapshot.chosenActionId,
    chosen_action_kind: params.snapshot.chosenActionKind,
    chosen_node_id: params.snapshot.chosenNodeId,
    chosen_tool_name: params.snapshot.chosenToolName,
    chosen_tool_capability_id: params.snapshot.chosenToolCapabilityId,
    chosen_tool_instance_id: params.snapshot.chosenToolInstanceId,
    chosen_action_probability: params.snapshot.chosenActionProbability,
    stop_probability: params.snapshot.stopProbability,
    stop_label: stopLabel,
    budget_context: budgetContextFromSnapshot(params.snapshot),
    route_context: routeContext,
    label_provenance: labelProvenance,
    evidence_spans: evidenceSpans,
    hard_negatives: hardNegatives,
    created_at: new Date(params.trace.createdAt).toISOString(),
  };
}

export function materializeRouteDecisionRowsFromTraceV1(params: {
  trace: DecisionTrace;
  routeFnVersion?: string | null;
  provenanceByDecisionPointId?: Record<string, AttributionTruthRecord | LabelProvenanceV1 | null | undefined>;
  defaultProvenance?: AttributionTruthRecord | LabelProvenanceV1 | null;
}): RouteDecisionRowV1[] {
  const snapshots = params.trace.routeTrace?.selectionMetadata?.decisionPointSnapshots ?? [];
  const provenanceByDecisionPointId = params.provenanceByDecisionPointId ?? {};
  return snapshots.map((snapshot) => materializeRouteDecisionRowFromSnapshotV1({
    trace: params.trace,
    snapshot,
    routeFnVersion: params.routeFnVersion ?? null,
    provenance: provenanceByDecisionPointId[snapshot.decisionPointId] ?? params.defaultProvenance ?? null,
  }));
}

function validateSchema(contract: string, schema: unknown, value: unknown): ContractValidationResultV1 {
  const valid = Value.Check(schema as never, value);
  return {
    contract,
    valid,
    issues: valid ? [] : [`${contract}/schema_validation_failed`],
  };
}

export function validateLabelProvenanceV1(value: unknown): ContractValidationResultV1 {
  return validateSchema("route_label_provenance.v1", LabelProvenanceSchemaV1, value);
}

export function validateRouteDecisionRowV1(value: unknown): ContractValidationResultV1 {
  const contract = "route_decision_row.v1";
  const base = validateSchema(contract, RouteDecisionRowSchemaV1, value);
  if (!base.valid) {
    return base;
  }

  const row = value as RouteDecisionRowV1;
  const candidateIds = new Set(row.local_action_set.map((candidate) => candidate.action_id));
  const chosenCandidate = row.local_action_set.find((candidate) => candidate.action_id === row.chosen_action_id);
  const issues: string[] = [];

  if (!chosenCandidate) {
    issues.push(`${contract}/chosen_action_id must reference an entry in local_action_set`);
  }

  if ((row.chosen_action_kind === "stop_local" || row.chosen_action_kind === "stop") && row.chosen_node_id !== null) {
    issues.push(`${contract}/chosen_node_id must be null for stop actions`);
  }

  if ((row.chosen_action_kind === "traverse" || row.chosen_action_kind === "tool_capability" || row.chosen_action_kind === "tool_instance") && row.chosen_node_id === null) {
    issues.push(`${contract}/chosen_node_id is required for non-stop actions`);
  }

  if (row.chosen_action_kind === "traverse") {
    if (row.chosen_node_id && !candidateIds.has(row.chosen_action_id)) {
      issues.push(`${contract}/chosen_action_id must align with a traverse candidate`);
    }
  }

  if (!validateLabelProvenanceV1(row.label_provenance).valid) {
    issues.push(`${contract}/label_provenance failed validation`);
  }

  return issues.length === 0
    ? base
    : {
        contract,
        valid: false,
        issues: [...base.issues, ...issues],
      };
}

export interface RouteDecisionRowSummaryV1 {
  rowId: string;
  traceId: string;
  decisionPointId: string;
  decisionPointKind: RouteDecisionRowV1["decision_point_kind"];
  chosenActionKind: RouteDecisionRowV1["chosen_action_kind"];
  stopLabel: RouteDecisionRowV1["stop_label"];
  actionKindCounts: Record<RouteDecisionActionKindV1, number>;
  localActionCount: number;
  evidenceSpanCount: number;
  hardNegativeCount: number;
  provenanceState: AttributionTruthState;
  provenanceBasis: AttributionTruthMatchBasis;
}

export function summarizeRouteDecisionRowV1(row: RouteDecisionRowV1): RouteDecisionRowSummaryV1 {
  const actionKindCounts: Record<RouteDecisionActionKindV1, number> = {
    traverse: 0,
    tool_capability: 0,
    tool_instance: 0,
    stop_local: 0,
    stop: 0,
  };
  for (const candidate of row.local_action_set) {
    actionKindCounts[candidate.action_kind] += 1;
  }

  return {
    rowId: row.row_id,
    traceId: row.trace_id,
    decisionPointId: row.decision_point_id,
    decisionPointKind: row.decision_point_kind,
    chosenActionKind: row.chosen_action_kind,
    stopLabel: row.stop_label,
    actionKindCounts,
    localActionCount: row.local_action_set.length,
    evidenceSpanCount: row.evidence_spans.length,
    hardNegativeCount: row.hard_negatives.length,
    provenanceState: row.label_provenance.state,
    provenanceBasis: row.label_provenance.basis,
  };
}
