import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { Type, type Static } from "@sinclair/typebox";
import { Value } from "@sinclair/typebox/value";
import type { ContractValidationResultV1, RouteDecisionRowSummaryV1, RouteDecisionRowV1 } from "./route-rows.js";
import { summarizeRouteDecisionRowV1, validateRouteDecisionRowV1 } from "./route-rows.js";
import type { PolicySupervisionRowV1, PolicySupervisionSummaryV1 } from "./policy-supervision-rows.js";
import { PolicySupervisionRowSchemaV1, buildPolicySupervisionRowV1, summarizePolicySupervisionRowsV1, validatePolicySupervisionRowV1 } from "./policy-supervision-rows.js";

export const GRAPHIFY_TRAINING_REVIEW_BUNDLE_VERSION_V1 = 1 as const;
export const GRAPHIFY_TRAINING_REVIEW_BUNDLE_CONTRACT_V1 = "graphify_training_review_bundle.v1";

const REVIEW_TRUST_CLASSES = ["EXTRACTED", "INFERRED", "AMBIGUOUS"] as const;
const LIVE_ELIGIBLE_TRUST_CLASSES = ["EXTRACTED"] as const;
const REVIEW_ONLY_TRUST_CLASSES = ["INFERRED", "AMBIGUOUS"] as const;
const REVIEW_BLOCKED_EFFECTS = [
  "current_truth_write",
  "correction_like_memory",
  "live_eligible_edge",
  "hot_path_serve_integration",
] as const;

const REVIEW_STRONGER_TRUTH_LAYERS = ["runtime truth", "proof truth", "docs truth"] as const;
const REVIEW_CORRECTION_PRECEDENCE = [
  "explicit correction memory",
  "recent raw user/source turns",
  "raw proof/runtime evidence",
  "frozen docs truth",
] as const;
const ROUTE_OBJECTIVE_ORACLE_BEST_MODES = ["learned_route", "graph_prior_only", "tie", "unclear"] as const;
const ROUTE_OBJECTIVE_ACTIVATION_PREFERENCES = ["activate", "stop_local", "unclear"] as const;
const ROUTE_OBJECTIVE_PAIRWISE_PREFERENCES = [
  "learned_route_over_graph_prior_only",
  "graph_prior_only_over_learned_route",
  "tie",
  "unclear",
] as const;
const ROUTE_OBJECTIVE_PROJECTION_STATUSES = ["trace_only", "decision_point_review_required"] as const;
const ROUTE_OBJECTIVE_HARD_NEGATIVE_MODES = ["mine_decoy_actions", "mine_abstention_negatives", "review_only"] as const;
const ROUTE_OBJECTIVE_COST_SENSITIVITIES = ["low", "medium", "high"] as const;

function normalizeString(value: string | null | undefined): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeStringArray(values: Array<string | null | undefined> | null | undefined): string[] {
  if (!Array.isArray(values)) {
    return [];
  }
  return [...new Set(values.map((value) => normalizeString(value)).filter((value): value is string => value !== null))].sort();
}

function takeBounded<T>(values: readonly T[], limit: number): T[] {
  if (limit <= 0) {
    return [];
  }
  return values.slice(0, limit);
}

function hashStableParts(parts: Array<string | number | null | undefined>): string {
  return createHash("sha256").update(parts.map((part) => String(part ?? "")).join("\u001f")).digest("hex").slice(0, 16);
}

function stableJson(value: unknown): string {
  return JSON.stringify(value, (_key, inner) => {
    if (Array.isArray(inner)) {
      return inner;
    }
    if (inner && typeof inner === "object") {
      return Object.keys(inner as Record<string, unknown>)
        .sort()
        .reduce((acc, key) => {
          acc[key] = (inner as Record<string, unknown>)[key];
          return acc;
        }, {} as Record<string, unknown>);
    }
    return inner;
  });
}

function sha256Text(text: string): string {
  return `sha256:${createHash("sha256").update(String(text ?? ""), "utf8").digest("hex")}`;
}

export const GraphifyReviewTrustBoundarySchemaV1 = Type.Object(
  {
    authority: Type.Literal("graphify_proposal_truth"),
    stronger_truth_layers: Type.Array(Type.Union(REVIEW_STRONGER_TRUTH_LAYERS.map((value) => Type.Literal(value))), { minItems: 1 }),
    correction_precedence: Type.Array(Type.Union(REVIEW_CORRECTION_PRECEDENCE.map((value) => Type.Literal(value))), { minItems: 1 }),
    live_eligible_trust_classes: Type.Array(Type.Union(LIVE_ELIGIBLE_TRUST_CLASSES.map((value) => Type.Literal(value))), { minItems: 1 }),
    review_only_trust_classes: Type.Array(Type.Union(REVIEW_ONLY_TRUST_CLASSES.map((value) => Type.Literal(value))), { minItems: 1 }),
    blocked_effects: Type.Array(Type.Union(REVIEW_BLOCKED_EFFECTS.map((value) => Type.Literal(value))), { minItems: 1 }),
    review_mode: Type.Literal("candidate_only"),
    target_state_only: Type.Literal(true),
    live_eligible: Type.Literal(false),
    current_truth_writes: Type.Literal(false),
    correction_memory_writes: Type.Literal(false),
    hot_path_dependency: Type.Literal(false),
  },
  { additionalProperties: false },
);

export type GraphifyReviewBoundaryV1 = Static<typeof GraphifyReviewTrustBoundarySchemaV1>;

export const GraphifyNeighborhoodSummarySchemaV1 = Type.Object(
  {
    prior_id: Type.String({ minLength: 1 }),
    neighborhood_id: Type.String({ minLength: 1 }),
    source_artifact_id: Type.String({ minLength: 1 }),
    source_artifact_kind: Type.String({ minLength: 1 }),
    title: Type.String({ minLength: 1 }),
    subject_ids: Type.Array(Type.String({ minLength: 1 })),
    evidence_pointer_ids: Type.Array(Type.String({ minLength: 1 })),
    rationale_pointer_ids: Type.Array(Type.String({ minLength: 1 })),
    source_artifact_path: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    source_meta_path: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    trust_class: Type.Literal("EXTRACTED"),
  },
  { additionalProperties: false },
);

export type GraphifyNeighborhoodSummaryV1 = Static<typeof GraphifyNeighborhoodSummarySchemaV1>;

export const GraphifyNeighborhoodContextSchemaV1 = Type.Object(
  {
    context_id: Type.String({ minLength: 1 }),
    source_bundle_id: Type.String({ minLength: 1 }),
    source_bundle_hash: Type.String({ minLength: 1 }),
    graphify_run_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    graphify_version: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    graphify_command: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    trust_class: Type.Literal("EXTRACTED"),
    review_mode: Type.Literal("candidate_only"),
    hub_prior_ids: Type.Array(Type.String({ minLength: 1 })),
    neighborhood_prior_ids: Type.Array(Type.String({ minLength: 1 })),
    source_artifact_ids: Type.Array(Type.String({ minLength: 1 })),
    subject_ids: Type.Array(Type.String({ minLength: 1 })),
    evidence_pointer_ids: Type.Array(Type.String({ minLength: 1 })),
    rationale_pointer_ids: Type.Array(Type.String({ minLength: 1 })),
    neighborhoods: Type.Array(GraphifyNeighborhoodSummarySchemaV1),
    review_boundary: GraphifyReviewTrustBoundarySchemaV1,
  },
  { additionalProperties: false },
);

export type GraphifyNeighborhoodContextV1 = Static<typeof GraphifyNeighborhoodContextSchemaV1>;

export const GraphifyHardNegativeSupportSchemaV1 = Type.Object(
  {
    support_id: Type.String({ minLength: 1 }),
    route_row_id: Type.String({ minLength: 1 }),
    negative_value: Type.String({ minLength: 1 }),
    supporting_hub_prior_ids: Type.Array(Type.String({ minLength: 1 })),
    supporting_neighborhood_prior_ids: Type.Array(Type.String({ minLength: 1 })),
    supporting_evidence_pointer_ids: Type.Array(Type.String({ minLength: 1 })),
    summary: Type.String({ minLength: 1 }),
    trust_class: Type.Literal("EXTRACTED"),
    review_mode: Type.Literal("candidate_only"),
  },
  { additionalProperties: false },
);

export type GraphifyHardNegativeSupportV1 = Static<typeof GraphifyHardNegativeSupportSchemaV1>;

export const GraphifyProvenanceGapHintSchemaV1 = Type.Object(
  {
    hint_id: Type.String({ minLength: 1 }),
    route_row_ids: Type.Array(Type.String({ minLength: 1 })),
    source_artifact_id: Type.String({ minLength: 1 }),
    source_artifact_kind: Type.String({ minLength: 1 }),
    title: Type.String({ minLength: 1 }),
    summary: Type.String({ minLength: 1 }),
    claim_ids: Type.Array(Type.String({ minLength: 1 })),
    evidence_ids: Type.Array(Type.String({ minLength: 1 })),
    trust_class: Type.Literal("AMBIGUOUS"),
    review_mode: Type.Literal("review_only"),
    confidence: Type.Union([Type.Number({ minimum: 0, maximum: 1 }), Type.Null()]),
  },
  { additionalProperties: false },
);

export type GraphifyProvenanceGapHintV1 = Static<typeof GraphifyProvenanceGapHintSchemaV1>;

export const GraphifyRouteObjectiveHintSchemaV1 = Type.Object(
  {
    focus_lane: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    strict_hard_memory_eligible: Type.Boolean(),
    oracle_best_mode: Type.Union([
      ...ROUTE_OBJECTIVE_ORACLE_BEST_MODES.map((value) => Type.Literal(value)),
      Type.Null(),
    ]),
    pairwise_preference: Type.Union(ROUTE_OBJECTIVE_PAIRWISE_PREFERENCES.map((value) => Type.Literal(value))),
    activation_preference: Type.Union(ROUTE_OBJECTIVE_ACTIVATION_PREFERENCES.map((value) => Type.Literal(value))),
    preference_weight: Type.Number({ minimum: 0 }),
    utility_delta_vs_graph_prior_only: Type.Union([Type.Number(), Type.Null()]),
    cost_sensitivity: Type.Union([
      ...ROUTE_OBJECTIVE_COST_SENSITIVITIES.map((value) => Type.Literal(value)),
      Type.Null(),
    ]),
    projection_status: Type.Union(ROUTE_OBJECTIVE_PROJECTION_STATUSES.map((value) => Type.Literal(value))),
    positive_action_ids: Type.Array(Type.String({ minLength: 1 })),
    negative_action_ids: Type.Array(Type.String({ minLength: 1 })),
    hard_negative_mining_mode: Type.Union(ROUTE_OBJECTIVE_HARD_NEGATIVE_MODES.map((value) => Type.Literal(value))),
    notes: Type.Array(Type.String({ minLength: 1 })),
  },
  { additionalProperties: false },
);

export type GraphifyRouteObjectiveHintV1 = Static<typeof GraphifyRouteObjectiveHintSchemaV1>;

export interface GraphifyRouteObjectiveTraceLabelLikeV1 {
  traceId?: string | null;
  focusLane?: string | null;
  strictHardMemoryEligible?: boolean | null;
  oracleBestMode?: "learned_route" | "graph_prior_only" | "tie" | "unclear" | null;
  utilityDeltaVsBaseline?: number | null;
  utilityDelta?: number | null;
  netUtilityDelta?: number | null;
  costSensitive?: "low" | "medium" | "high" | null;
}

const RouteRowCoreSchemaV1 = Type.Object(
  {
    row_id: Type.String({ minLength: 1 }),
    trace_id: Type.String({ minLength: 1 }),
    episode_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    decision_point_id: Type.String({ minLength: 1 }),
    query_text: Type.String({ minLength: 1 }),
    route_fn_version: Type.String({ minLength: 1 }),
    chosen_action_id: Type.String({ minLength: 1 }),
    chosen_action_kind: Type.Union([
      Type.Literal("traverse"),
      Type.Literal("tool_capability"),
      Type.Literal("tool_instance"),
      Type.Literal("stop_local"),
      Type.Literal("stop"),
    ]),
    chosen_node_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    stop_label: Type.Union([Type.Literal("CONTINUE"), Type.Literal("STOP_LOCAL"), Type.Literal("STOP")]),
    hard_negatives: Type.Array(Type.String({ minLength: 1 })),
    label_provenance: Type.Object(
      {
        provenance_id: Type.String({ minLength: 1 }),
        state: Type.String({ minLength: 1 }),
        basis: Type.String({ minLength: 1 }),
        confidence: Type.Union([Type.Number({ minimum: 0, maximum: 1 }), Type.Null()]),
        binding_mode: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
        attribution_quality: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
        feedback_richness: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
        trace_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
        episode_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
        decision_point_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
      },
      { additionalProperties: true },
    ),
  },
  { additionalProperties: false },
);

const GraphifyTrainingReviewRowSchemaV1 = Type.Object(
  {
    row_id: Type.String({ minLength: 1 }),
    route_row_id: Type.String({ minLength: 1 }),
    route_row_summary: Type.Object(
      {
        rowId: Type.String({ minLength: 1 }),
        traceId: Type.String({ minLength: 1 }),
        decisionPointId: Type.String({ minLength: 1 }),
        decisionPointKind: Type.Union([Type.Literal("seed"), Type.Literal("local")]),
        chosenActionKind: Type.Union([
          Type.Literal("traverse"),
          Type.Literal("tool_capability"),
          Type.Literal("tool_instance"),
          Type.Literal("stop_local"),
          Type.Literal("stop"),
        ]),
        stopLabel: Type.Union([Type.Literal("CONTINUE"), Type.Literal("STOP_LOCAL"), Type.Literal("STOP")]),
        actionKindCounts: Type.Record(
          Type.String({ minLength: 1 }),
          Type.Number({ minimum: 0 }),
        ),
        localActionCount: Type.Number({ minimum: 0 }),
        evidenceSpanCount: Type.Number({ minimum: 0 }),
        hardNegativeCount: Type.Number({ minimum: 0 }),
        provenanceState: Type.String({ minLength: 1 }),
        provenanceBasis: Type.String({ minLength: 1 }),
      },
      { additionalProperties: false },
    ),
    route_row_core: RouteRowCoreSchemaV1,
    route_objective_hint: Type.Union([GraphifyRouteObjectiveHintSchemaV1, Type.Null()]),
    graphify_neighborhood_context: GraphifyNeighborhoodContextSchemaV1,
    hard_negative_supports: Type.Array(GraphifyHardNegativeSupportSchemaV1),
    provenance_gap_hint_ids: Type.Array(Type.String({ minLength: 1 })),
    review_notes: Type.Array(Type.String({ minLength: 1 })),
  },
  { additionalProperties: false },
);

export type GraphifyTrainingReviewRowV1 = Static<typeof GraphifyTrainingReviewRowSchemaV1>;

export const GraphifyTrainingReviewBundleSchemaV1 = Type.Object(
  {
    schema_version: Type.Literal(GRAPHIFY_TRAINING_REVIEW_BUNDLE_VERSION_V1),
    contract: Type.Literal(GRAPHIFY_TRAINING_REVIEW_BUNDLE_CONTRACT_V1),
    bundle_id: Type.String({ minLength: 1 }),
    source_bundle_id: Type.String({ minLength: 1 }),
    source_bundle_hash: Type.String({ minLength: 1 }),
    graphify_run_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    graphify_version: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    graphify_command: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    created_at: Type.String({ minLength: 1 }),
    review_boundary: GraphifyReviewTrustBoundarySchemaV1,
    row_count: Type.Number({ minimum: 0 }),
    included_row_count: Type.Number({ minimum: 0 }),
    truncated: Type.Boolean(),
    neighborhood_context_count: Type.Number({ minimum: 0 }),
    provenance_gap_hint_count: Type.Number({ minimum: 0 }),
    hard_negative_support_count: Type.Number({ minimum: 0 }),
    policy_supervision_row_count: Type.Number({ minimum: 0 }),
    route_row_ids: Type.Array(Type.String({ minLength: 1 })),
    rows: Type.Array(GraphifyTrainingReviewRowSchemaV1),
    provenance_gap_hints: Type.Array(GraphifyProvenanceGapHintSchemaV1),
    policy_supervision_rows: Type.Array(PolicySupervisionRowSchemaV1),
  },
  { additionalProperties: false },
);

export type GraphifyTrainingReviewBundleV1 = Static<typeof GraphifyTrainingReviewBundleSchemaV1>;

export interface GraphifyImportSliceNeighborhoodPriorLikeV1 {
  priorId?: string | null;
  trustClass?: string | null;
  kind?: string | null;
  neighborhoodId?: string | null;
  label?: string | null;
  title?: string | null;
  artifactKind?: string | null;
  subjectIds?: string[] | null;
  sourceRoots?: string[] | null;
  sourceArtifactId?: string | null;
  sourceArtifactPath?: string | null;
  sourceMetaPath?: string | null;
  evidencePointerIds?: string[] | null;
  rationalePointerIds?: string[] | null;
  authority?: string | null;
  derivation?: string | null;
}

export interface GraphifyImportSliceLikeV1 {
  sourceBundleId?: string | null;
  sourceBundleHash?: string | null;
  graphifyRunId?: string | null;
  graphifyVersion?: string | null;
  graphifyCommand?: string | null;
  candidatePackInput?: {
    importedPriors?: {
      hubPriors?: GraphifyImportSliceNeighborhoodPriorLikeV1[] | null;
      neighborhoodPriors?: GraphifyImportSliceNeighborhoodPriorLikeV1[] | null;
    } | null;
  } | null;
  importSlice?: {
    hubPriors?: GraphifyImportSliceNeighborhoodPriorLikeV1[] | null;
    neighborhoodPriors?: GraphifyImportSliceNeighborhoodPriorLikeV1[] | null;
  } | null;
  truthBoundary?: {
    authority?: string | null;
    strongerTruthLayers?: string[] | null;
    correctionPrecedence?: string[] | null;
    allowedTrustClasses?: string[] | null;
    blockedTrustClasses?: string[] | null;
    blockedEffects?: string[] | null;
  } | null;
}

function readNeighborhoodPriors(importSlice: GraphifyImportSliceLikeV1): GraphifyImportSliceNeighborhoodPriorLikeV1[] {
  const candidatePriors = importSlice.candidatePackInput?.importedPriors?.neighborhoodPriors;
  if (Array.isArray(candidatePriors) && candidatePriors.length > 0) {
    return candidatePriors;
  }
  const directPriors = importSlice.importSlice?.neighborhoodPriors;
  return Array.isArray(directPriors) ? directPriors : [];
}

function readHubPriors(importSlice: GraphifyImportSliceLikeV1): GraphifyImportSliceNeighborhoodPriorLikeV1[] {
  const candidatePriors = importSlice.candidatePackInput?.importedPriors?.hubPriors;
  if (Array.isArray(candidatePriors) && candidatePriors.length > 0) {
    return candidatePriors;
  }
  const directPriors = importSlice.importSlice?.hubPriors;
  return Array.isArray(directPriors) ? directPriors : [];
}

function buildReviewBoundary(): GraphifyReviewBoundaryV1 {
  return {
    authority: "graphify_proposal_truth",
    stronger_truth_layers: [...REVIEW_STRONGER_TRUTH_LAYERS],
    correction_precedence: [...REVIEW_CORRECTION_PRECEDENCE],
    live_eligible_trust_classes: [...LIVE_ELIGIBLE_TRUST_CLASSES],
    review_only_trust_classes: [...REVIEW_ONLY_TRUST_CLASSES],
    blocked_effects: [...REVIEW_BLOCKED_EFFECTS],
    review_mode: "candidate_only",
    target_state_only: true,
    live_eligible: false,
    current_truth_writes: false,
    correction_memory_writes: false,
    hot_path_dependency: false,
  };
}

function buildNeighborhoodSummary(prior: GraphifyImportSliceNeighborhoodPriorLikeV1): GraphifyNeighborhoodSummaryV1 {
  const evidencePointerIds = normalizeStringArray(prior.evidencePointerIds);
  const rationalePointerIds = normalizeStringArray(prior.rationalePointerIds);
  return {
    prior_id: normalizeString(prior.priorId) ?? `neighborhood-prior-${hashStableParts([prior.neighborhoodId, prior.sourceArtifactId, prior.title])}`,
    neighborhood_id: normalizeString(prior.neighborhoodId) ?? normalizeString(prior.sourceArtifactId) ?? `neighborhood-${hashStableParts([prior.title, prior.label])}`,
    source_artifact_id: normalizeString(prior.sourceArtifactId) ?? normalizeString(prior.neighborhoodId) ?? `artifact-${hashStableParts([prior.title, prior.label])}`,
    source_artifact_kind: normalizeString(prior.artifactKind) ?? normalizeString(prior.kind) ?? "neighborhood_summary",
    title: normalizeString(prior.title) ?? normalizeString(prior.label) ?? "Graphify neighborhood",
    subject_ids: normalizeStringArray(prior.subjectIds),
    evidence_pointer_ids: evidencePointerIds,
    rationale_pointer_ids: rationalePointerIds,
    source_artifact_path: normalizeString(prior.sourceArtifactPath),
    source_meta_path: normalizeString(prior.sourceMetaPath),
    trust_class: "EXTRACTED",
  };
}

function buildNeighborhoodContext(importSlice: GraphifyImportSliceLikeV1): GraphifyNeighborhoodContextV1 {
  const neighborhoods = takeBounded(readNeighborhoodPriors(importSlice).map((prior) => buildNeighborhoodSummary(prior)), 3);
  const hubs = takeBounded(readHubPriors(importSlice).map((prior) => normalizeString(prior.priorId) ?? normalizeString(prior.neighborhoodId) ?? normalizeString(prior.sourceArtifactId) ?? "hub-prior"), 3);
  const neighborhoodPriorIds = neighborhoods.map((neighborhood) => neighborhood.prior_id);
  const sourceArtifactIds = [...new Set(neighborhoods.map((neighborhood) => neighborhood.source_artifact_id))].sort();
  const subjectIds = normalizeStringArray(neighborhoods.flatMap((neighborhood) => neighborhood.subject_ids));
  const evidencePointerIds = normalizeStringArray(neighborhoods.flatMap((neighborhood) => neighborhood.evidence_pointer_ids));
  const rationalePointerIds = normalizeStringArray(neighborhoods.flatMap((neighborhood) => neighborhood.rationale_pointer_ids));
  const contextId = `graphify-neighborhood-context-${hashStableParts([
    importSlice.sourceBundleId,
    importSlice.sourceBundleHash,
    importSlice.graphifyRunId,
    importSlice.graphifyVersion,
    JSON.stringify(neighborhoodPriorIds),
  ])}`;

  return {
    context_id: contextId,
    source_bundle_id: normalizeString(importSlice.sourceBundleId) ?? "unknown-source-bundle",
    source_bundle_hash: normalizeString(importSlice.sourceBundleHash) ?? sha256Text(stableJson(neighborhoods)),
    graphify_run_id: normalizeString(importSlice.graphifyRunId),
    graphify_version: normalizeString(importSlice.graphifyVersion),
    graphify_command: normalizeString(importSlice.graphifyCommand),
    trust_class: "EXTRACTED",
    review_mode: "candidate_only",
    hub_prior_ids: hubs,
    neighborhood_prior_ids: neighborhoodPriorIds,
    source_artifact_ids: sourceArtifactIds,
    subject_ids: subjectIds,
    evidence_pointer_ids: evidencePointerIds,
    rationale_pointer_ids: rationalePointerIds,
    neighborhoods,
    review_boundary: buildReviewBoundary(),
  };
}

function readPackManifest(bundleRoot: string): { packId: string | null; artifacts: Array<Record<string, unknown>> } {
  const manifestPath = path.join(bundleRoot, "pack.manifest.json");
  if (!existsSync(manifestPath)) {
    throw new Error(`graphify training bridge expected pack.manifest.json at ${manifestPath}`);
  }
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as { packId?: string | null; artifacts?: Array<Record<string, unknown>> };
  return {
    packId: normalizeString(manifest.packId),
    artifacts: Array.isArray(manifest.artifacts) ? manifest.artifacts : [],
  };
}

function readArtifactMeta(bundleRoot: string, artifactSummary: Record<string, unknown>): Record<string, unknown> | null {
  const metaPathRaw = typeof artifactSummary.metaPath === "string" ? artifactSummary.metaPath : null;
  const artifactId = typeof artifactSummary.artifactId === "string" ? artifactSummary.artifactId : null;
  const metaPath = metaPathRaw ? path.resolve(bundleRoot, metaPathRaw) : artifactId ? path.join(bundleRoot, "artifacts", artifactId, "artifact.meta.json") : null;
  if (!metaPath || !existsSync(metaPath)) {
    return null;
  }
  return JSON.parse(readFileSync(metaPath, "utf8")) as Record<string, unknown>;
}

function collectProvenanceGapHints(params: {
  compiledArtifactPackRoot: string;
  routeRowIds: string[];
  sourceBundleId: string;
  sourceBundleHash: string;
  limit: number;
}): GraphifyProvenanceGapHintV1[] {
  const manifest = readPackManifest(params.compiledArtifactPackRoot);
  const provenanceArtifacts = manifest.artifacts
    .filter((artifact) => artifact.kind === "provenance_gap_report")
    .map((artifact) => readArtifactMeta(params.compiledArtifactPackRoot, artifact))
    .filter((artifact): artifact is Record<string, unknown> => artifact !== null);

  const hints: GraphifyProvenanceGapHintV1[] = [];
  for (const artifact of provenanceArtifacts) {
    const artifactTitle = normalizeString(typeof artifact.title === "string" ? artifact.title : null);
    const artifactSummary = normalizeString(typeof artifact.summary === "string" ? artifact.summary : null);
    const artifactId = normalizeString(typeof artifact.artifactId === "string" ? artifact.artifactId : null) ?? `provenance-gap-${hashStableParts([params.sourceBundleId, params.sourceBundleHash, artifactTitle ?? artifactSummary ?? "provenance-gap"])}`;
    const sourceArtifactKind = normalizeString(typeof artifact.kind === "string" ? artifact.kind : null) ?? "provenance_gap_report";
    const title = artifactTitle ?? "Graphify provenance gap report";
    const artifactEvidenceIds = normalizeStringArray(Array.isArray(artifact.evidence) ? artifact.evidence.map((evidence) => typeof evidence === "object" && evidence !== null ? (evidence as { evidenceId?: string }).evidenceId ?? null : null) : []);
    const claims = Array.isArray(artifact.claims)
      ? artifact.claims.filter((claim): claim is Record<string, unknown> => typeof claim === "object" && claim !== null)
      : [];
    for (const claim of claims) {
      if (hints.length >= params.limit) {
        break;
      }
      const claimText = normalizeString(typeof claim.text === "string" ? claim.text : typeof claim.summary === "string" ? claim.summary : null) ?? title;
      const claimId = normalizeString(typeof claim.claimId === "string" ? claim.claimId : null) ?? `claim-${hashStableParts([artifactId, claimText, JSON.stringify(claim)])}`;
      const evidenceIds = normalizeStringArray(Array.isArray((claim as { evidenceIds?: string[] }).evidenceIds) ? (claim as { evidenceIds?: string[] }).evidenceIds : artifactEvidenceIds);
      hints.push({
        hint_id: `gap-${hashStableParts([params.sourceBundleId, params.sourceBundleHash, artifactId, claimId])}`,
        route_row_ids: [...params.routeRowIds],
        source_artifact_id: artifactId,
        source_artifact_kind: sourceArtifactKind,
        title,
        summary: claimText,
        claim_ids: [claimId],
        evidence_ids: evidenceIds,
        trust_class: "AMBIGUOUS",
        review_mode: "review_only",
        confidence: typeof (claim as { confidence?: number }).confidence === "number" && Number.isFinite((claim as { confidence?: number }).confidence) ? Math.max(0, Math.min(1, (claim as { confidence?: number }).confidence as number)) : null,
      });
    }
  }

  if (hints.length === 0) {
    return [];
  }

  return hints.slice(0, params.limit);
}

function buildHardNegativeSupports(params: {
  routeRow: RouteDecisionRowV1;
  neighborhoodContext: GraphifyNeighborhoodContextV1;
  limit: number;
}): GraphifyHardNegativeSupportV1[] {
  const neighborhoodIds = takeBounded(params.neighborhoodContext.neighborhood_prior_ids, 2);
  const hubIds = takeBounded(params.neighborhoodContext.hub_prior_ids, 2);
  const evidenceIds = takeBounded(params.neighborhoodContext.evidence_pointer_ids, 4);
  return takeBounded(params.routeRow.hard_negatives, params.limit).map((negativeValue, index) => ({
    support_id: `negative-support-${hashStableParts([params.routeRow.row_id, negativeValue, index])}`,
    route_row_id: params.routeRow.row_id,
    negative_value: negativeValue,
    supporting_hub_prior_ids: hubIds,
    supporting_neighborhood_prior_ids: neighborhoodIds,
    supporting_evidence_pointer_ids: evidenceIds,
    summary: `Graphify neighborhood context keeps the route-row hard negative ${negativeValue} review-only and beneath canonical route-row truth.`,
    trust_class: "EXTRACTED",
    review_mode: "candidate_only",
  }));
}

function normalizeCostSensitivity(value: string | null | undefined): "low" | "medium" | "high" | null {
  if (value === "low" || value === "medium" || value === "high") {
    return value;
  }
  return null;
}

function normalizeOracleBestMode(value: string | null | undefined): "learned_route" | "graph_prior_only" | "tie" | "unclear" | null {
  if (value === "learned_route" || value === "graph_prior_only" || value === "tie" || value === "unclear") {
    return value;
  }
  return null;
}

function firstFiniteNumber(...values: Array<number | null | undefined>): number | null {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

function pairwisePreferenceFromOracleBestMode(oracleBestMode: "learned_route" | "graph_prior_only" | "tie" | "unclear" | null): GraphifyRouteObjectiveHintV1["pairwise_preference"] {
  if (oracleBestMode === "learned_route") {
    return "learned_route_over_graph_prior_only";
  }
  if (oracleBestMode === "graph_prior_only") {
    return "graph_prior_only_over_learned_route";
  }
  if (oracleBestMode === "tie") {
    return "tie";
  }
  return "unclear";
}

function activationPreferenceFromOracleBestMode(oracleBestMode: "learned_route" | "graph_prior_only" | "tie" | "unclear" | null): GraphifyRouteObjectiveHintV1["activation_preference"] {
  if (oracleBestMode === "learned_route") {
    return "activate";
  }
  if (oracleBestMode === "graph_prior_only" || oracleBestMode === "tie") {
    return "stop_local";
  }
  return "unclear";
}

function weightFromTraceObjective(input: {
  utilityDelta: number | null;
  costSensitivity: "low" | "medium" | "high" | null;
}): number {
  if (input.utilityDelta !== null) {
    return Number((1 + Math.abs(input.utilityDelta)).toFixed(6));
  }
  if (input.costSensitivity === "high") {
    return 2;
  }
  if (input.costSensitivity === "medium") {
    return 1.5;
  }
  return 1;
}

export function buildGraphifyRouteObjectiveHintV1(params: {
  routeRow: RouteDecisionRowV1;
  traceLabel: GraphifyRouteObjectiveTraceLabelLikeV1 | null | undefined;
}): GraphifyRouteObjectiveHintV1 | null {
  const traceLabel = params.traceLabel ?? null;
  if (!traceLabel) {
    return null;
  }

  const oracleBestMode = normalizeOracleBestMode(traceLabel.oracleBestMode ?? null);
  const activationPreference = activationPreferenceFromOracleBestMode(oracleBestMode);
  const pairwisePreference = pairwisePreferenceFromOracleBestMode(oracleBestMode);
  const costSensitivity = normalizeCostSensitivity(traceLabel.costSensitive ?? null);
  const utilityDelta = firstFiniteNumber(
    traceLabel.netUtilityDelta,
    traceLabel.utilityDeltaVsBaseline,
    traceLabel.utilityDelta,
  );
  const stopActionIds = params.routeRow.local_action_set
    .filter((candidate) => candidate.action_kind === "stop_local" || candidate.action_kind === "stop")
    .map((candidate) => candidate.action_id);
  const nonStopActionIds = params.routeRow.local_action_set
    .filter((candidate) => candidate.action_kind !== "stop_local" && candidate.action_kind !== "stop")
    .map((candidate) => candidate.action_id);

  const positiveActionIds = activationPreference === "activate"
    ? (params.routeRow.chosen_action_kind === "stop_local" || params.routeRow.chosen_action_kind === "stop"
        ? []
        : [params.routeRow.chosen_action_id])
    : activationPreference === "stop_local"
      ? stopActionIds
      : [];

  const negativeActionIds = activationPreference === "activate"
    ? normalizeStringArray([...stopActionIds, ...params.routeRow.hard_negatives])
    : activationPreference === "stop_local"
      ? normalizeStringArray([...nonStopActionIds, ...params.routeRow.hard_negatives])
      : normalizeStringArray(params.routeRow.hard_negatives);

  return {
    focus_lane: normalizeString(traceLabel.focusLane) ?? null,
    strict_hard_memory_eligible: traceLabel.strictHardMemoryEligible === true,
    oracle_best_mode: oracleBestMode,
    pairwise_preference: pairwisePreference,
    activation_preference: activationPreference,
    preference_weight: weightFromTraceObjective({ utilityDelta, costSensitivity }),
    utility_delta_vs_graph_prior_only: utilityDelta,
    cost_sensitivity: costSensitivity,
    projection_status: "trace_only",
    positive_action_ids: normalizeStringArray(positiveActionIds),
    negative_action_ids: negativeActionIds,
    hard_negative_mining_mode: activationPreference === "activate"
      ? "mine_decoy_actions"
      : activationPreference === "stop_local"
        ? "mine_abstention_negatives"
        : "review_only",
    notes: [
      "Trace-level route-objective hint only; decision-point projection still needs explicit review or stronger evidence.",
      "Preference weight is magnitude-based when utility delta is known, otherwise it falls back to cost sensitivity.",
      activationPreference === "activate"
        ? "Chosen non-stop action is exposed as a positive seed action because learned_route was judged oracle-best for the trace."
        : activationPreference === "stop_local"
          ? "STOP_LOCAL is exposed as the preferred abstention family because graph_prior_only or tie-cheaper was judged oracle-best for the trace."
          : "Oracle-best-mode is unclear, so this row remains review-only for route-objective use.",
    ],
  };
}

export function materializeGraphifyTrainingReviewBundleV1(params: {
  routeRows: RouteDecisionRowV1[];
  graphifyImportSlice: GraphifyImportSliceLikeV1;
  compiledArtifactPackRoot: string;
  routeObjectiveLabelsByTraceId?: Record<string, GraphifyRouteObjectiveTraceLabelLikeV1 | null | undefined>;
  maxRows?: number;
  maxProvenanceGapHints?: number;
  maxHardNegativeSupportsPerRow?: number;
  generatedAt?: string;
}): GraphifyTrainingReviewBundleV1 {
  const rowLimit = typeof params.maxRows === "number" && Number.isFinite(params.maxRows) ? Math.max(0, Math.floor(params.maxRows)) : 12;
  const hardNegativeLimit = typeof params.maxHardNegativeSupportsPerRow === "number" && Number.isFinite(params.maxHardNegativeSupportsPerRow) ? Math.max(0, Math.floor(params.maxHardNegativeSupportsPerRow)) : 4;
  const provenanceGapLimit = typeof params.maxProvenanceGapHints === "number" && Number.isFinite(params.maxProvenanceGapHints) ? Math.max(0, Math.floor(params.maxProvenanceGapHints)) : 4;
  const reviewBoundary = buildReviewBoundary();
  const neighborhoodContext = buildNeighborhoodContext(params.graphifyImportSlice);
  const includedRows = takeBounded(params.routeRows, rowLimit);
  const routeRowIds = includedRows.map((row) => row.row_id);

  const provenanceGapHints = collectProvenanceGapHints({
    compiledArtifactPackRoot: params.compiledArtifactPackRoot,
    routeRowIds,
    sourceBundleId: neighborhoodContext.source_bundle_id,
    sourceBundleHash: neighborhoodContext.source_bundle_hash,
    limit: provenanceGapLimit,
  });

  const provenanceGapHintIds = provenanceGapHints.map((hint) => hint.hint_id);
  const rows = includedRows.map((routeRow) => {
    const validation = validateRouteDecisionRowV1(routeRow);
    if (!validation.valid) {
      throw new Error(`invalid route row ${routeRow.row_id}: ${validation.issues.join("; ")}`);
    }
    const routeRowSummary = summarizeRouteDecisionRowV1(routeRow);
    const hardNegativeSupports = buildHardNegativeSupports({
      routeRow,
      neighborhoodContext,
      limit: hardNegativeLimit,
    });
    const routeObjectiveHint = buildGraphifyRouteObjectiveHintV1({
      routeRow,
      traceLabel: params.routeObjectiveLabelsByTraceId?.[routeRow.trace_id] ?? null,
    });
    const rowCore = {
      row_id: routeRow.row_id,
      trace_id: routeRow.trace_id,
      episode_id: routeRow.episode_id,
      decision_point_id: routeRow.decision_point_id,
      query_text: routeRow.query_text,
      route_fn_version: routeRow.route_fn_version,
      chosen_action_id: routeRow.chosen_action_id,
      chosen_action_kind: routeRow.chosen_action_kind,
      chosen_node_id: routeRow.chosen_node_id,
      stop_label: routeRow.stop_label,
      hard_negatives: [...routeRow.hard_negatives],
      label_provenance: {
        provenance_id: routeRow.label_provenance.provenance_id,
        state: routeRow.label_provenance.state,
        basis: routeRow.label_provenance.basis,
        confidence: routeRow.label_provenance.confidence,
        binding_mode: routeRow.label_provenance.binding_mode,
        attribution_quality: routeRow.label_provenance.attribution_quality,
        feedback_richness: routeRow.label_provenance.feedback_richness,
        trace_id: routeRow.label_provenance.trace_id,
        episode_id: routeRow.label_provenance.episode_id,
        decision_point_id: routeRow.label_provenance.decision_point_id,
      },
    };

      return {
        row_id: routeRow.row_id,
        route_row_id: routeRow.row_id,
        route_row_summary: routeRowSummary,
        route_row_core: rowCore,
        route_objective_hint: routeObjectiveHint,
        graphify_neighborhood_context: neighborhoodContext,
        hard_negative_supports: hardNegativeSupports,
        provenance_gap_hint_ids: [...provenanceGapHintIds],
        review_notes: [
          "Graphify is review-only here and never outranks the canonical route row.",
          "Only EXTRACTED neighborhood context may contribute to candidate training input.",
          "Provenance-gap hints remain blocked from live-eligible import and stay on the review side.",
          ...(routeObjectiveHint ? ["Route-objective hints are trace-level scaffolds only and do not replace decision-point teacher labels."] : []),
        ],
      };
  });

  const policySupervisionRows: PolicySupervisionRowV1[] = [];
  for (const routeRow of includedRows) {
    const traceLabel = params.routeObjectiveLabelsByTraceId?.[routeRow.trace_id] ?? null;
    if (traceLabel) {
      const psRow = buildPolicySupervisionRowV1({ routeRow, traceLabel });
      const psValidation = validatePolicySupervisionRowV1(psRow);
      if (psValidation.valid) {
        policySupervisionRows.push(psRow);
      }
    }
  }

  const sourceBundleId = normalizeString(params.graphifyImportSlice.sourceBundleId) ?? neighborhoodContext.source_bundle_id;
  const sourceBundleHash = normalizeString(params.graphifyImportSlice.sourceBundleHash) ?? neighborhoodContext.source_bundle_hash;
  const generatedAt = params.generatedAt ?? new Date().toISOString();

  const bundle: GraphifyTrainingReviewBundleV1 = {
    schema_version: GRAPHIFY_TRAINING_REVIEW_BUNDLE_VERSION_V1,
    contract: GRAPHIFY_TRAINING_REVIEW_BUNDLE_CONTRACT_V1,
    bundle_id: `graphify-training-review-${hashStableParts([sourceBundleId, sourceBundleHash, routeRowIds.join(",")])}`,
    source_bundle_id: sourceBundleId,
    source_bundle_hash: sourceBundleHash,
    graphify_run_id: normalizeString(params.graphifyImportSlice.graphifyRunId),
    graphify_version: normalizeString(params.graphifyImportSlice.graphifyVersion),
    graphify_command: normalizeString(params.graphifyImportSlice.graphifyCommand),
    created_at: generatedAt,
    review_boundary: reviewBoundary,
    row_count: params.routeRows.length,
    included_row_count: rows.length,
    truncated: rows.length < params.routeRows.length,
    neighborhood_context_count: neighborhoodContext.neighborhoods.length,
    provenance_gap_hint_count: provenanceGapHints.length,
    hard_negative_support_count: rows.reduce((count, row) => count + row.hard_negative_supports.length, 0),
    policy_supervision_row_count: policySupervisionRows.length,
    route_row_ids: routeRowIds,
    rows,
    provenance_gap_hints: provenanceGapHints,
    policy_supervision_rows: policySupervisionRows,
  };

  return bundle;
}

function validateSchema(contract: string, schema: unknown, value: unknown): ContractValidationResultV1 {
  const valid = Value.Check(schema as never, value);
  return {
    contract,
    valid,
    issues: valid ? [] : [`${contract}/schema_validation_failed`],
  };
}

export function validateGraphifyTrainingReviewRowV1(value: unknown): ContractValidationResultV1 {
  return validateSchema("graphify_training_review_row.v1", GraphifyTrainingReviewRowSchemaV1, value);
}

export function validateGraphifyTrainingReviewBundleV1(value: unknown): ContractValidationResultV1 {
  const contract = GRAPHIFY_TRAINING_REVIEW_BUNDLE_CONTRACT_V1;
  const base = validateSchema(contract, GraphifyTrainingReviewBundleSchemaV1, value);
  if (!base.valid) {
    return base;
  }

  const bundle = value as GraphifyTrainingReviewBundleV1;
  const rowIds = new Set(bundle.route_row_ids);
  const hintIds = new Set(bundle.provenance_gap_hints.map((hint) => hint.hint_id));
  const issues: string[] = [];

  if (bundle.rows.length !== bundle.included_row_count) {
    issues.push(`${contract}/included_row_count must match rows.length`);
  }

  for (const row of bundle.rows) {
    if (row.row_id !== row.route_row_id || row.row_id !== row.route_row_summary.rowId || row.row_id !== row.route_row_core.row_id) {
      issues.push(`${contract}/row id mismatch for ${row.row_id}`);
    }
    if (!rowIds.has(row.route_row_id)) {
      issues.push(`${contract}/route_row_ids must include ${row.route_row_id}`);
    }
    if (!validateGraphifyTrainingReviewRowV1(row).valid) {
      issues.push(`${contract}/row ${row.row_id} failed validation`);
    }
    for (const hintId of row.provenance_gap_hint_ids) {
      if (!hintIds.has(hintId)) {
        issues.push(`${contract}/row ${row.row_id} references unknown provenance gap hint ${hintId}`);
      }
    }
  }

  return issues.length === 0
    ? base
    : {
        contract,
        valid: false,
        issues: [...base.issues, ...issues],
      };
}

export interface GraphifyTrainingReviewSummaryV1 {
  bundleId: string;
  rowCount: number;
  includedRowCount: number;
  truncated: boolean;
  neighborhoodContextCount: number;
  provenanceGapHintCount: number;
  hardNegativeSupportCount: number;
  policySupervisionRowCount: number;
  policySupervisionSummary: PolicySupervisionSummaryV1;
  routeRowIds: string[];
}

export function summarizeGraphifyTrainingReviewBundleV1(bundle: GraphifyTrainingReviewBundleV1): GraphifyTrainingReviewSummaryV1 {
  return {
    bundleId: bundle.bundle_id,
    rowCount: bundle.row_count,
    includedRowCount: bundle.included_row_count,
    truncated: bundle.truncated,
    neighborhoodContextCount: bundle.neighborhood_context_count,
    provenanceGapHintCount: bundle.provenance_gap_hint_count,
    hardNegativeSupportCount: bundle.hard_negative_support_count,
    policySupervisionRowCount: bundle.policy_supervision_row_count,
    policySupervisionSummary: summarizePolicySupervisionRowsV1(bundle.policy_supervision_rows),
    routeRowIds: [...bundle.route_row_ids],
  };
}
