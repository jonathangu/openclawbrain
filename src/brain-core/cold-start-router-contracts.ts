/**
 * Cold-start learned-router contract surfaces.
 *
 * This file freezes the highest-leverage data contracts for the shipped
 * router prior: governed datasets, route-decision supervision, teacher and
 * verifier labels, graph compiler inputs/outputs, compact router manifests,
 * migration snapshots, replay records, promotion decisions, and proof bundles.
 */

import { Type, type Static, type TSchema } from "@sinclair/typebox";
import { Value } from "@sinclair/typebox/value";

export const COLD_START_CONTRACT_VERSION_V1 = 1 as const;

export const COLD_START_SOURCE_FAMILIES_V1 = [
  "docs",
  "repo",
  "memory",
  "tools",
  "qa",
  "agent_traces",
] as const;

export type ColdStartSourceFamilyV1 = (typeof COLD_START_SOURCE_FAMILIES_V1)[number];

export const COLD_START_APPROVAL_STATUSES_V1 = [
  "proposed",
  "under_review",
  "approved_train",
  "approved_eval_only",
  "rejected",
  "archived",
] as const;

export type ColdStartApprovalStatusV1 = (typeof COLD_START_APPROVAL_STATUSES_V1)[number];

export const COLD_START_COMMERCIAL_USE_STATUSES_V1 = [
  "allowed",
  "restricted",
  "unknown",
] as const;

export type ColdStartCommercialUseStatusV1 = (typeof COLD_START_COMMERCIAL_USE_STATUSES_V1)[number];

export const COLD_START_REDISTRIBUTION_STATUSES_V1 = [
  "allowed",
  "restricted",
  "prohibited",
  "unknown",
] as const;

export type ColdStartRedistributionStatusV1 = (typeof COLD_START_REDISTRIBUTION_STATUSES_V1)[number];

export const COLD_START_PII_RISKS_V1 = ["none", "low", "medium", "high", "unknown"] as const;

export type ColdStartPiiRiskV1 = (typeof COLD_START_PII_RISKS_V1)[number];

export const COLD_START_BENCHMARK_SPLIT_STATUSES_V1 = [
  "train",
  "eval_only",
  "holdout",
  "unknown",
] as const;

export type ColdStartBenchmarkSplitStatusV1 =
  (typeof COLD_START_BENCHMARK_SPLIT_STATUSES_V1)[number];

export const COLD_START_STOP_LABELS_V1 = ["CONTINUE", "STOP_LOCAL", "STOP"] as const;

export type ColdStartStopLabelV1 = (typeof COLD_START_STOP_LABELS_V1)[number];

export const COLD_START_CANDIDATE_TYPES_V1 = [
  "memory_node",
  "doc_chunk",
  "tool",
  "graph_node",
  "repo_node",
  "file",
  "issue",
  "pr",
  "trace",
] as const;

export type ColdStartCandidateTypeV1 = (typeof COLD_START_CANDIDATE_TYPES_V1)[number];

export const COLD_START_ROUTE_ACTION_KINDS_V1 = ["traverse", "tool"] as const;

export type ColdStartRouteActionKindV1 = (typeof COLD_START_ROUTE_ACTION_KINDS_V1)[number];

export const COLD_START_PACK_TYPES_V1 = ["base", "user_delta", "mixed"] as const;

export type ColdStartPackTypeV1 = (typeof COLD_START_PACK_TYPES_V1)[number];

export const COLD_START_PROMOTION_DECISIONS_V1 = [
  "promote",
  "hold",
  "rollback",
] as const;

export type ColdStartPromotionDecisionV1 =
  (typeof COLD_START_PROMOTION_DECISIONS_V1)[number];

export const COLD_START_REPLAY_VERDICTS_V1 = ["pass", "warn", "fail"] as const;

export type ColdStartReplayVerdictV1 = (typeof COLD_START_REPLAY_VERDICTS_V1)[number];

export const COLD_START_PROOF_BUNDLE_REQUIRED_FILES_V1 = [
  "summary.md",
  "status.json",
  "surface-map.json",
  "proposal-report.json",
  "verdict.json",
] as const;

export type ColdStartProofBundleRequiredFileV1 =
  (typeof COLD_START_PROOF_BUNDLE_REQUIRED_FILES_V1)[number];

export const EvidenceSpanRefSchemaV1 = Type.Object(
  {
    source_ref: Type.String({ minLength: 1 }),
    start: Type.Integer({ minimum: 0 }),
    end: Type.Integer({ minimum: 0 }),
    excerpt: Type.Optional(Type.String()),
  },
  { additionalProperties: false },
);

export type EvidenceSpanRefV1 = Static<typeof EvidenceSpanRefSchemaV1>;

export const RouteCandidateSchemaV1 = Type.Object(
  {
    candidate_id: Type.String({ minLength: 1 }),
    candidate_type: Type.Union([
      Type.Literal("memory_node"),
      Type.Literal("doc_chunk"),
      Type.Literal("tool"),
      Type.Literal("graph_node"),
      Type.Literal("repo_node"),
      Type.Literal("file"),
      Type.Literal("issue"),
      Type.Literal("pr"),
      Type.Literal("trace"),
    ]),
    authority: Type.Optional(Type.String({ minLength: 1 })),
    freshness: Type.Optional(Type.String({ minLength: 1 })),
    token_cost: Type.Optional(Type.Integer({ minimum: 0 })),
    score_hint: Type.Optional(Type.Number()),
  },
  { additionalProperties: false },
);

export type RouteCandidateV1 = Static<typeof RouteCandidateSchemaV1>;

export const RouteTeacherActionSchemaV1 = Type.Union([
  Type.Object(
    {
      kind: Type.Literal("traverse"),
      target_ids: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
    },
    { additionalProperties: false },
  ),
  Type.Object(
    {
      kind: Type.Literal("tool"),
      tool_name: Type.String({ minLength: 1 }),
      tool_args_ref: Type.Optional(Type.String({ minLength: 1 })),
    },
    { additionalProperties: false },
  ),
]);

export type RouteTeacherActionV1 = Static<typeof RouteTeacherActionSchemaV1>;

export const RouteProvenanceSchemaV1 = Type.Object(
  {
    dataset: Type.String({ minLength: 1 }),
    source_license: Type.String({ minLength: 1 }),
    source_family: Type.Optional(
      Type.Union([
        Type.Literal("docs"),
        Type.Literal("repo"),
        Type.Literal("memory"),
        Type.Literal("tools"),
        Type.Literal("qa"),
        Type.Literal("agent_traces"),
      ]),
    ),
    source_snapshot_ref: Type.Optional(Type.String({ minLength: 1 })),
    recorded_by: Type.String({ minLength: 1 }),
    recorded_at: Type.String({ minLength: 1 }),
    review_status: Type.Union([
      Type.Literal("proposed"),
      Type.Literal("under_review"),
      Type.Literal("approved_train"),
      Type.Literal("approved_eval_only"),
      Type.Literal("rejected"),
      Type.Literal("archived"),
    ]),
  },
  { additionalProperties: false },
);

export type RouteProvenanceV1 = Static<typeof RouteProvenanceSchemaV1>;

export const RouteDecisionRowSchemaV1 = Type.Object(
  {
    row_id: Type.String({ minLength: 1 }),
    dataset_id: Type.String({ minLength: 1 }),
    query: Type.String({ minLength: 1 }),
    cursor_path: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
    candidate_set: Type.Array(RouteCandidateSchemaV1, { minItems: 1 }),
    teacher_action: RouteTeacherActionSchemaV1,
    stop_label: Type.Union([
      Type.Literal("CONTINUE"),
      Type.Literal("STOP_LOCAL"),
      Type.Literal("STOP"),
    ]),
    evidence_spans: Type.Array(EvidenceSpanRefSchemaV1, { minItems: 1 }),
    hard_negatives: Type.Array(Type.String({ minLength: 1 })),
    outcome_gain: Type.Number(),
    provenance: RouteProvenanceSchemaV1,
    split_tag: Type.String({ minLength: 1 }),
    created_at: Type.String({ minLength: 1 }),
  },
  { additionalProperties: false },
);

export type RouteDecisionRowV1 = Static<typeof RouteDecisionRowSchemaV1>;

export const DataRegistryEntrySchemaV1 = Type.Object(
  {
    dataset_id: Type.String({ minLength: 1 }),
    source_family: Type.Union([
      Type.Literal("docs"),
      Type.Literal("repo"),
      Type.Literal("memory"),
      Type.Literal("tools"),
      Type.Literal("qa"),
      Type.Literal("agent_traces"),
    ]),
    upstream_url: Type.String({ minLength: 1 }),
    original_creator: Type.String({ minLength: 1 }),
    license: Type.String({ minLength: 1 }),
    commercial_use_status: Type.Union([
      Type.Literal("allowed"),
      Type.Literal("restricted"),
      Type.Literal("unknown"),
    ]),
    redistribution_status: Type.Union([
      Type.Literal("allowed"),
      Type.Literal("restricted"),
      Type.Literal("prohibited"),
      Type.Literal("unknown"),
    ]),
    pii_risk: Type.Union([
      Type.Literal("none"),
      Type.Literal("low"),
      Type.Literal("medium"),
      Type.Literal("high"),
      Type.Literal("unknown"),
    ]),
    benchmark_split_status: Type.Union([
      Type.Literal("train"),
      Type.Literal("eval_only"),
      Type.Literal("holdout"),
      Type.Literal("unknown"),
    ]),
    approval_status: Type.Union([
      Type.Literal("proposed"),
      Type.Literal("under_review"),
      Type.Literal("approved_train"),
      Type.Literal("approved_eval_only"),
      Type.Literal("rejected"),
      Type.Literal("archived"),
    ]),
    reviewer: Type.String({ minLength: 1 }),
    immutable_snapshot_ref: Type.String({ minLength: 1 }),
    exact_files: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
    file_hashes: Type.Record(Type.String({ minLength: 1 }), Type.String({ minLength: 1 })),
    allowed_uses: Type.Array(Type.String({ minLength: 1 })),
    disallowed_uses: Type.Array(Type.String({ minLength: 1 })),
    notes: Type.Array(Type.String({ minLength: 1 })),
    created_at: Type.String({ minLength: 1 }),
    updated_at: Type.String({ minLength: 1 }),
  },
  { additionalProperties: false },
);

export type DataRegistryEntryV1 = Static<typeof DataRegistryEntrySchemaV1>;

export const TeacherLabelContractSchemaV1 = Type.Object(
  {
    label_id: Type.String({ minLength: 1 }),
    dataset_id: Type.String({ minLength: 1 }),
    row_id: Type.String({ minLength: 1 }),
    best_next_node_ids: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
    best_next_tool_name: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    stop_label: Type.Union([
      Type.Literal("CONTINUE"),
      Type.Literal("STOP_LOCAL"),
      Type.Literal("STOP"),
    ]),
    evidence_spans: Type.Array(EvidenceSpanRefSchemaV1, { minItems: 1 }),
    hard_negatives: Type.Array(Type.String({ minLength: 1 })),
    confidence: Type.Number({ minimum: 0, maximum: 1 }),
    rationale: Type.String({ minLength: 1 }),
    created_at: Type.String({ minLength: 1 }),
  },
  { additionalProperties: false },
);

export type TeacherLabelContractV1 = Static<typeof TeacherLabelContractSchemaV1>;

export const VerifierCheckSchemaV1 = Type.Object(
  {
    check_id: Type.String({ minLength: 1 }),
    status: Type.Union([
      Type.Literal("pass"),
      Type.Literal("warn"),
      Type.Literal("fail"),
    ]),
    summary: Type.String({ minLength: 1 }),
    evidence_refs: Type.Array(Type.String({ minLength: 1 })),
  },
  { additionalProperties: false },
);

export type VerifierCheckV1 = Static<typeof VerifierCheckSchemaV1>;

export const VerifierContractSchemaV1 = Type.Object(
  {
    verifier_id: Type.String({ minLength: 1 }),
    label_id: Type.String({ minLength: 1 }),
    row_id: Type.String({ minLength: 1 }),
    candidate_set_digest: Type.String({ minLength: 1 }),
    checks: Type.Array(VerifierCheckSchemaV1, { minItems: 1 }),
    passed: Type.Boolean(),
    issues: Type.Array(Type.String({ minLength: 1 })),
    explicit_correction_priority_honored: Type.Boolean(),
    created_at: Type.String({ minLength: 1 }),
  },
  { additionalProperties: false },
);

export type VerifierContractV1 = Static<typeof VerifierContractSchemaV1>;

export const GraphNodeSchemaV1 = Type.Object(
  {
    node_kind: Type.String({ minLength: 1 }),
    required_fields: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
    optional_fields: Type.Array(Type.String({ minLength: 1 })),
    notes: Type.Array(Type.String({ minLength: 1 })),
  },
  { additionalProperties: false },
);

export type GraphNodeSchemaV1Type = Static<typeof GraphNodeSchemaV1>;

export const GraphEdgeSchemaV1 = Type.Object(
  {
    edge_kind: Type.String({ minLength: 1 }),
    required_fields: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
    optional_fields: Type.Array(Type.String({ minLength: 1 })),
    notes: Type.Array(Type.String({ minLength: 1 })),
  },
  { additionalProperties: false },
);

export type GraphEdgeSchemaV1Type = Static<typeof GraphEdgeSchemaV1>;

export const GraphNeighborhoodPackSchemaV1 = Type.Object(
  {
    pack_id: Type.String({ minLength: 1 }),
    artifact_ref: Type.String({ minLength: 1 }),
    graph_ref: Type.String({ minLength: 1 }),
    radius_hops: Type.Integer({ minimum: 0 }),
    frontier_limit: Type.Integer({ minimum: 0 }),
  },
  { additionalProperties: false },
);

export type GraphNeighborhoodPackV1 = Static<typeof GraphNeighborhoodPackSchemaV1>;

export const GraphCompilerContractSchemaV1 = Type.Object(
  {
    compiler_id: Type.String({ minLength: 1 }),
    source_family: Type.Union([
      Type.Literal("docs"),
      Type.Literal("repo"),
      Type.Literal("memory"),
      Type.Literal("tools"),
      Type.Literal("qa"),
      Type.Literal("agent_traces"),
    ]),
    input_snapshot_ref: Type.String({ minLength: 1 }),
    node_schema: Type.Array(GraphNodeSchemaV1, { minItems: 1 }),
    edge_schema: Type.Array(GraphEdgeSchemaV1, { minItems: 1 }),
    provenance_rules: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
    output_neighborhood_pack: GraphNeighborhoodPackSchemaV1,
    compiler_version: Type.String({ minLength: 1 }),
  },
  { additionalProperties: false },
);

export type GraphCompilerContractV1 = Static<typeof GraphCompilerContractSchemaV1>;

export const RouterArtifactManifestSchemaV1 = Type.Object(
  {
    schema_version: Type.Literal(COLD_START_CONTRACT_VERSION_V1),
    artifact_id: Type.String({ minLength: 1 }),
    artifact_version: Type.String({ minLength: 1 }),
    pack_type: Type.Union([
      Type.Literal("base"),
      Type.Literal("user_delta"),
      Type.Literal("mixed"),
    ]),
    base_model_ref: Type.String({ minLength: 1 }),
    weights_ref: Type.String({ minLength: 1 }),
    calibration_ref: Type.String({ minLength: 1 }),
    feature_normalizers_ref: Type.String({ minLength: 1 }),
    source_priors_ref: Type.String({ minLength: 1 }),
    safety_rules_ref: Type.String({ minLength: 1 }),
    compatible_runtime_version: Type.String({ minLength: 1 }),
    training_data_refs: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
    replay_gate_refs: Type.Array(Type.String({ minLength: 1 })),
    prior_base_artifact_id: Type.Optional(Type.String({ minLength: 1 })),
    prior_base_artifact_checksum: Type.Optional(Type.String({ minLength: 1 })),
    artifact_checksum: Type.String({ minLength: 1 }),
    created_at: Type.String({ minLength: 1 }),
    router_identity: Type.Optional(Type.String({ minLength: 1 })),
  },
  { additionalProperties: false },
);

export type RouterArtifactManifestV1 = Static<typeof RouterArtifactManifestSchemaV1>;

export const MigrationSnapshotSchemaV1 = Type.Object(
  {
    schema_version: Type.Literal(COLD_START_CONTRACT_VERSION_V1),
    snapshot_id: Type.String({ minLength: 1 }),
    current_live_pack_id: Type.String({ minLength: 1 }),
    current_router_identity: Type.String({ minLength: 1 }),
    current_router_checksum: Type.String({ minLength: 1 }),
    graph_snapshot_ref: Type.String({ minLength: 1 }),
    feedback_log_export_ref: Type.String({ minLength: 1 }),
    user_delta_state_ref: Type.String({ minLength: 1 }),
    recent_overlay_ref: Type.String({ minLength: 1 }),
    rollback_key: Type.String({ minLength: 1 }),
    proof_refs: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
    timestamp: Type.String({ minLength: 1 }),
  },
  { additionalProperties: false },
);

export type MigrationSnapshotV1 = Static<typeof MigrationSnapshotSchemaV1>;

export const ReplayPolicyComparisonSchemaV1 = Type.Object(
  {
    old_live: Type.String({ minLength: 1 }),
    base_only: Type.String({ minLength: 1 }),
    mixed: Type.String({ minLength: 1 }),
  },
  { additionalProperties: false },
);

export type ReplayPolicyComparisonV1 = Static<typeof ReplayPolicyComparisonSchemaV1>;

export const ReplayRunRecordSchemaV1 = Type.Object(
  {
    schema_version: Type.Literal(COLD_START_CONTRACT_VERSION_V1),
    run_id: Type.String({ minLength: 1 }),
    scenario_id: Type.String({ minLength: 1 }),
    compared_policies: ReplayPolicyComparisonSchemaV1,
    metrics: Type.Record(Type.String({ minLength: 1 }), Type.Number()),
    regression_flags: Type.Record(Type.String({ minLength: 1 }), Type.Boolean()),
    verdict: Type.Union([
      Type.Literal("pass"),
      Type.Literal("warn"),
      Type.Literal("fail"),
    ]),
    proof_refs: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
    timestamp: Type.String({ minLength: 1 }),
  },
  { additionalProperties: false },
);

export type ReplayRunRecordV1 = Static<typeof ReplayRunRecordSchemaV1>;

export const PromotionGateResultSchemaV1 = Type.Object(
  {
    gate_id: Type.String({ minLength: 1 }),
    verdict: Type.Union([
      Type.Literal("pass"),
      Type.Literal("warn"),
      Type.Literal("fail"),
    ]),
    summary: Type.String({ minLength: 1 }),
    proof_refs: Type.Array(Type.String({ minLength: 1 })),
  },
  { additionalProperties: false },
);

export type PromotionGateResultV1 = Static<typeof PromotionGateResultSchemaV1>;

export const PromotionDecisionSchemaV1 = Type.Object(
  {
    schema_version: Type.Literal(COLD_START_CONTRACT_VERSION_V1),
    decision_id: Type.String({ minLength: 1 }),
    candidate_artifact_id: Type.String({ minLength: 1 }),
    gate_results: Type.Array(PromotionGateResultSchemaV1, { minItems: 1 }),
    decision: Type.Union([
      Type.Literal("promote"),
      Type.Literal("hold"),
      Type.Literal("rollback"),
    ]),
    reviewer: Type.String({ minLength: 1 }),
    approver: Type.String({ minLength: 1 }),
    rollback_binding: Type.String({ minLength: 1 }),
    proof_bundle_ref: Type.String({ minLength: 1 }),
    timestamp: Type.String({ minLength: 1 }),
  },
  { additionalProperties: false },
);

export type PromotionDecisionV1 = Static<typeof PromotionDecisionSchemaV1>;

export const ProofBundleArtifactChecksumsSchemaV1 = Type.Object(
  {
    "summary.md": Type.String({ minLength: 1 }),
    "status.json": Type.String({ minLength: 1 }),
    "surface-map.json": Type.String({ minLength: 1 }),
    "proposal-report.json": Type.String({ minLength: 1 }),
    "verdict.json": Type.String({ minLength: 1 }),
  },
  { additionalProperties: false },
);

export type ProofBundleArtifactChecksumsV1 = Static<typeof ProofBundleArtifactChecksumsSchemaV1>;

export const ProofBundleSchemaV1 = Type.Object(
  {
    schema_version: Type.Literal(COLD_START_CONTRACT_VERSION_V1),
    bundle_id: Type.String({ minLength: 1 }),
    artifact_id: Type.String({ minLength: 1 }),
    owner: Type.String({ minLength: 1 }),
    environment: Type.String({ minLength: 1 }),
    source_refs: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
    shipped_state_label: Type.String({ minLength: 1 }),
    target_state_label: Type.String({ minLength: 1 }),
    required_files: Type.Array(
      Type.Union([
        Type.Literal("summary.md"),
        Type.Literal("status.json"),
        Type.Literal("surface-map.json"),
        Type.Literal("proposal-report.json"),
        Type.Literal("verdict.json"),
      ]),
      { minItems: COLD_START_PROOF_BUNDLE_REQUIRED_FILES_V1.length, maxItems: COLD_START_PROOF_BUNDLE_REQUIRED_FILES_V1.length },
    ),
    artifact_checksums: ProofBundleArtifactChecksumsSchemaV1,
    bounded_summary_guard: Type.Object(
      {
        max_chars: Type.Integer({ minimum: 0 }),
        max_lines: Type.Integer({ minimum: 0 }),
        enforced: Type.Boolean(),
      },
      { additionalProperties: false },
    ),
    proof_refs: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
    timestamp: Type.String({ minLength: 1 }),
  },
  { additionalProperties: false },
);

export type ProofBundleV1 = Static<typeof ProofBundleSchemaV1>;

export interface ContractValidationResultV1 {
  contract: string;
  valid: boolean;
  issues: string[];
}

function formatValidationIssues(contract: string, schema: TSchema, value: unknown): string[] {
  return [...Value.Errors(schema, value)].map((error) => {
    const path = error.path && error.path.length > 0 ? error.path : "/";
    return `${contract}${path}: ${error.message}`;
  });
}

function validateSchema(contract: string, schema: TSchema, value: unknown): ContractValidationResultV1 {
  if (Value.Check(schema, value)) {
    return { contract, valid: true, issues: [] };
  }
  return { contract, valid: false, issues: formatValidationIssues(contract, schema, value) };
}

function toSet(values: readonly string[]): Set<string> {
  return new Set(values.map((value) => value.trim()).filter(Boolean));
}

function sortedDifference(left: readonly string[], right: readonly string[]): string[] {
  const rightSet = toSet(right);
  return [...toSet(left)].filter((value) => !rightSet.has(value)).sort();
}

export function validateDataRegistryEntryV1(value: unknown): ContractValidationResultV1 {
  const contract = "cold_start_data_registry_entry.v1";
  const base = validateSchema(contract, DataRegistryEntrySchemaV1, value);
  if (!base.valid) {
    return base;
  }

  const entry = value as DataRegistryEntryV1;
  const exactFiles = toSet(entry.exact_files);
  const missing = [...exactFiles].filter((file) => !(file in entry.file_hashes)).sort();
  const extra = sortedDifference(Object.keys(entry.file_hashes), entry.exact_files);

  if (missing.length === 0 && extra.length === 0) {
    return base;
  }

  return {
    contract,
    valid: false,
    issues: [
      ...base.issues,
      ...(missing.length > 0 ? [`${contract}/exact_files missing hashes for: ${missing.join(", ")}`] : []),
      ...(extra.length > 0 ? [`${contract}/file_hashes has extra entries not listed in exact_files: ${extra.join(", ")}`] : []),
    ],
  };
}

export function validateRouteDecisionRowV1(value: unknown): ContractValidationResultV1 {
  const contract = "cold_start_route_decision_row.v1";
  const base = validateSchema(contract, RouteDecisionRowSchemaV1, value);
  if (!base.valid) {
    return base;
  }

  const row = value as RouteDecisionRowV1;
  const candidateIds = new Set(row.candidate_set.map((candidate) => candidate.candidate_id));

  if (row.teacher_action.kind === "traverse") {
    const missingTargets = row.teacher_action.target_ids.filter((targetId) => !candidateIds.has(targetId));
    if (missingTargets.length === 0) {
      return base;
    }
    return {
      contract,
      valid: false,
      issues: [
        ...base.issues,
        `${contract}/teacher_action target_ids not present in candidate_set: ${missingTargets.join(", ")}`,
      ],
    };
  }

  if (row.teacher_action.kind === "tool" && row.teacher_action.tool_name.trim().length === 0) {
    return {
      contract,
      valid: false,
      issues: [...base.issues, `${contract}/teacher_action tool_name must be non-empty`],
    };
  }

  return base;
}

export function validateTeacherLabelContractV1(value: unknown): ContractValidationResultV1 {
  return validateSchema("cold_start_teacher_label.v1", TeacherLabelContractSchemaV1, value);
}

export function validateVerifierContractV1(value: unknown): ContractValidationResultV1 {
  const contract = "cold_start_verifier_contract.v1";
  const base = validateSchema(contract, VerifierContractSchemaV1, value);
  if (!base.valid) {
    return base;
  }

  const verifier = value as VerifierContractV1;
  const hasFail = verifier.checks.some((check) => check.status === "fail");
  const hasPass = verifier.checks.some((check) => check.status === "pass");

  if (verifier.passed && hasFail) {
    return {
      contract,
      valid: false,
      issues: [...base.issues, `${contract}/passed cannot be true when a check has status fail`],
    };
  }

  if (!verifier.passed && !hasFail) {
    return {
      contract,
      valid: false,
      issues: [...base.issues, `${contract}/passed is false but no failing check was recorded`],
    };
  }

  if (verifier.explicit_correction_priority_honored === false && hasPass) {
    return {
      contract,
      valid: false,
      issues: [...base.issues, `${contract}/explicit_correction_priority_honored must be true for a passing verifier`],
    };
  }

  return base;
}

export function validateGraphCompilerContractV1(value: unknown): ContractValidationResultV1 {
  return validateSchema("cold_start_graph_compiler.v1", GraphCompilerContractSchemaV1, value);
}

export function validateRouterArtifactManifestV1(value: unknown): ContractValidationResultV1 {
  const contract = "cold_start_router_artifact_manifest.v1";
  const base = validateSchema(contract, RouterArtifactManifestSchemaV1, value);
  if (!base.valid) {
    return base;
  }

  const manifest = value as RouterArtifactManifestV1;
  const hasPriorBaseArtifactId = manifest.prior_base_artifact_id !== undefined;
  const hasPriorBaseArtifactChecksum = manifest.prior_base_artifact_checksum !== undefined;
  if (hasPriorBaseArtifactId !== hasPriorBaseArtifactChecksum) {
    return {
      contract,
      valid: false,
      issues: [
        ...base.issues,
        `${contract}/prior_base_artifact_id and prior_base_artifact_checksum must be provided together`,
      ],
    };
  }

  return base;
}

export function validateMigrationSnapshotV1(value: unknown): ContractValidationResultV1 {
  return validateSchema("cold_start_migration_snapshot.v1", MigrationSnapshotSchemaV1, value);
}

export function validateReplayRunRecordV1(value: unknown): ContractValidationResultV1 {
  return validateSchema("cold_start_replay_run_record.v1", ReplayRunRecordSchemaV1, value);
}

export function validatePromotionDecisionV1(value: unknown): ContractValidationResultV1 {
  return validateSchema("cold_start_promotion_decision.v1", PromotionDecisionSchemaV1, value);
}

export function validateProofBundleV1(value: unknown): ContractValidationResultV1 {
  const contract = "cold_start_proof_bundle.v1";
  const base = validateSchema(contract, ProofBundleSchemaV1, value);
  if (!base.valid) {
    return base;
  }

  const bundle = value as ProofBundleV1;
  const requiredFiles = toSet(COLD_START_PROOF_BUNDLE_REQUIRED_FILES_V1);
  const declaredRequiredFiles = toSet(bundle.required_files);
  const checksumFiles = toSet(Object.keys(bundle.artifact_checksums));

  const missingRequired = [...requiredFiles].filter((file) => !declaredRequiredFiles.has(file)).sort();
  const extraRequired = [...declaredRequiredFiles].filter((file) => !requiredFiles.has(file)).sort();
  const missingChecksums = [...requiredFiles].filter((file) => !checksumFiles.has(file)).sort();
  const extraChecksums = [...checksumFiles].filter((file) => !requiredFiles.has(file)).sort();

  if (missingRequired.length === 0 && extraRequired.length === 0 && missingChecksums.length === 0 && extraChecksums.length === 0) {
    return base;
  }

  return {
    contract,
    valid: false,
    issues: [
      ...base.issues,
      ...(missingRequired.length > 0 ? [`${contract}/required_files missing: ${missingRequired.join(", ")}`] : []),
      ...(extraRequired.length > 0 ? [`${contract}/required_files has unexpected entries: ${extraRequired.join(", ")}`] : []),
      ...(missingChecksums.length > 0 ? [`${contract}/artifact_checksums missing: ${missingChecksums.join(", ")}`] : []),
      ...(extraChecksums.length > 0 ? [`${contract}/artifact_checksums has unexpected entries: ${extraChecksums.join(", ")}`] : []),
    ],
  };
}

export interface DataRegistryEntrySummaryV1 {
  datasetId: string;
  sourceFamily: ColdStartSourceFamilyV1;
  approvalStatus: ColdStartApprovalStatusV1;
  piiRisk: ColdStartPiiRiskV1;
  fileCount: number;
  fileHashCount: number;
  allowedUseCount: number;
  disallowedUseCount: number;
}

export function summarizeDataRegistryEntryV1(entry: DataRegistryEntryV1): DataRegistryEntrySummaryV1 {
  return {
    datasetId: entry.dataset_id,
    sourceFamily: entry.source_family,
    approvalStatus: entry.approval_status,
    piiRisk: entry.pii_risk,
    fileCount: entry.exact_files.length,
    fileHashCount: Object.keys(entry.file_hashes).length,
    allowedUseCount: entry.allowed_uses.length,
    disallowedUseCount: entry.disallowed_uses.length,
  };
}

export interface RouteDecisionRowSummaryV1 {
  rowId: string;
  datasetId: string;
  candidateCount: number;
  evidenceSpanCount: number;
  hardNegativeCount: number;
  teacherActionKind: ColdStartRouteActionKindV1;
  stopLabel: ColdStartStopLabelV1;
  outcomeGain: number;
  splitTag: string;
  provenanceDataset: string;
  approvalStatus: ColdStartApprovalStatusV1;
}

export function summarizeRouteDecisionRowV1(row: RouteDecisionRowV1): RouteDecisionRowSummaryV1 {
  return {
    rowId: row.row_id,
    datasetId: row.dataset_id,
    candidateCount: row.candidate_set.length,
    evidenceSpanCount: row.evidence_spans.length,
    hardNegativeCount: row.hard_negatives.length,
    teacherActionKind: row.teacher_action.kind,
    stopLabel: row.stop_label,
    outcomeGain: row.outcome_gain,
    splitTag: row.split_tag,
    provenanceDataset: row.provenance.dataset,
    approvalStatus: row.provenance.review_status,
  };
}

export interface TeacherLabelContractSummaryV1 {
  labelId: string;
  rowId: string;
  bestNextNodeCount: number;
  hasBestNextTool: boolean;
  stopLabel: ColdStartStopLabelV1;
  evidenceSpanCount: number;
  hardNegativeCount: number;
  confidence: number;
}

export function summarizeTeacherLabelContractV1(
  label: TeacherLabelContractV1,
): TeacherLabelContractSummaryV1 {
  return {
    labelId: label.label_id,
    rowId: label.row_id,
    bestNextNodeCount: label.best_next_node_ids.length,
    hasBestNextTool: label.best_next_tool_name !== null,
    stopLabel: label.stop_label,
    evidenceSpanCount: label.evidence_spans.length,
    hardNegativeCount: label.hard_negatives.length,
    confidence: label.confidence,
  };
}

export interface VerifierContractSummaryV1 {
  verifierId: string;
  labelId: string;
  rowId: string;
  passed: boolean;
  checkCount: number;
  failingCheckCount: number;
  issueCount: number;
  explicitCorrectionPriorityHonored: boolean;
}

export function summarizeVerifierContractV1(verifier: VerifierContractV1): VerifierContractSummaryV1 {
  return {
    verifierId: verifier.verifier_id,
    labelId: verifier.label_id,
    rowId: verifier.row_id,
    passed: verifier.passed,
    checkCount: verifier.checks.length,
    failingCheckCount: verifier.checks.filter((check) => check.status === "fail").length,
    issueCount: verifier.issues.length,
    explicitCorrectionPriorityHonored: verifier.explicit_correction_priority_honored,
  };
}

export interface GraphCompilerContractSummaryV1 {
  compilerId: string;
  sourceFamily: ColdStartSourceFamilyV1;
  nodeSchemaCount: number;
  edgeSchemaCount: number;
  provenanceRuleCount: number;
  neighborhoodPackId: string;
  compilerVersion: string;
}

export function summarizeGraphCompilerContractV1(
  compiler: GraphCompilerContractV1,
): GraphCompilerContractSummaryV1 {
  return {
    compilerId: compiler.compiler_id,
    sourceFamily: compiler.source_family,
    nodeSchemaCount: compiler.node_schema.length,
    edgeSchemaCount: compiler.edge_schema.length,
    provenanceRuleCount: compiler.provenance_rules.length,
    neighborhoodPackId: compiler.output_neighborhood_pack.pack_id,
    compilerVersion: compiler.compiler_version,
  };
}

export interface RouterArtifactManifestSummaryV1 {
  artifactId: string;
  packType: ColdStartPackTypeV1;
  trainingDataRefCount: number;
  replayGateRefCount: number;
  priorBaseArtifactId: string | null;
  priorBaseArtifactChecksum: string | null;
  checksum: string;
  runtimeVersion: string;
}

export function summarizeRouterArtifactManifestV1(
  manifest: RouterArtifactManifestV1,
): RouterArtifactManifestSummaryV1 {
  return {
    artifactId: manifest.artifact_id,
    packType: manifest.pack_type,
    trainingDataRefCount: manifest.training_data_refs.length,
    replayGateRefCount: manifest.replay_gate_refs.length,
    priorBaseArtifactId: manifest.prior_base_artifact_id ?? null,
    priorBaseArtifactChecksum: manifest.prior_base_artifact_checksum ?? null,
    checksum: manifest.artifact_checksum,
    runtimeVersion: manifest.compatible_runtime_version,
  };
}

export interface MigrationSnapshotSummaryV1 {
  snapshotId: string;
  livePackId: string;
  routerIdentity: string;
  proofRefCount: number;
  rollbackKey: string;
}

export function summarizeMigrationSnapshotV1(snapshot: MigrationSnapshotV1): MigrationSnapshotSummaryV1 {
  return {
    snapshotId: snapshot.snapshot_id,
    livePackId: snapshot.current_live_pack_id,
    routerIdentity: snapshot.current_router_identity,
    proofRefCount: snapshot.proof_refs.length,
    rollbackKey: snapshot.rollback_key,
  };
}

export interface ReplayRunRecordSummaryV1 {
  runId: string;
  scenarioId: string;
  comparedPolicies: ReplayPolicyComparisonV1;
  verdict: ColdStartReplayVerdictV1;
  metricCount: number;
  regressionFlagCount: number;
}

export function summarizeReplayRunRecordV1(record: ReplayRunRecordV1): ReplayRunRecordSummaryV1 {
  return {
    runId: record.run_id,
    scenarioId: record.scenario_id,
    comparedPolicies: record.compared_policies,
    verdict: record.verdict,
    metricCount: Object.keys(record.metrics).length,
    regressionFlagCount: Object.keys(record.regression_flags).length,
  };
}

export interface PromotionDecisionSummaryV1 {
  decisionId: string;
  candidateArtifactId: string;
  gateCount: number;
  failingGateCount: number;
  decision: ColdStartPromotionDecisionV1;
  proofBundleRef: string;
}

export function summarizePromotionDecisionV1(
  decision: PromotionDecisionV1,
): PromotionDecisionSummaryV1 {
  return {
    decisionId: decision.decision_id,
    candidateArtifactId: decision.candidate_artifact_id,
    gateCount: decision.gate_results.length,
    failingGateCount: decision.gate_results.filter((gate) => gate.verdict === "fail").length,
    decision: decision.decision,
    proofBundleRef: decision.proof_bundle_ref,
  };
}

export interface ProofBundleSummaryV1 {
  bundleId: string;
  artifactId: string;
  requiredFileCount: number;
  missingRequiredFiles: ColdStartProofBundleRequiredFileV1[];
  sourceRefCount: number;
  summaryGuardEnforced: boolean;
}

export function summarizeProofBundleV1(bundle: ProofBundleV1): ProofBundleSummaryV1 {
  const requiredFiles = COLD_START_PROOF_BUNDLE_REQUIRED_FILES_V1;
  const declaredRequired = toSet(bundle.required_files);
  const checksums = toSet(Object.keys(bundle.artifact_checksums));
  const missingRequiredFiles = requiredFiles.filter((file) => !declaredRequired.has(file) || !checksums.has(file));

  return {
    bundleId: bundle.bundle_id,
    artifactId: bundle.artifact_id,
    requiredFileCount: bundle.required_files.length,
    missingRequiredFiles,
    sourceRefCount: bundle.source_refs.length,
    summaryGuardEnforced: bundle.bounded_summary_guard.enforced,
  };
}

export function summarizeColdStartRouterContractHealthV1(params: {
  registryEntry?: DataRegistryEntryV1 | null;
  routeRow?: RouteDecisionRowV1 | null;
  teacherLabel?: TeacherLabelContractV1 | null;
  verifier?: VerifierContractV1 | null;
  compiler?: GraphCompilerContractV1 | null;
  routerManifest?: RouterArtifactManifestV1 | null;
  migrationSnapshot?: MigrationSnapshotV1 | null;
  replayRun?: ReplayRunRecordV1 | null;
  promotionDecision?: PromotionDecisionV1 | null;
  proofBundle?: ProofBundleV1 | null;
}): Record<string, unknown> {
  return {
    registryEntry: params.registryEntry ? summarizeDataRegistryEntryV1(params.registryEntry) : null,
    routeRow: params.routeRow ? summarizeRouteDecisionRowV1(params.routeRow) : null,
    teacherLabel: params.teacherLabel ? summarizeTeacherLabelContractV1(params.teacherLabel) : null,
    verifier: params.verifier ? summarizeVerifierContractV1(params.verifier) : null,
    compiler: params.compiler ? summarizeGraphCompilerContractV1(params.compiler) : null,
    routerManifest: params.routerManifest ? summarizeRouterArtifactManifestV1(params.routerManifest) : null,
    migrationSnapshot: params.migrationSnapshot ? summarizeMigrationSnapshotV1(params.migrationSnapshot) : null,
    replayRun: params.replayRun ? summarizeReplayRunRecordV1(params.replayRun) : null,
    promotionDecision: params.promotionDecision ? summarizePromotionDecisionV1(params.promotionDecision) : null,
    proofBundle: params.proofBundle ? summarizeProofBundleV1(params.proofBundle) : null,
  };
}
