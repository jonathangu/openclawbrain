import { describe, expect, it } from "vitest";
import { Value } from "@sinclair/typebox/value";
import type {
  DataRegistryEntryV1,
  GraphCompilerContractV1,
  MigrationSnapshotV1,
  ProofBundleV1,
  PromotionDecisionV1,
  ReplayRunRecordV1,
  RouterArtifactManifestV1,
  RouteDecisionRowV1,
  TeacherLabelContractV1,
  VerifierContractV1,
} from "../../src/brain-core/cold-start-router-contracts.js";
import {
  DataRegistryEntrySchemaV1,
  GraphCompilerContractSchemaV1,
  MigrationSnapshotSchemaV1,
  ProofBundleSchemaV1,
  PromotionDecisionSchemaV1,
  ReplayRunRecordSchemaV1,
  RouterArtifactManifestSchemaV1,
  RouteDecisionRowSchemaV1,
  TeacherLabelContractSchemaV1,
  VerifierContractSchemaV1,
  COLD_START_PROOF_BUNDLE_REQUIRED_FILES_V1,
  summarizeColdStartRouterContractHealthV1,
  summarizeDataRegistryEntryV1,
  summarizeGraphCompilerContractV1,
  summarizeMigrationSnapshotV1,
  summarizeProofBundleV1,
  summarizePromotionDecisionV1,
  summarizeReplayRunRecordV1,
  summarizeRouterArtifactManifestV1,
  summarizeRouteDecisionRowV1,
  summarizeTeacherLabelContractV1,
  summarizeVerifierContractV1,
  validateDataRegistryEntryV1,
  validateGraphCompilerContractV1,
  validateMigrationSnapshotV1,
  validateProofBundleV1,
  validatePromotionDecisionV1,
  validateReplayRunRecordV1,
  validateRouterArtifactManifestV1,
  validateRouteDecisionRowV1,
  validateTeacherLabelContractV1,
  validateVerifierContractV1,
} from "../../src/brain-core/cold-start-router-contracts.js";

const dataRegistryEntry: DataRegistryEntryV1 = {
  dataset_id: "dataset_hotpotqa_v1",
  source_family: "qa",
  upstream_url: "https://example.org/hotpotqa-v1",
  original_creator: "HotpotQA authors",
  license: "CC BY-SA 4.0",
  commercial_use_status: "allowed",
  redistribution_status: "allowed",
  pii_risk: "none",
  benchmark_split_status: "eval_only",
  approval_status: "approved_eval_only",
  reviewer: "operator",
  immutable_snapshot_ref: "snapshot:hotpotqa@sha256:abc123",
  exact_files: ["data/train.json", "data/dev.json"],
  file_hashes: {
    "data/train.json": "sha256:111",
    "data/dev.json": "sha256:222",
  },
  allowed_uses: ["route supervision", "teacher templates"],
  disallowed_uses: ["redistribution of private dumps"],
  notes: ["clean public QA source"],
  created_at: "2026-04-05T16:00:00Z",
  updated_at: "2026-04-05T16:05:00Z",
};

const routeDecisionRow: RouteDecisionRowV1 = {
  row_id: "row_001",
  dataset_id: dataRegistryEntry.dataset_id,
  query: "Find the user's last shipping-address correction and update the invoice draft",
  cursor_path: ["thread:billing", "mem:user_profile"],
  candidate_set: [
    { candidate_id: "mem:shipping_history", candidate_type: "memory_node", authority: "human" },
    { candidate_id: "doc:invoice_policy", candidate_type: "doc_chunk", authority: "operator_policy" },
    { candidate_id: "tool:gmail_search", candidate_type: "tool", freshness: "runtime" },
  ],
  teacher_action: {
    kind: "traverse",
    target_ids: ["mem:shipping_history"],
  },
  stop_label: "CONTINUE",
  evidence_spans: [
    { source_ref: "message_183", start: 21, end: 72 },
    { source_ref: "memory_entry_77", start: 0, end: 44 },
  ],
  hard_negatives: ["doc:invoice_policy"],
  outcome_gain: 0.84,
  provenance: {
    dataset: dataRegistryEntry.dataset_id,
    source_license: dataRegistryEntry.license,
    source_family: dataRegistryEntry.source_family,
    source_snapshot_ref: dataRegistryEntry.immutable_snapshot_ref,
    recorded_by: "operator",
    recorded_at: "2026-04-05T16:06:00Z",
    review_status: dataRegistryEntry.approval_status,
  },
  split_tag: "eval",
  created_at: "2026-04-05T16:06:30Z",
};

const teacherLabel: TeacherLabelContractV1 = {
  label_id: "label_001",
  dataset_id: routeDecisionRow.dataset_id,
  row_id: routeDecisionRow.row_id,
  best_next_node_ids: ["mem:shipping_history"],
  best_next_tool_name: null,
  stop_label: "CONTINUE",
  evidence_spans: routeDecisionRow.evidence_spans,
  hard_negatives: routeDecisionRow.hard_negatives,
  confidence: 0.93,
  rationale: "The correction memory is the highest-authority next hop.",
  created_at: "2026-04-05T16:07:00Z",
};

const verifier: VerifierContractV1 = {
  verifier_id: "verifier_001",
  label_id: teacherLabel.label_id,
  row_id: routeDecisionRow.row_id,
  candidate_set_digest: "sha256:candidate-set-001",
  checks: [
    { check_id: "evidence_reachable", status: "pass", summary: "evidence spans exist", evidence_refs: ["message_183"] },
    { check_id: "candidate_alignment", status: "pass", summary: "target ids are in the candidate set", evidence_refs: ["mem:shipping_history"] },
  ],
  passed: true,
  issues: [],
  explicit_correction_priority_honored: true,
  created_at: "2026-04-05T16:07:30Z",
};

const graphCompiler: GraphCompilerContractV1 = {
  compiler_id: "repo-compiler-v1",
  source_family: "repo",
  input_snapshot_ref: "snapshot:repo@sha256:def456",
  node_schema: [
    {
      node_kind: "file",
      required_fields: ["path", "content_hash"],
      optional_fields: ["language", "package"],
      notes: ["Files become neighborhood nodes"],
    },
  ],
  edge_schema: [
    {
      edge_kind: "imports",
      required_fields: ["source", "target"],
      optional_fields: ["symbol"],
      notes: ["Import edges support code graph traversal"],
    },
  ],
  provenance_rules: ["immutable snapshot", "explicit upstream URL", "license recorded"],
  output_neighborhood_pack: {
    pack_id: "repo-pack-001",
    artifact_ref: "artifact:repo-pack@sha256:fff111",
    graph_ref: "graph:repo@sha256:aaa222",
    radius_hops: 2,
    frontier_limit: 64,
  },
  compiler_version: "repo-graph-compiler@1.0.0",
};

const routerArtifactManifest: RouterArtifactManifestV1 = {
  schema_version: 1,
  artifact_id: "router-artifact-001",
  artifact_version: "0.1.0",
  pack_type: "mixed",
  base_model_ref: "base-router@sha256:base001",
  weights_ref: "weights@sha256:w001",
  calibration_ref: "calibration@sha256:c001",
  feature_normalizers_ref: "normalizers@sha256:n001",
  source_priors_ref: "priors@sha256:p001",
  safety_rules_ref: "safety@sha256:s001",
  compatible_runtime_version: "openclawbrain-runtime@0.4.44",
  training_data_refs: ["dataset_hotpotqa_v1", "dataset_repos_v1"],
  replay_gate_refs: ["replay:gate:001"],
  prior_base_artifact_id: "router-artifact-prior-000",
  prior_base_artifact_checksum: "sha256:router-artifact-prior-000",
  artifact_checksum: "sha256:router-artifact-001",
  created_at: "2026-04-05T16:08:00Z",
  router_identity: "router:gen1:mixed",
};

const migrationSnapshot: MigrationSnapshotV1 = {
  schema_version: 1,
  snapshot_id: "migration_001",
  current_live_pack_id: "pack_live_001",
  current_router_identity: "router:gen0:personalized",
  current_router_checksum: "sha256:router-live-000",
  graph_snapshot_ref: "graph:snapshot@sha256:g001",
  feedback_log_export_ref: "feedback:export@sha256:f001",
  user_delta_state_ref: "delta:user@sha256:d001",
  recent_overlay_ref: "overlay:recent@sha256:o001",
  rollback_key: "rollback:001",
  proof_refs: ["proof:before", "proof:after"],
  timestamp: "2026-04-05T16:09:00Z",
};

const replayRunRecord: ReplayRunRecordV1 = {
  schema_version: 1,
  run_id: "replay_001",
  scenario_id: "upgrade-preserve-user-brain",
  compared_policies: {
    old_live: "router:gen0:personalized",
    base_only: "router:gen1:base",
    mixed: "router:gen1:mixed",
  },
  metrics: {
    top1_accuracy: 0.91,
    stop_f1: 0.88,
    latency_ms: 42,
  },
  regression_flags: {
    explicit_correction_regression: false,
    budget_regression: false,
  },
  verdict: "pass",
  proof_refs: ["proof:replay:001"],
  timestamp: "2026-04-05T16:10:00Z",
};

const promotionDecision: PromotionDecisionV1 = {
  schema_version: 1,
  decision_id: "promotion_001",
  candidate_artifact_id: routerArtifactManifest.artifact_id,
  gate_results: [
    {
      gate_id: "replay_gate",
      verdict: "pass",
      summary: "mixed policy is non-regressive on user replay",
      proof_refs: ["proof:replay:001"],
    },
  ],
  decision: "promote",
  reviewer: "operator",
  approver: "jonathan",
  rollback_binding: migrationSnapshot.rollback_key,
  proof_bundle_ref: "proof-bundle:001",
  timestamp: "2026-04-05T16:11:00Z",
};

const proofBundle: ProofBundleV1 = {
  schema_version: 1,
  bundle_id: "proof_bundle_001",
  artifact_id: routerArtifactManifest.artifact_id,
  owner: "operator",
  environment: "openclawbrain-lane-b2",
  source_refs: ["src:brain-core/cold-start-router-contracts.ts", "test/brain-core/cold-start-router-contracts.test.ts"],
  shipped_state_label: "fresh-install",
  target_state_label: "upgrade-safe-mixed-router",
  required_files: [...COLD_START_PROOF_BUNDLE_REQUIRED_FILES_V1],
  artifact_checksums: {
    "summary.md": "sha256:sum001",
    "status.json": "sha256:stat001",
    "surface-map.json": "sha256:surf001",
    "proposal-report.json": "sha256:prop001",
    "verdict.json": "sha256:verd001",
  },
  bounded_summary_guard: {
    max_chars: 4000,
    max_lines: 80,
    enforced: true,
  },
  proof_refs: ["proof:bundle:001"],
  timestamp: "2026-04-05T16:12:00Z",
};

const invalidRouteRow: RouteDecisionRowV1 = {
  ...routeDecisionRow,
  teacher_action: { kind: "traverse", target_ids: ["mem:missing_target"] },
};

const invalidDataRegistryEntry: DataRegistryEntryV1 = {
  ...dataRegistryEntry,
  file_hashes: {
    ...dataRegistryEntry.file_hashes,
    "data/extra.json": "sha256:333",
  },
};

const invalidProofBundle: ProofBundleV1 = {
  ...proofBundle,
  required_files: ["summary.md", "status.json", "surface-map.json", "proposal-report.json", "summary.md"],
};

describe("cold-start router contracts", () => {
  it("accepts the canonical route-decision, registry, teacher, verifier, compiler, manifest, migration, replay, promotion, and proof-bundle surfaces", () => {
    expect(Value.Check(DataRegistryEntrySchemaV1, dataRegistryEntry)).toBe(true);
    expect(Value.Check(RouteDecisionRowSchemaV1, routeDecisionRow)).toBe(true);
    expect(Value.Check(TeacherLabelContractSchemaV1, teacherLabel)).toBe(true);
    expect(Value.Check(VerifierContractSchemaV1, verifier)).toBe(true);
    expect(Value.Check(GraphCompilerContractSchemaV1, graphCompiler)).toBe(true);
    expect(Value.Check(RouterArtifactManifestSchemaV1, routerArtifactManifest)).toBe(true);
    expect(Value.Check(MigrationSnapshotSchemaV1, migrationSnapshot)).toBe(true);
    expect(Value.Check(ReplayRunRecordSchemaV1, replayRunRecord)).toBe(true);
    expect(Value.Check(PromotionDecisionSchemaV1, promotionDecision)).toBe(true);
    expect(Value.Check(ProofBundleSchemaV1, proofBundle)).toBe(true);

    expect(validateDataRegistryEntryV1(dataRegistryEntry)).toMatchObject({ valid: true });
    expect(validateRouteDecisionRowV1(routeDecisionRow)).toMatchObject({ valid: true });
    expect(validateTeacherLabelContractV1(teacherLabel)).toMatchObject({ valid: true });
    expect(validateVerifierContractV1(verifier)).toMatchObject({ valid: true });
    expect(validateGraphCompilerContractV1(graphCompiler)).toMatchObject({ valid: true });
    expect(validateRouterArtifactManifestV1(routerArtifactManifest)).toMatchObject({ valid: true });
    expect(validateMigrationSnapshotV1(migrationSnapshot)).toMatchObject({ valid: true });
    expect(validateReplayRunRecordV1(replayRunRecord)).toMatchObject({ valid: true });
    expect(validatePromotionDecisionV1(promotionDecision)).toMatchObject({ valid: true });
    expect(validateProofBundleV1(proofBundle)).toMatchObject({ valid: true });

    expect(summarizeDataRegistryEntryV1(dataRegistryEntry)).toMatchObject({
      datasetId: "dataset_hotpotqa_v1",
      sourceFamily: "qa",
      approvalStatus: "approved_eval_only",
      fileCount: 2,
      fileHashCount: 2,
    });
    expect(summarizeRouteDecisionRowV1(routeDecisionRow)).toMatchObject({
      rowId: "row_001",
      candidateCount: 3,
      teacherActionKind: "traverse",
      stopLabel: "CONTINUE",
      approvalStatus: "approved_eval_only",
    });
    expect(summarizeTeacherLabelContractV1(teacherLabel)).toMatchObject({
      labelId: "label_001",
      bestNextNodeCount: 1,
      hasBestNextTool: false,
      confidence: 0.93,
    });
    expect(summarizeVerifierContractV1(verifier)).toMatchObject({
      verifierId: "verifier_001",
      passed: true,
      checkCount: 2,
      failingCheckCount: 0,
      explicitCorrectionPriorityHonored: true,
    });
    expect(summarizeGraphCompilerContractV1(graphCompiler)).toMatchObject({
      compilerId: "repo-compiler-v1",
      sourceFamily: "repo",
      neighborhoodPackId: "repo-pack-001",
      compilerVersion: "repo-graph-compiler@1.0.0",
    });
    expect(summarizeRouterArtifactManifestV1(routerArtifactManifest)).toMatchObject({
      artifactId: "router-artifact-001",
      packType: "mixed",
      trainingDataRefCount: 2,
      replayGateRefCount: 1,
      priorBaseArtifactId: "router-artifact-prior-000",
      priorBaseArtifactChecksum: "sha256:router-artifact-prior-000",
      warmStartApplied: false,
      warmStartFromArtifactId: null,
      warmStartFromArtifactChecksum: null,
      warmStartSummary: null,
      checksum: "sha256:router-artifact-001",
    });
    expect(summarizeMigrationSnapshotV1(migrationSnapshot)).toMatchObject({
      snapshotId: "migration_001",
      livePackId: "pack_live_001",
      routerIdentity: "router:gen0:personalized",
      proofRefCount: 2,
    });
    expect(summarizeReplayRunRecordV1(replayRunRecord)).toMatchObject({
      runId: "replay_001",
      scenarioId: "upgrade-preserve-user-brain",
      verdict: "pass",
      metricCount: 3,
      regressionFlagCount: 2,
    });
    expect(summarizePromotionDecisionV1(promotionDecision)).toMatchObject({
      decisionId: "promotion_001",
      candidateArtifactId: routerArtifactManifest.artifact_id,
      gateCount: 1,
      failingGateCount: 0,
      decision: "promote",
    });
    expect(summarizeProofBundleV1(proofBundle)).toMatchObject({
      bundleId: "proof_bundle_001",
      artifactId: routerArtifactManifest.artifact_id,
      requiredFileCount: 5,
      missingRequiredFiles: [],
      sourceRefCount: 2,
      summaryGuardEnforced: true,
    });

    expect(
      summarizeColdStartRouterContractHealthV1({
        registryEntry: dataRegistryEntry,
        routeRow: routeDecisionRow,
        teacherLabel,
        verifier,
        compiler: graphCompiler,
        routerManifest: routerArtifactManifest,
        migrationSnapshot,
        replayRun: replayRunRecord,
        promotionDecision,
        proofBundle,
      }),
    ).toMatchObject({
      routeRow: {
        candidateCount: 3,
        teacherActionKind: "traverse",
      },
      routerManifest: {
        packType: "mixed",
      },
      proofBundle: {
        missingRequiredFiles: [],
      },
    });
  });

  it("rejects a router manifest when only one prior-lineage field is present", () => {
    const missingChecksumManifest = { ...routerArtifactManifest } as Record<string, unknown>;
    delete missingChecksumManifest.prior_base_artifact_checksum;

    const validation = validateRouterArtifactManifestV1(missingChecksumManifest);
    expect(validation.valid).toBe(false);
    expect(validation.issues.join(" ")).toContain("prior_base_artifact_id and prior_base_artifact_checksum must be provided together");
  });

  it("rejects a router manifest when warm-start metadata is incomplete", () => {
    const invalidWarmStartManifest = {
      ...routerArtifactManifest,
      warm_start_applied: true,
      warm_start_from_artifact_id: "router-artifact-prior-000",
      warm_start_from_artifact_checksum: "sha256:router-artifact-prior-000",
    } as Record<string, unknown>;

    const validation = validateRouterArtifactManifestV1(invalidWarmStartManifest);
    expect(validation.valid).toBe(false);
    expect(validation.issues.join(" ")).toContain("warm_start_applied=true requires warm_start_from_artifact_id, warm_start_from_artifact_checksum, and warm_start_summary");
  });

  it("fails semantic validation when a row routes to a missing target, a registry row carries an extra hash, and a proof bundle omits a required file", () => {
    const routeValidation = validateRouteDecisionRowV1(invalidRouteRow);
    expect(routeValidation.valid).toBe(false);
    expect(routeValidation.issues.join("\n")).toContain("target_ids not present in candidate_set");

    const registryValidation = validateDataRegistryEntryV1(invalidDataRegistryEntry);
    expect(registryValidation.valid).toBe(false);
    expect(registryValidation.issues.join("\n")).toContain("extra entries not listed in exact_files");

    const proofValidation = validateProofBundleV1(invalidProofBundle);
    expect(proofValidation.valid).toBe(false);
    expect(proofValidation.issues.join("\n")).toContain("required_files missing");
  });

  it("blocks verifier contracts that declare a pass without a failing check", () => {
    const blocked: VerifierContractV1 = {
      ...verifier,
      passed: false,
      checks: verifier.checks.map((check) => ({ ...check, status: "warn" })),
    };

    const validation = validateVerifierContractV1(blocked);
    expect(validation.valid).toBe(false);
    expect(validation.issues.join("\n")).toContain("no failing check was recorded");
  });
});
