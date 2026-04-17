import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import type {
  RouteDecisionRowV1,
  RouterArtifactManifestSummaryV1,
  RouterArtifactManifestV1,
} from "./cold-start-router-contracts.ts";
import {
  summarizeRouterArtifactManifestV1,
  validateRouterArtifactManifestV1,
} from "./cold-start-router-contracts.ts";
import {
  COLD_START_ROUTER_ARTIFACT_LAYOUT_V1,
  COLD_START_ROUTER_BASE_MODEL_CONTRACT_V1,
  COLD_START_ROUTER_CALIBRATION_CONTRACT_V1,
  COLD_START_ROUTER_FEATURE_NORMALIZERS_CONTRACT_V1,
  COLD_START_ROUTER_SAFETY_RULES_CONTRACT_V1,
  COLD_START_ROUTER_SOURCE_PRIORS_CONTRACT_V1,
  COLD_START_ROUTER_WEIGHTS_CONTRACT_V1,
  scoreColdStartRouteRowV1,
  type ColdStartRouterBaseModelV1,
  type ColdStartRouterCalibrationV1,
  type ColdStartRouterFeatureNormalizersV1,
  type ColdStartRouterModelV1,
  type ColdStartRouterSafetyRulesV1,
  type ColdStartRouterSourcePriorsV1,
} from "./cold-start-router-trainer.ts";
import { COLD_START_ROUTER_LIVE_POLICY_INITIALIZER_CONTRACT_V1 } from "./graph.js";
import {
  normalizePolicySupervisionRowsForReplayV1,
  validatePolicySupervisionRowV1,
  type HardNegativeClass,
  type NormalizedPolicySupervisionReplayExpectationV1,
  type PolicySupervisionRowType,
  type PolicySupervisionRowV1,
} from "./policy-supervision-rows.ts";

export interface ColdStartRouterReplayGateLoadIssueV1 {
  code: string;
  detail: string;
}

export interface ColdStartRouterReplayGatePolicyExpectationResultV1 {
  policyRowId: string;
  routeRowId: string | null;
  rowType: PolicySupervisionRowType;
  focusLane: string | null;
  weight: number;
  hardNegativeClass: HardNegativeClass | null;
  oracleBestMode: PolicySupervisionRowV1["oracle_best_mode"];
  expectedActivated: boolean | null;
  actualActivated: boolean;
  expectedAbstained: boolean | null;
  actualAbstained: boolean;
  expectedStopLocal: boolean | null;
  actualStopLocal: boolean;
  passed: boolean;
  issues: string[];
}

export interface ColdStartRouterReplayGateLaneSummaryV1 {
  lane: string;
  policyExpectationCount: number;
  passedPolicyExpectationCount: number;
  failedPolicyExpectationCount: number;
  totalWeight: number;
  activationExpectationCount: number;
  activationMatchCount: number;
  abstainExpectationCount: number;
  abstainMatchCount: number;
  stopLocalExpectationCount: number;
  stopLocalMatchCount: number;
}

export interface ColdStartRouterReplayGateRowResultV1 {
  rowId: string;
  teacherActionKind: RouteDecisionRowV1["teacher_action"]["kind"];
  expectedActivated: boolean | null;
  actualActivated: boolean | null;
  actualAbstained: boolean | null;
  expectedTopCandidateId: string | null;
  actualTopCandidateId: string | null;
  expectedStopLabel: RouteDecisionRowV1["stop_label"];
  actualStopLabel: RouteDecisionRowV1["stop_label"];
  actualStopReason: string | null;
  actualStopLocal: boolean | null;
  decisionConfidence: number | null;
  activationProbability: number | null;
  abstentionProbability: number | null;
  predictedUtility: number | null;
  predictedRegretOfAbstaining: number | null;
  stopLocalProbability: number | null;
  topCandidateProbability: number;
  stopProbability: number;
  activationThreshold: number | null;
  abstentionThreshold: number | null;
  expectedUtilityThreshold: number | null;
  stopLocalThreshold: number | null;
  gateEvaluated: boolean;
  routeRowDiagnosticPassed: boolean;
  routeRowDiagnosticIssues: string[];
  policyExpectationCount: number;
  policyExpectationPassCount: number;
  policyExpectationResults: ColdStartRouterReplayGatePolicyExpectationResultV1[];
  passed: boolean;
  issues: string[];
}

export interface ColdStartRouterReplayGateVerdictV1 {
  artifactDir: string;
  manifestSummary: RouterArtifactManifestSummaryV1 | null;
  passed: boolean;
  verdict: "pass" | "warn" | "fail";
  summary: string;
  evaluatedRowCount: number;
  passedRowCount: number;
  failedRowCount: number;
  skippedRowCount: number;
  policyExpectationCount: number;
  passedPolicyExpectationCount: number;
  failedPolicyExpectationCount: number;
  loadIssues: ColdStartRouterReplayGateLoadIssueV1[];
  laneSummaries: ColdStartRouterReplayGateLaneSummaryV1[];
  rowResults: ColdStartRouterReplayGateRowResultV1[];
}

function expectedActivatedForReplayRow(row: RouteDecisionRowV1): boolean | null {
  switch (row.teacher_action.kind) {
    case "traverse":
      return row.stop_label === "CONTINUE";
    case "tool":
    case "stop":
      return false;
    default:
      return null;
  }
}

function findTraverseProbability(params: {
  candidateId: string | null;
  actionDistribution: ReturnType<typeof scoreColdStartRouteRowV1>["policyDistribution"]["actions"];
}): number {
  if (!params.candidateId) {
    return 0;
  }
  return params.actionDistribution.find((entry) => (
    entry.action.type === "traverse"
      && entry.action.candidate?.candidate_id === params.candidateId
  ))?.probability ?? 0;
}

const REPLAY_GATE_PRIORITY_LANES = ["felt_resume_25", "must_not_fire_100"] as const;

function booleanStateLabel(value: boolean): "on" | "off" {
  return value ? "on" : "off";
}

function buildPolicyExpectationIssues(params: {
  expectation: NormalizedPolicySupervisionReplayExpectationV1;
  actualActivated: boolean;
  actualAbstained: boolean;
  actualStopLocal: boolean;
}): string[] {
  const issues: string[] = [];

  if (params.expectation.expectedActivated !== null && params.actualActivated !== params.expectation.expectedActivated) {
    issues.push(`activation ${booleanStateLabel(params.actualActivated)} != expected ${booleanStateLabel(params.expectation.expectedActivated)}`);
  }
  if (params.expectation.expectedAbstained !== null && params.actualAbstained !== params.expectation.expectedAbstained) {
    issues.push(`abstain ${booleanStateLabel(params.actualAbstained)} != expected ${booleanStateLabel(params.expectation.expectedAbstained)}`);
  }
  if (params.expectation.expectedStopLocal !== null && params.actualStopLocal !== params.expectation.expectedStopLocal) {
    issues.push(`stop_local ${booleanStateLabel(params.actualStopLocal)} != expected ${booleanStateLabel(params.expectation.expectedStopLocal)}`);
  }

  return issues;
}

function summarizePolicyExpectationLanes(
  expectationResults: readonly ColdStartRouterReplayGatePolicyExpectationResultV1[],
): ColdStartRouterReplayGateLaneSummaryV1[] {
  const laneSummaries = new Map<string, ColdStartRouterReplayGateLaneSummaryV1>();

  for (const result of expectationResults) {
    if (!result.focusLane) {
      continue;
    }
    const laneSummary = laneSummaries.get(result.focusLane) ?? {
      lane: result.focusLane,
      policyExpectationCount: 0,
      passedPolicyExpectationCount: 0,
      failedPolicyExpectationCount: 0,
      totalWeight: 0,
      activationExpectationCount: 0,
      activationMatchCount: 0,
      abstainExpectationCount: 0,
      abstainMatchCount: 0,
      stopLocalExpectationCount: 0,
      stopLocalMatchCount: 0,
    };
    laneSummary.policyExpectationCount += 1;
    laneSummary.totalWeight = Number((laneSummary.totalWeight + result.weight).toFixed(6));
    if (result.passed) {
      laneSummary.passedPolicyExpectationCount += 1;
    } else {
      laneSummary.failedPolicyExpectationCount += 1;
    }
    if (result.expectedActivated !== null) {
      laneSummary.activationExpectationCount += 1;
      if (result.actualActivated === result.expectedActivated) {
        laneSummary.activationMatchCount += 1;
      }
    }
    if (result.expectedAbstained !== null) {
      laneSummary.abstainExpectationCount += 1;
      if (result.actualAbstained === result.expectedAbstained) {
        laneSummary.abstainMatchCount += 1;
      }
    }
    if (result.expectedStopLocal !== null) {
      laneSummary.stopLocalExpectationCount += 1;
      if (result.actualStopLocal === result.expectedStopLocal) {
        laneSummary.stopLocalMatchCount += 1;
      }
    }
    laneSummaries.set(result.focusLane, laneSummary);
  }

  const priorityLaneOrder = new Map(REPLAY_GATE_PRIORITY_LANES.map((lane, index) => [lane, index] as const));
  return [...laneSummaries.values()].sort((left, right) => {
    const leftPriority = priorityLaneOrder.get(left.lane);
    const rightPriority = priorityLaneOrder.get(right.lane);
    if (leftPriority !== undefined || rightPriority !== undefined) {
      if (leftPriority === undefined) {
        return 1;
      }
      if (rightPriority === undefined) {
        return -1;
      }
      return leftPriority - rightPriority;
    }
    return left.lane.localeCompare(right.lane);
  });
}

export interface ColdStartRouterLoadedArtifactV1 {
  artifactDir: string;
  manifestPath: string;
  manifest: RouterArtifactManifestV1;
  manifestSummary: RouterArtifactManifestSummaryV1;
  baseModelPath: string;
  baseModel: ColdStartRouterBaseModelV1;
  weightsPath: string;
  model: ColdStartRouterModelV1;
  calibrationPath: string;
  calibration: ColdStartRouterCalibrationV1;
  featureNormalizersPath: string;
  featureNormalizers: ColdStartRouterFeatureNormalizersV1;
  sourcePriorsPath: string;
  sourcePriors: ColdStartRouterSourcePriorsV1;
  safetyRulesPath: string;
  safetyRules: ColdStartRouterSafetyRulesV1;
}

function sha256Text(value: string): string {
  return `sha256:${createHash("sha256").update(value, "utf8").digest("hex")}`;
}

function buildLogicalBaseModelDigest(baseModel: ColdStartRouterBaseModelV1): string {
  return sha256Text(JSON.stringify({
    contract: baseModel.contract,
    artifactId: baseModel.artifactId,
    artifactVersion: baseModel.artifactVersion,
    packType: baseModel.packType,
    compatibleRuntimeVersion: baseModel.compatibleRuntimeVersion,
    routerIdentity: baseModel.routerIdentity,
    createdAt: baseModel.createdAt,
    training: baseModel.training,
  }));
}

function readJsonFile<T>(filePath: string): { value: T; text: string; digest: string } {
  const text = readFileSync(filePath, "utf8");
  return {
    value: JSON.parse(text) as T,
    text,
    digest: sha256Text(text),
  };
}

function parseRefDigest(ref: string): { fileName: string; digest: string } | null {
  const [fileName, digest] = ref.split("#", 2);
  if (!fileName || !digest) {
    return null;
  }
  return { fileName, digest };
}

function verifyRefDigest(params: {
  ref: string;
  fileName: string;
  digest: string;
}): string | null {
  const parsed = parseRefDigest(params.ref);
  if (!parsed) {
    return `expected ${params.fileName} reference with digest but got ${params.ref}`;
  }
  if (parsed.fileName !== params.fileName) {
    return `expected ${params.fileName} reference but got ${parsed.fileName}`;
  }
  if (parsed.digest !== params.digest) {
    return `digest mismatch for ${params.fileName}: manifest=${parsed.digest} actual=${params.digest}`;
  }
  return null;
}

function recordIssue(issues: ColdStartRouterReplayGateLoadIssueV1[], code: string, detail: string): void {
  issues.push({ code, detail });
}

export function loadColdStartRouterArtifactV1(artifactDir: string): {
  artifact: ColdStartRouterLoadedArtifactV1 | null;
  issues: ColdStartRouterReplayGateLoadIssueV1[];
} {
  const issues: ColdStartRouterReplayGateLoadIssueV1[] = [];
  const manifestPath = path.join(artifactDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.manifest);
  const weightsPath = path.join(artifactDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.weights);
  const calibrationPath = path.join(artifactDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.calibration);
  const featureNormalizersPath = path.join(artifactDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.featureNormalizers);
  const sourcePriorsPath = path.join(artifactDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.sourcePriors);
  const safetyRulesPath = path.join(artifactDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.safetyRules);
  const baseModelPath = path.join(artifactDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.baseModel);

  for (const [fileName, filePath] of [
    [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.manifest, manifestPath],
    [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.weights, weightsPath],
    [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.calibration, calibrationPath],
    [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.featureNormalizers, featureNormalizersPath],
    [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.sourcePriors, sourcePriorsPath],
    [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.safetyRules, safetyRulesPath],
    [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.baseModel, baseModelPath],
  ] as const) {
    if (!existsSync(filePath)) {
      recordIssue(issues, "missing_file", `${fileName} is missing at ${filePath}`);
    }
  }

  if (issues.length > 0) {
    return { artifact: null, issues };
  }

  const manifestRead = readJsonFile<RouterArtifactManifestV1>(manifestPath);
  const manifestValidation = validateRouterArtifactManifestV1(manifestRead.value);
  if (!manifestValidation.valid) {
    recordIssue(issues, "invalid_manifest", manifestValidation.issues.join("; "));
  }

  const weightsRead = readJsonFile<ColdStartRouterModelV1>(weightsPath);
  const calibrationRead = readJsonFile<ColdStartRouterCalibrationV1>(calibrationPath);
  const featureNormalizersRead = readJsonFile<ColdStartRouterFeatureNormalizersV1>(featureNormalizersPath);
  const sourcePriorsRead = readJsonFile<ColdStartRouterSourcePriorsV1>(sourcePriorsPath);
  const safetyRulesRead = readJsonFile<ColdStartRouterSafetyRulesV1>(safetyRulesPath);
  const baseModelRead = readJsonFile<ColdStartRouterBaseModelV1>(baseModelPath);

  const manifest = manifestRead.value;
  const model = weightsRead.value;
  const calibration = calibrationRead.value;
  const featureNormalizers = featureNormalizersRead.value;
  const sourcePriors = sourcePriorsRead.value;
  const safetyRules = safetyRulesRead.value;
  const baseModel = baseModelRead.value;

  if (model.contract !== COLD_START_ROUTER_WEIGHTS_CONTRACT_V1) {
    recordIssue(issues, "invalid_weights_contract", `weights.json contract ${model.contract} != ${COLD_START_ROUTER_WEIGHTS_CONTRACT_V1}`);
  }
  if (calibration.contract !== COLD_START_ROUTER_CALIBRATION_CONTRACT_V1) {
    recordIssue(issues, "invalid_calibration_contract", `calibration.json contract ${calibration.contract} != ${COLD_START_ROUTER_CALIBRATION_CONTRACT_V1}`);
  }
  if (featureNormalizers.contract !== COLD_START_ROUTER_FEATURE_NORMALIZERS_CONTRACT_V1) {
    recordIssue(issues, "invalid_feature_normalizers_contract", `feature-normalizers.json contract ${featureNormalizers.contract} != ${COLD_START_ROUTER_FEATURE_NORMALIZERS_CONTRACT_V1}`);
  }
  if (sourcePriors.contract !== COLD_START_ROUTER_SOURCE_PRIORS_CONTRACT_V1) {
    recordIssue(issues, "invalid_source_priors_contract", `source-priors.json contract ${sourcePriors.contract} != ${COLD_START_ROUTER_SOURCE_PRIORS_CONTRACT_V1}`);
  }
  if (safetyRules.contract !== COLD_START_ROUTER_SAFETY_RULES_CONTRACT_V1) {
    recordIssue(issues, "invalid_safety_rules_contract", `safety-rules.json contract ${safetyRules.contract} != ${COLD_START_ROUTER_SAFETY_RULES_CONTRACT_V1}`);
  }
  if (baseModel.contract !== COLD_START_ROUTER_BASE_MODEL_CONTRACT_V1) {
    recordIssue(issues, "invalid_base_model_contract", `base-model.json contract ${baseModel.contract} != ${COLD_START_ROUTER_BASE_MODEL_CONTRACT_V1}`);
  }
  if (model.livePolicyInitializer?.contract !== COLD_START_ROUTER_LIVE_POLICY_INITIALIZER_CONTRACT_V1) {
    recordIssue(issues, "invalid_live_policy_initializer_contract", `weights.livePolicyInitializer.contract ${model.livePolicyInitializer?.contract ?? "missing"} != ${COLD_START_ROUTER_LIVE_POLICY_INITIALIZER_CONTRACT_V1}`);
  }
  if (!Array.isArray((model as { toolActionPriors?: unknown }).toolActionPriors)) {
    recordIssue(issues, "missing_tool_action_priors", "weights.toolActionPriors must be an array");
  }
  if (!Array.isArray((model as { toolActionSets?: unknown }).toolActionSets)) {
    recordIssue(issues, "missing_tool_action_sets", "weights.toolActionSets must be an array");
  }

  const expectedBaseModelDigest = buildLogicalBaseModelDigest(baseModel);

  const manifestRefChecks = [
    verifyRefDigest({ ref: manifest.weights_ref, fileName: COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.weights, digest: weightsRead.digest }),
    verifyRefDigest({ ref: manifest.calibration_ref, fileName: COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.calibration, digest: calibrationRead.digest }),
    verifyRefDigest({ ref: manifest.feature_normalizers_ref, fileName: COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.featureNormalizers, digest: featureNormalizersRead.digest }),
    verifyRefDigest({ ref: manifest.source_priors_ref, fileName: COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.sourcePriors, digest: sourcePriorsRead.digest }),
    verifyRefDigest({ ref: manifest.safety_rules_ref, fileName: COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.safetyRules, digest: safetyRulesRead.digest }),
    verifyRefDigest({ ref: manifest.base_model_ref, fileName: COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.baseModel, digest: expectedBaseModelDigest }),
  ];

  for (const issue of manifestRefChecks.filter((value): value is string => value !== null)) {
    recordIssue(issues, "manifest_ref_mismatch", issue);
  }

  const recomputedChecksum = sha256Text(JSON.stringify({
    schema_version: manifest.schema_version,
    artifact_id: manifest.artifact_id,
    artifact_version: manifest.artifact_version,
    pack_type: manifest.pack_type,
    base_model_ref: manifest.base_model_ref,
    weights_ref: manifest.weights_ref,
    calibration_ref: manifest.calibration_ref,
    feature_normalizers_ref: manifest.feature_normalizers_ref,
    source_priors_ref: manifest.source_priors_ref,
    safety_rules_ref: manifest.safety_rules_ref,
    compatible_runtime_version: manifest.compatible_runtime_version,
    training_data_refs: manifest.training_data_refs,
    replay_gate_refs: manifest.replay_gate_refs,
    ...(manifest.prior_base_artifact_id !== undefined ? { prior_base_artifact_id: manifest.prior_base_artifact_id } : {}),
    ...(manifest.prior_base_artifact_checksum !== undefined ? { prior_base_artifact_checksum: manifest.prior_base_artifact_checksum } : {}),
    warm_start_applied: manifest.warm_start_applied,
    ...(manifest.warm_start_from_artifact_id !== undefined ? { warm_start_from_artifact_id: manifest.warm_start_from_artifact_id } : {}),
    ...(manifest.warm_start_from_artifact_checksum !== undefined ? { warm_start_from_artifact_checksum: manifest.warm_start_from_artifact_checksum } : {}),
    ...(manifest.warm_start_summary !== undefined ? { warm_start_summary: manifest.warm_start_summary } : {}),
    created_at: manifest.created_at,
    router_identity: manifest.router_identity,
    trainingSummary: model.training,
    modelRefDigests: {
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.baseModel]: expectedBaseModelDigest,
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.weights]: weightsRead.digest,
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.calibration]: calibrationRead.digest,
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.featureNormalizers]: featureNormalizersRead.digest,
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.sourcePriors]: sourcePriorsRead.digest,
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.safetyRules]: safetyRulesRead.digest,
    },
  }));
  if (recomputedChecksum !== manifest.artifact_checksum) {
    recordIssue(issues, "manifest_checksum_mismatch", `artifact_checksum mismatch: manifest=${manifest.artifact_checksum} actual=${recomputedChecksum}`);
  }

  if (model.artifactId !== manifest.artifact_id) {
    recordIssue(issues, "artifact_id_mismatch", `weights artifactId ${model.artifactId} != manifest artifact_id ${manifest.artifact_id}`);
  }
  if (model.artifactVersion !== manifest.artifact_version) {
    recordIssue(issues, "artifact_version_mismatch", `weights artifactVersion ${model.artifactVersion} != manifest artifact_version ${manifest.artifact_version}`);
  }
  if (model.packType !== manifest.pack_type) {
    recordIssue(issues, "pack_type_mismatch", `weights packType ${model.packType} != manifest pack_type ${manifest.pack_type}`);
  }
  if (model.compatibleRuntimeVersion !== manifest.compatible_runtime_version) {
    recordIssue(issues, "runtime_version_mismatch", `weights runtime ${model.compatibleRuntimeVersion} != manifest compatible_runtime_version ${manifest.compatible_runtime_version}`);
  }
  if (model.routerIdentity !== (manifest.router_identity ?? null)) {
    recordIssue(issues, "router_identity_mismatch", `weights routerIdentity ${model.routerIdentity ?? "null"} != manifest router_identity ${manifest.router_identity ?? "null"}`);
  }
  if (model.training.totalRows !== baseModel.training.totalRows || model.training.usedRows !== baseModel.training.usedRows) {
    recordIssue(issues, "training_summary_mismatch", "base model training summary does not match weights training summary");
  }
  if (baseModel.weightsRef !== manifest.weights_ref ||
      baseModel.calibrationRef !== manifest.calibration_ref ||
      baseModel.featureNormalizersRef !== manifest.feature_normalizers_ref ||
      baseModel.sourcePriorsRef !== manifest.source_priors_ref ||
      baseModel.safetyRulesRef !== manifest.safety_rules_ref) {
    recordIssue(issues, "base_model_ref_mismatch", "base-model refs do not match manifest refs");
  }

  if (issues.length > 0) {
    return { artifact: null, issues };
  }

  return {
    artifact: {
      artifactDir,
      manifestPath,
      manifest,
      manifestSummary: summarizeRouterArtifactManifestV1(manifest),
      baseModelPath,
      baseModel,
      weightsPath,
      model,
      calibrationPath,
      calibration,
      featureNormalizersPath,
      featureNormalizers,
      sourcePriorsPath,
      sourcePriors,
      safetyRulesPath,
      safetyRules,
    },
    issues,
  };
}

export function replayColdStartRouterArtifactV1(params: {
  artifactDir: string;
  routeRows: RouteDecisionRowV1[];
  policySupervisionRows?: PolicySupervisionRowV1[];
}): ColdStartRouterReplayGateVerdictV1 {
  const loaded = loadColdStartRouterArtifactV1(params.artifactDir);
  if (!loaded.artifact) {
    return {
      artifactDir: params.artifactDir,
      manifestSummary: null,
      passed: false,
      verdict: "fail",
      summary: `load failed: ${loaded.issues.map((issue) => issue.detail).join("; ")}`,
      evaluatedRowCount: 0,
      passedRowCount: 0,
      failedRowCount: 0,
      skippedRowCount: 0,
      policyExpectationCount: 0,
      passedPolicyExpectationCount: 0,
      failedPolicyExpectationCount: 0,
      loadIssues: loaded.issues,
      laneSummaries: [],
      rowResults: [],
    };
  }

  const rowResults: ColdStartRouterReplayGateRowResultV1[] = [];
  const loadIssues = loaded.issues;
  const validPolicyRows: PolicySupervisionRowV1[] = [];

  for (const policyRow of params.policySupervisionRows ?? []) {
    const validation = validatePolicySupervisionRowV1(policyRow);
    if (!validation.valid) {
      recordIssue(loadIssues, "invalid_policy_supervision_row", `${policyRow.row_id}: ${validation.issues.join("; ")}`);
      continue;
    }
    validPolicyRows.push(policyRow);
  }

  const routeRowIds = new Set(params.routeRows.map((row) => row.row_id));
  const policyExpectationsByRouteRowId = new Map<string, NormalizedPolicySupervisionReplayExpectationV1[]>();
  for (const expectation of normalizePolicySupervisionRowsForReplayV1(validPolicyRows)) {
    if (!expectation.routeRowId) {
      recordIssue(
        loadIssues,
        "invalid_policy_supervision_row",
        `${expectation.policyRowId}: trace_slice.route_row_id is required for replay gating`,
      );
      continue;
    }
    if (!routeRowIds.has(expectation.routeRowId)) {
      recordIssue(
        loadIssues,
        "missing_policy_route_row",
        `${expectation.policyRowId}: route row ${expectation.routeRowId} is not present in replay input`,
      );
      continue;
    }
    const expectationsForRouteRow = policyExpectationsByRouteRowId.get(expectation.routeRowId) ?? [];
    expectationsForRouteRow.push(expectation);
    policyExpectationsByRouteRowId.set(expectation.routeRowId, expectationsForRouteRow);
  }

  const policyMode = policyExpectationsByRouteRowId.size > 0;
  let skippedRowCount = 0;

  for (const row of params.routeRows) {
    const linkedPolicyExpectations = policyExpectationsByRouteRowId.get(row.row_id) ?? [];
    const expectedActivated = expectedActivatedForReplayRow(row);
    const expectedTopCandidateId = row.teacher_action.kind === "traverse"
      ? row.teacher_action.target_ids[0]?.trim() ?? null
      : null;
    const shouldScoreRow = row.teacher_action.kind === "traverse" || linkedPolicyExpectations.length > 0;

    if (!shouldScoreRow) {
      const gateEvaluated = !policyMode;
      const routeRowDiagnosticIssues = [`unsupported teacher_action kind ${row.teacher_action.kind}`];
      skippedRowCount += 1;
      rowResults.push({
        rowId: row.row_id,
        teacherActionKind: row.teacher_action.kind,
        expectedActivated,
        actualActivated: null,
        actualAbstained: null,
        expectedTopCandidateId,
        actualTopCandidateId: null,
        expectedStopLabel: row.stop_label,
        actualStopLabel: row.stop_label,
        actualStopReason: null,
        actualStopLocal: null,
        decisionConfidence: null,
        activationProbability: null,
        abstentionProbability: null,
        predictedUtility: null,
        predictedRegretOfAbstaining: null,
        stopLocalProbability: null,
        topCandidateProbability: 0,
        stopProbability: 0,
        activationThreshold: null,
        abstentionThreshold: null,
        expectedUtilityThreshold: null,
        stopLocalThreshold: null,
        gateEvaluated,
        routeRowDiagnosticPassed: false,
        routeRowDiagnosticIssues,
        policyExpectationCount: 0,
        policyExpectationPassCount: 0,
        policyExpectationResults: [],
        passed: gateEvaluated ? false : true,
        issues: gateEvaluated ? routeRowDiagnosticIssues : [],
      });
      continue;
    }

    const scoring = scoreColdStartRouteRowV1({ model: loaded.artifact.model, row });
    const actualTopCandidateId = scoring.rankedCandidates[0]?.candidate.candidate_id ?? null;
    const actualStopLabel = scoring.stopPrediction.label;
    const actualActivated = scoring.decisionSummary.activated;
    const actualAbstained = !actualActivated;
    const actualStopLocal = actualStopLabel === "STOP_LOCAL" || scoring.decisionSummary.stopReason === "stop_local";
    const topCandidateProbability = findTraverseProbability({
      candidateId: actualTopCandidateId,
      actionDistribution: scoring.policyDistribution.actions,
    });
    const stopProbability = scoring.policyDistribution.stopAction.probability;
    const routeRowDiagnosticIssues: string[] = [];
    if (row.teacher_action.kind === "traverse") {
      if (!expectedTopCandidateId) {
        routeRowDiagnosticIssues.push("missing expected traverse target");
      } else if (actualTopCandidateId !== expectedTopCandidateId) {
        routeRowDiagnosticIssues.push(`top candidate ${actualTopCandidateId ?? "none"} != expected ${expectedTopCandidateId}`);
      }
    }
    if (expectedActivated !== null && actualActivated !== expectedActivated) {
      routeRowDiagnosticIssues.push(`activation ${booleanStateLabel(actualActivated)} != expected ${booleanStateLabel(expectedActivated)}`);
    }
    if (actualStopLabel !== row.stop_label) {
      routeRowDiagnosticIssues.push(`stop label ${actualStopLabel} != expected ${row.stop_label}`);
    }
    const policyExpectationResults = linkedPolicyExpectations.map((expectation) => {
      const issues = buildPolicyExpectationIssues({
        expectation,
        actualActivated,
        actualAbstained,
        actualStopLocal,
      });
      return {
        policyRowId: expectation.policyRowId,
        routeRowId: expectation.routeRowId,
        rowType: expectation.rowType,
        focusLane: expectation.focusLane,
        weight: expectation.weight,
        hardNegativeClass: expectation.hardNegativeClass,
        oracleBestMode: expectation.oracleBestMode,
        expectedActivated: expectation.expectedActivated,
        actualActivated,
        expectedAbstained: expectation.expectedAbstained,
        actualAbstained,
        expectedStopLocal: expectation.expectedStopLocal,
        actualStopLocal,
        passed: issues.length === 0,
        issues,
      } satisfies ColdStartRouterReplayGatePolicyExpectationResultV1;
    });
    const policyExpectationIssues = policyExpectationResults.flatMap((result) => result.issues.map((issue) => `[${result.policyRowId}] ${issue}`));
    const policyExpectationPassCount = policyExpectationResults.filter((result) => result.passed).length;
    const gateEvaluated = policyMode ? policyExpectationResults.length > 0 : true;
    const gateIssues = policyMode ? policyExpectationIssues : routeRowDiagnosticIssues;
    if (!gateEvaluated) {
      skippedRowCount += 1;
    }

    rowResults.push({
      rowId: row.row_id,
      teacherActionKind: row.teacher_action.kind,
      expectedActivated,
      actualActivated,
      actualAbstained,
      expectedTopCandidateId,
      actualTopCandidateId,
      expectedStopLabel: row.stop_label,
      actualStopLabel,
      actualStopReason: scoring.decisionSummary.stopReason,
      actualStopLocal,
      decisionConfidence: scoring.decisionSummary.confidence,
      activationProbability: scoring.decisionSummary.activationProbability,
      abstentionProbability: scoring.decisionSummary.abstentionProbability,
      predictedUtility: scoring.decisionSummary.predictedUtility,
      predictedRegretOfAbstaining: scoring.decisionSummary.predictedRegretOfAbstaining,
      stopLocalProbability: scoring.decisionSummary.stopLocalProbability,
      topCandidateProbability,
      stopProbability,
      activationThreshold: scoring.decisionSummary.activationThreshold,
      abstentionThreshold: scoring.decisionSummary.abstentionThreshold,
      expectedUtilityThreshold: scoring.decisionSummary.expectedUtilityThreshold,
      stopLocalThreshold: scoring.decisionSummary.stopLocalThreshold,
      gateEvaluated,
      routeRowDiagnosticPassed: routeRowDiagnosticIssues.length === 0,
      routeRowDiagnosticIssues,
      policyExpectationCount: policyExpectationResults.length,
      policyExpectationPassCount,
      policyExpectationResults,
      passed: gateEvaluated ? gateIssues.length === 0 : true,
      issues: gateEvaluated ? gateIssues : [],
    });
  }

  const policyExpectationResults = rowResults.flatMap((rowResult) => rowResult.policyExpectationResults);
  const laneSummaries = summarizePolicyExpectationLanes(policyExpectationResults);
  const policyExpectationCount = policyExpectationResults.length;
  const passedPolicyExpectationCount = policyExpectationResults.filter((result) => result.passed).length;
  const failedPolicyExpectationCount = policyExpectationCount - passedPolicyExpectationCount;
  const evaluatedRowCount = rowResults.filter((rowResult) => rowResult.gateEvaluated).length;
  const passedRowCount = rowResults.filter((rowResult) => rowResult.gateEvaluated && rowResult.passed).length;
  const failedRowCount = rowResults.filter((rowResult) => rowResult.gateEvaluated && !rowResult.passed).length;
  const hasStructuralIssue = loadIssues.length > 0;
  const hasReplayFailure = policyExpectationCount > 0
    ? failedPolicyExpectationCount > 0
    : failedRowCount > 0;
  const passed = !hasStructuralIssue && !hasReplayFailure && evaluatedRowCount > 0;
  const verdict: ColdStartRouterReplayGateVerdictV1["verdict"] = hasStructuralIssue || hasReplayFailure
    ? "fail"
    : skippedRowCount > 0 || evaluatedRowCount === 0
      ? "warn"
      : "pass";
  const routeActivationMatchCount = rowResults.filter((rowResult) => (
    rowResult.expectedActivated !== null
      && rowResult.actualActivated !== null
      && rowResult.expectedActivated === rowResult.actualActivated
  )).length;
  const routeActivationComparableCount = rowResults.filter((rowResult) => (
    rowResult.expectedActivated !== null
      && rowResult.actualActivated !== null
  )).length;
  const policyActivationMatchCount = policyExpectationResults.filter((result) => (
    result.expectedActivated !== null && result.actualActivated === result.expectedActivated
  )).length;
  const policyActivationComparableCount = policyExpectationResults.filter((result) => result.expectedActivated !== null).length;
  const policyAbstainMatchCount = policyExpectationResults.filter((result) => (
    result.expectedAbstained !== null && result.actualAbstained === result.expectedAbstained
  )).length;
  const policyAbstainComparableCount = policyExpectationResults.filter((result) => result.expectedAbstained !== null).length;
  const policyStopLocalMatchCount = policyExpectationResults.filter((result) => (
    result.expectedStopLocal !== null && result.actualStopLocal === result.expectedStopLocal
  )).length;
  const policyStopLocalComparableCount = policyExpectationResults.filter((result) => result.expectedStopLocal !== null).length;
  const laneSummaryText = laneSummaries
    .filter((laneSummary) => REPLAY_GATE_PRIORITY_LANES.includes(laneSummary.lane as (typeof REPLAY_GATE_PRIORITY_LANES)[number]))
    .map((laneSummary) => `${laneSummary.lane} ${laneSummary.passedPolicyExpectationCount}/${laneSummary.policyExpectationCount}`)
    .join(", ");
  const policySummary = policyExpectationCount > 0
    ? `; policy expectations ${passedPolicyExpectationCount}/${policyExpectationCount} passed${laneSummaryText.length > 0 ? ` (${laneSummaryText})` : ""}`
    : "";
  const activationSummary = policyExpectationCount > 0
    ? [
      `policy activation matches ${policyActivationMatchCount}/${policyActivationComparableCount}`,
      policyAbstainComparableCount > 0 ? `abstain matches ${policyAbstainMatchCount}/${policyAbstainComparableCount}` : null,
      policyStopLocalComparableCount > 0 ? `stop_local matches ${policyStopLocalMatchCount}/${policyStopLocalComparableCount}` : null,
    ].filter((value): value is string => value !== null).join(", ")
    : `activation matches ${routeActivationMatchCount}/${routeActivationComparableCount}`;

  return {
    artifactDir: params.artifactDir,
    manifestSummary: loaded.artifact.manifestSummary,
    passed,
    verdict,
    summary: `${passedRowCount}/${evaluatedRowCount} replay rows passed${skippedRowCount > 0 ? `, ${skippedRowCount} skipped` : ""}; ${activationSummary}${policySummary}`,
    evaluatedRowCount,
    passedRowCount,
    failedRowCount,
    skippedRowCount,
    policyExpectationCount,
    passedPolicyExpectationCount,
    failedPolicyExpectationCount,
    loadIssues,
    laneSummaries,
    rowResults,
  };
}
