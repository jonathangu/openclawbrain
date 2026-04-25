import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";
import type { ColdStartPackTypeV1, RouteDecisionRowV1, RouterArtifactManifestV1 } from "./cold-start-router-contracts.ts";
import { validateRouterArtifactManifestV1 } from "./cold-start-router-contracts.ts";
import {
  COLD_START_ROUTER_LIVE_POLICY_INITIALIZER_CONTRACT_V1,
  materializeColdStartRouterLivePolicyGraphV1,
} from "./graph.js";
import type {
  ColdStartRouterBaseModelV1,
  ColdStartRouterCalibrationV1,
  ColdStartRouterFeatureNormalizersV1,
  ColdStartRouterModelV1,
  ColdStartRouterSafetyRulesV1,
  ColdStartRouterScoringResultV1,
  ColdStartRouterSourcePriorsV1,
} from "./cold-start-router-trainer.ts";
import {
  COLD_START_ROUTER_ARTIFACT_LAYOUT_V1,
  COLD_START_ROUTER_BASE_MODEL_CONTRACT_V1,
  COLD_START_ROUTER_CALIBRATION_CONTRACT_V1,
  COLD_START_ROUTER_FEATURE_NORMALIZERS_CONTRACT_V1,
  COLD_START_ROUTER_SAFETY_RULES_CONTRACT_V1,
  COLD_START_ROUTER_SOURCE_PRIORS_CONTRACT_V1,
  COLD_START_ROUTER_WEIGHTS_CONTRACT_V1,
  scoreColdStartRouteRowV1,
} from "./cold-start-router-trainer.ts";

function sha256Text(value: string): string {
  return `sha256:${createHash("sha256").update(value, "utf8").digest("hex")}`;
}

function readJsonArtifact<T>(filePath: string): { value: T; text: string; digest: string } {
  const text = readFileSync(filePath, "utf8");
  return {
    value: JSON.parse(text) as T,
    text,
    digest: sha256Text(text),
  };
}

function parseArtifactRef(ref: string): { refPath: string; digest: string | null } {
  const normalized = ref.trim();
  const hashIndex = normalized.indexOf("#");
  if (hashIndex < 0) {
    return { refPath: normalized, digest: null };
  }
  return {
    refPath: normalized.slice(0, hashIndex).trim(),
    digest: normalized.slice(hashIndex + 1).trim() || null,
  };
}

function deepEqualJson(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

function assertRefPath(ref: string, expectedFileName: string, label: string): { refPath: string; digest: string | null } {
  const parsed = parseArtifactRef(ref);
  assert(path.basename(parsed.refPath) === expectedFileName, `${label} ref must point at ${expectedFileName}, got ${parsed.refPath}`);
  return parsed;
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

function buildArtifactChecksum(params: {
  manifest: Omit<RouterArtifactManifestV1, "artifact_checksum">;
  training: ColdStartRouterBaseModelV1["training"];
  modelRefDigests: Record<string, string>;
}): string {
  return sha256Text(JSON.stringify({
    ...params.manifest,
    trainingSummary: params.training,
    modelRefDigests: params.modelRefDigests,
  }));
}

function readAndValidateArtifactFile<T>(params: {
  filePath: string;
  expectedFileName: string;
  label: string;
  contract: string;
}): { value: T; text: string; digest: string } {
  const artifact = readJsonArtifact<T>(params.filePath);
  const value = artifact.value as Record<string, unknown>;
  assert(typeof value === "object" && value !== null, `${params.label} must be an object`);
  assert(value.contract === params.contract, `${params.label} contract must be ${params.contract}`);
  assert(path.basename(params.filePath) === params.expectedFileName, `${params.label} file must be named ${params.expectedFileName}`);
  return artifact;
}

function normalizeCalibrationForRuntime(
  calibration: ColdStartRouterCalibrationV1,
): ColdStartRouterCalibrationV1 {
  const hasLegacyMissingGateFields = calibration.activationThreshold === undefined
    && calibration.abstentionThreshold === undefined
    && calibration.expectedUtilityThreshold === undefined
    && calibration.stopLocalThreshold === undefined
    && calibration.interventionHead === undefined;

  return {
    ...calibration,
    activationThreshold: calibration.activationThreshold ?? (hasLegacyMissingGateFields ? 0 : 0.45),
    abstentionThreshold: calibration.abstentionThreshold ?? (hasLegacyMissingGateFields ? 1 : 0.55),
    expectedUtilityThreshold: calibration.expectedUtilityThreshold ?? 0,
    stopLocalThreshold: calibration.stopLocalThreshold ?? 0.5,
    interventionHead: calibration.interventionHead ?? {
      decisionPolicyMode: "router_blended",
      freezeCandidateSelection: false,
      freezeStopLocal: false,
      featureProfile: "default_router",
    },
  };
}

function normalizeLivePolicyInitializerForRuntime(
  initializer: ColdStartRouterModelV1["livePolicyInitializer"],
): ColdStartRouterModelV1["livePolicyInitializer"] {
  return {
    ...initializer,
    semanticClassSeedWeights: initializer.semanticClassSeedWeights ?? [],
    semanticClassEdgeWeights: initializer.semanticClassEdgeWeights ?? [],
  };
}

export interface ColdStartRouterRuntimeArtifactBundleV1 {
  artifactDir: string;
  manifestPath: string;
  manifest: RouterArtifactManifestV1;
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

export interface ColdStartRouterSelectionResultV1 extends ColdStartRouterScoringResultV1 {
  selectedCandidateIds: string[];
  stopped: boolean;
}

export interface ColdStartRouterArtifactRuntimeTruthV1 {
  artifactId: string;
  artifactVersion: string;
  artifactChecksum: string;
  packType: ColdStartPackTypeV1;
  routerIdentity: string | null;
  priorBaseArtifactId: string | null;
  priorBaseArtifactChecksum: string | null;
  mixedPackFromBaseArtifactId: string | null;
  summary: string;
}

export function summarizeColdStartRouterArtifactManifestRuntimeTruthV1(
  manifest: RouterArtifactManifestV1,
): ColdStartRouterArtifactRuntimeTruthV1 {
  const priorBaseArtifactId = manifest.prior_base_artifact_id ?? null;
  const priorBaseArtifactChecksum = manifest.prior_base_artifact_checksum ?? null;
  const mixedPackFromBaseArtifactId = manifest.pack_type === "mixed"
    ? priorBaseArtifactId
    : null;

  return {
    artifactId: manifest.artifact_id,
    artifactVersion: manifest.artifact_version,
    artifactChecksum: manifest.artifact_checksum,
    packType: manifest.pack_type,
    routerIdentity: manifest.router_identity ?? null,
    priorBaseArtifactId,
    priorBaseArtifactChecksum,
    mixedPackFromBaseArtifactId,
    summary: [
      `artifact=${manifest.artifact_id}@${manifest.artifact_version}`,
      `pack=${manifest.pack_type}`,
      `checksum=${manifest.artifact_checksum}`,
      `prior=${priorBaseArtifactId ?? "none"}`,
      `mixedFrom=${mixedPackFromBaseArtifactId ?? "none"}`,
    ].join("; "),
  };
}

export function summarizeColdStartRouterArtifactBundleRuntimeTruthV1(
  artifactBundle: ColdStartRouterRuntimeArtifactBundleV1,
): ColdStartRouterArtifactRuntimeTruthV1 {
  return summarizeColdStartRouterArtifactManifestRuntimeTruthV1(artifactBundle.manifest);
}

export function loadColdStartRouterArtifactBundleV1(artifactDir: string): ColdStartRouterRuntimeArtifactBundleV1 {
  const resolvedArtifactDir = path.resolve(artifactDir);
  const manifestPath = path.join(resolvedArtifactDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.manifest);
  const baseModelPath = path.join(resolvedArtifactDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.baseModel);
  const weightsPath = path.join(resolvedArtifactDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.weights);
  const calibrationPath = path.join(resolvedArtifactDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.calibration);
  const featureNormalizersPath = path.join(resolvedArtifactDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.featureNormalizers);
  const sourcePriorsPath = path.join(resolvedArtifactDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.sourcePriors);
  const safetyRulesPath = path.join(resolvedArtifactDir, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.safetyRules);

  const manifestArtifact = readJsonArtifact<RouterArtifactManifestV1>(manifestPath);
  const manifestValidation = validateRouterArtifactManifestV1(manifestArtifact.value);
  assert(manifestValidation.valid, `router manifest failed validation: ${manifestValidation.issues.join("; ")}`);

  const baseModelArtifact = readAndValidateArtifactFile<ColdStartRouterBaseModelV1>({
    filePath: baseModelPath,
    expectedFileName: COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.baseModel,
    label: "base model",
    contract: COLD_START_ROUTER_BASE_MODEL_CONTRACT_V1,
  });
  const weightsArtifact = readAndValidateArtifactFile<ColdStartRouterModelV1>({
    filePath: weightsPath,
    expectedFileName: COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.weights,
    label: "weights",
    contract: COLD_START_ROUTER_WEIGHTS_CONTRACT_V1,
  });
  const calibrationArtifact = readAndValidateArtifactFile<ColdStartRouterCalibrationV1>({
    filePath: calibrationPath,
    expectedFileName: COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.calibration,
    label: "calibration",
    contract: COLD_START_ROUTER_CALIBRATION_CONTRACT_V1,
  });
  const featureNormalizersArtifact = readAndValidateArtifactFile<ColdStartRouterFeatureNormalizersV1>({
    filePath: featureNormalizersPath,
    expectedFileName: COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.featureNormalizers,
    label: "feature normalizers",
    contract: COLD_START_ROUTER_FEATURE_NORMALIZERS_CONTRACT_V1,
  });
  const sourcePriorsArtifact = readAndValidateArtifactFile<ColdStartRouterSourcePriorsV1>({
    filePath: sourcePriorsPath,
    expectedFileName: COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.sourcePriors,
    label: "source priors",
    contract: COLD_START_ROUTER_SOURCE_PRIORS_CONTRACT_V1,
  });
  const safetyRulesArtifact = readAndValidateArtifactFile<ColdStartRouterSafetyRulesV1>({
    filePath: safetyRulesPath,
    expectedFileName: COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.safetyRules,
    label: "safety rules",
    contract: COLD_START_ROUTER_SAFETY_RULES_CONTRACT_V1,
  });

  const manifest = manifestArtifact.value;
  const baseModel = baseModelArtifact.value;
  const rawModel = weightsArtifact.value;
  const rawCalibration = calibrationArtifact.value;
  const featureNormalizers = featureNormalizersArtifact.value;
  const sourcePriors = sourcePriorsArtifact.value;
  const safetyRules = safetyRulesArtifact.value;

  const calibration = normalizeCalibrationForRuntime(rawCalibration);
  const model: ColdStartRouterModelV1 = {
    ...rawModel,
    calibration,
    livePolicyInitializer: normalizeLivePolicyInitializerForRuntime(rawModel.livePolicyInitializer),
  };

  assert(rawModel.calibration.contract === COLD_START_ROUTER_CALIBRATION_CONTRACT_V1, "weights.calibration must carry the calibration contract");
  assert(rawModel.featureNormalizers.contract === COLD_START_ROUTER_FEATURE_NORMALIZERS_CONTRACT_V1, "weights.featureNormalizers must carry the feature normalizer contract");
  assert(rawModel.sourcePriors.contract === COLD_START_ROUTER_SOURCE_PRIORS_CONTRACT_V1, "weights.sourcePriors must carry the source priors contract");
  assert(rawModel.safetyRules.contract === COLD_START_ROUTER_SAFETY_RULES_CONTRACT_V1, "weights.safetyRules must carry the safety rules contract");
  assert(rawModel.livePolicyInitializer?.contract === COLD_START_ROUTER_LIVE_POLICY_INITIALIZER_CONTRACT_V1, "weights.livePolicyInitializer must carry the live policy initializer contract");
  assert(Array.isArray(rawModel.toolActionPriors), "weights.toolActionPriors must be an array");
  assert(Array.isArray(rawModel.toolActionSets), "weights.toolActionSets must be an array");
  assert(deepEqualJson(rawModel.calibration, rawCalibration), "weights.calibration must match calibration.json exactly");
  assert(deepEqualJson(rawModel.featureNormalizers, featureNormalizers), "weights.featureNormalizers must match feature-normalizers.json exactly");
  assert(deepEqualJson(rawModel.sourcePriors, sourcePriors), "weights.sourcePriors must match source-priors.json exactly");
  assert(deepEqualJson(rawModel.safetyRules, safetyRules), "weights.safetyRules must match safety-rules.json exactly");
  assert(deepEqualJson(model.training, baseModel.training), "weights.training must match base-model.json training summary exactly");
  assert(baseModel.artifactId === manifest.artifact_id, "base-model artifactId must match the manifest");
  assert(baseModel.artifactVersion === manifest.artifact_version, "base-model artifactVersion must match the manifest");
  assert(baseModel.packType === manifest.pack_type, "base-model packType must match the manifest");
  assert(baseModel.compatibleRuntimeVersion === manifest.compatible_runtime_version, "base-model runtime version must match the manifest");
  assert(baseModel.routerIdentity === (manifest.router_identity ?? null), "base-model routerIdentity must match the manifest");
  assert(model.artifactId === manifest.artifact_id, "weights artifactId must match the manifest");
  assert(model.artifactVersion === manifest.artifact_version, "weights artifactVersion must match the manifest");
  assert(model.packType === manifest.pack_type, "weights packType must match the manifest");
  assert(model.compatibleRuntimeVersion === manifest.compatible_runtime_version, "weights runtime version must match the manifest");
  assert(model.routerIdentity === (manifest.router_identity ?? null), "weights routerIdentity must match the manifest");

  const baseModelRef = assertRefPath(manifest.base_model_ref, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.baseModel, "base model");
  const weightsRef = assertRefPath(manifest.weights_ref, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.weights, "weights");
  const calibrationRef = assertRefPath(manifest.calibration_ref, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.calibration, "calibration");
  const featureNormalizersRef = assertRefPath(manifest.feature_normalizers_ref, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.featureNormalizers, "feature normalizers");
  const sourcePriorsRef = assertRefPath(manifest.source_priors_ref, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.sourcePriors, "source priors");
  const safetyRulesRef = assertRefPath(manifest.safety_rules_ref, COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.safetyRules, "safety rules");

  const expectedBaseModelDigest = buildLogicalBaseModelDigest(baseModel);
  assert(baseModelRef.digest === expectedBaseModelDigest, "base-model ref digest does not match the logical base model digest");
  assert(weightsRef.digest === weightsArtifact.digest, "weights ref digest does not match weights.json");
  assert(calibrationRef.digest === calibrationArtifact.digest, "calibration ref digest does not match calibration.json");
  assert(featureNormalizersRef.digest === featureNormalizersArtifact.digest, "feature normalizers ref digest does not match feature-normalizers.json");
  assert(sourcePriorsRef.digest === sourcePriorsArtifact.digest, "source priors ref digest does not match source-priors.json");
  assert(safetyRulesRef.digest === safetyRulesArtifact.digest, "safety rules ref digest does not match safety-rules.json");

  const expectedChecksum = buildArtifactChecksum({
    manifest: {
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
      ...(manifest.router_identity !== undefined ? { router_identity: manifest.router_identity } : {}),
    },
    training: baseModel.training,
    modelRefDigests: {
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.baseModel]: expectedBaseModelDigest,
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.weights]: weightsArtifact.digest,
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.calibration]: calibrationArtifact.digest,
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.featureNormalizers]: featureNormalizersArtifact.digest,
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.sourcePriors]: sourcePriorsArtifact.digest,
      [COLD_START_ROUTER_ARTIFACT_LAYOUT_V1.safetyRules]: safetyRulesArtifact.digest,
    },
  });
  assert(manifest.artifact_checksum === expectedChecksum, "artifact checksum does not match the loaded bundle");

  return {
    artifactDir: resolvedArtifactDir,
    manifestPath,
    manifest,
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
  };
}

export function scoreColdStartRouteRowFromArtifactBundleV1(params: {
  artifactBundle: ColdStartRouterRuntimeArtifactBundleV1;
  row: RouteDecisionRowV1;
}): ColdStartRouterScoringResultV1 {
  return scoreColdStartRouteRowV1({ model: params.artifactBundle.model, row: params.row });
}

export function selectColdStartRouteCandidateIdsFromArtifactBundleV1(params: {
  artifactBundle: ColdStartRouterRuntimeArtifactBundleV1;
  row: RouteDecisionRowV1;
  /**
   * Optional replay/serving budget for callers that can safely consume more than
   * one context block. The default remains one to preserve the historical cold
   * start router contract for direct runtime callers.
   */
  maxCandidateIds?: number | null;
  /**
   * When selecting multiple candidates, keep only candidates close enough to the
   * top router score. This prevents broad over-firing while fixing the replay
   * under-selection case where one useful block was too narrow.
   */
  multiSelectScoreWindow?: number | null;
}): ColdStartRouterSelectionResultV1 {
  const scoring = scoreColdStartRouteRowFromArtifactBundleV1(params);
  const bestTraverse = scoring.rankedCandidates[0] ?? null;
  const stopped = !scoring.decisionSummary.activated;
  const requestedLimit = Math.max(1, Math.floor(params.maxCandidateIds ?? 1));
  const scoreWindow = Math.max(0, params.multiSelectScoreWindow ?? 0.35);
  const selectedCandidateIds = stopped || bestTraverse === null
    ? []
    : scoring.rankedCandidates
      .filter((candidate) => bestTraverse.score - candidate.score <= scoreWindow)
      .slice(0, requestedLimit)
      .map((candidate) => candidate.candidate.candidate_id);
  return {
    ...scoring,
    selectedCandidateIds,
    stopped,
  };
}

export function materializeColdStartRouterLivePolicyFromArtifactBundleV1(params: {
  artifactBundle: ColdStartRouterRuntimeArtifactBundleV1;
  row: RouteDecisionRowV1;
}) {
  return materializeColdStartRouterLivePolicyGraphV1({
    initializer: params.artifactBundle.model.livePolicyInitializer,
    row: params.row,
    applyResumeGateReplaySemanticFallbackBoost:
      params.artifactBundle.model.calibration.interventionHead?.featureProfile === "resume_gate_v1",
  });
}
