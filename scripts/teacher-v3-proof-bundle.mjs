#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { captureTeacherV3ReplayOutcomes } from "./teacher-v3-replay-outcomes.mjs";
import {
  describeTeacherCanaryActivationGuardV1,
  describeTeacherCanaryRolloutPlanV1,
  describeTeacherProposalReplayGateMatrixV1,
} from "../src/brain-core/teacher-v3-contracts.js";
import { summarizeRouterMigrationComparisonV1 } from "../src/brain-core/router-migration.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const workspaceRoot = path.resolve(repoRoot, "..");

export const TEACHER_V3_PROOF_BUNDLE_LAYOUT = {
  summary: "summary.md",
  status: "status.json",
  surfaceMap: "surface-map.json",
  proposalReport: "proposal-report.json",
  verdict: "verdict.json",
};

export const TEACHER_V3_PROOF_BUNDLE_CONTRACT = "teacher_v3_proof_bundle.v1";
export const TEACHER_V3_PROOF_STATUS_CONTRACT = "teacher_v3_proof_bundle_status.v1";
export const TEACHER_V3_PROOF_SURFACE_MAP_CONTRACT = "teacher_v3_surface_map.v1";
export const TEACHER_V3_PROPOSAL_REPORT_CONTRACT = "teacher_v3_proposal_report.v1";
export const TEACHER_V3_PROOF_VERDICT_CONTRACT = "teacher_v3_proof_bundle_verdict.v1";

export const DEFAULT_TEACHER_V3_PROOF_BUNDLE_PARENT = path.join(
  workspaceRoot,
  "artifacts",
  "teacher-v3-proof",
);

function sha256Text(text) {
  return `sha256:${createHash("sha256").update(String(text ?? ""), "utf8").digest("hex")}`;
}

function renderJson(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function ensureDir(dirPath) {
  if (!existsSync(dirPath)) {
    mkdirSync(dirPath, { recursive: true });
  }
}

function normalizeText(value) {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeBoolean(value) {
  return value === true ? true : value === false ? false : null;
}

function normalizeNumber(value) {
  return Number.isFinite(value) ? Number(value) : null;
}

function normalizeArray(value) {
  return Array.isArray(value) ? value.filter((item) => item !== null && item !== undefined) : [];
}

function normalizeShadowReplaySummary(value, fallback = {}) {
  if (!value || typeof value !== "object") {
    return null;
  }

  const proposalClass = normalizeText(value.proposalClass);
  if (proposalClass !== "mutation" && proposalClass !== "forgetting") {
    return null;
  }

  const reviewMode = normalizeText(value.reviewMode) ?? "shadow_only";
  const rollback = value.rollback && typeof value.rollback === "object"
    ? {
      strategy: normalizeText(value.rollback.strategy),
      restored: normalizeBoolean(value.rollback.restored),
      summary: normalizeText(value.rollback.summary),
      before: value.rollback.before ?? null,
      after: value.rollback.after ?? null,
    }
    : null;

  return {
    proposalId: normalizeText(value.proposalId) ?? normalizeText(fallback.proposalId),
    proposalClass,
    reviewMode,
    shadowOnly: normalizeBoolean(value.shadowOnly) ?? reviewMode === "shadow_only",
    promotionBypass: normalizeBoolean(value.promotionBypass) ?? false,
    rollbackKey: normalizeText(value.rollbackKey) ?? normalizeText(fallback.rollbackKey),
    applied: normalizeBoolean(value.applied),
    reversible: normalizeBoolean(value.reversible),
    replayOutcome: normalizeText(value.replayOutcome),
    summary: normalizeText(value.summary),
    candidateStateKind: normalizeText(value.candidateStateKind),
    before: value.before ?? null,
    after: value.after ?? null,
    rollback,
    target: value.target ?? null,
    decision: value.decision ?? null,
    guardrail: normalizeText(value.guardrail),
    reason: normalizeText(value.reason),
    requestedTransition: normalizeText(value.requestedTransition),
    applications: Array.isArray(value.applications)
      ? value.applications.map((application) => ({
        index: normalizeNumber(application?.index),
        proposalId: normalizeText(application?.proposalId),
        proposalKind: normalizeText(application?.proposalKind),
        applied: normalizeBoolean(application?.applied),
        reversible: normalizeBoolean(application?.reversible),
        reason: normalizeText(application?.reason),
        before: application?.before ?? null,
        after: application?.after ?? null,
        operationKinds: normalizeArray(application?.operationKinds).map((operationKind) => normalizeText(operationKind)).filter(Boolean),
      }))
      : undefined,
  };
}

function relativeWorkspacePath(value) {
  const text = normalizeText(value);
  if (!text) {
    return null;
  }
  if (!path.isAbsolute(text)) {
    return text;
  }
  return path.relative(workspaceRoot, text);
}

function timestampToken(date = new Date()) {
  const resolvedDate = date instanceof Date ? date : new Date(date);
  return resolvedDate.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z").replace("T", "-");
}

function summarizeRuntimeStatus(runtimeStatus) {
  const currentPackMetadata = runtimeStatus && typeof runtimeStatus.currentPackMetadata === "object"
    ? runtimeStatus.currentPackMetadata
    : null;
  const operatorHealth = runtimeStatus && typeof runtimeStatus.operatorHealth === "object"
    ? runtimeStatus.operatorHealth
    : null;
  const learningHealth = runtimeStatus && typeof runtimeStatus.learningHealth === "object"
    ? runtimeStatus.learningHealth
    : null;
  const lastReplayGateVerdict = runtimeStatus && typeof runtimeStatus.lastReplayGateVerdict === "object"
    ? runtimeStatus.lastReplayGateVerdict
    : null;
  const lastPromotionVerdict = runtimeStatus && typeof runtimeStatus.lastPromotionVerdict === "object"
    ? runtimeStatus.lastPromotionVerdict
    : null;
  const lastAssemblyDecision = runtimeStatus && typeof runtimeStatus.lastAssemblyDecision === "object"
    ? runtimeStatus.lastAssemblyDecision
    : null;
  const lastPrefetchDecision = runtimeStatus && typeof runtimeStatus.lastPrefetchDecision === "object"
    ? runtimeStatus.lastPrefetchDecision
    : null;

  return {
    serveState: normalizeText(runtimeStatus?.serveState),
    currentPackVersion: normalizeNumber(runtimeStatus?.currentPackVersion),
    currentPackPromotedAt: normalizeText(runtimeStatus?.currentPackPromotedAt),
    currentPackMetadata: {
      reason: normalizeText(currentPackMetadata?.reason),
      kind: normalizeText(currentPackMetadata?.kind),
      promotedAt: normalizeText(currentPackMetadata?.promotedAt),
      rolledBack: normalizeBoolean(currentPackMetadata?.rolledBack),
    },
    teacherConfigured: normalizeBoolean(runtimeStatus?.teacherConfigured),
    teacherProvider: normalizeText(runtimeStatus?.teacherProvider),
    teacherModel: normalizeText(runtimeStatus?.teacherModel),
    teacherConfigError: normalizeText(runtimeStatus?.teacherConfigError),
    operatorHealth: {
      status: normalizeText(operatorHealth?.status),
      detail: normalizeText(operatorHealth?.detail),
    },
    learningHealth: {
      status: normalizeText(learningHealth?.status),
      detail: normalizeText(learningHealth?.detail),
    },
    routeTraceCount: normalizeNumber(runtimeStatus?.routeTraceCount),
    supervisionCount: normalizeNumber(runtimeStatus?.supervisionCount),
    recentTraceCount: normalizeNumber(runtimeStatus?.recentTraceCount),
    pendingLabels: normalizeNumber(runtimeStatus?.pendingLabels),
    pendingObservations: normalizeNumber(runtimeStatus?.pendingObservations),
    mutationBacklog: runtimeStatus?.mutationBacklog && typeof runtimeStatus.mutationBacklog === "object"
      ? runtimeStatus.mutationBacklog
      : null,
    lastCompileReportSummary: normalizeText(runtimeStatus?.lastCompileReportSummary),
    lastAssemblyDecision: {
      summary: normalizeText(lastAssemblyDecision?.summary),
      verdict: normalizeText(lastAssemblyDecision?.verdict),
    },
    lastPrefetchDecision: {
      summary: normalizeText(lastPrefetchDecision?.summary),
      verdict: normalizeText(lastPrefetchDecision?.verdict),
    },
    lastPromotionReason: normalizeText(runtimeStatus?.lastPromotionReason),
    lastPromotionVerdict: {
      verdict: normalizeText(lastPromotionVerdict?.verdict),
      summary: normalizeText(lastPromotionVerdict?.summary),
    },
    lastReplayFailureReason: normalizeText(runtimeStatus?.lastReplayFailureReason),
    lastReplayGateVerdict: {
      verdict: normalizeText(lastReplayGateVerdict?.verdict),
      summary: normalizeText(lastReplayGateVerdict?.summary),
    },
  };
}

function summarizeOperatorProofTruth(operatorProof) {
  const verdict = operatorProof?.verdict && typeof operatorProof.verdict === "object"
    ? operatorProof.verdict
    : null;
  return {
    bundleDir: relativeWorkspacePath(operatorProof?.bundleDir),
    command: normalizeText(operatorProof?.command),
    summary: normalizeText(operatorProof?.summary),
    verdict: normalizeText(verdict?.verdict),
    severity: normalizeText(verdict?.severity),
    why: normalizeText(verdict?.why),
    runtimeLoadProofPath: relativeWorkspacePath(operatorProof?.runtimeLoadProofPath),
    runtimeLoadProofExists: normalizeBoolean(operatorProof?.runtimeLoadProofExists),
    stepCount: normalizeNumber(operatorProof?.stepCount),
    postBundleCount: normalizeNumber(operatorProof?.postBundleCount),
    missingProofCount: normalizeNumber(Array.isArray(verdict?.missingProofs) ? verdict.missingProofs.length : null),
  };
}

function summarizeDocsTruth(docsTruth) {
  return {
    path: relativeWorkspacePath(docsTruth?.path),
    title: normalizeText(docsTruth?.title),
    state: "shipped",
    summary: normalizeText(docsTruth?.summary),
  };
}

function formatProposalClassLabel(proposalClass) {
  const normalized = normalizeText(proposalClass) ?? "teacher";
  return `${normalized.charAt(0).toUpperCase()}${normalized.slice(1)} lane`;
}

function buildTeacherV3CanaryRollout(plan, replaySummary = null) {
  const proposalClass = normalizeText(plan?.proposalClass) ?? "compiler";
  const rollbackKey = normalizeText(plan?.rollbackKey) ?? null;
  const candidatePackVersion = normalizeNumber(plan?.candidatePackVersion)
    ?? normalizeNumber(replaySummary?.candidatePackVersion);
  const candidatePackId = normalizeText(plan?.candidatePackId)
    ?? normalizeText(replaySummary?.candidatePackId);
  const candidatePackText = candidatePackId ?? candidatePackVersion ?? "unbound";
  const laneLabel = formatProposalClassLabel(proposalClass);
  const rolloutMode = normalizeText(plan?.rolloutMode) ?? "off";
  const enabled = normalizeBoolean(plan?.enabled) ?? false;
  const guardrails = Array.isArray(plan?.guardrails) && plan.guardrails.length > 0
    ? plan.guardrails
    : [
      "Keep the rollout plan target-state only until it is explicitly shipped.",
      "Default rolloutMode stays off.",
      "Do not use the canary plan to change live serving without separate replay and rollback proof.",
      "Bind any candidate pack by durable version or id, never by ad hoc display labels.",
    ];

  return {
    proposalClass,
    surfaceState: normalizeText(plan?.surfaceState) ?? "target",
    rolloutMode,
    enabled,
    disabledByDefault: rolloutMode === "off" && enabled === false,
    rollbackBound: rollbackKey !== null,
    rollbackKey,
    candidatePackVersion: candidatePackVersion ?? null,
    candidatePackId,
    shippedStateSummary: normalizeText(plan?.shippedStateSummary)
      ?? `${laneLabel}: shipped runtime serves only promoted packs; no canary live rollout is shipped.`,
    targetStateSummary: normalizeText(plan?.targetStateSummary)
      ?? `${laneLabel}: the canary plan stays explicit, replayable, and off by default until a later tranche opts it in.`,
    bindingSummary: `${laneLabel}: rollback-bound to ${rollbackKey ?? "missing"}; candidate pack ${candidatePackText}.`,
    guardrails,
  };
}

function buildTeacherV3ProposalSeed(input) {
  const bundleId = normalizeText(input.bundleId) ?? `teacher-v3-proof-${timestampToken(input.bundleStartedAt)}`;
  const runtimeStatus = summarizeRuntimeStatus(input.runtimeStatus ?? null);
  const operatorProof = summarizeOperatorProofTruth(input.operatorProof ?? null);
  const docsTruth = summarizeDocsTruth(input.docsTruth ?? null);
  const runtimeStatusCommand = normalizeText(input.runtimeStatusCommand) ?? "runtime-status";
  const operatorProofCommand = normalizeText(input.operatorProofCommand) ?? "openclawbrain proof";
  const subjectIds = normalizeArray(input.subjectIds).map((subject) => normalizeText(subject)).filter(Boolean);
  const proposalRecordSubjectIds = normalizeArray(input.proposalRecord?.subjectIds)
    .map((subject) => normalizeText(subject))
    .filter(Boolean);
  const proposalSubjectIds = subjectIds.length > 0
    ? subjectIds
    : proposalRecordSubjectIds.length > 0
      ? proposalRecordSubjectIds
      : ["surface:runtime-truth", "surface:proof-truth", "surface:docs-truth"];
  const proposalClass = normalizeText(input.proposalClass) ?? normalizeText(input.proposalRecord?.proposalClass) ?? "lint";
  const proposalLane = normalizeText(input.proposalLane) ?? normalizeText(input.proposalRecord?.proposalLane) ?? proposalClass;
  const producerVersion = normalizeText(input.producerVersion) ?? "openclawbrain-proof-capture";
  const proposalId = normalizeText(input.proposalId) ?? bundleId;
  const sourceBundleId = operatorProof.bundleDir ?? normalizeText(input.operatorProof?.bundleDir) ?? null;
  const proposalRecord = input.proposalRecord ?? null;
  const proposalRecordEvidence = normalizeArray(proposalRecord?.evidence);
  const proposalRecordCounterevidence = normalizeArray(proposalRecord?.counterevidence);
  const proposalRecordReplaySuites = normalizeArray(proposalRecord?.replaySuites);
  const proposalRecordReplaySummary = proposalRecord?.replaySummary ?? null;
  const rollbackKey = normalizeText(input.proposalRecord?.rollbackKey) ?? `rollback:teacher-v3-proof-bundle:${bundleId}`;
  const shadowReplaySummary = normalizeShadowReplaySummary(input.shadowReplay ?? proposalRecord?.shadowReplay ?? null, {
    proposalId,
    rollbackKey,
  });
  const routerMigrationComparison = input.routerMigrationComparison
    ? summarizeRouterMigrationComparisonV1({
      ...input.routerMigrationComparison,
      proofBundleFiles: normalizeArray(input.routerMigrationComparison.proofBundleFiles).length > 0
        ? normalizeArray(input.routerMigrationComparison.proofBundleFiles)
        : Object.values(TEACHER_V3_PROOF_BUNDLE_LAYOUT),
    })
    : proposalRecord?.migrationComparison
      ? summarizeRouterMigrationComparisonV1({
        ...proposalRecord.migrationComparison,
        proofBundleFiles: normalizeArray(proposalRecord.migrationComparison.proofBundleFiles).length > 0
          ? normalizeArray(proposalRecord.migrationComparison.proofBundleFiles)
          : Object.values(TEACHER_V3_PROOF_BUNDLE_LAYOUT),
      })
      : null;
  const canaryCandidatePackVersion = proposalRecordReplaySummary?.candidatePackVersion ?? shadowReplaySummary?.candidatePackVersion ?? null;
  const canaryCandidatePackId = proposalRecordReplaySummary?.candidatePackId ?? shadowReplaySummary?.candidatePackId ?? null;
  const baseCanaryPlan = proposalRecord?.canaryRollout
    ?? describeTeacherCanaryRolloutPlanV1({
      proposalClass,
      rollbackKey,
      candidatePackVersion: canaryCandidatePackVersion ?? undefined,
      candidatePackId: canaryCandidatePackId ?? undefined,
    });
  const canaryRollout = buildTeacherV3CanaryRollout(baseCanaryPlan, proposalRecordReplaySummary);
  const canaryActivationGuard = describeTeacherCanaryActivationGuardV1({
    proposalId,
    proposalClass,
    rollbackKey,
    canaryRollout,
    replaySummary: proposalRecordReplaySummary,
    proofBundle: proposalRecord?.proofBundle ?? null,
  });
  const idempotencyKey = sha256Text(JSON.stringify({
    bundleId,
    proposalId,
    proposalClass,
    proposalLane,
    producerVersion,
    basePackVersion: runtimeStatus.currentPackVersion,
    baseGraphHash: runtimeStatus.currentPackMetadata?.kind ?? null,
    runtimeStatusCommand,
    operatorProofCommand,
    runtimeLoadProofPath: operatorProof.runtimeLoadProofPath,
    docsTruthPath: docsTruth.path,
  }));

  return {
    recordSource: input.proposalRecord?.recordSource ?? "runtime-capture",
    bundleId,
    proposalId,
    proposalLane,
    proposalClass,
    status: normalizeText(input.proposalStatus) ?? normalizeText(input.proposalRecord?.status) ?? "validated",
    lineage: {
      proposalClass,
      basePackVersion: runtimeStatus.currentPackVersion,
      baseGraphHash: normalizeText(input.proposalRecord?.lineage?.baseGraphHash) ?? null,
      producerVersion,
      producerBuildId: normalizeText(input.proposalRecord?.lineage?.producerBuildId) ?? null,
      promptHash: normalizeText(input.proposalRecord?.lineage?.promptHash) ?? sha256Text(runtimeStatusCommand + operatorProofCommand + docsTruth.path),
      templateId: normalizeText(input.proposalRecord?.lineage?.templateId) ?? "teacher-v3-proof-bundle-v1",
      scope: normalizeText(input.proposalRecord?.lineage?.scope) ?? "runtime-proof-adoption",
      profile: normalizeText(input.proposalRecord?.lineage?.profile) ?? null,
      idempotencyKey,
      sourceBundleId,
      parentProposalIds: normalizeArray(input.proposalRecord?.lineage?.parentProposalIds)
        .map((parentId) => normalizeText(parentId))
        .filter(Boolean),
    },
    subjectIds: proposalSubjectIds,
    evidence: proposalRecordEvidence.length > 0
      ? proposalRecordEvidence
      : [
        {
          evidenceId: "runtime-status",
          sourceKind: "repo",
          sourceId: runtimeStatusCommand,
          authority: "raw_source",
          derivation: "teacher_inference",
          excerpt: "Runtime status snapshot captured for the Teacher v3 proof bundle.",
          sourceHash: sha256Text(JSON.stringify(runtimeStatus)),
        },
        {
          evidenceId: "operator-proof",
          sourceKind: "repo",
          sourceId: operatorProofCommand,
          authority: "raw_source",
          derivation: "teacher_compilation",
          excerpt: "Operator proof bundle captured alongside the runtime snapshot.",
          sourceHash: sha256Text(JSON.stringify(operatorProof)),
        },
        {
          evidenceId: "docs-truth",
          sourceKind: "file",
          sourceId: normalizeText(input.docsTruth?.path) ?? "docs/architecture/teacher-v3-proof.md",
          authority: "raw_source",
          derivation: "teacher_compilation",
          excerpt: "Teacher v3 proof-surface design note used as the docs truth anchor.",
          sourceHash: sha256Text(JSON.stringify(docsTruth)),
        },
      ],
    counterevidence: proposalRecordCounterevidence,
    payload: input.proposalRecord?.payload ?? {
      runtimeStatus,
      operatorProof,
      docsTruth,
      bundleArtifacts: [
        TEACHER_V3_PROOF_BUNDLE_LAYOUT.summary,
        TEACHER_V3_PROOF_BUNDLE_LAYOUT.status,
        TEACHER_V3_PROOF_BUNDLE_LAYOUT.surfaceMap,
        TEACHER_V3_PROOF_BUNDLE_LAYOUT.proposalReport,
        TEACHER_V3_PROOF_BUNDLE_LAYOUT.verdict,
      ],
    },
    confidence: Number.isFinite(proposalRecord?.confidence)
      ? Number(proposalRecord.confidence)
      : 0.92,
    replaySuites: proposalRecordReplaySuites.length > 0
      ? proposalRecordReplaySuites
      : [
        "runtime-status-surface",
        "operator-proof-surface",
        "teacher-v3-docs-truth",
      ],
    rollbackKey,
    runtimeStatusCommand,
    operatorProofCommand,
    createdAt: normalizeText(input.proposalRecord?.createdAt) ?? input.bundleStartedAt,
    resolvedAt: normalizeText(input.proposalRecord?.resolvedAt) ?? input.bundleStartedAt,
    updatedAt: normalizeText(input.proposalRecord?.updatedAt) ?? input.bundleStartedAt,
    gate1Seam: {
      present: Boolean(input.proposalRecord?.gate1Seam?.present),
      recordSource: input.proposalRecord?.gate1Seam?.recordSource ?? "runtime-capture",
      note: input.proposalRecord?.gate1Seam?.note
        ?? "Gate 1 persistence is not wired in this branch; this bundle is seeded from runtime/proof inputs and can later be replaced by a stored proposal record.",
    },
    docsTruth,
    runtimeTruth: runtimeStatus,
    proofTruth: operatorProof,
    replaySummary: proposalRecordReplaySummary,
    routerMigrationComparison,
    canaryRollout,
    canaryActivationGuard,
    shadowReplaySummary,
    proposalReviewMode: proposalClass === "mutation" || proposalClass === "forgetting" || proposalClass === "correction"
      ? "shadow_only"
      : "promotable",
  };
}

function buildSurfaceMap(seed, bundlePaths) {
  const canaryRolloutNote = seed.canaryRollout
    ? `${seed.canaryRollout.surfaceState}/${seed.canaryRollout.rolloutMode}/${seed.canaryRollout.enabled ? "enabled" : "disabled"}; ${seed.canaryRollout.bindingSummary}`
    : "canary rollout unavailable";
  const migrationComparisonNote = seed.routerMigrationComparison
    ? `router migration comparison across old_live/base_only/mixed; decision=${seed.routerMigrationComparison.comparison.decision}`
    : null;

  const observedSurfaces = [
    {
      id: "runtime-truth",
      state: "shipped",
      kind: "runtime_truth",
      source: seed.runtimeStatusCommand,
      note: [
        "Live runtime snapshot captured from the shipped status surface.",
        seed.runtimeTruth.currentPackVersion !== null
          ? `currentPackVersion=${seed.runtimeTruth.currentPackVersion}`
          : null,
        seed.runtimeTruth.serveState ? `serveState=${seed.runtimeTruth.serveState}` : null,
      ].filter(Boolean).join("; "),
    },
    {
      id: "operator-proof-truth",
      state: "shipped",
      kind: "proof_truth",
      source: seed.operatorProofCommand,
      note: [
        "Host-anchored proof bundle emitted by the shipped proof capture path.",
        seed.proofTruth.verdict ? `verdict=${seed.proofTruth.verdict}` : null,
        seed.proofTruth.runtimeLoadProofExists !== null
          ? `runtimeLoadProof=${seed.proofTruth.runtimeLoadProofExists ? "present" : "missing"}`
          : null,
      ].filter(Boolean).join("; "),
    },
    {
      id: "docs-truth",
      state: "shipped",
      kind: "docs_truth",
      source: seed.docsTruth.path,
      note: "Teacher v3 proof-surface design note that keeps shipped truth distinct from target-state proposal surfaces.",
    },
  ];

  const bundleArtifacts = [
    {
      id: "teacher-v3-proof-summary",
      state: "target",
      kind: "proposal_truth",
      source: bundlePaths.summary,
      note: "bounded human summary",
    },
    {
      id: "teacher-v3-proof-status",
      state: "target",
      kind: "proposal_truth",
      source: bundlePaths.status,
      note: `thin machine status with canary rollout and gate matrix surfaced as ${canaryRolloutNote}`,
    },
    {
      id: "teacher-v3-proof-surface-map",
      state: "target",
      kind: "proposal_truth",
      source: bundlePaths.surfaceMap,
      note: "shipped-vs-target inventory",
    },
    {
      id: "teacher-v3-proof-proposal-report",
      state: "target",
      kind: "proposal_truth",
      source: bundlePaths.proposalReport,
      note: [
        `machine-readable proposal report with canary rollout and explicit gate matrix coverage surfaced as ${canaryRolloutNote}`,
        migrationComparisonNote,
      ].filter(Boolean).join("; "),
    },
    {
      id: "teacher-v3-proof-verdict",
      state: "target",
      kind: "proposal_truth",
      source: bundlePaths.verdict,
      note: [
        `review verdict; canary rollout remains target-only/off/rollback-bound (${canaryRolloutNote})`,
        migrationComparisonNote,
      ].filter(Boolean).join("; "),
    },
  ];

  return {
    contract: TEACHER_V3_PROOF_SURFACE_MAP_CONTRACT,
    bundleId: seed.bundleId,
    observedSurfaces,
    bundleArtifacts,
    counts: {
      observedSurfaceCount: observedSurfaces.length,
      shippedSurfaceCount: observedSurfaces.filter((surface) => surface.state === "shipped").length,
      targetSurfaceCount: bundleArtifacts.length,
      totalSurfaceCount: observedSurfaces.length + bundleArtifacts.length,
    },
  };
}

function buildProposalReport(seed, surfaceMap, bundlePaths, replayCapture) {
  const publicationSafeArtifacts = [
    {
      artifactId: "teacher-v3-proof-summary",
      kind: "summary",
      path: bundlePaths.summary,
      redactions: ["raw stdout/stderr", "secret-bearing values"],
      containsRawLogs: false,
    },
    {
      artifactId: "teacher-v3-proof-status",
      kind: "status",
      path: bundlePaths.status,
      redactions: ["raw stdout/stderr", "secret-bearing values"],
      containsRawLogs: false,
    },
    {
      artifactId: "teacher-v3-proof-surface-map",
      kind: "surface-map",
      path: bundlePaths.surfaceMap,
      redactions: ["raw source payloads"],
      containsRawLogs: false,
    },
    {
      artifactId: "teacher-v3-proof-proposal-report",
      kind: "metadata",
      path: bundlePaths.proposalReport,
      redactions: ["raw stdout/stderr", "secret-bearing values"],
      containsRawLogs: false,
    },
    {
      artifactId: "teacher-v3-proof-verdict",
      kind: "verdict",
      path: bundlePaths.verdict,
      redactions: ["raw stdout/stderr", "secret-bearing values"],
      containsRawLogs: false,
    },
  ];

  const runtimeTruth = seed.runtimeTruth;
  const proofTruth = seed.proofTruth;
  const docsTruth = seed.docsTruth;
  const replayRecommendation = seed.shadowReplaySummary?.proposalClass === "mutation"
    ? "Keep mutation replay on the candidate graph and verify rollback before considering canary activation."
    : seed.shadowReplaySummary?.proposalClass === "forgetting"
      ? "Keep forgetting replay on the retention state machine and preserve the hard-delete guardrail before considering canary activation."
      : "Thread candidate-state replay and rollback binding in the next tranche before considering canary activation.";
  const migrationRecommendation = seed.routerMigrationComparison
    ? seed.routerMigrationComparison.comparison.decision === "promote"
      ? "Mixed router migration passed with explicit correction protection and rollback binding; keep the proof bundle and rollback handle attached to the promoted path."
      : seed.routerMigrationComparison.explicitCorrectionProtection.blockers.length > 0
        ? "Hold the existing live policy; the mixed candidate regressed explicit corrections and must not be promoted."
        : "Hold the existing live policy until the mixed candidate wins the historical replay set and preserves user value."
    : null;
  const recommendations = normalizeArray(seed.proposalRecord?.recommendations).length > 0
    ? normalizeArray(seed.proposalRecord.recommendations)
    : [
      seed.gate1Seam.present
        ? "Preserve the persisted proposal record seam and load it directly once Gate 1 lands."
        : "Add Gate 1 proposal persistence so the synthetic runtime-capture record can be replaced by a stored proposal row.",
      "Keep the bundle publication-safe and bounded; never spill raw logs into the target-state artifacts.",
      replayRecommendation,
      ...(migrationRecommendation ? [migrationRecommendation] : []),
    ];

  const replayGateMatrix = describeTeacherProposalReplayGateMatrixV1({
    proposalId: seed.proposalId,
    proposalClass: seed.proposalClass,
    rollbackKey: seed.rollbackKey,
    proofBundleId: seed.bundleId,
    replaySummaryId: seed.replaySummary?.replayId ?? null,
    replaySummary: seed.replaySummary
      ? {
        status: seed.replaySummary.status,
        reviewMode: seed.replaySummary.reviewMode,
        beforeScore: seed.replaySummary.beforeScore,
        afterScore: seed.replaySummary.afterScore,
        scoreDelta: seed.replaySummary.scoreDelta,
        candidatePackId: seed.replaySummary.candidatePackId,
        candidatePackVersion: seed.replaySummary.candidatePackVersion,
        summary: seed.replaySummary.summary,
      }
      : null,
    replayOutcomeSummary: replayCapture.summary,
    shadowReplay: seed.shadowReplaySummary,
    replaySuites: seed.replaySuites,
    surfaceMap: surfaceMap.observedSurfaces,
    evidenceLinks: seed.evidence,
    counterevidenceLinks: seed.counterevidence,
  });

  return {
    contract: TEACHER_V3_PROPOSAL_REPORT_CONTRACT,
    bundleId: seed.bundleId,
    proposal: {
      proposalId: seed.proposalId,
      proposalLane: seed.proposalLane,
      proposalClass: seed.proposalClass,
      status: seed.status,
      reviewMode: seed.proposalReviewMode,
      lineage: seed.lineage,
      subjectIds: seed.subjectIds,
      rollbackKey: seed.rollbackKey,
      replaySuites: seed.replaySuites,
      replayOutcomeCount: replayCapture.summary.replayOutcomeCount,
      canaryRollout: seed.canaryRollout,
      confidence: seed.confidence,
      recordSource: seed.recordSource,
      replaySummary: seed.replaySummary,
      canaryActivationGuard: seed.canaryActivationGuard,
    },
    replayGateMatrix,
    replayGate: {
      proposalClass: seed.proposalClass,
      reviewMode: seed.proposalReviewMode,
      dimensions: {
        truthInvariants: {
          name: "truth_invariants",
          summary: "Keep derived output subordinate to explicit authority.",
          requirements: [
            "Explicit correction memory still outranks teacher synthesis.",
            "Mixed router migration cannot override preserved explicit corrections.",
            "The live path stays read-only to the proposal.",
            "Evidence refs stay attached to any non-trivial claim.",
          ],
        },
        attributionFloor: {
          name: "attribution_floor",
          summary: "Every proposed change needs clear evidence coverage.",
          requirements: [
            "Every proposal carries durable evidence refs.",
            "Source ids must be stable record ids, not display labels.",
            "Unattributed payload stays out of promotion.",
          ],
        },
        boundedness: {
          name: "boundedness",
          summary: "Keep the reviewable surface compact and inspectable.",
          requirements: [
            "Proposal subject sets stay finite and small.",
            "Payloads avoid raw corpus dumps and unbounded excerpts.",
            "Replay fits inside a single review pass.",
          ],
        },
        reversibility: {
          name: "reversibility",
          summary: "Preserve rollback and replay identity.",
          requirements: [
            "RollbackKey identifies the reversible path.",
            "Prior state remains recoverable for replay.",
            "Rollback retains the prior live router, user value, and proof references.",
            "Rejected or superseded proposals keep lineage.",
          ],
        },
      },
    },
    runtimeTruth,
    proofTruth,
    docsTruth,
    shadowReplay: seed.shadowReplaySummary,
    surfaceMap: surfaceMap.observedSurfaces,
    surfaceCounts: surfaceMap.counts,
    evidenceLinks: seed.evidence,
    counterevidenceLinks: seed.counterevidence,
    replaySummary: seed.replaySummary,
    routerMigrationComparison: seed.routerMigrationComparison,
    canaryRollout: seed.canaryRollout,
    recommendations,
    gate1Seam: seed.gate1Seam,
    publicationSafeArtifacts,
    createdAt: seed.createdAt,
    updatedAt: seed.updatedAt ?? seed.resolvedAt,
    runtimeTruthSummary: {
      currentPackVersion: runtimeTruth.currentPackVersion,
      serveState: runtimeTruth.serveState,
      teacherConfigured: runtimeTruth.teacherConfigured,
      operatorHealthStatus: runtimeTruth.operatorHealth.status,
      learningHealthStatus: runtimeTruth.learningHealth.status,
      routeTraceCount: runtimeTruth.routeTraceCount,
      supervisionCount: runtimeTruth.supervisionCount,
      lastReplayGateVerdict: runtimeTruth.lastReplayGateVerdict.verdict,
      lastPromotionVerdict: runtimeTruth.lastPromotionVerdict.verdict,
    },
    proofTruthSummary: {
      verdict: proofTruth.verdict,
      severity: proofTruth.severity,
      runtimeLoadProofExists: proofTruth.runtimeLoadProofExists,
      postBundleCount: proofTruth.postBundleCount,
      stepCount: proofTruth.stepCount,
    },
    replayOutcomes: replayCapture.outcomes,
    replayOutcomeSummary: replayCapture.summary,
    routerMigrationComparison: seed.routerMigrationComparison,
  };
}

function buildStatusReport(seed, surfaceMap, proposalReport) {
  return {
    contract: TEACHER_V3_PROOF_STATUS_CONTRACT,
    bundleId: seed.bundleId,
    proposalId: seed.proposalId,
    proposalLane: seed.proposalLane,
    proposalClass: seed.proposalClass,
    proposalStatus: seed.status,
    reviewMode: seed.proposalReviewMode,
    surfaceCounts: surfaceMap.counts,
    runtimeTruth: proposalReport.runtimeTruthSummary,
    proofTruth: proposalReport.proofTruthSummary,
    docsTruth: proposalReport.docsTruth,
    replayOutcomeSummary: proposalReport.replayOutcomeSummary,
    replaySummary: proposalReport.replaySummary,
    gateMatrix: proposalReport.replayGateMatrix,
    routerMigrationComparison: proposalReport.routerMigrationComparison,
    canaryRollout: proposalReport.proposal.canaryRollout,
    canaryActivationGuard: proposalReport.proposal.canaryActivationGuard,
    gate1Seam: seed.gate1Seam,
    shadowReplay: proposalReport.shadowReplay
      ? {
        proposalClass: proposalReport.shadowReplay.proposalClass,
        reviewMode: proposalReport.shadowReplay.reviewMode,
        shadowOnly: proposalReport.shadowReplay.shadowOnly,
        rollbackKey: proposalReport.shadowReplay.rollbackKey,
        summary: proposalReport.shadowReplay.summary,
      }
      : null,
    recommendations: proposalReport.recommendations,
    publicationSafeArtifacts: proposalReport.publicationSafeArtifacts.map((artifact) => ({
      artifactId: artifact.artifactId,
      kind: artifact.kind,
      path: artifact.path,
      redactions: artifact.redactions,
      containsRawLogs: artifact.containsRawLogs,
    })),
    createdAt: seed.createdAt,
    updatedAt: seed.updatedAt ?? seed.resolvedAt ?? seed.createdAt,
  };
}

function buildVerdictReport(seed, statusReport, proposalReport) {
  const proofSeverity = proposalReport.proofTruthSummary.severity;
  const runtimeReady = statusReport.runtimeTruth.currentPackVersion !== null
    || statusReport.runtimeTruth.serveState !== null
    || statusReport.runtimeTruth.teacherConfigured !== null;
  const proofReady = proposalReport.proofTruthSummary.runtimeLoadProofExists !== null;
  const migrationComparison = proposalReport.routerMigrationComparison;
  const migrationBlocked = migrationComparison?.comparison.blocked === true;
  const blocking = !runtimeReady || !proofReady || migrationBlocked;
  const shadowReplay = proposalReport.shadowReplay;
  const shadowReplayVerdict = shadowReplay
    ? {
      proposalClass: shadowReplay.proposalClass,
      reviewMode: shadowReplay.reviewMode,
      shadowOnly: shadowReplay.shadowOnly,
      promotionBypass: seed.shadowReplaySummary?.promotionBypass ?? false,
      rollbackKey: shadowReplay.rollbackKey,
      summary: shadowReplay.summary,
      replayOutcome: seed.shadowReplaySummary?.replayOutcome ?? null,
      rollbackRestored: seed.shadowReplaySummary?.rollback?.restored ?? null,
    }
    : null;

  return {
    contract: TEACHER_V3_PROOF_VERDICT_CONTRACT,
    bundleId: seed.bundleId,
    verdict: blocking ? "rejected" : "reviewable",
    severity: blocking ? "blocking" : proofSeverity === "blocking" ? "warn" : "info",
    why: blocking
      ? !runtimeReady || !proofReady
        ? "runtime or proof truth could not be summarized into a bounded bundle"
        : migrationComparison?.comparison.blockers?.length > 0
          ? `router migration comparison is blocked: ${migrationComparison.comparison.blockers.join("; ")}`
          : "router migration comparison could not promote the mixed policy"
      : shadowReplay
        ? seed.gate1Seam.present
          ? `runtime, proof, and docs truth were summarized; ${shadowReplay.proposalClass} replay stayed shadow-only with explicit rollback semantics and the record can still be loaded from storage (${proposalReport.replayOutcomeSummary.summary})`
          : `runtime, proof, and docs truth were summarized; ${shadowReplay.proposalClass} replay stayed shadow-only with explicit rollback semantics and no promotion bypass (${proposalReport.replayOutcomeSummary.summary})`
        : migrationComparison
          ? migrationComparison.comparison.decision === "promote"
            ? `runtime, proof, and docs truth were summarized; mixed router migration replay passed with explicit correction protection and rollback binding (${migrationComparison.comparison.summary})`
            : `runtime, proof, and docs truth were summarized; mixed router migration remains holdback-only because ${migrationComparison.comparison.summary}`
          : seed.gate1Seam.present
          ? `runtime, proof, and docs truth were summarized; Gate 1 persistence is already wired so the record can be loaded from storage (${proposalReport.replayOutcomeSummary.summary})`
          : `runtime, proof, and docs truth were summarized; Gate 1 persistence is still pending so the bundle remains a derived review surface (${proposalReport.replayOutcomeSummary.summary})`,
    reviewMode: seed.proposalReviewMode,
    gate1Seam: seed.gate1Seam,
    replayOutcomeSummary: proposalReport.replayOutcomeSummary,
    routerMigrationComparison: migrationComparison,
    shadowReplay: shadowReplayVerdict,
    canaryRollout: proposalReport.proposal.canaryRollout,
    canaryActivationGuard: proposalReport.proposal.canaryActivationGuard,
    blockers: blocking ? [
      !runtimeReady ? "missing runtime truth summary" : null,
      !proofReady ? "missing proof truth summary" : null,
      ...(migrationBlocked ? migrationComparison.comparison.blockers : []),
    ].filter(Boolean) : [],
    targetStateOnly: !seed.gate1Seam.present,
    createdAt: seed.createdAt,
    updatedAt: seed.updatedAt ?? seed.resolvedAt ?? seed.createdAt,
  };
}

function buildSummaryMarkdown(seed, statusReport, verdictReport, bundlePaths) {
  const lines = [
    "# Teacher v3 proof bundle",
    "",
    `- bundle: \`${seed.bundleId}\``,
    `- proposal: \`${seed.proposalId}\` (${seed.proposalClass}, ${seed.status})`,
    `- review mode: **${seed.proposalReviewMode}**`,
    ...(seed.replaySummary ? [
      `- replay status: **${seed.replaySummary.status}**`,
      `- replay score: ${seed.replaySummary.beforeScore.toFixed(3)} → ${seed.replaySummary.afterScore.toFixed(3)} (Δ ${(seed.replaySummary.scoreDelta).toFixed(3)})`,
      `- candidate pack: ${seed.replaySummary.candidatePackId ?? seed.replaySummary.candidatePackVersion ?? "unbound"}`,
    ] : []),
    `- verdict: **${verdictReport.verdict}**`,
    `- severity: **${verdictReport.severity}**`,
    `- runtime truth: \`${seed.runtimeStatusCommand}\``,
    `- proof truth: \`${seed.operatorProofCommand}\``,
    `- docs truth: \`${seed.docsTruth.path}\``,
    "",
    "## Canary rollout",
    `- surface state: ${statusReport.canaryRollout?.surfaceState ?? "target"}`,
    `- rollout mode: ${statusReport.canaryRollout?.rolloutMode ?? "off"}`,
    `- enabled: ${statusReport.canaryRollout?.enabled ? "yes" : "no"}`,
    `- disabled by default: ${statusReport.canaryRollout?.disabledByDefault ? "yes" : "no"}`,
    `- rollback bound: ${statusReport.canaryRollout?.rollbackBound ? "yes" : "no"}`,
    `- rollback key: \`${statusReport.canaryRollout?.rollbackKey ?? seed.rollbackKey}\``,
    `- candidate pack: ${statusReport.canaryRollout?.candidatePackId ?? statusReport.canaryRollout?.candidatePackVersion ?? "unbound"}`,
    `- binding: ${statusReport.canaryRollout?.bindingSummary ?? "rollback-bound and off by default"}`,
    `- guardrails: ${Array.isArray(statusReport.canaryRollout?.guardrails) ? statusReport.canaryRollout.guardrails.join("; ") : "none"}`,
    "",
    "## Surface counts",
    `- shipped surfaces: ${statusReport.surfaceCounts.shippedSurfaceCount}`,
    `- target bundle artifacts: ${statusReport.surfaceCounts.targetSurfaceCount}`,
    `- total referenced surfaces: ${statusReport.surfaceCounts.totalSurfaceCount}`,
    "",
    "## Replay outcomes",
    `- captured outcomes: ${statusReport.replayOutcomeSummary.replayOutcomeCount}`,
    `- replay suites: ${statusReport.replayOutcomeSummary.replaySuites.length > 0 ? statusReport.replayOutcomeSummary.replaySuites.join(", ") : "none"}`,
    `- results: pass=${statusReport.replayOutcomeSummary.resultCounts.pass}, warn=${statusReport.replayOutcomeSummary.resultCounts.warn}, fail=${statusReport.replayOutcomeSummary.resultCounts.fail}`,
    `- review modes: promotable=${statusReport.replayOutcomeSummary.reviewModeCounts.promotable}, shadow_only=${statusReport.replayOutcomeSummary.reviewModeCounts.shadow_only}`,
    `- sources: proposal_record=${statusReport.replayOutcomeSummary.sourceCounts.proposal_record}, proof_bundle=${statusReport.replayOutcomeSummary.sourceCounts.proof_bundle}, derived=${statusReport.replayOutcomeSummary.sourceCounts.derived}`,
    "",
    "## Gate matrix",
    ...gateMatrixMarkdownLines(statusReport.gateMatrix),
    "",
    ...(statusReport.routerMigrationComparison ? [
      "## Router migration comparison",
      ...routerMigrationMarkdownLines(statusReport.routerMigrationComparison),
      "",
    ] : []),
    "## Canary rollout",
    `- rollout mode: ${statusReport.canaryRollout?.rolloutMode ?? "off"}`,
    `- enabled: ${statusReport.canaryRollout?.enabled ? "yes" : "no"}`,
    `- candidate pack: ${statusReport.canaryRollout?.candidatePackId ?? statusReport.canaryRollout?.candidatePackVersion ?? "unbound"}`,
    `- activation: ${statusReport.canaryActivationGuard?.blocked ? "blocked" : statusReport.canaryActivationGuard?.requested ? "permitted" : "off by default"}`,
    statusReport.canaryActivationGuard?.summary ? `- guard summary: ${statusReport.canaryActivationGuard.summary}` : null,
    statusReport.canaryActivationGuard?.blockers?.length ? `- blockers: ${statusReport.canaryActivationGuard.blockers.join(", ")}` : null,
    "",
    "## Gate 1 seam",
    `- present: ${seed.gate1Seam.present ? "yes" : "no"}`,
    `- record source: ${seed.gate1Seam.recordSource}`,
    `- note: ${seed.gate1Seam.note}`,
    "",
  ];

  if (seed.shadowReplaySummary) {
    lines.push(
      "## Shadow replay",
      `- proposal class: \`${seed.shadowReplaySummary.proposalClass}\``,
      `- review mode: **${seed.shadowReplaySummary.reviewMode}**`,
      `- promotion bypass: ${seed.shadowReplaySummary.promotionBypass ? "yes" : "no"}`,
      `- rollback key: \`${seed.shadowReplaySummary.rollbackKey ?? seed.rollbackKey}\``,
      ...shadowReplayMarkdownLines(seed.shadowReplaySummary),
      "",
    );
  }

  lines.push(
    "## Publication-safe artifacts",
    ...proposalReportArtifactsLines(bundlePaths),
    "",
    "## Recommendations",
    ...statusReport.recommendations.map((item) => `- ${item}`),
  );

  return `${lines.join("\n")}\n`;
}

function proposalReportArtifactsLines(bundlePaths) {
  return [
    `- \`${bundlePaths.summary}\` — bounded human summary`,
    `- \`${bundlePaths.status}\` — thin machine status`,
    `- \`${bundlePaths.surfaceMap}\` — shipped-vs-target inventory`,
    `- \`${bundlePaths.proposalReport}\` — machine-readable proposal report with gate matrix coverage`,
    `- \`${bundlePaths.verdict}\` — review verdict`,
  ];
}

function routerMigrationMarkdownLines(migrationComparison) {
  if (!migrationComparison) {
    return [];
  }

  return [
    `- decision: **${migrationComparison.comparison.decision}**`,
    `- winner: \`${migrationComparison.comparison.winner}\``,
    `- support ratios: old_live=${migrationComparison.comparison.supportRatios.old_live?.toFixed?.(3) ?? "n/a"}, base_only=${migrationComparison.comparison.supportRatios.base_only?.toFixed?.(3) ?? "n/a"}, mixed=${migrationComparison.comparison.supportRatios.mixed?.toFixed?.(3) ?? "n/a"}`,
    `- old_live: ${migrationComparison.variants.old_live.summary}`,
    `- base_only: ${migrationComparison.variants.base_only.summary}`,
    `- mixed: ${migrationComparison.variants.mixed.summary}`,
    `- explicit correction protection: ${migrationComparison.explicitCorrectionProtection.protected ? "yes" : "no"} (${migrationComparison.explicitCorrectionProtection.summary})`,
    `- rollback: ${migrationComparison.rollback.available ? "available" : "incomplete"}; key=${migrationComparison.rollback.rollbackKey ?? "missing"}`,
    `- proof bundle expectations: ${migrationComparison.proofBundleExpectations.summary}`,
    migrationComparison.comparison.blockers.length > 0 ? `- blockers: ${migrationComparison.comparison.blockers.join(", ")}` : null,
  ].filter(Boolean);
}

function shadowReplayMarkdownLines(shadowReplay) {
  if (!shadowReplay) {
    return [];
  }

  if (shadowReplay.proposalClass === "mutation") {
    const before = shadowReplay.before ?? {};
    const after = shadowReplay.after ?? {};
    const rollback = shadowReplay.rollback ?? {};
    return [
      `- candidate graph: ${before.nodeCount ?? "?"} nodes / ${before.edgeCount ?? "?"} edges → ${after.nodeCount ?? "?"} nodes / ${after.edgeCount ?? "?"} edges`,
      `- applications: ${Array.isArray(shadowReplay.applications) ? shadowReplay.applications.length : 0}`,
      `- replay outcome: ${shadowReplay.replayOutcome ?? "n/a"}`,
      `- rollback: ${rollback.restored === true ? "restored to base graph" : "not restored"}`,
      rollback.summary ? `- rollback summary: ${rollback.summary}` : null,
    ].filter(Boolean);
  }

  const before = shadowReplay.before ?? {};
  const after = shadowReplay.after ?? {};
  const rollback = shadowReplay.rollback ?? {};
  return [
    `- retention state: ${before.retentionState ?? "?"} → ${after.retentionState ?? "?"}`,
    `- target source: ${shadowReplay.target?.sourceId ?? "?"} (${shadowReplay.target?.authority ?? "?"})`,
    `- requested transition: ${shadowReplay.requestedTransition ?? "n/a"}`,
    `- replay outcome: ${shadowReplay.replayOutcome ?? "n/a"}`,
    shadowReplay.guardrail ? `- guardrail: ${shadowReplay.guardrail}` : null,
    shadowReplay.reason ? `- reason: ${shadowReplay.reason}` : null,
    `- rollback: ${rollback.restored === true ? "restored prior retention state" : "not restored"}`,
    rollback.summary ? `- rollback summary: ${rollback.summary}` : null,
  ].filter(Boolean);
}

function gateMatrixMarkdownLines(gateMatrix) {
  if (!gateMatrix) {
    return [];
  }

  return [
    `- proposal class: \`${gateMatrix.proposalClass}\``,
    `- review mode: **${gateMatrix.reviewMode}**`,
    `- rollback key: \`${gateMatrix.rollbackKey ?? "missing"}\``,
    `- proof bundle id: \`${gateMatrix.proofBundleId ?? "missing"}\``,
    `- replay summary id: \`${gateMatrix.replaySummaryId ?? "missing"}\``,
    ...gateMatrix.rows.flatMap((row) => [
      `- ${row.name}: **${row.status}** (${row.coverage}) — ${row.summary}`,
      `  - evidence: ${row.evidenceSurfaceIds.length > 0 ? row.evidenceSurfaceIds.join(", ") : "none"}`,
      `  - proof: ${row.proofSurfaceIds.length > 0 ? row.proofSurfaceIds.join(", ") : "none"}`,
      row.notes.length > 0 ? `  - notes: ${row.notes.join("; ")}` : null,
    ].filter(Boolean)),
  ];
}

export function resolveTeacherV3ProofOutputDir({ outputDir = null, bundleStartedAt = new Date() } = {}) {
  if (typeof outputDir === "string" && outputDir.trim().length > 0) {
    return path.resolve(outputDir);
  }
  const startedAt = bundleStartedAt instanceof Date ? bundleStartedAt : new Date(bundleStartedAt);
  return path.join(
    DEFAULT_TEACHER_V3_PROOF_BUNDLE_PARENT,
    `teacher-v3-proof-${timestampToken(Number.isNaN(startedAt.getTime()) ? new Date() : startedAt)}`,
  );
}

export function buildTeacherV3ProofBundle(input) {
  const bundleStartedAt = input.bundleStartedAt instanceof Date
    ? input.bundleStartedAt.toISOString()
    : normalizeText(input.bundleStartedAt) ?? new Date().toISOString();
  const outputDir = path.resolve(input.outputDir ?? resolveTeacherV3ProofOutputDir({ bundleStartedAt }));
  const bundleId = normalizeText(input.bundleId) ?? path.basename(outputDir);
  const bundlePaths = {
    summary: relativeWorkspacePath(path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.summary)),
    status: relativeWorkspacePath(path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.status)),
    surfaceMap: relativeWorkspacePath(path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.surfaceMap)),
    proposalReport: relativeWorkspacePath(path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.proposalReport)),
    verdict: relativeWorkspacePath(path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.verdict)),
  };

  const seed = buildTeacherV3ProposalSeed({
    ...input,
    bundleId,
    bundleStartedAt,
  });
  const replayCapture = captureTeacherV3ReplayOutcomes({
    bundleId,
    proposalId: seed.proposalId,
    proposalClass: seed.proposalClass,
    reviewMode: seed.proposalReviewMode,
    replaySuites: seed.replaySuites,
    proofVerdict: seed.proofTruth,
    replayOutcomes: input.proposalRecord?.replayOutcomes,
    bundleStartedAt,
  });
  const surfaceMap = buildSurfaceMap(seed, bundlePaths);
  const proposalReport = buildProposalReport(seed, surfaceMap, bundlePaths, replayCapture);
  const statusReport = buildStatusReport(seed, surfaceMap, proposalReport);
  const verdictReport = buildVerdictReport(seed, statusReport, proposalReport);
  const summaryMarkdown = buildSummaryMarkdown(seed, statusReport, verdictReport, bundlePaths);

  return {
    bundleId,
    bundleStartedAt,
    outputDir,
    bundlePaths,
    seed,
    surfaceMap,
    proposalReport,
    statusReport,
    verdictReport,
    summaryMarkdown,
    files: {
      [TEACHER_V3_PROOF_BUNDLE_LAYOUT.summary]: summaryMarkdown,
      [TEACHER_V3_PROOF_BUNDLE_LAYOUT.status]: renderJson(statusReport),
      [TEACHER_V3_PROOF_BUNDLE_LAYOUT.surfaceMap]: renderJson(surfaceMap),
      [TEACHER_V3_PROOF_BUNDLE_LAYOUT.proposalReport]: renderJson(proposalReport),
      [TEACHER_V3_PROOF_BUNDLE_LAYOUT.verdict]: renderJson(verdictReport),
    },
  };
}

export function writeTeacherV3ProofBundle(outputDir, bundle) {
  ensureDir(outputDir);
  for (const [fileName, content] of Object.entries(bundle.files)) {
    writeFileSync(path.join(outputDir, fileName), content, "utf8");
  }
  return {
    outputDir,
    writtenFiles: Object.keys(bundle.files).map((fileName) => path.join(outputDir, fileName)),
  };
}

export function buildTeacherV3ProofBundleDigest(bundle) {
  return {
    bundleId: bundle.bundleId,
    files: Object.fromEntries(
      Object.entries(bundle.files).map(([fileName, content]) => [fileName, sha256Text(content)]),
    ),
  };
}
