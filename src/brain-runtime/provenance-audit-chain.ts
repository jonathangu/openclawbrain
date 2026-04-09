/**
 * Provenance audit chain for the OpenClawBrain operator-proof path.
 *
 * This is a read-only stitching surface: it composes the shipped serve-decision
 * / route-row truth with attribution, learning-update, and promotion/proof truth
 * without changing policy or teacher semantics.
 */

import type { DecisionTraceSelectionMetadataV4 } from "../brain-core/types.js";
import type { DecisionTrace } from "../brain-core/types.js";
import type { RecentDecisionTraceSummary } from "../brain-core/trace.js";
import {
  materializeRouteDecisionRowsFromTraceV1,
  summarizeRouteDecisionRowV1,
  type RouteDecisionRowSummaryV1,
  type RouteDecisionRowV1,
} from "../brain-core/route-rows.js";
import type { AttributionTruthSummary } from "../live-runtime-audit.js";

export const PROVENANCE_AUDIT_CHAIN_CONTRACT = "openclawbrain_provenance_audit_chain.v1" as const;
export const PROVENANCE_AUDIT_CHAIN_MAX_ROUTE_ROWS = 3;
export const PROVENANCE_AUDIT_CHAIN_MAX_UPDATE_DECISIONS = 3;
export const PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS = 180;

const PRECEDENCE_LABEL = "user_explicit correction > raw_source > teacher_inference";

function normalizeText(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function truncateText(value: string | null | undefined, maxChars = PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS): string | null {
  const text = normalizeText(value);
  if (text === null) {
    return null;
  }
  if (text.length <= maxChars) {
    return text;
  }
  return `${text.slice(0, Math.max(0, maxChars - 1))}…`;
}

function normalizeNumber(value: unknown): number | null {
  return Number.isFinite(value) ? Number(value) : null;
}

function normalizeBoolean(value: unknown): boolean | null {
  return value === true ? true : value === false ? false : null;
}

function normalizeArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function uniqueStrings(values: Array<string | null | undefined>, limit = 5): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const text = normalizeText(value);
    if (!text || seen.has(text)) {
      continue;
    }
    seen.add(text);
    out.push(text);
    if (out.length >= limit) {
      break;
    }
  }
  return out;
}

function summarizeRouteRow(row: RouteDecisionRowV1): RouteDecisionRowSummaryV1 & {
  precedenceLabel: string;
  detail: string;
} {
  const summary = summarizeRouteDecisionRowV1(row);
  const provenanceDetail = truncateText(row.label_provenance.detail, 96) ?? "n/a";
  return {
    ...summary,
    precedenceLabel: PRECEDENCE_LABEL,
    detail: truncateText(
      [
        `chosen=${summary.chosenActionKind}`,
        `stop=${summary.stopLabel}`,
        `provenance=${summary.provenanceState}/${summary.provenanceBasis}`,
        `detail=${provenanceDetail}`,
      ].join("; "),
      PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS,
    ) ?? "n/a",
  };
}

function toRouteTraceSnapshot(trace: DecisionTraceSelectionMetadataV4 | null | undefined, bundleId: string): DecisionTrace | null {
  const snapshots = normalizeArray(trace?.decisionPointSnapshots) as Array<Record<string, unknown>>;
  if (snapshots.length === 0) {
    return null;
  }

  const firstSnapshot = snapshots[0] ?? {};
  const routeContext = (firstSnapshot.routeContext ?? null) as Record<string, unknown> | null;
  const traceId = normalizeText(firstSnapshot.traceId) ?? normalizeText(firstSnapshot.decisionPointId) ?? `trace:${bundleId}`;
  const episodeId = normalizeText(firstSnapshot.episodeId) ?? null;
  const selectedPathNodeIds = Array.isArray(routeContext?.selectedPathNodeIds)
    ? [...routeContext.selectedPathNodeIds as string[]]
    : Array.isArray(routeContext?.selectedNodeIds)
      ? [...routeContext.selectedNodeIds as string[]]
      : [];
  const selectedTraversalNodeIds = Array.isArray(routeContext?.selectedTraversalNodeIds)
    ? [...routeContext.selectedTraversalNodeIds as string[]]
    : [...selectedPathNodeIds];
  const selectedSeedNodeIds = Array.isArray(routeContext?.selectedSeedNodeIds)
    ? [...routeContext.selectedSeedNodeIds as string[]]
    : [];
  const candidateNodeIds = Array.isArray(routeContext?.candidateNodeIds)
    ? [...routeContext.candidateNodeIds as string[]]
    : [];

  return {
    id: traceId,
    episodeId,
    packVersion: null,
    queryText: normalizeText(trace?.compileReportSummary) ?? normalizeText(trace?.decisionPointSummary) ?? `provenance audit chain ${bundleId}`,
    seedScores: [],
    trajectory: [],
    firedNodes: [],
    vetoedNodes: [],
    contextChars: normalizeNumber(trace?.queryChars) ?? 0,
    footer: "",
    routeTrace: {
      requestDigest: normalizeText(routeContext?.requestDigest) ?? traceId,
      conversationId: normalizeNumber(firstSnapshot.conversationId) ?? null,
      agentIdentity: null,
      activePackId: normalizeText(routeContext?.activePackId) ?? null,
      routerIdentity: normalizeText(routeContext?.routerIdentity) ?? null,
      candidateNodeIds,
      selectedNodeIds: selectedPathNodeIds,
      selectedTraversalNodeIds,
      selectedPathNodeIds,
      selectedSeedNodeIds,
      branchOutcomes: [],
      injectedNodeSummaries: [],
      sourceSummary: {
        injectedCount: 0,
        kinds: {},
        trusts: {},
        sourceUris: [],
        sourceRefs: [],
      },
      selectionMetadata: trace ?? null,
    },
    createdAt: Date.now(),
  } as DecisionTrace;
}

function normalizeUpdateDecision(decision: unknown): {
  episodeId: string | null;
  status: string | null;
  reason: string | null;
  routeUpdateCount: number | null;
  traceIds: string[];
  observationIds: string[];
  supervisionIds: string[];
  baselineBefore: number | null;
  baselineAfter: number | null;
  summary: string;
} | null {
  if (!decision || typeof decision !== "object") {
    return null;
  }

  const record = decision as Record<string, unknown>;
  const traceIds = uniqueStrings(normalizeArray(record.traceIds).map((value) => normalizeText(value as string)), 3);
  const observationIds = uniqueStrings(normalizeArray(record.observationIds).map((value) => normalizeText(value as string)), 3);
  const supervisionIds = uniqueStrings(normalizeArray(record.supervisionIds).map((value) => normalizeText(value as string)), 3);
  const status = normalizeText(record.status);
  const reason = truncateText(normalizeText(record.reason), 120);
  const routeUpdateCount = normalizeNumber(record.routeUpdateCount);
  const baselineBefore = normalizeNumber(record.baselineBefore);
  const baselineAfter = normalizeNumber(record.baselineAfter);

  return {
    episodeId: normalizeText(record.episodeId),
    status,
    reason,
    routeUpdateCount,
    traceIds,
    observationIds,
    supervisionIds,
    baselineBefore,
    baselineAfter,
    summary: truncateText(
      [
        status ? `status=${status}` : null,
        routeUpdateCount !== null ? `route_updates=${routeUpdateCount}` : null,
        baselineBefore !== null && baselineAfter !== null
          ? `baseline=${baselineBefore.toFixed(3)}→${baselineAfter.toFixed(3)}`
          : null,
        reason ? `reason=${reason}` : null,
      ].filter(Boolean).join("; "),
      PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS,
    ) ?? "n/a",
  };
}

function normalizeTeacherUpdateCycle(updateCycle: unknown): {
  generatedAt: string | null;
  eligibleEpisodeCount: number | null;
  appliedEpisodeCount: number | null;
  skippedEpisodeCount: number | null;
  skippedReasons: Record<string, number>;
  decisions: ReturnType<typeof normalizeUpdateDecision>[];
  detail: string;
} | null {
  if (!updateCycle || typeof updateCycle !== "object") {
    return null;
  }

  const record = updateCycle as Record<string, unknown>;
  const decisions = normalizeArray(record.decisions)
    .slice(0, PROVENANCE_AUDIT_CHAIN_MAX_UPDATE_DECISIONS)
    .map((decision) => normalizeUpdateDecision(decision))
    .filter((decision): decision is NonNullable<typeof decision> => decision !== null);

  const generatedAt = normalizeText(record.generatedAt);
  const eligibleEpisodeCount = normalizeNumber(record.eligibleEpisodeCount);
  const appliedEpisodeCount = normalizeNumber(record.appliedEpisodeCount);
  const skippedEpisodeCount = normalizeNumber(record.skippedEpisodeCount);
  const skippedReasons = record.skippedReasons && typeof record.skippedReasons === "object"
    ? Object.fromEntries(
      Object.entries(record.skippedReasons as Record<string, unknown>)
        .filter(([, value]) => Number.isFinite(value))
        .map(([key, value]) => [key, Number(value)]),
    )
    : {};
  const detail = truncateText(normalizeText(record.detail) ?? null, PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS)
    ?? "no teacher update cycle is available";

  return {
    generatedAt,
    eligibleEpisodeCount,
    appliedEpisodeCount,
    skippedEpisodeCount,
    skippedReasons,
    decisions,
    detail,
  };
}

function normalizePromotionVerdict(value: unknown): { verdict: string | null; summary: string | null } | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const record = value as Record<string, unknown>;
  return {
    verdict: normalizeText(record.verdict),
    summary: truncateText(normalizeText(record.summary) ?? normalizeText(record.why), PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS),
  };
}

function normalizeRetrainLineage(value: unknown): {
  priorBaseArtifactId: string | null;
  priorBaseArtifactVersion: string | null;
  priorBaseArtifactChecksum: string | null;
  candidateArtifactId: string | null;
  candidateArtifactVersion: string | null;
  candidateArtifactChecksum: string | null;
  priorRooted: boolean | null;
  promotionValid: boolean | null;
  residualUpdateCount: number | null;
  summary: string | null;
} | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const record = value as Record<string, unknown>;
  const priorBaseArtifactId = normalizeText(record.priorBaseArtifactId);
  const priorBaseArtifactVersion = normalizeText(record.priorBaseArtifactVersion);
  const priorBaseArtifactChecksum = normalizeText(record.priorBaseArtifactChecksum);
  const candidateArtifactId = normalizeText(record.candidateArtifactId);
  const candidateArtifactVersion = normalizeText(record.candidateArtifactVersion);
  const candidateArtifactChecksum = normalizeText(record.candidateArtifactChecksum);
  const priorRooted = normalizeBoolean(record.priorRooted);
  const promotionValid = normalizeBoolean(record.promotionValid);
  const residualUpdateCount = normalizeNumber(record.residualUpdateCount);

  return {
    priorBaseArtifactId,
    priorBaseArtifactVersion,
    priorBaseArtifactChecksum,
    candidateArtifactId,
    candidateArtifactVersion,
    candidateArtifactChecksum,
    priorRooted,
    promotionValid,
    residualUpdateCount,
    summary: truncateText(normalizeText(record.summary), PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS),
  };
}

function normalizeProofTruth(value: unknown): {
  bundleDir: string | null;
  command: string | null;
  summary: string | null;
  verdict: string | null;
  severity: string | null;
  why: string | null;
  runtimeLoadProofPath: string | null;
  runtimeLoadProofExists: boolean | null;
  stepCount: number | null;
  postBundleCount: number | null;
} | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const record = value as Record<string, unknown>;
  const verdict = normalizePromotionVerdict(record.verdict);
  return {
    bundleDir: normalizeText(record.bundleDir),
    command: normalizeText(record.command),
    summary: truncateText(normalizeText(record.summary), PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS),
    verdict: verdict?.verdict ?? null,
    severity: normalizeText(record.verdict && typeof record.verdict === "object" ? (record.verdict as Record<string, unknown>).severity : null),
    why: truncateText(normalizeText(record.verdict && typeof record.verdict === "object" ? (record.verdict as Record<string, unknown>).why : null), PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS),
    runtimeLoadProofPath: normalizeText(record.runtimeLoadProofPath),
    runtimeLoadProofExists: normalizeBoolean(record.runtimeLoadProofExists),
    stepCount: normalizeNumber(record.stepCount),
    postBundleCount: normalizeNumber(record.postBundleCount),
  };
}

function summarizeRouteDecisionRows(
  selectionMetadata: DecisionTraceSelectionMetadataV4 | null | undefined,
  bundleId: string,
): Array<ReturnType<typeof summarizeRouteRow>> {
  const syntheticTrace = toRouteTraceSnapshot(selectionMetadata, bundleId);
  if (!syntheticTrace) {
    return [];
  }
  return materializeRouteDecisionRowsFromTraceV1({
    trace: syntheticTrace,
    routeFnVersion: normalizeText(selectionMetadata?.routerIdentity) ?? null,
    provenanceByDecisionPointId: {},
    defaultProvenance: null,
  })
    .slice(0, PROVENANCE_AUDIT_CHAIN_MAX_ROUTE_ROWS)
    .map((row) => summarizeRouteRow(row));
}

function summarizeAttributionTruth(attributionTruth: AttributionTruthSummary | null | undefined): {
  contract: string | null;
  visible: boolean | null;
  primaryState: string | null;
  detail: string | null;
  latest: {
    ambiguous: {
      observationId: string;
      episodeId: string;
      traceId: string | null;
      bindingMode: string;
      attributionQuality: string;
      feedbackRichness: string;
      confidence: number | null;
      reason: string | null;
      evaluatedAt: number | null;
      precedenceLabel: string;
    } | null;
    unmatched: {
      observationId: string;
      episodeId: string;
      traceId: string | null;
      bindingMode: string;
      attributionQuality: string;
      feedbackRichness: string;
      confidence: number | null;
      reason: string | null;
      evaluatedAt: number | null;
      precedenceLabel: string;
    } | null;
    followupPending: {
      observationId: string;
      episodeId: string;
      traceId: string | null;
      status: string;
      gate: string;
      reason: string;
      feedbackRichness: string;
      createdAt: number;
    } | null;
  };
  summary: string;
} {
  if (!attributionTruth) {
    return {
      contract: null,
      visible: null,
      primaryState: null,
      detail: null,
      latest: {
        ambiguous: null,
        unmatched: null,
        followupPending: null,
      },
      summary: "attribution truth is unavailable",
    };
  }

  const latestAmbiguous = attributionTruth.latest.ambiguous
    ? {
      ...attributionTruth.latest.ambiguous,
      precedenceLabel: PRECEDENCE_LABEL,
    }
    : null;
  const latestUnmatched = attributionTruth.latest.unmatched
    ? {
      ...attributionTruth.latest.unmatched,
      precedenceLabel: PRECEDENCE_LABEL,
    }
    : null;
  const followupPending = attributionTruth.latest.followupPending
    ? {
      ...attributionTruth.latest.followupPending,
    }
    : null;

  return {
    contract: attributionTruth.contract,
    visible: attributionTruth.visible,
    primaryState: attributionTruth.primaryState,
    detail: truncateText(attributionTruth.detail, PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS),
    latest: {
      ambiguous: latestAmbiguous,
      unmatched: latestUnmatched,
      followupPending,
    },
    summary: truncateText(
      attributionTruth.detail,
      PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS,
    ) ?? "attribution truth is not available",
  };
}

function summarizePromotionAndProofTruth(params: {
  promotionStory: unknown;
  lastPromotionReason: unknown;
  lastPromotionVerdict: unknown;
  lastReplayGateVerdict: unknown;
  retrainLineage: unknown;
  proofTruth: unknown;
}): {
  currentPackVersion: number | null;
  lastPromotionReason: string | null;
  lastPromotionVerdict: { verdict: string | null; summary: string | null } | null;
  lastReplayGateVerdict: { verdict: string | null; summary: string | null } | null;
  retrainLineage: ReturnType<typeof normalizeRetrainLineage>;
  proofTruth: ReturnType<typeof normalizeProofTruth>;
  summary: string;
} {
  const promotionStory = params.promotionStory && typeof params.promotionStory === "object"
    ? params.promotionStory as Record<string, unknown>
    : null;
  const currentPackVersion = normalizeNumber(promotionStory?.summary && typeof promotionStory.summary === "object"
    ? (promotionStory.summary as Record<string, unknown>).currentPackVersion
    : null);
  const lastPromotionVerdict = normalizePromotionVerdict(params.lastPromotionVerdict);
  const lastReplayGateVerdict = normalizePromotionVerdict(params.lastReplayGateVerdict);
  const retrainLineage = normalizeRetrainLineage(params.retrainLineage);
  const proofTruth = normalizeProofTruth(params.proofTruth);
  const lastPromotionReason = truncateText(normalizeText(params.lastPromotionReason), PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS);
  const summary = truncateText(
    [
      lastPromotionVerdict?.verdict ? `promotion=${lastPromotionVerdict.verdict}` : null,
      lastReplayGateVerdict?.verdict ? `replay_gate=${lastReplayGateVerdict.verdict}` : null,
      retrainLineage?.summary ? `lineage=${retrainLineage.summary}` : null,
      proofTruth?.verdict ? `proof=${proofTruth.verdict}` : null,
    ].filter(Boolean).join("; "),
    PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS,
  ) ?? "promotion/proof truth is not available";

  return {
    currentPackVersion,
    lastPromotionReason,
    lastPromotionVerdict,
    lastReplayGateVerdict,
    retrainLineage,
    proofTruth,
    summary,
  };
}

export interface ProvenanceAuditChainInputV1 {
  bundleId: string;
  generatedAt?: string | null;
  runtimeStatus: Record<string, unknown>;
  proofTruth: unknown;
}

export interface ProvenanceAuditChainV1 {
  contract: typeof PROVENANCE_AUDIT_CHAIN_CONTRACT;
  bundleId: string;
  generatedAt: string;
  precedence: {
    label: string;
    correctionAuthority: "user_explicit";
    rawAuthority: "raw_source";
    teacherSynthesis: "teacher_inference";
    note: string;
  };
  serveDecision: {
    traceId: string | null;
    episodeId: string | null;
    routeRowCount: number;
    routeRows: Array<ReturnType<typeof summarizeRouteRow>>;
    recentDecisionSummary: string | null;
    lastCompileReportSummary: string | null;
    summary: string;
  };
  attributionTruth: ReturnType<typeof summarizeAttributionTruth>;
  learningUpdate: {
    lastEvaluationCycle: unknown | null;
    lastUpdateCycle: ReturnType<typeof normalizeTeacherUpdateCycle>;
    queue: unknown | null;
    summary: string;
    precedenceLabel: string;
    linkedTraceIds: string[];
    linkedObservationIds: string[];
    linkedEpisodeIds: string[];
  };
  promotionProofTruth: ReturnType<typeof summarizePromotionAndProofTruth>;
  linkages: {
    traceId: string | null;
    routeRowIds: string[];
    observationIds: string[];
    updateEpisodeIds: string[];
    proofBundleDir: string | null;
    proofVerdict: string | null;
    restartSafe: boolean;
  };
  boundedness: {
    maxRouteRows: number;
    maxUpdateDecisions: number;
    maxTextChars: number;
    routeRowCount: number;
    routeRowSampleCount: number;
    updateDecisionCount: number;
    updateDecisionSampleCount: number;
  };
}

export function buildProvenanceAuditChainV1(input: ProvenanceAuditChainInputV1): ProvenanceAuditChainV1 {
  const runtimeStatus = input.runtimeStatus ?? {};
  const recentDecisionSummary = runtimeStatus.recentDecisionSummary && typeof runtimeStatus.recentDecisionSummary === "object"
    ? runtimeStatus.recentDecisionSummary as RecentDecisionTraceSummary
    : null;
  const lastTraceSelectionMetadata = runtimeStatus.lastTraceSelectionMetadata && typeof runtimeStatus.lastTraceSelectionMetadata === "object"
    ? runtimeStatus.lastTraceSelectionMetadata as DecisionTraceSelectionMetadataV4
    : null;
  const attributionTruth = summarizeAttributionTruth(
    runtimeStatus.attributionTruth && typeof runtimeStatus.attributionTruth === "object"
      ? runtimeStatus.attributionTruth as AttributionTruthSummary
      : null,
  );
  const teacherTruth = runtimeStatus.teacherTruth && typeof runtimeStatus.teacherTruth === "object"
    ? runtimeStatus.teacherTruth as Record<string, unknown>
    : null;
  const promotionStory = runtimeStatus.promotionStory ?? null;
  const continuousLearning = runtimeStatus.continuousLearning && typeof runtimeStatus.continuousLearning === "object"
    ? runtimeStatus.continuousLearning as Record<string, unknown>
    : null;
  const retrain = continuousLearning?.retrain && typeof continuousLearning.retrain === "object"
    ? continuousLearning.retrain as Record<string, unknown>
    : null;
  const lastUpdateCycle = normalizeTeacherUpdateCycle(teacherTruth?.lastUpdateCycle);
  const promotionProofTruth = summarizePromotionAndProofTruth({
    promotionStory,
    lastPromotionReason: runtimeStatus.lastPromotionReason,
    lastPromotionVerdict: runtimeStatus.lastPromotionVerdict,
    lastReplayGateVerdict: runtimeStatus.lastReplayGateVerdict,
    retrainLineage: retrain?.lineage ?? null,
    proofTruth: input.proofTruth,
  });
  const routeRows = summarizeRouteDecisionRows(lastTraceSelectionMetadata, input.bundleId);
  const traceId = routeRows[0]?.traceId ?? null;
  const episodeId = routeRows[0]?.decisionPointId ? (toRouteTraceSnapshot(lastTraceSelectionMetadata, input.bundleId)?.episodeId ?? null) : null;
  const rowIds = routeRows.map((row) => row.rowId);
  const linkedObservationIds = uniqueStrings(
    lastUpdateCycle?.decisions.flatMap((decision) => decision.observationIds) ?? [],
    PROVENANCE_AUDIT_CHAIN_MAX_UPDATE_DECISIONS,
  );
  const linkedTraceIds = uniqueStrings(
    [
      ...routeRows.map((row) => row.traceId),
      ...(lastUpdateCycle?.decisions.flatMap((decision) => decision.traceIds) ?? []),
    ],
    5,
  );
  const linkedEpisodeIds = uniqueStrings(
    [
      episodeId,
      ...(lastUpdateCycle?.decisions.map((decision) => decision.episodeId) ?? []),
    ],
    5,
  );
  const updateDecisionCount = lastUpdateCycle?.decisions.length ?? 0;
  const updateDecisionSampleCount = Math.min(updateDecisionCount, PROVENANCE_AUDIT_CHAIN_MAX_UPDATE_DECISIONS);

  const lastEvaluationCycle = teacherTruth?.lastEvaluationCycle ?? null;
  const queue = teacherTruth?.queue ?? null;
  const recentDecisionDetail = recentDecisionSummary?.detail ? truncateText(recentDecisionSummary.detail) : null;
  const serveSummary = truncateText(
    [
      traceId ? `trace=${traceId}` : null,
      routeRows.length > 0 ? `${routeRows.length} route row(s)` : null,
      routeRows[0] ? `latest=${routeRows[0].decisionPointId}/${routeRows[0].chosenActionKind}` : null,
      attributionTruth.primaryState ? `attribution=${attributionTruth.primaryState}` : null,
      promotionProofTruth.proofTruth?.verdict ? `proof=${promotionProofTruth.proofTruth.verdict}` : null,
    ].filter(Boolean).join("; "),
    PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS,
  ) ?? "serve decision truth is not available";

  const learningSummary = truncateText(
    [
      lastUpdateCycle?.detail ?? null,
      lastUpdateCycle?.generatedAt ? `generated_at=${lastUpdateCycle.generatedAt}` : null,
      updateDecisionCount > 0 ? `${updateDecisionCount} decision(s)` : null,
    ].filter(Boolean).join("; "),
    PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS,
  ) ?? "learning update truth is not available";

  return {
    contract: PROVENANCE_AUDIT_CHAIN_CONTRACT,
    bundleId: input.bundleId,
    generatedAt: normalizeText(input.generatedAt) ?? new Date().toISOString(),
    precedence: {
      label: PRECEDENCE_LABEL,
      correctionAuthority: "user_explicit",
      rawAuthority: "raw_source",
      teacherSynthesis: "teacher_inference",
      note: "read-only audit stitching keeps user_explicit correction and raw_source above teacher synthesis",
    },
    serveDecision: {
      traceId,
      episodeId,
      routeRowCount: routeRows.length,
      routeRows,
      recentDecisionSummary: recentDecisionDetail,
      lastCompileReportSummary: truncateText(normalizeText(runtimeStatus.lastCompileReportSummary), PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS),
      summary: serveSummary,
    },
    attributionTruth,
    learningUpdate: {
      lastEvaluationCycle,
      lastUpdateCycle,
      queue,
      summary: learningSummary,
      precedenceLabel: PRECEDENCE_LABEL,
      linkedTraceIds,
      linkedObservationIds,
      linkedEpisodeIds,
    },
    promotionProofTruth,
    linkages: {
      traceId,
      routeRowIds: rowIds,
      observationIds: linkedObservationIds,
      updateEpisodeIds: linkedEpisodeIds,
      proofBundleDir: promotionProofTruth.proofTruth?.bundleDir ?? null,
      proofVerdict: promotionProofTruth.proofTruth?.verdict ?? null,
      restartSafe: Boolean(traceId)
        && routeRows.length > 0
        && linkedTraceIds.length > 0
        && (promotionProofTruth.proofTruth?.verdict ?? null) !== null,
    },
    boundedness: {
      maxRouteRows: PROVENANCE_AUDIT_CHAIN_MAX_ROUTE_ROWS,
      maxUpdateDecisions: PROVENANCE_AUDIT_CHAIN_MAX_UPDATE_DECISIONS,
      maxTextChars: PROVENANCE_AUDIT_CHAIN_MAX_TEXT_CHARS,
      routeRowCount: lastTraceSelectionMetadata?.decisionPointSnapshots?.length ?? routeRows.length,
      routeRowSampleCount: routeRows.length,
      updateDecisionCount,
      updateDecisionSampleCount,
    },
  };
}

export function renderProvenanceAuditChainMarkdownV1(chain: ProvenanceAuditChainV1): string {
  const lines: string[] = [
    "# Provenance audit chain",
    "",
    `- contract: \`${chain.contract}\``,
    `- bundle: \`${chain.bundleId}\``,
    `- generated at: ${chain.generatedAt}`,
    `- precedence: ${chain.precedence.label}`,
    `- note: ${chain.precedence.note}`,
    "",
    "## Serve decision / route rows",
    `- trace: ${chain.serveDecision.traceId ?? "unknown"}`,
    `- episode: ${chain.serveDecision.episodeId ?? "unknown"}`,
    `- rows captured: ${chain.serveDecision.routeRowCount}`,
    chain.serveDecision.recentDecisionSummary ? `- recent decision summary: ${chain.serveDecision.recentDecisionSummary}` : null,
    chain.serveDecision.lastCompileReportSummary ? `- compile report summary: ${chain.serveDecision.lastCompileReportSummary}` : null,
    `- summary: ${chain.serveDecision.summary}`,
  ].filter((line): line is string => line !== null);

  for (const row of chain.serveDecision.routeRows) {
    lines.push(
      `- row ${row.rowId}: ${row.detail}`,
      `  - provenance: ${row.provenanceState}/${row.provenanceBasis}`,
      `  - precedence: ${row.precedenceLabel}`,
    );
  }

  lines.push(
    "",
    "## Attribution truth",
    `- contract: ${chain.attributionTruth.contract ?? "unknown"}`,
    `- visible: ${chain.attributionTruth.visible === null ? "unknown" : chain.attributionTruth.visible ? "yes" : "no"}`,
    `- primary state: ${chain.attributionTruth.primaryState ?? "unknown"}`,
    `- summary: ${chain.attributionTruth.summary}`,
  );

  if (chain.attributionTruth.latest.ambiguous) {
    lines.push(
      `- latest ambiguous: ${chain.attributionTruth.latest.ambiguous.observationId} / ${chain.attributionTruth.latest.ambiguous.episodeId} (${chain.attributionTruth.latest.ambiguous.bindingMode}, ${chain.attributionTruth.latest.ambiguous.attributionQuality})`,
      `  - reason: ${truncateText(chain.attributionTruth.latest.ambiguous.reason, 120) ?? "n/a"}`,
      `  - precedence: ${chain.attributionTruth.latest.ambiguous.precedenceLabel}`,
    );
  }
  if (chain.attributionTruth.latest.unmatched) {
    lines.push(
      `- latest unmatched: ${chain.attributionTruth.latest.unmatched.observationId} / ${chain.attributionTruth.latest.unmatched.episodeId} (${chain.attributionTruth.latest.unmatched.bindingMode}, ${chain.attributionTruth.latest.unmatched.attributionQuality})`,
      `  - reason: ${truncateText(chain.attributionTruth.latest.unmatched.reason, 120) ?? "n/a"}`,
      `  - precedence: ${chain.attributionTruth.latest.unmatched.precedenceLabel}`,
    );
  }
  if (chain.attributionTruth.latest.followupPending) {
    lines.push(
      `- latest follow-up pending: ${chain.attributionTruth.latest.followupPending.observationId} / ${chain.attributionTruth.latest.followupPending.episodeId} (${chain.attributionTruth.latest.followupPending.status}, ${chain.attributionTruth.latest.followupPending.gate})`,
      `  - reason: ${truncateText(chain.attributionTruth.latest.followupPending.reason, 120) ?? "n/a"}`,
    );
  }

  lines.push(
    "",
    "## Learning update",
    `- summary: ${chain.learningUpdate.summary}`,
    `- precedence: ${chain.learningUpdate.precedenceLabel}`,
    `- linked trace ids: ${chain.learningUpdate.linkedTraceIds.length > 0 ? chain.learningUpdate.linkedTraceIds.join(", ") : "none"}`,
    `- linked observation ids: ${chain.learningUpdate.linkedObservationIds.length > 0 ? chain.learningUpdate.linkedObservationIds.join(", ") : "none"}`,
    `- linked episode ids: ${chain.learningUpdate.linkedEpisodeIds.length > 0 ? chain.learningUpdate.linkedEpisodeIds.join(", ") : "none"}`,
  );

  if (chain.learningUpdate.lastUpdateCycle) {
    lines.push(
      `- update generated at: ${chain.learningUpdate.lastUpdateCycle.generatedAt ?? "unknown"}`,
      `- update decisions captured: ${chain.learningUpdate.lastUpdateCycle.decisions.length}`,
    );
    for (const decision of chain.learningUpdate.lastUpdateCycle.decisions) {
      lines.push(
        `  - decision: ${decision.summary}`,
        `    - trace ids: ${decision.traceIds.length > 0 ? decision.traceIds.join(", ") : "none"}`,
        `    - observation ids: ${decision.observationIds.length > 0 ? decision.observationIds.join(", ") : "none"}`,
        `    - supervision ids: ${decision.supervisionIds.length > 0 ? decision.supervisionIds.join(", ") : "none"}`,
      );
    }
  }

  lines.push(
    "",
    "## Promotion / proof truth",
    `- current pack version: ${chain.promotionProofTruth.currentPackVersion ?? "unknown"}`,
    `- last promotion reason: ${chain.promotionProofTruth.lastPromotionReason ?? "unknown"}`,
    `- last promotion verdict: ${chain.promotionProofTruth.lastPromotionVerdict?.verdict ?? "unknown"}`,
    chain.promotionProofTruth.lastPromotionVerdict?.summary ? `- promotion summary: ${chain.promotionProofTruth.lastPromotionVerdict.summary}` : null,
    `- last replay gate verdict: ${chain.promotionProofTruth.lastReplayGateVerdict?.verdict ?? "unknown"}`,
    chain.promotionProofTruth.lastReplayGateVerdict?.summary ? `- replay gate summary: ${chain.promotionProofTruth.lastReplayGateVerdict.summary}` : null,
    chain.promotionProofTruth.retrainLineage?.summary ? `- retrain lineage: ${chain.promotionProofTruth.retrainLineage.summary}` : null,
    chain.promotionProofTruth.retrainLineage?.priorBaseArtifactId ? `- prior base artifact: ${chain.promotionProofTruth.retrainLineage.priorBaseArtifactId}${chain.promotionProofTruth.retrainLineage.priorBaseArtifactVersion ? `@${chain.promotionProofTruth.retrainLineage.priorBaseArtifactVersion}` : ""}` : null,
    chain.promotionProofTruth.retrainLineage?.priorBaseArtifactChecksum ? `- prior base checksum: ${chain.promotionProofTruth.retrainLineage.priorBaseArtifactChecksum}` : null,
    chain.promotionProofTruth.retrainLineage?.candidateArtifactId ? `- candidate artifact: ${chain.promotionProofTruth.retrainLineage.candidateArtifactId}${chain.promotionProofTruth.retrainLineage.candidateArtifactVersion ? `@${chain.promotionProofTruth.retrainLineage.candidateArtifactVersion}` : ""}` : null,
    chain.promotionProofTruth.retrainLineage?.candidateArtifactChecksum ? `- current router checksum: ${chain.promotionProofTruth.retrainLineage.candidateArtifactChecksum}` : null,
    typeof chain.promotionProofTruth.retrainLineage?.priorRooted === "boolean" ? `- prior-rooted: ${chain.promotionProofTruth.retrainLineage.priorRooted ? "yes" : "no"}` : null,
    typeof chain.promotionProofTruth.retrainLineage?.promotionValid === "boolean" ? `- promotion-valid: ${chain.promotionProofTruth.retrainLineage.promotionValid ? "yes" : "no"}` : null,
    chain.promotionProofTruth.retrainLineage?.residualUpdateCount !== null && chain.promotionProofTruth.retrainLineage?.residualUpdateCount !== undefined ? `- residual updates: ${chain.promotionProofTruth.retrainLineage.residualUpdateCount}` : null,
    `- proof verdict: ${chain.promotionProofTruth.proofTruth?.verdict ?? "unknown"}`,
    chain.promotionProofTruth.proofTruth?.summary ? `- proof summary: ${chain.promotionProofTruth.proofTruth.summary}` : null,
    chain.promotionProofTruth.proofTruth?.why ? `- proof why: ${chain.promotionProofTruth.proofTruth.why}` : null,
    chain.promotionProofTruth.proofTruth?.runtimeLoadProofPath ? `- runtime load proof: ${chain.promotionProofTruth.proofTruth.runtimeLoadProofPath}` : null,
    `- restart safe linkage: ${chain.linkages.restartSafe ? "yes" : "no"}`,
  );

  lines.push(
    "",
    "## Linkages",
    `- route row ids: ${chain.linkages.routeRowIds.length > 0 ? chain.linkages.routeRowIds.join(", ") : "none"}`,
    `- observation ids: ${chain.linkages.observationIds.length > 0 ? chain.linkages.observationIds.join(", ") : "none"}`,
    `- update episode ids: ${chain.linkages.updateEpisodeIds.length > 0 ? chain.linkages.updateEpisodeIds.join(", ") : "none"}`,
    `- proof bundle dir: ${chain.linkages.proofBundleDir ?? "unknown"}`,
    `- proof verdict: ${chain.linkages.proofVerdict ?? "unknown"}`,
    "",
    "## Boundedness",
    `- max route rows: ${chain.boundedness.maxRouteRows}`,
    `- route rows observed: ${chain.boundedness.routeRowCount}`,
    `- route rows captured: ${chain.boundedness.routeRowSampleCount}`,
    `- max update decisions: ${chain.boundedness.maxUpdateDecisions}`,
    `- update decisions observed: ${chain.boundedness.updateDecisionCount}`,
    `- update decisions captured: ${chain.boundedness.updateDecisionSampleCount}`,
    `- text cap: ${chain.boundedness.maxTextChars} chars`,
  );

  return `${lines.filter((line): line is string => line !== null).join("\n")}\n`;
}
