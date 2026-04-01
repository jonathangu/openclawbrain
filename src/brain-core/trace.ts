/**
 * Decision trace recording and footer generation.
 */

import { createHash, randomUUID } from "node:crypto";
import type {
  BrainCompileReportV1,
  BrainPersistenceMode,
  BrainNode,
  BrainObservationToolResult,
  BrainObservationBindingMode,
  BrainPrefetchDecision,
  DecisionTraceBranchOutcome,
  DecisionRouteTrace,
  DecisionTrace,
  DecisionTraceInjectedNodeSummary,
  NodeKind,
  RecentPrefetchSummary,
  TrustLevel,
} from "./types.js";
import type { TraverseResult } from "./traverse.js";
import { resolveStopTruth } from "./trajectory-stop.js";

const ROUTER_IDENTITY = "brain-graph-traverse.v2";
const TRACE_PREVIEW_CHARS = 160;
const COMPILE_PREVIEW_CHARS = 96;
const COMPILE_MAX_BUCKET_ITEMS = 5;
const COMPILE_MAX_REASON_KEYS = 5;

function hashValue(value: string): string {
  return createHash("sha256").update(value).digest("hex").slice(0, 16);
}

function hashQuery(queryText: string): string {
  return hashValue(queryText);
}

function truncatePreview(content: string): string {
  const normalized = content.replace(/\s+/g, " ").trim();
  if (normalized.length <= TRACE_PREVIEW_CHARS) {
    return normalized;
  }
  return `${normalized.slice(0, TRACE_PREVIEW_CHARS - 1)}…`;
}

function isRedactedSurface(value: string): boolean {
  return /^\[redacted [^\]]+\]$/u.test(value);
}

export function redactTextSurface(label: string, value: string | null | undefined): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim();
  if (normalized.length === 0) {
    return "";
  }
  if (isRedactedSurface(normalized)) {
    return normalized;
  }
  return `[redacted ${label} chars=${normalized.length} sha256=${hashValue(normalized)}]`;
}

export function toProvenanceRef(value: string | null | undefined, fallback: string): string {
  const basis = typeof value === "string" && value.trim().length > 0 ? value.trim() : fallback;
  return `prov_${hashValue(basis)}`;
}

export function redactInjectedNodeSummary(
  summary: DecisionTraceInjectedNodeSummary,
): DecisionTraceInjectedNodeSummary {
  return {
    ...summary,
    provenanceRef: summary.provenanceRef ?? toProvenanceRef(summary.sourceUri, summary.nodeId),
    sourceUri: null,
    contentPreview: redactTextSurface("source_content", summary.contentPreview) ?? "",
  };
}

function cloneInjectedNodeSummary(summary: DecisionTraceInjectedNodeSummary): DecisionTraceInjectedNodeSummary {
  return {
    nodeId: summary.nodeId,
    kind: summary.kind,
    trust: summary.trust,
    provenanceRef: summary.provenanceRef,
    sourceUri: summary.sourceUri,
    tags: [...summary.tags],
    tokenCount: summary.tokenCount,
    contentPreview: summary.contentPreview,
  };
}

function cloneBranchOutcome(outcome: DecisionTraceBranchOutcome): DecisionTraceBranchOutcome {
  return {
    sourceNodeId: outcome.sourceNodeId,
    expansionIndex: outcome.expansionIndex,
    selectionSubstepCount: outcome.selectionSubstepCount,
    continued: outcome.continued,
    selectedTargetIds: [...outcome.selectedTargetIds],
    acceptedTargetIds: [...outcome.acceptedTargetIds],
    vetoedTargetIds: [...outcome.vetoedTargetIds],
    droppedTargetIds: [...outcome.droppedTargetIds],
    stopTruth: outcome.stopTruth,
    stopReason: outcome.stopReason,
    terminationReason: outcome.terminationReason,
    proof: outcome.proof,
  };
}

function buildPersistenceMode(persistRawSurfaces: boolean): BrainPersistenceMode {
  return persistRawSurfaces ? "redacted_with_operator_audit" : "redacted";
}

export function redactToolResult(result: BrainObservationToolResult): BrainObservationToolResult {
  return {
    ...result,
    input: redactTextSurface("tool_input", result.input),
    output: redactTextSurface("tool_output", result.output),
    excerpt: redactTextSurface("tool_excerpt", result.excerpt),
  };
}

export function redactRouteTrace(
  routeTrace: DecisionRouteTrace | null | undefined,
  queryText?: string | null,
  persistRawSurfaces = false,
): DecisionRouteTrace | null {
  if (!routeTrace) {
    return null;
  }

  const injectedNodeSummaries = routeTrace.injectedNodeSummaries.map(redactInjectedNodeSummary);
  return {
    ...routeTrace,
    persistenceMode: buildPersistenceMode(persistRawSurfaces),
    branchOutcomes: (routeTrace.branchOutcomes ?? []).map(cloneBranchOutcome),
    injectedNodeSummaries,
    sourceSummary: {
      ...routeTrace.sourceSummary,
      sourceUris: [],
      sourceRefs: [...new Set(injectedNodeSummaries.map((node) => node.provenanceRef).filter((value): value is string => Boolean(value)))],
    },
    operatorAudit: persistRawSurfaces
      ? {
          queryText: queryText ?? "",
          injectedNodeSummaries: routeTrace.injectedNodeSummaries.map(cloneInjectedNodeSummary),
        }
      : null,
  };
}

export function redactDecisionTrace(trace: DecisionTrace, persistRawSurfaces = false): DecisionTrace {
  return {
    ...trace,
    queryText: redactTextSurface("query", trace.queryText) ?? "",
    routeTrace: redactRouteTrace(trace.routeTrace, trace.queryText, persistRawSurfaces),
  };
}

function countBy<T extends string>(values: T[]): Partial<Record<T, number>> {
  const counts: Partial<Record<T, number>> = {};
  for (const value of values) {
    counts[value] = (counts[value] ?? 0) + 1;
  }
  return counts;
}

function candidateNodeIds(traversalResult: TraverseResult): string[] {
  const ids = new Set<string>();
  for (const seed of traversalResult.seedScores) {
    ids.add(seed.nodeId);
  }
  for (const expansion of traversalResult.trajectory) {
    for (const substep of expansion.substeps) {
      for (const candidate of substep.candidates) {
        if (candidate.action.type === "traverse") {
          ids.add(candidate.action.targetNodeId);
        }
      }
    }
  }
  return [...ids];
}

function selectedTraversalNodeIds(traversalResult: TraverseResult): string[] {
  return traversalResult.trajectory.flatMap((expansion) => expansion.acceptedTargets);
}

function countStopTruths(traversalResult: TraverseResult): {
  chosenStopCount: number;
  forcedStopCount: number;
} {
  let chosenStopCount = 0;
  let forcedStopCount = 0;

  for (const expansion of traversalResult.trajectory) {
    for (const substep of expansion.substeps) {
      const stopTruth = resolveStopTruth(substep);
      if (stopTruth === "forced") {
        forcedStopCount += 1;
        continue;
      }
      if (stopTruth === "chosen") {
        chosenStopCount += 1;
      }
    }
  }

  return { chosenStopCount, forcedStopCount };
}

function countDroppedProposalReasons(traversalResult: TraverseResult): {
  droppedProposalCount: number;
  droppedProposalReasons: Record<string, number> | null;
} {
  const droppedProposalReasons: Record<string, number> = {};
  let droppedProposalCount = 0;

  for (const expansion of traversalResult.trajectory) {
    for (const outcome of expansion.proposalOutcomes ?? []) {
      if (outcome.outcome !== "dropped") {
        continue;
      }
      droppedProposalCount += 1;
      droppedProposalReasons[outcome.reason] = (droppedProposalReasons[outcome.reason] ?? 0) + 1;
    }
  }

  return {
    droppedProposalCount,
    droppedProposalReasons: Object.keys(droppedProposalReasons).length > 0 ? droppedProposalReasons : null,
  };
}

function formatBranchSource(sourceNodeId: string | null): string {
  return sourceNodeId ?? "start";
}

function formatPolicyVisibility(substep: TraverseResult["trajectory"][number]["substeps"][number] | null): string {
  const policyState = substep?.stateSnapshot.policyState;
  if (!substep || !policyState) {
    return "";
  }

  const parts = [
    `pressure=${policyState.pressureLevel.toFixed(2)}`,
    `budget=${policyState.budgetUsedFraction.toFixed(2)}`,
    `frontier=${policyState.frontierPressure.toFixed(2)}`,
    `stop_p=${substep.stopProbability.toFixed(2)}`,
  ];
  if (policyState.pendingSelectionCount > 0) {
    parts.push(`pending=${policyState.pendingSelectionCount}`);
  }
  return ` [${parts.join(" ")}]`;
}

function buildBranchProof(outcome: {
  sourceNodeId: string | null;
  acceptedTargetIds: string[];
  vetoedTargetIds: string[];
  droppedTargetIds: string[];
  stopTruth: "chosen" | "forced" | null;
  terminationReason: string | null;
}, stopSubstep: TraverseResult["trajectory"][number]["substeps"][number] | null): string {
  const continuedSegment = outcome.acceptedTargetIds.length > 0
    ? `continued via ${outcome.acceptedTargetIds.join(",")}`
    : "stopped without continuation";
  const stopSegment = outcome.stopTruth === null
    ? "ended without explicit stop truth"
    : `${outcome.stopTruth} stop (${outcome.terminationReason ?? "unknown"})`;
  return `branch ${formatBranchSource(outcome.sourceNodeId)} ${continuedSegment} then ${stopSegment}${formatPolicyVisibility(stopSubstep)}; accepted=${outcome.acceptedTargetIds.length} vetoed=${outcome.vetoedTargetIds.length} dropped=${outcome.droppedTargetIds.length}`;
}

function summarizeBranchOutcomes(traversalResult: TraverseResult): {
  branchOutcomes: DecisionTraceBranchOutcome[];
  branchOutcomeSummary: NonNullable<DecisionRouteTrace["selectionMetadata"]["branchOutcomeSummary"]>;
} {
  const branchOutcomes = traversalResult.trajectory.map((expansion) => {
    const stopSubstep = [...expansion.substeps]
      .reverse()
      .find((substep) => substep.chosenAction.type === "stop_local") ?? null;
    const stopTruth = stopSubstep ? resolveStopTruth(stopSubstep) : null;
    const stopReason = stopSubstep?.stopReason ?? null;
    const terminationReason = expansion.terminationReason ?? stopReason;
    const vetoedTargetIds = expansion.vetoedTargets.map((target) => target.targetNodeId);
    const droppedTargetIds = (expansion.proposalOutcomes ?? [])
      .filter((outcome) => outcome.outcome === "dropped")
      .map((outcome) => outcome.targetNodeId);
    return {
      sourceNodeId: expansion.sourceNodeId,
      expansionIndex: expansion.expansionIndex,
      selectionSubstepCount: expansion.substeps.length,
      continued: expansion.acceptedTargets.length > 0,
      selectedTargetIds: [...expansion.selectedTargets],
      acceptedTargetIds: [...expansion.acceptedTargets],
      vetoedTargetIds,
      droppedTargetIds,
      stopTruth,
      stopReason,
      terminationReason,
      proof: buildBranchProof({
        sourceNodeId: expansion.sourceNodeId,
        acceptedTargetIds: expansion.acceptedTargets,
        vetoedTargetIds,
        droppedTargetIds,
        stopTruth,
        terminationReason,
      }, stopSubstep),
    };
  });

  const terminationReasons: Record<string, number> = {};
  for (const outcome of branchOutcomes) {
    if (typeof outcome.terminationReason !== "string" || outcome.terminationReason.length === 0) {
      continue;
    }
    terminationReasons[outcome.terminationReason] = (terminationReasons[outcome.terminationReason] ?? 0) + 1;
  }
  const branchCount = branchOutcomes.length;
  const continuingBranchCount = branchOutcomes.filter((outcome) => outcome.continued).length;
  const stoppedWithoutProgressCount = branchCount - continuingBranchCount;
  const chosenStopBranchCount = branchOutcomes.filter((outcome) => outcome.stopTruth === "chosen").length;
  const forcedStopBranchCount = branchOutcomes.filter((outcome) => outcome.stopTruth === "forced").length;
  const reasonSummary = Object.entries(terminationReasons)
    .map(([reason, count]) => `${reason}=${count}`)
    .join(", ");

  return {
    branchOutcomes,
    branchOutcomeSummary: {
      branchCount,
      continuingBranchCount,
      stoppedWithoutProgressCount,
      chosenStopBranchCount,
      forcedStopBranchCount,
      terminationReasons: Object.keys(terminationReasons).length > 0 ? terminationReasons : null,
      detail: branchCount === 0
        ? "no traced branches"
        : `${continuingBranchCount}/${branchCount} branches continued; ${stoppedWithoutProgressCount}/${branchCount} stopped without continuation; chosen=${chosenStopBranchCount}; forced=${forcedStopBranchCount}${reasonSummary.length > 0 ? `; reasons ${reasonSummary}` : ""}`,
    },
  };
}

function summarizeInjectedNode(node: BrainNode): DecisionTraceInjectedNodeSummary {
  return {
    nodeId: node.id,
    kind: node.kind,
    trust: node.trust,
    provenanceRef: toProvenanceRef(node.sourceUri, node.id),
    sourceUri: node.sourceUri,
    tags: [...node.tags],
    tokenCount: node.tokenCount,
    contentPreview: truncatePreview(node.content),
  };
}

function truncateCompilePreview(content: string): string {
  const normalized = content.replace(/\s+/g, " ").trim();
  if (normalized.length <= COMPILE_PREVIEW_CHARS) {
    return normalized;
  }
  return `${normalized.slice(0, COMPILE_PREVIEW_CHARS - 1)}…`;
}

function takeTopHistogramEntries(
  histogram: Record<string, number> | null | undefined,
  limit = COMPILE_MAX_REASON_KEYS,
): { entries: Record<string, number>; overflow: number } {
  const source = Object.entries(histogram ?? {})
    .filter(([, count]) => Number.isFinite(count) && count > 0)
    .sort(([leftKey, leftCount], [rightKey, rightCount]) => (
      rightCount === leftCount ? leftKey.localeCompare(rightKey) : rightCount - leftCount
    ));
  const entries: Record<string, number> = {};
  let overflow = 0;
  source.forEach(([key, count], index) => {
    if (index < limit) {
      entries[key] = count;
      return;
    }
    overflow += count;
  });
  return { entries, overflow };
}

function buildCompileSummaryLine(report: BrainCompileReportV1): string {
  const stage = report.decision.brainDropStage
    ?? report.decision.interruptionStage
    ?? (report.budget.contextClipped ? "injection" : "selection");
  const clipped = report.budget.contextClipped === true ? "yes" : "no";
  const failOpen = report.decision.queryInterrupted === true || report.decision.servedPartial === true
    ? "yes"
    : "no";
  const elapsed = report.timing.compileElapsedMs ?? 0;
  const deadline = report.timing.compileDeadlineMs === null
    ? "n/a"
    : `${report.timing.compileDeadlineMs}ms`;
  const bindingMode = report.bindingMode ?? "unknown";
  const mode = report.decision.mode ?? "unknown";
  return `[brain compile] mode=${mode} stage=${stage} selected=${report.counters.selectedNodeCount} dropped=${report.counters.droppedNodeCount} prefetched=${report.counters.prefetchedNodeCount} compressed=${report.counters.compressedNodeCount} clipped=${clipped} fail_open=${failOpen} elapsed=${elapsed}ms deadline=${deadline} pack=${report.activePackId ?? "n/a"} bind=${bindingMode} trace=${report.traceId ?? "n/a"} ep=${report.episodeId ?? "n/a"}`;
}

function buildCompileItem(params: {
  nodeId: string;
  provenanceRef: string | null;
  kind: BrainCompileReportV1["buckets"]["selected"][number]["kind"];
  trust: BrainCompileReportV1["buckets"]["selected"][number]["trust"];
  tokenCount: number | null;
  preview: string | null;
  state: BrainCompileReportV1["buckets"]["selected"][number]["state"];
  reason: string | null;
  sourceUri?: string | null;
  stage?: BrainCompileReportV1["buckets"]["selected"][number]["stage"];
  fitStrategy?: BrainCompileReportV1["buckets"]["selected"][number]["fitStrategy"];
  compressionMode?: string | null;
}): BrainCompileReportV1["buckets"]["selected"][number] {
  return {
    nodeId: params.nodeId,
    provenanceRef: params.provenanceRef,
    kind: params.kind,
    trust: params.trust,
    tokenCount: params.tokenCount,
    preview: params.preview,
    state: params.state,
    reason: params.reason,
    ...(params.sourceUri !== undefined ? { sourceUri: params.sourceUri } : {}),
    ...(params.stage !== undefined ? { stage: params.stage } : {}),
    ...(params.fitStrategy !== undefined ? { fitStrategy: params.fitStrategy } : {}),
    ...(params.compressionMode !== undefined ? { compressionMode: params.compressionMode } : {}),
  };
}

export function buildBrainCompileReport(params: {
  routeTrace: DecisionRouteTrace | null | undefined;
  decision?: {
    mode?: string | null;
    bindingMode?: BrainObservationBindingMode | null;
    traceId?: string | null;
    episodeId?: string | null;
  };
  lookupNode?: (nodeId: string) => BrainNode | null | undefined;
}): BrainCompileReportV1 | null {
  const routeTrace = params.routeTrace;
  if (!routeTrace?.selectionMetadata) {
    return null;
  }

  const selectedIds = new Set(routeTrace.selectedNodeIds);
  const selectedTraversalIds = routeTrace.selectedTraversalNodeIds.length;
  const candidateIds = routeTrace.candidateNodeIds;
  const droppedIds = candidateIds.filter((nodeId) => !selectedIds.has(nodeId));
  const selectedItems = routeTrace.injectedNodeSummaries
    .slice(0, COMPILE_MAX_BUCKET_ITEMS)
    .map((summary) => buildCompileItem({
      nodeId: summary.nodeId,
      provenanceRef: summary.provenanceRef,
      kind: summary.kind,
      trust: summary.trust,
      tokenCount: summary.tokenCount,
      preview: truncateCompilePreview(summary.contentPreview),
      state: "selected",
      reason: null,
      sourceUri: summary.sourceUri,
      stage: routeTrace.selectionMetadata.brainDropStage ?? null,
      fitStrategy: routeTrace.selectionMetadata.fitStrategy ?? null,
    }));

  const canExposeRawSource = routeTrace.persistenceMode === "redacted_with_operator_audit";
  const droppedItems = droppedIds
    .slice(0, COMPILE_MAX_BUCKET_ITEMS)
    .map((nodeId) => {
      const node = params.lookupNode?.(nodeId) ?? null;
      return buildCompileItem({
        nodeId,
        provenanceRef: node ? toProvenanceRef(node.sourceUri, node.id) : null,
        kind: node?.kind ?? null,
        trust: node?.trust ?? null,
        tokenCount: node?.tokenCount ?? null,
        preview: canExposeRawSource && node ? truncateCompilePreview(node.content) : null,
        state: "dropped",
        reason: routeTrace.selectionMetadata.fittingDropReasons && Object.keys(routeTrace.selectionMetadata.fittingDropReasons).length > 0
          ? Object.keys(routeTrace.selectionMetadata.fittingDropReasons)[0] ?? "not_selected"
          : "not_selected",
        sourceUri: canExposeRawSource ? (node?.sourceUri ?? null) : null,
        stage: routeTrace.selectionMetadata.brainDropStage ?? null,
        fitStrategy: routeTrace.selectionMetadata.fitStrategy ?? null,
      });
    });

  const selectedNodeCount = routeTrace.injectedNodeSummaries.length;
  const droppedNodeCount = Math.max(0, droppedIds.length);
  const prefetchedNodeCount = 0;
  const compressedNodeCount = 0;
  const branchSummary = routeTrace.selectionMetadata.branchOutcomeSummary ?? null;
  const droppedReasonHistogram = routeTrace.selectionMetadata.fittingDropReasons
    ? Object.fromEntries(Object.entries(routeTrace.selectionMetadata.fittingDropReasons).map(([key, count]) => [key, count]))
    : (droppedNodeCount > 0 ? { not_selected: droppedNodeCount } : {});
  const droppedProposalReasons = routeTrace.selectionMetadata.droppedProposalReasons ?? null;
  const terminationReasons = branchSummary?.terminationReasons ?? null;
  const limitedDroppedReasons = takeTopHistogramEntries(droppedReasonHistogram);
  const limitedDroppedProposalReasons = takeTopHistogramEntries(droppedProposalReasons);
  const limitedPrefetchedReasons = takeTopHistogramEntries({});
  const limitedCompressionReasons = takeTopHistogramEntries({});
  const limitedTerminationReasons = takeTopHistogramEntries(terminationReasons);
  const reasonOverflowCount =
    limitedDroppedReasons.overflow
    + limitedDroppedProposalReasons.overflow
    + limitedPrefetchedReasons.overflow
    + limitedCompressionReasons.overflow
    + limitedTerminationReasons.overflow;

  const report: BrainCompileReportV1 = {
    schemaVersion: 1,
    summary: "",
    traceId: params.decision?.traceId ?? null,
    episodeId: params.decision?.episodeId ?? null,
    requestDigest: routeTrace.requestDigest ?? null,
    activePackId: routeTrace.activePackId ?? null,
    routerIdentity: routeTrace.routerIdentity ?? null,
    bindingMode: params.decision?.bindingMode ?? null,
    decision: {
      mode: params.decision?.mode ?? null,
      brainDropReason: routeTrace.selectionMetadata.brainDropReason ?? null,
      brainDropStage: routeTrace.selectionMetadata.brainDropStage ?? null,
      queryInterrupted: routeTrace.selectionMetadata.queryInterrupted ?? null,
      interruptionStage: routeTrace.selectionMetadata.interruptionStage ?? null,
      interruptionReason: routeTrace.selectionMetadata.interruptionReason ?? null,
      servedPartial: routeTrace.selectionMetadata.servedPartial ?? null,
    },
    timing: {
      compileElapsedMs: routeTrace.selectionMetadata.compileElapsedMs ?? null,
      compileDeadlineMs: routeTrace.selectionMetadata.compileDeadlineMs ?? null,
      compileDeadlineHit: routeTrace.selectionMetadata.compileDeadlineHit ?? null,
      routeSelectionMs: routeTrace.selectionMetadata.routeSelectionMs ?? null,
      totalQueryMs: routeTrace.selectionMetadata.totalQueryMs ?? null,
    },
    budget: {
      budgetFraction: routeTrace.selectionMetadata.budgetFraction ?? null,
      budgetChars: routeTrace.selectionMetadata.budgetChars ?? null,
      queryBudgetChars: routeTrace.selectionMetadata.queryBudgetChars ?? null,
      maxContextChars: routeTrace.selectionMetadata.maxContextChars ?? null,
      injectedChars: routeTrace.selectionMetadata.injectedChars ?? null,
      droppedChars: routeTrace.selectionMetadata.droppedChars ?? null,
      contextClipped: routeTrace.selectionMetadata.contextClipped ?? null,
      fitStrategy: routeTrace.selectionMetadata.fitStrategy ?? null,
    },
    counters: {
      candidateNodeCount: candidateIds.length,
      selectedNodeCount,
      selectedTraversalNodeCount: selectedTraversalIds,
      selectedSeedNodeCount: routeTrace.selectedSeedNodeIds.length,
      droppedNodeCount,
      prefetchedNodeCount,
      compressedNodeCount,
      droppedProposalCount: routeTrace.selectionMetadata.droppedProposalCount ?? 0,
      branchCount: branchSummary?.branchCount ?? routeTrace.branchOutcomes.length,
      continuingBranchCount: branchSummary?.continuingBranchCount ?? routeTrace.branchOutcomes.filter((outcome) => outcome.continued).length,
      sourceRefCount: routeTrace.sourceSummary.sourceRefs.length,
    },
    reasons: {
      droppedNodeReasons: limitedDroppedReasons.entries,
      droppedProposalReasons: limitedDroppedProposalReasons.entries,
      prefetchedReasons: limitedPrefetchedReasons.entries,
      compressionReasons: limitedCompressionReasons.entries,
      terminationReasons: limitedTerminationReasons.entries,
    },
    buckets: {
      selected: selectedItems,
      dropped: droppedItems,
      prefetched: [],
      compressed: [],
    },
    overflow: {
      selectedOverflowCount: Math.max(0, routeTrace.injectedNodeSummaries.length - selectedItems.length),
      droppedOverflowCount: Math.max(0, droppedIds.length - droppedItems.length),
      prefetchedOverflowCount: 0,
      compressedOverflowCount: 0,
      reasonOverflowCount,
    },
    boundedness: {
      maxItemsPerBucket: COMPILE_MAX_BUCKET_ITEMS,
      maxReasonKeys: COMPILE_MAX_REASON_KEYS,
      maxPreviewChars: COMPILE_PREVIEW_CHARS,
      maxSourceRefs: COMPILE_MAX_BUCKET_ITEMS,
    },
  };
  report.summary = buildCompileSummaryLine(report);
  return report;
}

export function rewriteBrainCompileReportSummary(
  report: BrainCompileReportV1,
  params: {
    mode?: string | null;
    bindingMode?: BrainObservationBindingMode | null;
  },
): BrainCompileReportV1 {
  const next: BrainCompileReportV1 = {
    ...report,
    bindingMode: params.bindingMode ?? report.bindingMode,
    decision: {
      ...report.decision,
      ...(params.mode !== undefined ? { mode: params.mode } : {}),
    },
  };
  next.summary = buildCompileSummaryLine(next);
  return next;
}

export type RecentDecisionOutcome =
  | "served_full"
  | "served_clipped"
  | "partial_fail_open"
  | "partial_fail_open_clipped"
  | "interrupted_without_partial";

export interface RecentDecisionRateSummary {
  count: number;
  rate: number | null;
}

export interface RecentDecisionTraceSummary {
  windowSize: number;
  sampleSize: number;
  histograms: {
    decisionOutcome: Record<RecentDecisionOutcome, number>;
    brainDropReason: Record<string, number>;
    interruptionStage: Record<string, number>;
    fitStrategy: Record<string, number>;
    queryEmbeddingSource: Record<string, number>;
  };
  branchBehavior: {
    branchCount: number;
    continuingBranchCount: number;
    histograms: {
      stopTruth: Record<string, number>;
      terminationReason: Record<string, number>;
    };
    detail: string;
  };
  clipRate: RecentDecisionRateSummary;
  failOpenRate: RecentDecisionRateSummary;
  detail: string;
}

function emptyPrefetchStateHistogram(): Record<BrainPrefetchDecision["state"], number> {
  return {
    scheduled: 0,
    materialized: 0,
    hit: 0,
    miss: 0,
    stale: 0,
    invalidated: 0,
    dropped: 0,
  };
}

export function summarizeRecentPrefetchDecisions(
  prefetchDecisions: BrainPrefetchDecision[],
  windowSize = prefetchDecisions.length,
): RecentPrefetchSummary {
  const histograms: RecentPrefetchSummary["histograms"] = {
    state: emptyPrefetchStateHistogram(),
    kind: {},
    budgetClass: {},
    summaryRoutingMode: {},
    invalidationReason: {},
  };

  const sample = prefetchDecisions.slice(-windowSize);
  let hitCount = 0;
  let staleCount = 0;
  let invalidationCount = 0;

  for (const decision of sample) {
    histograms.state[decision.state] += 1;
    if (decision.kind) {
      incrementHistogram(histograms.kind, decision.kind);
    }
    if (decision.budgetClass) {
      incrementHistogram(histograms.budgetClass, decision.budgetClass);
    }
    if (decision.summaryRoutingMode) {
      incrementHistogram(histograms.summaryRoutingMode, decision.summaryRoutingMode);
    }
    if (decision.invalidatedReason) {
      incrementHistogram(histograms.invalidationReason, decision.invalidatedReason);
    }
    if (decision.state === "hit") {
      hitCount += 1;
    }
    if (decision.state === "stale") {
      staleCount += 1;
    }
    if (decision.state === "invalidated") {
      invalidationCount += 1;
    }
  }

  return {
    windowSize,
    sampleSize: sample.length,
    histograms,
    hitRate: buildRateSummary(hitCount, sample.length),
    staleRate: buildRateSummary(staleCount, sample.length),
    invalidationRate: buildRateSummary(invalidationCount, sample.length),
    detail: sample.length === 0
      ? "no recent prefetch decisions"
      : `${hitCount}/${sample.length} hits; ${staleCount}/${sample.length} stale; ${invalidationCount}/${sample.length} invalidated`,
  };
}

function incrementHistogram(histogram: Record<string, number>, key: string): void {
  histogram[key] = (histogram[key] ?? 0) + 1;
}

function isClippedDecision(
  selectionMetadata: DecisionRouteTrace["selectionMetadata"] | null | undefined,
): boolean {
  return selectionMetadata?.contextClipped === true
    || selectionMetadata?.brainDropReason === "injection_cap_clipped";
}

function isFailOpenDecision(
  selectionMetadata: DecisionRouteTrace["selectionMetadata"] | null | undefined,
): boolean {
  if (!selectionMetadata) {
    return false;
  }
  if (selectionMetadata.servedPartial === true || selectionMetadata.queryInterrupted === true) {
    return true;
  }
  switch (selectionMetadata.brainDropReason) {
    case "assembly_fail_open":
    case "deadline_before_query":
    case "deadline_after_query":
    case "deadline_before_injection":
      return true;
    default:
      return false;
  }
}

function classifyRecentDecisionOutcome(
  selectionMetadata: DecisionRouteTrace["selectionMetadata"] | null | undefined,
): RecentDecisionOutcome {
  const clipped = isClippedDecision(selectionMetadata);
  if (selectionMetadata?.servedPartial === true) {
    return clipped ? "partial_fail_open_clipped" : "partial_fail_open";
  }
  if (selectionMetadata?.queryInterrupted === true || isFailOpenDecision(selectionMetadata)) {
    return "interrupted_without_partial";
  }
  return clipped ? "served_clipped" : "served_full";
}

function buildRateSummary(count: number, sampleSize: number): RecentDecisionRateSummary {
  return {
    count,
    rate: sampleSize > 0 ? count / sampleSize : null,
  };
}

export function summarizeRecentDecisionTraces(
  traces: DecisionTrace[],
  windowSize = traces.length,
): RecentDecisionTraceSummary {
  const histograms: RecentDecisionTraceSummary["histograms"] = {
    decisionOutcome: {
      served_full: 0,
      served_clipped: 0,
      partial_fail_open: 0,
      partial_fail_open_clipped: 0,
      interrupted_without_partial: 0,
    },
    brainDropReason: {},
    interruptionStage: {},
    fitStrategy: {},
    queryEmbeddingSource: {},
  };
  const branchStopTruth: Record<string, number> = {};
  const branchTerminationReason: Record<string, number> = {};

  let sampleSize = 0;
  let clippedCount = 0;
  let failOpenCount = 0;
  let branchCount = 0;
  let continuingBranchCount = 0;

  for (const trace of traces) {
    const selectionMetadata = trace.routeTrace?.selectionMetadata ?? null;
    if (!selectionMetadata) {
      continue;
    }
    sampleSize += 1;
    histograms.decisionOutcome[classifyRecentDecisionOutcome(selectionMetadata)] += 1;
    incrementHistogram(histograms.brainDropReason, selectionMetadata.brainDropReason ?? "none");
    incrementHistogram(histograms.interruptionStage, selectionMetadata.interruptionStage ?? "none");
    incrementHistogram(histograms.fitStrategy, selectionMetadata.fitStrategy ?? "none");
    incrementHistogram(histograms.queryEmbeddingSource, selectionMetadata.queryEmbeddingSource ?? "unknown");
    if (isClippedDecision(selectionMetadata)) {
      clippedCount += 1;
    }
    if (isFailOpenDecision(selectionMetadata)) {
      failOpenCount += 1;
    }
    for (const branchOutcome of trace.routeTrace?.branchOutcomes ?? []) {
      branchCount += 1;
      if (branchOutcome.continued) {
        continuingBranchCount += 1;
      }
      incrementHistogram(branchStopTruth, branchOutcome.stopTruth ?? "unknown");
      incrementHistogram(branchTerminationReason, branchOutcome.terminationReason ?? "unknown");
    }
  }

  const clippedRate = buildRateSummary(clippedCount, sampleSize);
  const failOpenRate = buildRateSummary(failOpenCount, sampleSize);

  return {
    windowSize,
    sampleSize,
    histograms,
    branchBehavior: {
      branchCount,
      continuingBranchCount,
      histograms: {
        stopTruth: branchStopTruth,
        terminationReason: branchTerminationReason,
      },
      detail: branchCount === 0
        ? "no recent branch stop/continue traces"
        : `${continuingBranchCount}/${branchCount} recent branches continued; stop truths ${Object.entries(branchStopTruth).map(([truth, count]) => `${truth}=${count}`).join(", ")}; reasons ${Object.entries(branchTerminationReason).map(([reason, count]) => `${reason}=${count}`).join(", ")}`,
    },
    clipRate: clippedRate,
    failOpenRate,
    detail: sampleSize === 0
      ? "no recent traced decisions"
      : `${clippedCount}/${sampleSize} clipped and ${failOpenCount}/${sampleSize} fail-open or interrupted across the recent decision window`,
  };
}

function buildRouteTrace(params: {
  traversalResult: TraverseResult;
  queryText: string;
  conversationId: number | null;
  packVersion: number | null;
  budgetChars: number;
  maxHops: number;
  maxFanoutPerNode: number;
  maxFrontierSize: number;
  embeddingMs: number | null;
  routeSelectionMs: number | null;
  totalQueryMs: number | null;
  queryEmbeddingSource: "provided" | "runtime";
  selectedNodes: BrainNode[];
  persistRawSurfaces: boolean;
}): DecisionRouteTrace {
  const interruption = params.traversalResult.interruption ?? null;
  const candidateIds = candidateNodeIds(params.traversalResult);
  const selectedIds = params.traversalResult.firedNodes.map((node) => node.nodeId);
  const selectedTraversalIds = selectedTraversalNodeIds(params.traversalResult);
  const selectedSeedNodeIds = params.traversalResult.seedScores
    .filter((seed) => seed.selected)
    .map((seed) => seed.nodeId);
  const chosenSeedNodeId = selectedSeedNodeIds.length === 1 ? selectedSeedNodeIds[0] : null;
  const { chosenStopCount, forcedStopCount } = countStopTruths(params.traversalResult);
  const { branchOutcomes, branchOutcomeSummary } = summarizeBranchOutcomes(params.traversalResult);
  const { droppedProposalCount, droppedProposalReasons } = countDroppedProposalReasons(params.traversalResult);
  const selectionSubstepCount = params.traversalResult.trajectory.reduce(
    (count, expansion) => count + expansion.substeps.length,
    0,
  );
  const injectedNodeSummaries = params.selectedNodes.map((node) => summarizeInjectedNode(node));

  const rawRouteTrace: DecisionRouteTrace = {
    persistenceMode: buildPersistenceMode(params.persistRawSurfaces),
    requestDigest: hashQuery(params.queryText),
    conversationId: params.conversationId,
    activePackId: params.packVersion === null ? null : `brain-pack-v${params.packVersion}`,
    routerIdentity: ROUTER_IDENTITY,
    candidateNodeIds: candidateIds,
    selectedNodeIds: selectedIds,
    selectedTraversalNodeIds: selectedTraversalIds,
    selectedPathNodeIds: selectedTraversalIds,
    selectedSeedNodeIds,
    branchOutcomes,
    injectedNodeSummaries,
    sourceSummary: {
      injectedCount: injectedNodeSummaries.length,
      kinds: countBy(injectedNodeSummaries.map((node) => node.kind as NodeKind)),
      trusts: countBy(injectedNodeSummaries.map((node) => node.trust as TrustLevel)),
      sourceUris: [...new Set(injectedNodeSummaries.flatMap((node) => node.sourceUri ? [node.sourceUri] : []))],
      sourceRefs: [...new Set(injectedNodeSummaries.map((node) => node.provenanceRef).filter((value): value is string => Boolean(value)))],
    },
    operatorAudit: params.persistRawSurfaces
      ? {
          queryText: params.queryText,
          injectedNodeSummaries: injectedNodeSummaries.map(cloneInjectedNodeSummary),
        }
      : null,
    selectionMetadata: {
      traceSliceVersion: 4,
      queryChars: params.queryText.length,
      budgetChars: params.budgetChars,
      maxHops: params.maxHops,
      maxFanoutPerNode: params.maxFanoutPerNode,
      maxFrontierSize: params.maxFrontierSize,
      seedCount: params.traversalResult.seedScores.length,
      seedSelectionCount: selectedSeedNodeIds.length,
      candidateCount: candidateIds.length,
      hopCount: selectedTraversalIds.length,
      expansionCount: params.traversalResult.trajectory.length,
      selectionSubstepCount,
      firedCount: params.traversalResult.firedNodes.length,
      vetoedCount: params.traversalResult.vetoedNodes.length,
      chosenSeedNodeId,
      selectedSeedNodeIds,
      routeSelectionMs: params.routeSelectionMs,
      embeddingMs: params.embeddingMs,
      totalQueryMs: params.totalQueryMs,
      queryEmbeddingSource: params.queryEmbeddingSource,
      chosenStopCount,
      forcedStopCount,
      branchOutcomeSummary,
      droppedProposalCount,
      droppedProposalReasons,
      interruption,
      queryInterrupted: interruption?.interrupted ?? false,
      interruptionStage: interruption?.stage ?? null,
      interruptionReason: interruption?.reason ?? null,
      servedPartial: interruption?.servedPartial ?? false,
      interruptionAccounting: params.traversalResult.interruptionAccounting ?? null,
    },
  };

  return params.persistRawSurfaces
    ? rawRouteTrace
    : (redactRouteTrace(rawRouteTrace, params.queryText, false) ?? rawRouteTrace);
}

export function recordTrace(params: {
  traversalResult: TraverseResult;
  queryText: string;
  episodeId: string | null;
  conversationId: number | null;
  packVersion: number | null;
  budgetChars: number;
  maxHops: number;
  maxFanoutPerNode: number;
  maxFrontierSize: number;
  embeddingMs: number | null;
  routeSelectionMs: number | null;
  totalQueryMs: number | null;
  queryEmbeddingSource: "provided" | "runtime";
  selectedNodes: BrainNode[];
  persistRawSurfaces?: boolean;
}): DecisionTrace {
  const persistRawSurfaces = params.persistRawSurfaces ?? false;
  return {
    id: `bt_${randomUUID().slice(0, 8)}`,
    episodeId: params.episodeId,
    packVersion: params.packVersion,
    queryText: persistRawSurfaces
      ? params.queryText
      : (redactTextSurface("query", params.queryText) ?? ""),
    seedScores: params.traversalResult.seedScores,
    trajectory: params.traversalResult.trajectory,
    firedNodes: params.traversalResult.firedNodes.map((n) => n.nodeId),
    vetoedNodes: params.traversalResult.vetoedNodes.map((v) => v.nodeId),
    contextChars: params.traversalResult.contextChars,
    footer: params.traversalResult.footer,
    routeTrace: buildRouteTrace({ ...params, persistRawSurfaces }),
    createdAt: Date.now(),
  };
}

export function generateFooter(params: {
  packVersion: number;
  seedCount: number;
  seedSelectionCount: number;
  expansionCount: number;
  firedCount: number;
  vetoCount: number;
  contextChars: number;
  traceId: string;
}): string {
  return `Brain v${params.packVersion} · ${params.seedCount} seed candidates · ${params.seedSelectionCount} seed picks · ${params.expansionCount} expansions · ${params.firedCount} fired · ${params.vetoCount} veto · ${params.contextChars} chars · trace ${params.traceId}`;
}
