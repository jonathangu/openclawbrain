/**
 * Decision trace recording and footer generation.
 */

import { createHash, randomUUID } from "node:crypto";
import type {
  BrainPersistenceMode,
  BrainNode,
  BrainObservationToolResult,
  DecisionTraceBranchOutcome,
  DecisionRouteTrace,
  DecisionTrace,
  DecisionTraceInjectedNodeSummary,
  NodeKind,
  TrustLevel,
} from "./types.js";
import type { TraverseResult } from "./traverse.js";
import { resolveStopTruth } from "./trajectory-stop.js";

const ROUTER_IDENTITY = "brain-graph-traverse.v2";
const TRACE_PREVIEW_CHARS = 160;

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
      traceSliceVersion: 3,
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
