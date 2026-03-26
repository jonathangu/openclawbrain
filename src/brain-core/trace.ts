/**
 * Decision trace recording and footer generation.
 */

import { createHash, randomUUID } from "node:crypto";
import type {
  BrainPersistenceMode,
  BrainNode,
  BrainObservationToolResult,
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
      droppedProposalCount,
      droppedProposalReasons,
      interruption,
      queryInterrupted: interruption?.interrupted ?? false,
      interruptionStage: interruption?.stage ?? null,
      interruptionReason: interruption?.reason ?? null,
      servedPartial: interruption?.servedPartial ?? false,
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
