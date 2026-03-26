/**
 * Decision trace recording and footer generation.
 */

import { createHash, randomUUID } from "node:crypto";
import type {
  BrainNode,
  DecisionRouteTrace,
  DecisionTrace,
  NodeKind,
  TrustLevel,
} from "./types.js";
import type { TraverseResult } from "./traverse.js";

const ROUTER_IDENTITY = "brain-graph-traverse.v2";
const TRACE_PREVIEW_CHARS = 160;

function hashQuery(queryText: string): string {
  return createHash("sha256").update(queryText).digest("hex").slice(0, 16);
}

function truncatePreview(content: string): string {
  const normalized = content.replace(/\s+/g, " ").trim();
  if (normalized.length <= TRACE_PREVIEW_CHARS) {
    return normalized;
  }
  return `${normalized.slice(0, TRACE_PREVIEW_CHARS - 1)}…`;
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
      if (substep.chosenAction.type !== "stop_local") {
        continue;
      }
      if (substep.stopTruth === "forced") {
        forcedStopCount += 1;
        continue;
      }
      chosenStopCount += 1;
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
  const injectedNodeSummaries = params.selectedNodes.map((node) => ({
    nodeId: node.id,
    kind: node.kind,
    trust: node.trust,
    sourceUri: node.sourceUri,
    tags: [...node.tags],
    tokenCount: node.tokenCount,
    contentPreview: truncatePreview(node.content),
  }));

  return {
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
    },
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
      queryInterrupted: false,
      interruptionStage: null,
      interruptionReason: null,
      servedPartial: false,
      chosenStopCount,
      forcedStopCount,
      droppedProposalCount,
      droppedProposalReasons,
      interruption,
      queryInterrupted: interruption?.interrupted ?? null,
      interruptionStage: interruption?.stage ?? null,
      interruptionReason: interruption?.reason ?? null,
      servedPartial: interruption?.servedPartial ?? null,
    },
  };
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
}): DecisionTrace {
  return {
    id: `bt_${randomUUID().slice(0, 8)}`,
    episodeId: params.episodeId,
    packVersion: params.packVersion,
    queryText: params.queryText,
    seedScores: params.traversalResult.seedScores,
    trajectory: params.traversalResult.trajectory,
    firedNodes: params.traversalResult.firedNodes.map((n) => n.nodeId),
    vetoedNodes: params.traversalResult.vetoedNodes.map((v) => v.nodeId),
    contextChars: params.traversalResult.contextChars,
    footer: params.traversalResult.footer,
    routeTrace: buildRouteTrace(params),
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
