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

const ROUTER_IDENTITY = "brain-graph-traverse.v1";
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
  for (const step of traversalResult.trajectory) {
    for (const candidate of step.candidates) {
      if (candidate.action.type === "traverse") {
        ids.add(candidate.action.targetNodeId);
      }
    }
  }
  return [...ids];
}

function selectedPathNodeIds(traversalResult: TraverseResult): string[] {
  return traversalResult.trajectory
    .flatMap((step) => step.chosenAction.type === "traverse" ? [step.chosenAction.targetNodeId] : [])
    .filter((nodeId, index, values) => values.indexOf(nodeId) === index);
}

function buildRouteTrace(params: {
  traversalResult: TraverseResult;
  queryText: string;
  conversationId: number | null;
  packVersion: number | null;
  budgetChars: number;
  maxHops: number;
  embeddingMs: number | null;
  routeSelectionMs: number | null;
  totalQueryMs: number | null;
  queryEmbeddingSource: "provided" | "runtime";
  selectedNodes: BrainNode[];
}): DecisionRouteTrace {
  const candidateIds = candidateNodeIds(params.traversalResult);
  const selectedIds = params.traversalResult.firedNodes.map((node) => node.nodeId);
  const selectedPathIds = selectedPathNodeIds(params.traversalResult);
  const chosenSeedNodeId = params.traversalResult.seedScores.find((seed) => seed.chosen)?.nodeId ?? null;
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
    selectedPathNodeIds: selectedPathIds,
    injectedNodeSummaries,
    sourceSummary: {
      injectedCount: injectedNodeSummaries.length,
      kinds: countBy(injectedNodeSummaries.map((node) => node.kind as NodeKind)),
      trusts: countBy(injectedNodeSummaries.map((node) => node.trust as TrustLevel)),
      sourceUris: [...new Set(injectedNodeSummaries.flatMap((node) => node.sourceUri ? [node.sourceUri] : []))],
    },
    selectionMetadata: {
      traceSliceVersion: 1,
      queryChars: params.queryText.length,
      budgetChars: params.budgetChars,
      maxHops: params.maxHops,
      seedCount: params.traversalResult.seedScores.length,
      candidateCount: candidateIds.length,
      hopCount: params.traversalResult.trajectory.length,
      firedCount: params.traversalResult.firedNodes.length,
      vetoedCount: params.traversalResult.vetoedNodes.length,
      chosenSeedNodeId,
      routeSelectionMs: params.routeSelectionMs,
      embeddingMs: params.embeddingMs,
      totalQueryMs: params.totalQueryMs,
      queryEmbeddingSource: params.queryEmbeddingSource,
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
  hopCount: number;
  firedCount: number;
  vetoCount: number;
  contextChars: number;
  traceId: string;
}): string {
  return `Brain v${params.packVersion} · ${params.seedCount} seeds · ${params.hopCount} hops · ${params.firedCount} fired · ${params.vetoCount} veto · ${params.contextChars} chars · trace ${params.traceId}`;
}
