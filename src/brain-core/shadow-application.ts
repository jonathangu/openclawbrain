/**
 * Shadow application substrate for Teacher v3 mutation proposals.
 *
 * The helper intentionally applies proposals only to graph copies / candidate
 * state and keeps the reversible operation log separate from live graph
 * mutation and persistence.
 */

import { createHash } from "node:crypto";
import type { BrainEdge, BrainNode, MutationProposal } from "./types.js";
import type { BrainGraph } from "./graph.js";

export type ShadowGraphOperation =
  | { kind: "insert_node"; node: BrainNode; edges?: BrainEdge[] }
  | { kind: "delete_node"; node: BrainNode; edges: BrainEdge[] }
  | { kind: "insert_edge"; edge: BrainEdge }
  | { kind: "delete_edge"; edge: BrainEdge };

export interface ShadowGraphSnapshot {
  nodeCount: number;
  edgeCount: number;
}

export interface ShadowMutationApplication {
  proposalId: string;
  proposalKind: MutationProposal["kind"];
  applied: boolean;
  reversible: boolean;
  reason: string | null;
  before: ShadowGraphSnapshot;
  after: ShadowGraphSnapshot;
  operations: ShadowGraphOperation[];
}

export interface ShadowCandidateState {
  baseGraph: BrainGraph;
  candidateGraph: BrainGraph;
  applications: ShadowMutationApplication[];
}

function cloneNode(node: BrainNode): BrainNode {
  return {
    ...node,
    embedding: node.embedding ? new Float32Array(node.embedding) : null,
    tags: [...node.tags],
    metadata: { ...node.metadata },
  };
}

function cloneEdge(edge: BrainEdge): BrainEdge {
  return {
    ...edge,
    metadata: { ...edge.metadata },
  };
}

function snapshotGraph(graph: BrainGraph): ShadowGraphSnapshot {
  return {
    nodeCount: graph.getAllNodes().length,
    edgeCount: graph.getAllEdges().length,
  };
}

function operationInverse(operation: ShadowGraphOperation): ShadowGraphOperation {
  switch (operation.kind) {
    case "insert_node":
      return { kind: "delete_node", node: cloneNode(operation.node), edges: (operation.edges ?? []).map(cloneEdge) };
    case "delete_node":
      return { kind: "insert_node", node: cloneNode(operation.node), edges: operation.edges.map(cloneEdge) };
    case "insert_edge":
      return { kind: "delete_edge", edge: cloneEdge(operation.edge) };
    case "delete_edge":
      return { kind: "insert_edge", edge: cloneEdge(operation.edge) };
  }
}

function applyGraphOperation(graph: BrainGraph, operation: ShadowGraphOperation): boolean {
  switch (operation.kind) {
    case "insert_node": {
      graph.addNode(cloneNode(operation.node));
      for (const edge of operation.edges ?? []) {
        graph.addEdge(cloneEdge(edge));
      }
      return true;
    }
    case "delete_node": {
      if (!graph.getNode(operation.node.id)) {
        return false;
      }
      graph.removeNode(operation.node.id);
      return true;
    }
    case "insert_edge": {
      graph.addEdge(cloneEdge(operation.edge));
      return true;
    }
    case "delete_edge": {
      const edge = graph
        .getOutgoingEdges(operation.edge.source)
        .find((candidate) =>
          candidate.target === operation.edge.target && candidate.kind === operation.edge.kind,
        );
      if (!edge) {
        return false;
      }
      graph.removeEdge(operation.edge.source, operation.edge.target, operation.edge.kind);
      return true;
    }
  }
}

function buildShadowOperations(
  graph: BrainGraph,
  proposal: MutationProposal,
): { operations: ShadowGraphOperation[]; reason: string | null } {
  const now = Date.now();
  const payload = proposal.proposal as Record<string, unknown>;

  switch (proposal.kind) {
    case "connect": {
      const source = typeof payload.nodeA === "string" ? payload.nodeA : null;
      const target = typeof payload.nodeB === "string" ? payload.nodeB : null;
      if (!source || !target) {
        return { operations: [], reason: "invalid connect proposal payload" };
      }

      const edge: BrainEdge = {
        source,
        target,
        kind: "learned",
        weight: 0.5,
        prior: 0.5,
        metadata: {
          mutationId: proposal.id,
          proposalKind: proposal.kind,
          shadowApplication: true,
        },
        decayedAt: now,
        createdAt: now,
      };

      return { operations: [{ kind: "insert_edge", edge }], reason: null };
    }

    case "prune": {
      const source = typeof payload.source === "string" ? payload.source : null;
      const target = typeof payload.target === "string" ? payload.target : null;
      if (!source || !target) {
        return { operations: [], reason: "invalid prune proposal payload" };
      }

      const kind = typeof payload.edgeKind === "string" ? (payload.edgeKind as BrainEdge["kind"]) : "learned";
      const edge = graph.getOutgoingEdges(source).find((candidate) => candidate.target === target && candidate.kind === kind);
      if (!edge) {
        return { operations: [], reason: "prune edge not present in candidate graph" };
      }

      return { operations: [{ kind: "delete_edge", edge: cloneEdge(edge) }], reason: null };
    }

    case "inject": {
      const content = typeof payload.content === "string" ? payload.content : String(payload.content ?? "");
      const firedNodes = Array.isArray(payload.firedNodes)
        ? payload.firedNodes.filter((value): value is string => typeof value === "string")
        : [];
      const nodeKind = typeof payload.nodeKind === "string"
        ? (payload.nodeKind as BrainNode["kind"])
        : "episode_anchor";
      const digest = createHash("sha256")
        .update(JSON.stringify({ proposalId: proposal.id, content, firedNodes, nodeKind }))
        .digest("hex")
        .slice(0, 12);

      const node: BrainNode = {
        id: `shadow_inject_${digest}`,
        kind: nodeKind,
        content,
        embedding: null,
        sourceUri: null,
        trust: "scanner",
        tags: ["episode-anchor", "shadow-application"],
        tokenCount: Math.ceil(content.length / 4),
        metadata: {
          mutationId: proposal.id,
          proposalKind: proposal.kind,
          shadowApplication: true,
          firedNodes,
        },
        createdAt: now,
        updatedAt: now,
      };

      const edges = firedNodes.map((target) => ({
        source: node.id,
        target,
        kind: "learned" as const,
        weight: 0.4,
        prior: 0.4,
        metadata: {
          mutationId: proposal.id,
          proposalKind: proposal.kind,
          shadowApplication: true,
        },
        decayedAt: now,
        createdAt: now,
      }));

      return {
        operations: [{ kind: "insert_node", node, edges }],
        reason: null,
      };
    }

    case "split":
    case "merge":
      return { operations: [], reason: `unsupported mutation kind: ${proposal.kind}` };
  }
}

function rollbackOperations(graph: BrainGraph, operations: ShadowGraphOperation[]): void {
  for (const operation of [...operations].reverse()) {
    const inverse = operationInverse(operation);
    if (!applyGraphOperation(graph, inverse)) {
      throw new Error(`shadow rollback failed for ${inverse.kind}`);
    }
  }
}

export function applyShadowMutationProposal(
  candidateGraph: BrainGraph,
  proposal: MutationProposal,
): ShadowMutationApplication {
  const before = snapshotGraph(candidateGraph);
  const { operations, reason } = buildShadowOperations(candidateGraph, proposal);

  if (reason !== null) {
    return {
      proposalId: proposal.id,
      proposalKind: proposal.kind,
      applied: false,
      reversible: true,
      reason,
      before,
      after: before,
      operations: [],
    };
  }

  const appliedOperations: ShadowGraphOperation[] = [];
  for (const operation of operations) {
    if (!applyGraphOperation(candidateGraph, operation)) {
      rollbackOperations(candidateGraph, appliedOperations);
      const afterRollback = snapshotGraph(candidateGraph);
      return {
        proposalId: proposal.id,
        proposalKind: proposal.kind,
        applied: false,
        reversible: true,
        reason: `failed to apply shadow ${operation.kind}`,
        before,
        after: afterRollback,
        operations: [],
      };
    }
    appliedOperations.push(operation);
  }

  const after = snapshotGraph(candidateGraph);
  return {
    proposalId: proposal.id,
    proposalKind: proposal.kind,
    applied: true,
    reversible: true,
    reason: null,
    before,
    after,
    operations: appliedOperations,
  };
}

export function revertShadowMutationApplication(
  candidateGraph: BrainGraph,
  application: ShadowMutationApplication,
): void {
  if (!application.applied || application.operations.length === 0) {
    return;
  }

  rollbackOperations(candidateGraph, application.operations);
}

export function createShadowCandidateState(baseGraph: BrainGraph): ShadowCandidateState {
  return {
    baseGraph,
    candidateGraph: baseGraph.clone(),
    applications: [],
  };
}

export function applyShadowMutationProposalToState(
  state: ShadowCandidateState,
  proposal: MutationProposal,
): ShadowMutationApplication {
  const application = applyShadowMutationProposal(state.candidateGraph, proposal);
  state.applications.push(application);
  return application;
}

export function revertShadowCandidateState(
  state: ShadowCandidateState,
  untilIndex = 0,
): void {
  if (untilIndex < 0 || untilIndex > state.applications.length) {
    throw new RangeError(`untilIndex ${untilIndex} is outside the applied shadow history`);
  }

  for (let index = state.applications.length - 1; index >= untilIndex; index -= 1) {
    revertShadowMutationApplication(state.candidateGraph, state.applications[index]);
  }
  state.applications.splice(untilIndex);
}

export function resetShadowCandidateState(state: ShadowCandidateState): void {
  revertShadowCandidateState(state, 0);
}
