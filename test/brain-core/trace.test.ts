import { describe, expect, it } from "vitest";
import { recordTrace, summarizeRecentDecisionTraces } from "../../src/brain-core/trace.js";
import type {
  BrainNode,
  SeedScore,
  TrajectoryExpansion,
  TrajectoryStopReason,
  TrajectoryStopTruth,
} from "../../src/brain-core/types.js";
import type { TraverseResult } from "../../src/brain-core/traverse.js";

function makeNode(id: string): BrainNode {
  return {
    id,
    kind: "chunk",
    content: `node ${id}`,
    embedding: new Float32Array([1, 0, 0]),
    sourceUri: "test.md",
    trust: "scanner",
    tags: [],
    tokenCount: 10,
    metadata: {},
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function makeSeedScore(nodeId: string, selected: boolean, selectionSubstepIndex: number | null): SeedScore {
  return {
    nodeId,
    priorScore: 1,
    learnedSeedWeight: 0,
    initialPolicyScore: 1,
    initialProbability: 0.8,
    latestPolicyScore: 1,
    latestProbability: 0.8,
    selected,
    selectionSubstepIndex,
  };
}

function makeStateSnapshot(sourceNodeId: string | null, expansionIndex: number, selectionIndex: number) {
  return {
    sourceNodeId,
    expansionIndex,
    selectionIndex,
    budgetRemaining: 100,
    initialBudget: 100,
    reservedTokenCost: 0,
    maxHops: 4,
    frontierSize: 0,
    frontierNodeIds: [],
    visitedCount: 0,
    firedCount: 0,
  };
}

function makeTraverseSubstep(
  sourceNodeId: string | null,
  expansionIndex: number,
  selectionIndex: number,
  targetNodeId: string,
) {
  return {
    stateSnapshot: makeStateSnapshot(sourceNodeId, expansionIndex, selectionIndex),
    candidates: [
      { action: { type: "traverse" as const, targetNodeId }, score: 1, probability: 0.8 },
      { action: { type: "stop_local" as const }, score: 0, probability: 0.2 },
    ],
    chosenAction: { type: "traverse" as const, targetNodeId },
    chosenActionProbability: 0.8,
    stopProbability: 0.2,
  };
}

function makeStopSubstep(
  sourceNodeId: string | null,
  expansionIndex: number,
  selectionIndex: number,
  stopTruth: TrajectoryStopTruth,
  stopReason: TrajectoryStopReason,
) {
  return {
    stateSnapshot: makeStateSnapshot(sourceNodeId, expansionIndex, selectionIndex),
    candidates: [
      { action: { type: "stop_local" as const }, score: 1, probability: 1 },
    ],
    chosenAction: { type: "stop_local" as const },
    chosenActionProbability: 1,
    stopProbability: 1,
    stopTruth,
    stopReason,
  };
}

function makeTrace(
  trajectory: TrajectoryExpansion[],
  firedNodeIds: string[],
  selectedNodes: BrainNode[],
  seedScores: SeedScore[] = [makeSeedScore("a", true, 0)],
) {
  const traversalResult: TraverseResult = {
    firedNodes: firedNodeIds.map((nodeId) => ({
      nodeId,
      kind: "chunk",
      content: `node ${nodeId}`,
      tokenCount: 10,
    })),
    vetoedNodes: [],
    trajectory,
    seedScores,
    contextChars: firedNodeIds.length * 6,
    footer: "Brain · 1 seed candidates · 1 seed picks · 2 expansions · 1 fired · 0 veto · 6 chars",
    interruption: null,
  };

  return recordTrace({
    traversalResult,
    queryText: "trace branch proof",
    episodeId: "ep-trace-branch-proof",
    conversationId: 7,
    packVersion: 2,
    budgetChars: 100,
    maxHops: 4,
    maxFanoutPerNode: 4,
    maxFrontierSize: 8,
    embeddingMs: 3,
    routeSelectionMs: 5,
    totalQueryMs: 8,
    queryEmbeddingSource: "provided",
    selectedNodes,
    persistRawSurfaces: false,
  });
}

describe("decision trace branch proofs", () => {
  it("records per-branch stop and continue proof lines plus a compact summary", () => {
    const trace = makeTrace([
      {
        sourceNodeId: null,
        expansionIndex: 0,
        frontierBefore: [],
        frontierAfter: ["a"],
        budgetBefore: 100,
        budgetAfter: 90,
        substeps: [
          makeTraverseSubstep(null, 0, 0, "a"),
          makeStopSubstep(null, 0, 1, "forced", "no_traversable_candidates"),
        ],
        selectedTargets: ["a"],
        acceptedTargets: ["a"],
        vetoedTargets: [],
        proposalOutcomes: [
          { targetNodeId: "a", outcome: "accepted", reason: "accepted" },
        ],
        terminationReason: "no_traversable_candidates",
      },
      {
        sourceNodeId: "a",
        expansionIndex: 1,
        frontierBefore: ["a"],
        frontierAfter: [],
        budgetBefore: 90,
        budgetAfter: 90,
        substeps: [
          makeStopSubstep("a", 1, 0, "chosen", "policy_stop"),
        ],
        selectedTargets: [],
        acceptedTargets: [],
        vetoedTargets: [{ targetNodeId: "c", reason: "inhibitory edge" }],
        proposalOutcomes: [
          { targetNodeId: "c", outcome: "vetoed", reason: "inhibitory edge" },
        ],
        terminationReason: "policy_stop",
      },
    ], ["a"], [makeNode("a")]);

    expect(trace.routeTrace?.branchOutcomes).toEqual([
      {
        sourceNodeId: null,
        expansionIndex: 0,
        selectionSubstepCount: 2,
        continued: true,
        selectedTargetIds: ["a"],
        acceptedTargetIds: ["a"],
        vetoedTargetIds: [],
        droppedTargetIds: [],
        stopTruth: "forced",
        stopReason: "no_traversable_candidates",
        terminationReason: "no_traversable_candidates",
        proof: "branch start continued via a then forced stop (no_traversable_candidates); accepted=1 vetoed=0 dropped=0",
      },
      {
        sourceNodeId: "a",
        expansionIndex: 1,
        selectionSubstepCount: 1,
        continued: false,
        selectedTargetIds: [],
        acceptedTargetIds: [],
        vetoedTargetIds: ["c"],
        droppedTargetIds: [],
        stopTruth: "chosen",
        stopReason: "policy_stop",
        terminationReason: "policy_stop",
        proof: "branch a stopped without continuation then chosen stop (policy_stop); accepted=0 vetoed=1 dropped=0",
      },
    ]);
    expect(trace.routeTrace?.selectionMetadata.branchOutcomeSummary).toEqual({
      branchCount: 2,
      continuingBranchCount: 1,
      stoppedWithoutProgressCount: 1,
      chosenStopBranchCount: 1,
      forcedStopBranchCount: 1,
      terminationReasons: {
        no_traversable_candidates: 1,
        policy_stop: 1,
      },
      detail: "1/2 branches continued; 1/2 stopped without continuation; chosen=1; forced=1; reasons no_traversable_candidates=1, policy_stop=1",
    });
  });

  it("aggregates branch stop and continue behavior across recent traced decisions", () => {
    const firstTrace = makeTrace([
      {
        sourceNodeId: null,
        expansionIndex: 0,
        frontierBefore: [],
        frontierAfter: ["a"],
        budgetBefore: 100,
        budgetAfter: 90,
        substeps: [
          makeTraverseSubstep(null, 0, 0, "a"),
          makeStopSubstep(null, 0, 1, "forced", "no_traversable_candidates"),
        ],
        selectedTargets: ["a"],
        acceptedTargets: ["a"],
        vetoedTargets: [],
        proposalOutcomes: [
          { targetNodeId: "a", outcome: "accepted", reason: "accepted" },
        ],
        terminationReason: "no_traversable_candidates",
      },
      {
        sourceNodeId: "a",
        expansionIndex: 1,
        frontierBefore: ["a"],
        frontierAfter: [],
        budgetBefore: 90,
        budgetAfter: 90,
        substeps: [
          makeStopSubstep("a", 1, 0, "chosen", "policy_stop"),
        ],
        selectedTargets: [],
        acceptedTargets: [],
        vetoedTargets: [],
        proposalOutcomes: [],
        terminationReason: "policy_stop",
      },
    ], ["a"], [makeNode("a")]);
    const secondTrace = makeTrace([
      {
        sourceNodeId: null,
        expansionIndex: 0,
        frontierBefore: [],
        frontierAfter: [],
        budgetBefore: 100,
        budgetAfter: 100,
        substeps: [
          makeStopSubstep(null, 0, 0, "forced", "frontier_cap"),
        ],
        selectedTargets: [],
        acceptedTargets: [],
        vetoedTargets: [],
        proposalOutcomes: [],
        terminationReason: "frontier_cap",
      },
    ], [], []);

    const summary = summarizeRecentDecisionTraces([firstTrace, secondTrace], 10);

    expect(summary.branchBehavior).toEqual({
      branchCount: 3,
      continuingBranchCount: 1,
      histograms: {
        stopTruth: {
          forced: 2,
          chosen: 1,
        },
        terminationReason: {
          no_traversable_candidates: 1,
          policy_stop: 1,
          frontier_cap: 1,
        },
      },
      detail: "1/3 recent branches continued; stop truths forced=2, chosen=1; reasons no_traversable_candidates=1, policy_stop=1, frontier_cap=1",
    });
  });
});
