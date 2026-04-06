import { describe, expect, it } from "vitest";
import {
  buildBrainCompileReport,
  createAttributionTruthRecord,
  recordTrace,
  summarizeRecentDecisionTraces,
  toAttributionTruthId,
  toAttributionUpdateId,
} from "../../src/brain-core/trace.js";
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

function makeToolNode(id: string, toolName: string): BrainNode {
  return {
    ...makeNode(id),
    kind: "toolcard",
    metadata: {
      toolName,
      toolArgsShape: "query,timeout",
    },
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
    maxFrontierSize: 4,
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
  it("exports decision-point snapshots with traverse, tool, and stop_local action sets", () => {
    const traversalResult: TraverseResult = {
      firedNodes: [
        { nodeId: "tool_1", kind: "toolcard", content: "tool node", tokenCount: 8 },
      ],
      vetoedNodes: [],
      trajectory: [
        {
          sourceNodeId: "a",
          expansionIndex: 0,
          frontierBefore: [],
          frontierAfter: ["tool_1"],
          budgetBefore: 100,
          budgetAfter: 92,
          substeps: [
            {
              stateSnapshot: makeStateSnapshot("a", 0, 0),
              candidates: [
                {
                  action: { type: "traverse" as const, targetNodeId: "doc_1" },
                  score: 1.25,
                  probability: 0.2,
                  scoreBreakdown: { totalScore: 1.25, seedPrior: 0.25 },
                },
                {
                  action: { type: "traverse" as const, targetNodeId: "tool_1" },
                  score: 2.5,
                  probability: 0.65,
                  scoreBreakdown: { totalScore: 2.5, toolActionPrior: 1.2 },
                },
                {
                  action: { type: "stop_local" as const },
                  score: 0.1,
                  probability: 0.15,
                  scoreBreakdown: { totalScore: 0.1, learnedStopWeight: -0.2 },
                },
              ],
              chosenAction: { type: "traverse" as const, targetNodeId: "tool_1" },
              chosenActionProbability: 0.65,
              stopProbability: 0.15,
            },
          ],
          selectedTargets: ["tool_1"],
          acceptedTargets: ["tool_1"],
          vetoedTargets: [],
          proposalOutcomes: [
            { targetNodeId: "tool_1", outcome: "accepted", reason: "accepted" },
          ],
          terminationReason: "policy_stop",
        },
      ],
      seedScores: [makeSeedScore("tool_1", false, null)],
      contextChars: 8,
      footer: "Brain · 1 seed candidates · 0 seed picks · 1 expansions · 1 fired · 0 veto · 8 chars",
      interruption: null,
      interruptionAccounting: null,
    };

    const trace = recordTrace({
      traversalResult,
      queryText: "find the tool",
      episodeId: "ep-decision-point-export",
      conversationId: 12,
      packVersion: 3,
      budgetChars: 100,
      maxHops: 4,
      maxFanoutPerNode: 4,
      maxFrontierSize: 8,
      embeddingMs: 1,
      routeSelectionMs: 2,
      totalQueryMs: 3,
      queryEmbeddingSource: "provided",
      selectedNodes: [makeToolNode("tool_1", "bash")],
      lookupNode: (nodeId) => {
        if (nodeId === "tool_1") {
          return makeToolNode("tool_1", "bash");
        }
        if (nodeId === "doc_1") {
          return makeNode("doc_1");
        }
        return null;
      },
      persistRawSurfaces: false,
    });

    const snapshots = trace.routeTrace?.selectionMetadata.decisionPointSnapshots ?? [];
    expect(snapshots).toHaveLength(1);
    expect(trace.routeTrace?.selectionMetadata.decisionPointSummary).toBe(
      "[brain decision points] total=1 actions traverse=1 tool=1 stop_local=1 stop=0",
    );
    expect(snapshots[0]).toMatchObject({
      schemaVersion: 1,
      decisionPointKind: "local",
      sourceNodeId: "a",
      chosenActionKind: "tool",
      chosenToolName: "bash",
      localActionSet: [
        expect.objectContaining({ actionKind: "traverse", nodeId: "doc_1" }),
        expect.objectContaining({ actionKind: "tool", nodeId: "tool_1", toolName: "bash" }),
        expect.objectContaining({ actionKind: "stop_local", nodeId: null }),
      ],
      budgetContext: expect.objectContaining({
        budgetRemaining: 100,
        initialBudget: 100,
        budgetUsed: 0,
        budgetUsedFraction: 0,
        routeSelectionMs: 2,
        totalQueryMs: 3,
      }),
      routeContext: expect.objectContaining({
        candidateNodeIds: ["tool_1", "doc_1"],
        selectedNodeIds: ["tool_1"],
        selectedTraversalNodeIds: ["tool_1"],
        selectedSeedNodeIds: [],
      }),
    });
  });

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
    const compileReport = buildBrainCompileReport({
      routeTrace: trace.routeTrace,
      decision: {
        mode: "use_brain",
        traceId: trace.id,
        episodeId: trace.episodeId,
      },
      lookupNode: (nodeId) => (nodeId === "a" ? makeNode(nodeId) : null),
    });
    expect(compileReport).toEqual(expect.objectContaining({
      schemaVersion: 1,
      summary: expect.stringContaining("[brain compile]"),
      counters: expect.objectContaining({
        selectedNodeCount: 1,
      }),
    }));
    expect(compileReport?.summary).toContain("q_budget=");
    expect(compileReport?.summary).toContain("inject_cap=");
    expect(compileReport?.summary).toContain("dropped_chars=");
  });

  it("attributes clipped selected nodes separately from traversal misses in the compile report", () => {
    const trace = makeTrace([
      {
        sourceNodeId: null,
        expansionIndex: 0,
        frontierBefore: [],
        frontierAfter: ["a", "b", "c"],
        budgetBefore: 100,
        budgetAfter: 70,
        substeps: [
          makeTraverseSubstep(null, 0, 0, "a"),
          makeTraverseSubstep(null, 0, 1, "b"),
          makeTraverseSubstep(null, 0, 2, "c"),
          makeStopSubstep(null, 0, 3, "forced", "fanout_cap"),
        ],
        selectedTargets: ["a", "b", "c"],
        acceptedTargets: ["a", "b", "c"],
        vetoedTargets: [],
        proposalOutcomes: [
          { targetNodeId: "a", outcome: "accepted", reason: "accepted" },
          { targetNodeId: "b", outcome: "accepted", reason: "accepted" },
          { targetNodeId: "c", outcome: "accepted", reason: "accepted" },
        ],
        terminationReason: "fanout_cap",
      },
    ], ["a", "b", "c"], [makeNode("a"), makeNode("b"), makeNode("c")]);

    if (!trace.routeTrace?.selectionMetadata) {
      throw new Error("expected route trace metadata");
    }
    trace.routeTrace = {
      ...trace.routeTrace,
      candidateNodeIds: ["a", "b", "c", "d"],
      selectionMetadata: {
        ...trace.routeTrace.selectionMetadata,
        brainDropReason: "deadline_after_query",
        brainDropStage: "query",
        contextClipped: true,
        fitStrategy: "structured_node_budget",
        fittedNodeCount: 2,
        droppedNodeCount: 2,
        fittingDropReasons: {
          omitted_for_partial_serve: 1,
        },
      },
    };

    const compileReport = buildBrainCompileReport({
      routeTrace: trace.routeTrace,
      decision: {
        mode: "partial_query_interruption",
        traceId: trace.id,
        episodeId: trace.episodeId,
      },
      lookupNode: (nodeId) => makeNode(nodeId),
    });

    expect(compileReport).toEqual(expect.objectContaining({
      counters: expect.objectContaining({
        selectedNodeCount: 2,
        droppedNodeCount: 2,
      }),
      reasons: expect.objectContaining({
        droppedNodeReasons: {
          omitted_for_partial_serve: 1,
          not_selected: 1,
        },
      }),
    }));
    expect(compileReport?.summary).toContain("q_budget=");
    expect(compileReport?.summary).toContain("inject_cap=");
    expect(compileReport?.buckets.selected.map((item) => item.nodeId)).toEqual(["a", "b"]);
    expect(compileReport?.buckets.dropped).toEqual(expect.arrayContaining([
      expect.objectContaining({
        nodeId: "c",
        reason: "omitted_for_partial_serve",
        fitStrategy: "structured_node_budget",
      }),
      expect.objectContaining({
        nodeId: "d",
        reason: "not_selected",
      }),
    ]));
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

  it("adds pressure visibility to branch proofs when policy-state trace data is present", () => {
    const trace = makeTrace([
      {
        sourceNodeId: "a",
        expansionIndex: 0,
        frontierBefore: ["a"],
        frontierAfter: [],
        budgetBefore: 90,
        budgetAfter: 90,
        substeps: [
          {
            ...makeStopSubstep("a", 0, 0, "chosen", "policy_stop"),
            stopProbability: 0.64,
            stateSnapshot: {
              ...makeStateSnapshot("a", 0, 0),
              pendingSelectionCount: 1,
              pendingTargetNodeIds: ["b"],
              policyState: {
                effectiveBudgetRemaining: 45,
                budgetUsedFraction: 0.55,
                frontierBacklogPressure: 0.5,
                frontierSaturation: 0.5,
                frontierPressure: 0.5,
                pressureLevel: 0.53,
                remainingExpansionSlots: 2,
                activeFrontierSize: 1,
                pendingSelectionCount: 1,
              },
            },
          },
        ],
        selectedTargets: [],
        acceptedTargets: [],
        vetoedTargets: [],
        proposalOutcomes: [],
        terminationReason: "policy_stop",
      },
    ], [], []);

    expect(trace.routeTrace?.branchOutcomes[0]?.proof).toContain(
      "chosen stop (policy_stop) [pressure=0.53 budget=0.55 frontier=0.50 stop_p=0.64 pending=1]",
    );
  });

  it("normalizes attribution truth ids and hashes from linkage inputs", () => {
    const updateIdA = toAttributionUpdateId({
      episodeId: "ep_attr",
      observationIds: ["bo_2", "bo_1", "bo_1"],
      supervisionIds: ["ts_2", "ts_1"],
      traceIds: ["bt_2", "bt_1"],
    });
    const updateIdB = toAttributionUpdateId({
      episodeId: "ep_attr",
      observationIds: ["bo_1", "bo_2"],
      supervisionIds: ["ts_1", "ts_2"],
      traceIds: ["bt_1", "bt_2"],
    });

    expect(updateIdA).toBe(updateIdB);

    const record = createAttributionTruthRecord({
      conversationId: 7,
      state: "ambiguous",
      observation: {
        observationId: "bo_1",
        episodeId: "ep_attr",
        conversationId: 7,
        traceId: "bt_1",
        bindingMode: "legacy_heuristic",
        requestDigest: "digest_1",
        serveDecisionRecordId: null,
        selectionDigest: "selection_1",
        turnCompileEventId: null,
        provenanceRef: null,
      },
      supervision: {
        supervisionId: "ts_1",
        episodeId: "ep_attr",
        conversationId: 7,
        source: "teacher",
        kind: "teacher_review",
        observationId: null,
        traceId: "bt_1",
        teacherTraceId: "bt_1",
        serveDecisionRecordId: null,
        selectionDigest: "selection_1",
        turnCompileEventId: null,
        bindingMode: "legacy_heuristic",
        attributionQuality: "fallback",
        feedbackRichness: "followup_only",
        traceRequestDigest: "digest_1",
        provenanceRef: null,
      },
      update: {
        updateId: updateIdA,
        episodeId: "ep_attr",
        observationIds: ["bo_2", "bo_1", "bo_1"],
        supervisionIds: ["ts_2", "ts_1"],
        traceIds: ["bt_2", "bt_1"],
        rewardSource: "teacher",
        attributionQuality: "mixed",
        feedbackRichness: "mixed",
        routeUpdateCount: 3,
        seedUpdateCount: 1,
        stopLocalUpdateCount: 1,
        edgeUpdateCount: 1,
        updateReason: "teacher fallback attribution updated 3 route weight(s)",
        provenanceRef: null,
      },
      linkage: {
        observationToSupervision: {
          state: "ambiguous",
          basis: "heuristic",
          confidence: 0.42,
          detail: "two viable observations remained after heuristic matching",
          candidateIds: ["bo_2", "bo_1", "bo_1"],
        },
        supervisionToUpdate: {
          state: "matched",
          basis: "manual",
          confidence: 1,
          detail: "update consumed the teacher supervision directly",
          candidateIds: ["upd_alt", updateIdA],
        },
      },
    });

    expect(record.attributionTruthId).toBe(toAttributionTruthId({
      observationId: "bo_1",
      supervisionId: "ts_1",
      updateId: updateIdA,
      episodeId: "ep_attr",
      state: "ambiguous",
    }));
    expect(record.update?.observationIds).toEqual(["bo_1", "bo_2"]);
    expect(record.update?.supervisionIds).toEqual(["ts_1", "ts_2"]);
    expect(record.update?.traceIds).toEqual(["bt_1", "bt_2"]);
    expect(record.linkage.observationToSupervision.candidateIds).toEqual(["bo_1", "bo_2"]);
    expect(record.linkage.supervisionToUpdate.candidateIds).toEqual([updateIdA, "upd_alt"]);
    expect(record.contentHash).toMatch(/^hash_/);
    expect(record.lineageHash).toMatch(/^lineage_/);
    expect(record.provenanceRef).toMatch(/^prov_/);
  });

  it("derives distinct automatic attribution truth ids for different truth states on the same lineage", () => {
    const delayed = createAttributionTruthRecord({
      conversationId: 7,
      state: "delayed",
      observation: {
        observationId: "bo_1",
        episodeId: "ep_attr",
        conversationId: 7,
        traceId: "bt_1",
        bindingMode: "trace_id",
        requestDigest: "digest_1",
        serveDecisionRecordId: null,
        selectionDigest: null,
        turnCompileEventId: null,
        provenanceRef: null,
      },
      linkage: {
        observationToSupervision: {
          state: "delayed",
          basis: "pending_observation",
          confidence: null,
          detail: "waiting for teacher supervision",
          candidateIds: [],
        },
        supervisionToUpdate: {
          state: "delayed",
          basis: "pending_update",
          confidence: null,
          detail: "no learner update yet",
          candidateIds: [],
        },
      },
    });
    const matched = createAttributionTruthRecord({
      conversationId: 7,
      state: "matched",
      observation: {
        observationId: "bo_1",
        episodeId: "ep_attr",
        conversationId: 7,
        traceId: "bt_1",
        bindingMode: "trace_id",
        requestDigest: "digest_1",
        serveDecisionRecordId: null,
        selectionDigest: null,
        turnCompileEventId: null,
        provenanceRef: null,
      },
      supervision: {
        supervisionId: "ts_1",
        episodeId: "ep_attr",
        conversationId: 7,
        source: "teacher",
        kind: "teacher_review",
        observationId: "bo_1",
        traceId: "bt_1",
        teacherTraceId: "bt_1",
        serveDecisionRecordId: null,
        selectionDigest: null,
        turnCompileEventId: null,
        bindingMode: "trace_id",
        attributionQuality: "fallback",
        feedbackRichness: "followup_only",
        traceRequestDigest: "digest_1",
        provenanceRef: null,
      },
      linkage: {
        observationToSupervision: {
          state: "matched",
          basis: "trace_id",
          confidence: 1,
          detail: "trace id bound the supervision",
          candidateIds: ["bo_1"],
        },
        supervisionToUpdate: {
          state: "matched",
          basis: "manual",
          confidence: 1,
          detail: "teacher supervision was consumed directly",
          candidateIds: ["upd_1"],
        },
      },
    });

    expect(delayed.attributionTruthId).not.toBe(matched.attributionTruthId);
    expect(delayed.lineageHash).toBe(matched.lineageHash);
    expect(delayed.contentHash).not.toBe(matched.contentHash);
  });
});
