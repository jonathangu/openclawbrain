import { describe, expect, it } from "vitest";
import { Value } from "@sinclair/typebox/value";
import { createAttributionTruthRecord, recordTrace } from "../../src/brain-core/trace.js";
import { materializeRouteDecisionRowsFromTraceV1, RouteDecisionRowSchemaV1, summarizeRouteDecisionRowV1, validateLabelProvenanceV1, validateRouteDecisionRowV1 } from "../../src/brain-core/route-rows.js";
import type { BrainNode, DecisionPointSnapshotV1, SeedScore, TrajectoryExpansion, TrajectoryStopReason, TrajectoryStopTruth } from "../../src/brain-core/types.js";
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

function makeToolNode(id: string, toolName: string, role: "capability" | "instance" = "capability"): BrainNode {
  return {
    ...makeNode(id),
    kind: "toolcard",
    metadata: {
      toolName,
      toolArgsShape: "query,timeout",
      toolRole: role,
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
    footer: "Brain · 1 seed candidates · 1 seed picks · 1 expansions · 1 fired · 0 veto · 6 chars",
    interruption: null,
  };

  return recordTrace({
    traversalResult,
    queryText: "find the tool",
    episodeId: "ep-route-row-bridge",
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
    selectedNodes,
    lookupNode: (nodeId) => {
      if (nodeId === "tool_1") {
        return makeToolNode("tool_1", "bash", "capability");
      }
      if (nodeId === "tool_2") {
        return makeToolNode("tool_2", "wttr", "instance");
      }
      if (nodeId === "doc_1") {
        return makeNode("doc_1");
      }
      return null;
    },
    persistRawSurfaces: false,
  });
}

function makeTraceLabelProvenance(traceId: string, decisionPointId: string) {
  return createAttributionTruthRecord({
    state: "matched",
    observation: {
      observationId: `obs_${decisionPointId}`,
      episodeId: "ep-route-row-bridge",
      conversationId: 12,
      traceId,
      bindingMode: "trace_id",
      requestDigest: "req_route_row_bridge",
      serveDecisionRecordId: decisionPointId,
      selectionDigest: "sel_route_row_bridge",
      turnCompileEventId: null,
      provenanceRef: null,
    },
    supervision: {
      supervisionId: `sup_${decisionPointId}`,
      episodeId: "ep-route-row-bridge",
      conversationId: 12,
      source: "teacher",
      kind: "teacher_review",
      observationId: `obs_${decisionPointId}`,
      traceId,
      teacherTraceId: traceId,
      serveDecisionRecordId: decisionPointId,
      selectionDigest: "sel_route_row_bridge",
      turnCompileEventId: null,
      bindingMode: "trace_id",
      attributionQuality: "exact",
      feedbackRichness: "followup_and_tool",
      traceRequestDigest: "req_route_row_bridge",
      provenanceRef: null,
    },
    update: {
      updateId: `upd_${decisionPointId}`,
      episodeId: "ep-route-row-bridge",
      observationIds: [`obs_${decisionPointId}`],
      supervisionIds: [`sup_${decisionPointId}`],
      traceIds: [traceId],
      rewardSource: "teacher",
      attributionQuality: "exact",
      feedbackRichness: "followup_and_tool",
      routeUpdateCount: 1,
      seedUpdateCount: 0,
      stopLocalUpdateCount: 1,
      edgeUpdateCount: 1,
      updateReason: "selected tool instance after capability gate",
      provenanceRef: null,
    },
    linkage: {
      observationToSupervision: {
        state: "matched",
        basis: "trace_id",
        confidence: 0.98,
        detail: "teacher supervision matches the traced decision point",
        candidateIds: [decisionPointId],
      },
      supervisionToUpdate: {
        state: "matched",
        basis: "selection_digest",
        confidence: 0.96,
        detail: "update follows the supervised decision point",
        candidateIds: [decisionPointId],
      },
    },
    createdAt: 1_746_000_000_000,
    updatedAt: 1_746_000_000_500,
  });
}

describe("route rows", () => {
  it("materializes a supervised route row from a traced decision point with traverse, tool capability, tool instance, and stop_local candidates", () => {
    const trace = makeTrace([
      {
        sourceNodeId: "a",
        expansionIndex: 0,
        frontierBefore: ["a"],
        frontierAfter: [],
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
                probability: 0.35,
                scoreBreakdown: { totalScore: 2.5, toolActionPrior: 1.2 },
              },
              {
                action: { type: "traverse" as const, targetNodeId: "tool_2" },
                score: 2.75,
                probability: 0.3,
                scoreBreakdown: { totalScore: 2.75, toolActionPrior: 1.35 },
              },
              {
                action: { type: "stop_local" as const },
                score: 0.1,
                probability: 0.15,
                scoreBreakdown: { totalScore: 0.1, learnedStopWeight: -0.2 },
              },
            ],
            chosenAction: { type: "traverse" as const, targetNodeId: "tool_2" },
            chosenActionProbability: 0.3,
            stopProbability: 0.15,
          },
        ],
        selectedTargets: ["tool_2"],
        acceptedTargets: ["tool_2"],
        vetoedTargets: [],
        proposalOutcomes: [{ targetNodeId: "tool_2", outcome: "accepted", reason: "accepted" }],
        terminationReason: "policy_stop",
      },
    ] as TrajectoryExpansion[], ["tool_2"], [makeToolNode("tool_2", "wttr", "instance")]);

    const snapshot: DecisionPointSnapshotV1 | undefined = trace.routeTrace?.selectionMetadata.decisionPointSnapshots?.[0];
    if (!snapshot) {
      throw new Error("expected a decision point snapshot");
    }

    const retryTraceId = trace.routeTrace?.selectionMetadata.retryIdentity?.traceId ?? trace.id;
    const provenance = makeTraceLabelProvenance(retryTraceId, snapshot.decisionPointId);
    const rows = materializeRouteDecisionRowsFromTraceV1({
      trace,
      provenanceByDecisionPointId: {
        [snapshot.decisionPointId]: provenance,
      },
    });

    expect(rows).toHaveLength(1);
    const row = rows[0]!;

    expect(Value.Check(RouteDecisionRowSchemaV1, row)).toBe(true);
    expect(validateRouteDecisionRowV1(row)).toMatchObject({ valid: true });
    expect(validateLabelProvenanceV1(row.label_provenance)).toMatchObject({ valid: true });
    expect(row.route_fn_version).toBe("brain-graph-traverse.v2");
    expect(row.chosen_action_kind).toBe("tool_instance");
    expect(row.stop_label).toBe("CONTINUE");
    expect(row.local_action_set.map((candidate) => candidate.action_kind)).toEqual([
      "traverse",
      "tool_capability",
      "tool_instance",
      "stop_local",
    ]);
    expect(row.label_provenance).toMatchObject({
      state: "matched",
      basis: "trace_id",
      source: "teacher",
      kind: "teacher_review",
      binding_mode: "trace_id",
      attribution_quality: "exact",
      feedback_richness: "followup_and_tool",
      trace_id: retryTraceId,
      decision_point_id: snapshot.decisionPointId,
    });
    expect(row.hard_negatives).toEqual(["doc_1", "tool_1"]);
    expect(row.evidence_spans[0]?.source_ref).toMatch(/^(?:req_route_row_bridge|prov_|[a-z0-9:_-]+)$/i);
    expect(summarizeRouteDecisionRowV1(row)).toMatchObject({
      traceId: retryTraceId,
      decisionPointId: snapshot.decisionPointId,
      chosenActionKind: "tool_instance",
      stopLabel: "CONTINUE",
      localActionCount: 4,
      hardNegativeCount: 2,
      provenanceState: "matched",
    });
  });

  it("keeps row ids and trace ids stable across repeated materialization of the same logical trace", () => {
    const firstTrace = makeTrace([
      {
        sourceNodeId: "a",
        expansionIndex: 0,
        frontierBefore: ["a"],
        frontierAfter: [],
        budgetBefore: 100,
        budgetAfter: 92,
        substeps: [
          {
            stateSnapshot: makeStateSnapshot("a", 0, 0),
            candidates: [
              { action: { type: "traverse" as const, targetNodeId: "doc_1" }, score: 1.25, probability: 0.2 },
              { action: { type: "traverse" as const, targetNodeId: "tool_2" }, score: 2.75, probability: 0.3 },
              { action: { type: "stop_local" as const }, score: 0.1, probability: 0.15 },
            ],
            chosenAction: { type: "traverse" as const, targetNodeId: "tool_2" },
            chosenActionProbability: 0.3,
            stopProbability: 0.15,
          },
        ],
        selectedTargets: ["tool_2"],
        acceptedTargets: ["tool_2"],
        vetoedTargets: [],
        proposalOutcomes: [{ targetNodeId: "tool_2", outcome: "accepted", reason: "accepted" }],
        terminationReason: "policy_stop",
      },
    ] as TrajectoryExpansion[], ["tool_2"], [makeToolNode("tool_2", "wttr", "instance")]);
    const secondTrace = makeTrace([
      {
        sourceNodeId: "a",
        expansionIndex: 0,
        frontierBefore: ["a"],
        frontierAfter: [],
        budgetBefore: 100,
        budgetAfter: 92,
        substeps: [
          {
            stateSnapshot: makeStateSnapshot("a", 0, 0),
            candidates: [
              { action: { type: "traverse" as const, targetNodeId: "doc_1" }, score: 1.25, probability: 0.2 },
              { action: { type: "traverse" as const, targetNodeId: "tool_2" }, score: 2.75, probability: 0.3 },
              { action: { type: "stop_local" as const }, score: 0.1, probability: 0.15 },
            ],
            chosenAction: { type: "traverse" as const, targetNodeId: "tool_2" },
            chosenActionProbability: 0.3,
            stopProbability: 0.15,
          },
        ],
        selectedTargets: ["tool_2"],
        acceptedTargets: ["tool_2"],
        vetoedTargets: [],
        proposalOutcomes: [{ targetNodeId: "tool_2", outcome: "accepted", reason: "accepted" }],
        terminationReason: "policy_stop",
      },
    ] as TrajectoryExpansion[], ["tool_2"], [makeToolNode("tool_2", "wttr", "instance")]);

    const firstSnapshot = firstTrace.routeTrace?.selectionMetadata.decisionPointSnapshots?.[0];
    const secondSnapshot = secondTrace.routeTrace?.selectionMetadata.decisionPointSnapshots?.[0];
    if (!firstSnapshot || !secondSnapshot) {
      throw new Error("expected decision point snapshots");
    }

    const firstRetryTraceId = firstTrace.routeTrace?.selectionMetadata.retryIdentity?.traceId ?? firstTrace.id;
    const secondRetryTraceId = secondTrace.routeTrace?.selectionMetadata.retryIdentity?.traceId ?? secondTrace.id;
    const firstRow = materializeRouteDecisionRowsFromTraceV1({
      trace: firstTrace,
      provenanceByDecisionPointId: {
        [firstSnapshot.decisionPointId]: makeTraceLabelProvenance(firstRetryTraceId, firstSnapshot.decisionPointId),
      },
    })[0]!;
    const secondRow = materializeRouteDecisionRowsFromTraceV1({
      trace: secondTrace,
      provenanceByDecisionPointId: {
        [secondSnapshot.decisionPointId]: makeTraceLabelProvenance(secondRetryTraceId, secondSnapshot.decisionPointId),
      },
    })[0]!;

    expect(firstRow.row_id).toBe(secondRow.row_id);
    expect(firstRow.trace_id).toBe(secondRow.trace_id);
    expect(firstRow.label_provenance.trace_id).toBe(secondRow.label_provenance.trace_id);
    expect(firstRow.decision_point_id).toBe(secondRow.decision_point_id);
    expect(firstRow.chosen_action_id).toBe(secondRow.chosen_action_id);
    expect(firstRow.row_id).toMatch(/^rr_[a-f0-9]{16}$/);
  });
});
