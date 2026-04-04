import { describe, expect, it } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import {
  applyShadowMutationProposalToState,
  createShadowCandidateState,
} from "../../src/brain-core/shadow-application.js";
import type { BrainNode, MutationProposal } from "../../src/brain-core/types.js";
import {
  summarizeTeacherForgettingShadowReplayV1,
  summarizeTeacherMutationShadowReplayV1,
} from "../../src/brain-core/teacher-v3-shadow-replay.js";

function makeNode(id: string, kind: BrainNode["kind"] = "chunk"): BrainNode {
  const now = Date.now();
  return {
    id,
    kind,
    content: id,
    embedding: null,
    sourceUri: null,
    trust: "scanner",
    tags: [id],
    tokenCount: id.length,
    metadata: {},
    createdAt: now,
    updatedAt: now,
  };
}

function makeGraph(): BrainGraph {
  const graph = new BrainGraph();
  graph.addNode(makeNode("a"));
  graph.addNode(makeNode("b"));
  graph.addNode(makeNode("c"));
  graph.addEdge({
    source: "a",
    target: "b",
    kind: "learned",
    weight: 0.8,
    prior: 0.5,
    metadata: { seed: true },
    decayedAt: Date.now(),
    createdAt: Date.now(),
  });
  return graph;
}

function makeProposal(kind: MutationProposal["kind"], proposal: Record<string, unknown>): MutationProposal {
  return {
    id: `mut_${kind}`,
    kind,
    proposal,
    evidence: null,
    expectedGain: 0.1,
    status: "pending",
    createdAt: Date.now(),
    resolvedAt: null,
  };
}

describe("teacher v3 shadow replay summaries", () => {
  it("summarizes mutation replay on a real candidate graph and records rollback restoration", () => {
    const baseGraph = makeGraph();
    const state = createShadowCandidateState(baseGraph);

    applyShadowMutationProposalToState(
      state,
      makeProposal("connect", { nodeA: "b", nodeB: "c" }),
    );
    applyShadowMutationProposalToState(
      state,
      makeProposal("inject", {
        nodeKind: "episode_anchor",
        content: "shadow note",
        firedNodes: ["a", "b"],
      }),
    );

    const summary = summarizeTeacherMutationShadowReplayV1({
      proposalId: "prop_mutation_shadow_01",
      rollbackKey: "rollback:teacher-v3:mutation:shadow",
      state,
    });

    expect(summary).toMatchObject({
      proposalClass: "mutation",
      reviewMode: "shadow_only",
      shadowOnly: true,
      promotionBypass: false,
      proposalId: "prop_mutation_shadow_01",
      rollbackKey: "rollback:teacher-v3:mutation:shadow",
      replayOutcome: "applied",
      applied: true,
      reversible: true,
    });
    expect(summary.before).toMatchObject({
      nodeCount: baseGraph.getAllNodes().length,
      edgeCount: baseGraph.getAllEdges().length,
    });
    expect(summary.after.nodeCount).toBeGreaterThan(summary.before.nodeCount);
    expect(summary.applications).toHaveLength(2);
    expect(summary.applications[0]?.operationKinds).toContain("insert_edge");
    expect(summary.rollback.restored).toBe(true);
    expect(summary.rollback.after).toEqual(summary.before);
    expect(summary.summary).toContain("shadow-only");
    expect(summary.summary).toContain("rollback");
  });

  it("summarizes forgetting replay over retention state and keeps the rollback explicit", () => {
    const summary = summarizeTeacherForgettingShadowReplayV1({
      proposalId: "prop_forgetting_shadow_01",
      rollbackKey: "rollback:teacher-v3:forgetting:shadow",
      current: "retained",
      target: {
        sourceId: "bn_source_01",
        sourceKind: "summary",
        authority: "raw_source",
      },
      requestedTransition: "archive",
    });

    expect(summary).toMatchObject({
      proposalClass: "forgetting",
      reviewMode: "shadow_only",
      shadowOnly: true,
      promotionBypass: false,
      proposalId: "prop_forgetting_shadow_01",
      rollbackKey: "rollback:teacher-v3:forgetting:shadow",
      replayOutcome: "applied",
      applied: true,
      reversible: true,
      guardrail: undefined,
    });
    expect(summary.before.retentionState).toBe("retained");
    expect(summary.after.retentionState).toBe("archived");
    expect(summary.decision.allowed).toBe(true);
    expect(summary.rollback.restored).toBe(true);
    expect(summary.rollback.after.retentionState).toBe("retained");
    expect(summary.summary).toContain("no promotion bypass");
    expect(summary.summary).toContain("shadow-only");
  });

  it("keeps user_explicit hard delete blocked in the forgetting replay summary", () => {
    const summary = summarizeTeacherForgettingShadowReplayV1({
      proposalId: "prop_forgetting_shadow_02",
      rollbackKey: "rollback:teacher-v3:forgetting:shadow-guardrail",
      current: "tombstoned",
      target: {
        sourceId: "bn_correction_01",
        sourceKind: "correction",
        authority: "user_explicit",
      },
      requestedTransition: "hard_delete",
    });

    expect(summary.replayOutcome).toBe("blocked");
    expect(summary.applied).toBe(false);
    expect(summary.decision.allowed).toBe(false);
    expect(summary.decision.guardrail).toBe("deny_hard_delete_user_explicit");
    expect(summary.guardrail).toBe("deny_hard_delete_user_explicit");
    expect(summary.reason).toContain("user_explicit correction memory");
    expect(summary.rollback.restored).toBe(true);
    expect(summary.rollback.after.retentionState).toBe("tombstoned");
    expect(summary.summary).toContain("blocked");
    expect(summary.summary).toContain("no promotion bypass");
  });
});
