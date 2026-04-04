import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { BrainGraph } from "../src/brain-core/graph.js";
import { createShadowCandidateState, applyShadowMutationProposalToState } from "../src/brain-core/shadow-application.js";
import type { BrainNode, MutationProposal } from "../src/brain-core/types.js";
import {
  summarizeTeacherForgettingShadowReplayV1,
  summarizeTeacherMutationShadowReplayV1,
} from "../src/brain-core/teacher-v3-shadow-replay.js";
import {
  describeTeacherCanaryActivationGuardV1,
  describeTeacherCanaryRolloutPlanV1,
  summarizeTeacherCanaryRolloutPlanV1,
} from "../src/brain-core/teacher-v3-contracts.js";

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

describe("teacher v3 shadow worked examples", () => {
  it("documents the mutation shadow lane with real replay and rollback semantics", () => {
    const artifactPath = path.join(process.cwd(), "artifacts", "teacher-v3-shadow-examples", "mutation-worked-example.md");
    const artifact = readFileSync(artifactPath, "utf8");

    const baseGraph = makeGraph();
    const state = createShadowCandidateState(baseGraph);
    applyShadowMutationProposalToState(state, makeProposal("connect", { nodeA: "b", nodeB: "c" }));
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

    expect(artifact).toContain("prop_mutation_shadow_01");
    expect(artifact).toContain("shadow_only");
    expect(artifact).toContain(summary.summary);
    expect(artifact).toContain("reset_shadow_candidate_state");
    expect(artifact).toContain("4 nodes / 4 edges");
    expect(artifact).toContain("3 nodes / 1 edge");
  });

  it("documents the forgetting shadow lane and its hard-delete guardrail", () => {
    const artifactPath = path.join(process.cwd(), "artifacts", "teacher-v3-shadow-examples", "forgetting-worked-example.md");
    const artifact = readFileSync(artifactPath, "utf8");

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

    const blocked = summarizeTeacherForgettingShadowReplayV1({
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

    expect(artifact).toContain("prop_forgetting_shadow_01");
    expect(artifact).toContain("shadow_only");
    expect(artifact).toContain(summary.summary);
    expect(artifact).toContain("restore_retention_state");
    expect(artifact).toContain("deny_hard_delete_user_explicit");
    expect(blocked.replayOutcome).toBe("blocked");
    expect(blocked.decision.guardrail).toBe("deny_hard_delete_user_explicit");
  });

  it("documents the canary boundary as target-state only and off by default", () => {
    const artifactPath = path.join(process.cwd(), "artifacts", "teacher-v3-shadow-examples", "promotion-canary-boundary.md");
    const artifact = readFileSync(artifactPath, "utf8");

    const plan = describeTeacherCanaryRolloutPlanV1({
      proposalClass: "mutation",
      rollbackKey: "rollback:teacher-v3:canary:mutation",
      candidatePackVersion: 8,
      candidatePackId: "candidate_pack_08",
    });
    const planSummary = summarizeTeacherCanaryRolloutPlanV1(plan).summary;
    const guardBlocked = describeTeacherCanaryActivationGuardV1({
      proposalId: "prop_canary_02",
      proposalClass: "mutation",
      rollbackKey: "rollback:teacher-v3:canary:mutation",
      canaryRollout: { ...plan, enabled: true, rolloutMode: "canary" },
      replaySummary: null,
      proofBundle: null,
    });

    expect(artifact).toContain("## What is promotable today");
    expect(artifact).toContain("## What stays shadow-only");
    expect(artifact).toContain(planSummary);
    expect(artifact).toContain("`rolloutMode`: `off`");
    expect(artifact).toContain("missing replay summary");
    expect(artifact).toContain("missing proof bundle");
    expect(artifact).toContain("missing proof rollback binding");
    expect(guardBlocked.blocked).toBe(true);
  });

  it("keeps the docs index honest about the shadow-only boundary", () => {
    const docsPath = path.join(process.cwd(), "docs", "architecture", "teacher-v3-shadow-worked-examples.md");
    const docs = readFileSync(docsPath, "utf8");

    expect(docs).toContain("mutation / forgetting / correction: shadow-only classes");
    expect(docs).toContain("canary: target-state only, off by default, rollback-bound");
    expect(docs).toContain("artifacts/teacher-v3-shadow-examples/mutation-worked-example.md");
    expect(docs).toContain("artifacts/teacher-v3-shadow-examples/promotion-canary-boundary.md");
  });
});
