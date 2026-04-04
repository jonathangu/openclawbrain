import { describe, expect, it } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import { buildTeacherProposalReplaySummaryV1, hashBrainGraphState } from "../../src/brain-core/teacher-v3-replay.js";
import type { BrainNode, Pack } from "../../src/brain-core/types.js";
import type { TeacherProposal } from "../../src/brain-core/teacher-v3-contracts.js";

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
  graph.addNode(makeNode("alpha"));
  graph.addNode(makeNode("beta"));
  graph.addNode(makeNode("gamma"));
  graph.addEdge({
    source: "alpha",
    target: "beta",
    kind: "learned",
    weight: 0.7,
    prior: 0.5,
    metadata: { seed: true },
    decayedAt: Date.now(),
    createdAt: Date.now(),
  });
  graph.addEdge({
    source: "beta",
    target: "gamma",
    kind: "learned",
    weight: 0.65,
    prior: 0.5,
    metadata: { seed: true },
    decayedAt: Date.now(),
    createdAt: Date.now(),
  });
  return graph;
}

function makePack(version: number): Pack {
  const health = {
    nodeCount: 3,
    edgeCount: 2,
    firedPerQuery: 1.8,
    dormantPercent: 0.12,
    orphanCount: 0,
  };
  return {
    version,
    nodeCount: 3,
    edgeCount: 2,
    healthJson: JSON.stringify(health),
    promotedAt: null,
    rolledBack: false,
    createdAt: Date.now(),
  };
}

function makeProposal(proposalClass: TeacherProposal["proposalClass"]): TeacherProposal {
  const isCompiler = proposalClass === "compiler";
  return {
    proposalId: `${proposalClass}_replay_01`,
    proposalClass,
    lane: proposalClass,
    status: "validated",
    lineage: {
      proposalClass,
      basePackVersion: 7,
      baseGraphHash: `graph_sha_${proposalClass}_base`,
      producerVersion: "teacher-v3@0.1.0",
      producerBuildId: `build_${proposalClass}_01`,
      promptHash: `prompt_sha_${proposalClass}_01`,
      templateId: `teacher-v3/${proposalClass}-v1`,
      scope: isCompiler ? "docs/architecture/compiler-replay" : "release-drift",
      profile: "default",
      idempotencyKey: `teacher-v3::${proposalClass}::replay-01`,
      sourceBundleId: `bundle_${proposalClass}_01`,
      parentProposalIds: [`seed_${proposalClass}_00`],
    },
    subjectIds: isCompiler
      ? ["doc:teacher-v3", "doc:replay-path"]
      : ["doc:README", "doc:proof-page"],
    evidence: [{
      evidenceId: `${proposalClass}_evidence_01`,
      sourceKind: "file",
      sourceId: `${proposalClass}/source.md`,
      authority: "raw_source",
      derivation: isCompiler ? "teacher_compilation" : "teacher_lint",
      excerpt: `${proposalClass} replay evidence`,
      sourceHash: `sha256:${proposalClass}-evidence`,
    }],
    counterevidence: isCompiler ? [] : [{
      evidenceId: "lint_counter_01",
      sourceKind: "file",
      sourceId: "docs/changelog.md",
      authority: "raw_source",
      derivation: "teacher_lint",
      excerpt: "lint counterevidence",
      sourceHash: "sha256:lint-counter-01",
    }],
    payload: {
      kind: proposalClass,
      summary: `${proposalClass} replay payload`,
    },
    expectedEffect: {
      retrieval: isCompiler ? "better" : "same",
      truthRisk: "low",
      tokenBudget: "same",
    },
    confidence: isCompiler ? 0.95 : 0.88,
    replaySuites: [
      `${proposalClass}-replay`,
      `${proposalClass}-candidate-pack`,
    ],
    rollbackKey: `rollback:${proposalClass}:replay-01`,
    replayGate: undefined,
    createdAt: "2026-04-03T18:26:00Z",
  };
}

describe("Teacher v3 proposal replay summaries", () => {
  it("builds a promotable compiler replay summary against real candidate state", () => {
    const proposal = makeProposal("compiler");
    const candidateGraph = makeGraph();
    const candidatePack = makePack(8);

    const summary = buildTeacherProposalReplaySummaryV1({
      proposal,
      candidateState: {
        candidatePack,
        candidatePackId: "candidate_pack_08",
        candidateGraph,
      },
      evaluatedAt: "2026-04-03T18:33:00Z",
    });

    expect(summary).toMatchObject({
      proposalId: proposal.proposalId,
      proposalClass: "compiler",
      status: "promotable",
      reviewMode: "promotable",
      basePackVersion: 7,
      candidatePackVersion: 8,
      candidatePackId: "candidate_pack_08",
      before: {
        phase: "before",
        surfaceState: "shipped",
      },
      after: {
        phase: "after",
        surfaceState: "target",
        packVersion: 8,
        packId: "candidate_pack_08",
      },
      classSummary: {
        kind: "compiler",
        reviewMode: "promotable",
        promotionDiscipline: "promotable",
      },
    });
    expect(summary.beforeScore).toBeLessThan(summary.afterScore);
    expect(summary.scoreDelta).toBeGreaterThan(0);
    expect(summary.after.graphHash).toBe(hashBrainGraphState(candidateGraph));
    expect(summary.summary).toContain("compiler replay accepted");
    expect(summary.classSummary.summary).toContain("promotable");
    expect(summary.classSummary.notes).toEqual(
      expect.arrayContaining([
        "basePackVersion=7",
        "baseGraphHash=graph_sha_compiler_base",
        "compiler proposals may be evaluated as promotable, but they are not auto-promoted",
      ]),
    );
  });

  it("keeps lint replay summaries bounded, report-only, and inspectable", () => {
    const proposal = makeProposal("lint");
    const candidateGraph = makeGraph();
    const candidatePack = makePack(8);

    const summary = buildTeacherProposalReplaySummaryV1({
      proposal,
      candidateState: {
        candidatePack,
        candidatePackId: "candidate_pack_08",
        candidateGraph,
      },
      evaluatedAt: "2026-04-03T18:34:00Z",
    });

    expect(summary).toMatchObject({
      proposalClass: "lint",
      status: "promotable",
      classSummary: {
        kind: "lint",
        reviewMode: "promotable",
        promotionDiscipline: "promotable",
      },
    });
    expect(summary.classSummary.counterevidenceCount).toBe(1);
    expect(summary.classSummary.replaySuites).toEqual(proposal.replaySuites);
    expect(summary.classSummary.summary).toContain("bounded report-only review");
    expect(summary.summary).toContain("lint replay accepted");
    expect(summary.beforeScore).toBeGreaterThan(0);
    expect(summary.afterScore).toBeGreaterThanOrEqual(summary.beforeScore);
  });
});
