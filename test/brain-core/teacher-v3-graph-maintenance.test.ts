import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import type { BrainNode } from "../../src/brain-core/types.js";
import type { EvidenceRefV1, TeacherProposal } from "../../src/brain-core/teacher-v3-contracts.js";
import {
  buildTeacherAddEdgeGraphMaintenanceProposalV1,
  replayTeacherGraphMaintenanceProposalV1,
  summarizeTeacherGraphMaintenanceLifecycleV1,
} from "../../src/brain-core/teacher-v3-graph-maintenance.js";
import { runBrainMigrations } from "../../src/brain-store/migrations.js";
import { BrainStore } from "../../src/brain-store/store.js";

const tempDirs: string[] = [];

function setupStore(): BrainStore {
  const dir = mkdtempSync(join(tmpdir(), "teacher-graph-maintenance-test-"));
  tempDirs.push(dir);
  const db = new DatabaseSync(join(dir, "test.db"));
  runBrainMigrations(db);
  return new BrainStore(db);
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function node(id: string): BrainNode {
  const now = Date.parse("2026-04-25T14:00:00Z");
  return {
    id,
    kind: "chunk",
    content: id,
    embedding: null,
    sourceUri: null,
    trust: "scanner",
    tags: ["teacher-v3", "graph-maintenance"],
    tokenCount: id.length,
    metadata: {},
    createdAt: now,
    updatedAt: now,
  };
}

function baseGraph(): BrainGraph {
  const graph = new BrainGraph();
  graph.addNode(node("concept_teacher_v3"));
  graph.addNode(node("concept_graph_prior"));
  graph.addNode(node("concept_replay_gate"));
  graph.addEdge({
    source: "concept_teacher_v3",
    target: "concept_replay_gate",
    kind: "learned",
    weight: 0.7,
    prior: 0.5,
    metadata: { seed: true },
    decayedAt: Date.parse("2026-04-25T14:00:00Z"),
    createdAt: Date.parse("2026-04-25T14:00:00Z"),
  });
  return graph;
}

const evidence: EvidenceRefV1 = {
  evidenceId: "evi_graph_maintenance_add_edge_01",
  sourceKind: "file",
  sourceId: "docs/architecture/teacher-v3.md#teacher-as-compiler",
  authority: "raw_source",
  derivation: "teacher_mutation_proposal",
  excerpt: "Teacher v3 may propose structural graph changes off-path, behind replay gates.",
  sourceHash: "sha256:graph-maintenance-add-edge",
  capturedAt: "2026-04-25T14:00:00Z",
};

function addEdgeProposal(): TeacherProposal {
  return buildTeacherAddEdgeGraphMaintenanceProposalV1({
    proposalId: "prop_graph_add_edge_01",
    sourceNodeId: "concept_teacher_v3",
    targetNodeId: "concept_graph_prior",
    subjectIds: ["concept_teacher_v3", "concept_graph_prior"],
    evidence: [evidence],
    expectedEffect: {
      retrieval: "better",
      truthRisk: "low",
      tokenBudget: "same",
    },
    confidence: 0.82,
    replaySuites: ["teacher-v3-graph-maintenance-shadow", "teacher-v3-rollback-smoke"],
    rollbackKey: "rollback:teacher-v3:graph-maintenance:add-edge:01",
    lineage: {
      proposalClass: "mutation",
      basePackVersion: 48,
      baseGraphHash: "sha256:base-graph-before-add-edge",
      producerVersion: "teacher-v3@0.4.48",
      producerBuildId: "t287-lane-c",
      promptHash: "sha256:teacher-graph-maintenance-prompt",
      templateId: "teacher-v3/graph-maintenance/add-edge-v1",
      scope: "teacher-v3-graph-maintenance",
      profile: "shadow-only",
      idempotencyKey: "teacher-v3::graph-maintenance::add-edge::concept_teacher_v3::concept_graph_prior",
      sourceBundleId: "bundle_teacher_graph_maintenance_01",
    },
    createdAt: "2026-04-25T14:00:00Z",
  });
}

describe("Teacher v3 graph-maintenance proposal lifecycle", () => {
  it("persists, loads, shadow-replays, and summarizes one add_edge graph maintenance proposal", () => {
    const store = setupStore();
    const proposal = addEdgeProposal();

    store.insertTeacherProposal(proposal);
    const loaded = store.getTeacherProposal(proposal.proposalId);
    expect(loaded).toMatchObject({
      proposalId: "prop_graph_add_edge_01",
      proposalClass: "mutation",
      proposalKind: "add_edge",
      status: "proposed",
      lifecycleState: "proposed",
      safeClassMode: "shadow_only",
      subjectIds: ["concept_teacher_v3", "concept_graph_prior"],
      evidence: [expect.objectContaining({ evidenceId: evidence.evidenceId })],
      expectedEffect: { retrieval: "better", truthRisk: "low", tokenBudget: "same" },
      replaySuites: ["teacher-v3-graph-maintenance-shadow", "teacher-v3-rollback-smoke"],
      rollbackKey: "rollback:teacher-v3:graph-maintenance:add-edge:01",
    });

    const replayed = replayTeacherGraphMaintenanceProposalV1({
      proposal: loaded!,
      baseGraph: baseGraph(),
      evaluatedAt: "2026-04-25T14:02:00Z",
    });

    expect(replayed.shadowReplay).toMatchObject({
      proposalClass: "mutation",
      reviewMode: "shadow_only",
      shadowOnly: true,
      promotionBypass: false,
      applied: true,
      reversible: true,
      replayOutcome: "applied",
      rollback: {
        restored: true,
      },
    });
    expect(replayed.shadowReplay.after.edgeCount).toBe(replayed.shadowReplay.before.edgeCount + 1);
    expect(replayed.replaySummary).toMatchObject({
      proposalId: proposal.proposalId,
      proposalClass: "mutation",
      status: "shadow_scored",
      reviewMode: "shadow_only",
      classSummary: {
        promotionDiscipline: "shadow_only",
        promotionBypass: false,
        rollback: { restored: true },
      },
    });
    expect(replayed.replaySummary.summary).toContain("no live self-editing");

    store.updateTeacherProposalStatus({
      proposalId: proposal.proposalId,
      status: "shadow_scored",
      replaySummary: replayed.replaySummary,
    });

    const loadedReplayed = store.getTeacherProposal(proposal.proposalId);
    expect(loadedReplayed).toMatchObject({
      status: "shadow_scored",
      lifecycleState: "replayed",
      replaySummary: {
        replayId: "treplay_graph_prop_graph_add_edge_01",
        status: "shadow_scored",
        reviewMode: "shadow_only",
      },
    });

    const lifecycle = summarizeTeacherGraphMaintenanceLifecycleV1({
      proposal: loadedReplayed!,
      replaySummary: replayed.replaySummary,
      shadowReplay: replayed.shadowReplay,
    });
    expect(lifecycle).toMatchObject({
      contract: "teacher_v3_graph_maintenance_lifecycle.v1",
      proposalId: proposal.proposalId,
      proposalClass: "mutation",
      proposalKind: "add_edge",
      lifecycleState: "replayed",
      status: "shadow_scored",
      safeClassMode: "shadow_only",
      replayOutcome: "applied",
      rollbackRestored: true,
      promotionBypass: false,
      liveSelfEditingEnabled: false,
    });
    expect(lifecycle.evidenceIds).toEqual([evidence.evidenceId]);
    expect(lifecycle.boundary).toContain("does not write to the live graph");
  });

  it("blocks graph maintenance proposals that try to leave shadow-only mode", () => {
    const store = setupStore();
    const proposal = {
      ...addEdgeProposal(),
      safeClassMode: "promotable" as const,
    };

    expect(() => store.insertTeacherProposal(proposal)).toThrow(/safeClassMode must be shadow_only/);
  });

  it("blocks promoted mutation status and missing subject-node replay", () => {
    const store = setupStore();
    const promoted = {
      ...addEdgeProposal(),
      proposalId: "prop_graph_add_edge_bad_promoted",
      lineage: {
        ...addEdgeProposal().lineage,
        idempotencyKey: "teacher-v3::graph-maintenance::bad-promoted",
      },
      status: "promoted" as const,
      lifecycleState: "promoted" as const,
    };
    expect(() => store.insertTeacherProposal(promoted)).toThrow(/must remain shadow-only/);

    const proposal = addEdgeProposal();
    expect(() => replayTeacherGraphMaintenanceProposalV1({
      proposal,
      baseGraph: new BrainGraph(),
      evaluatedAt: "2026-04-25T14:03:00Z",
    })).toThrow(/references missing subject nodes/);
  });
});
