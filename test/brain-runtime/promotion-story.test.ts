import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MutationProposal } from "../../src/brain-core/types.js";
import { runBrainMigrations } from "../../src/brain-store/migrations.js";
import { BrainStore } from "../../src/brain-store/store.js";
import { buildPromotionStory } from "../../src/brain-runtime/promotion-story.js";

const tempDirs: string[] = [];

function makeTempDir(prefix: string): string {
  const dir = mkdtempSync(join(tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

function makeMutation(params: {
  id: string;
  kind: MutationProposal["kind"];
  proposal: MutationProposal["proposal"];
  status?: MutationProposal["status"];
  expectedGain?: number | null;
}): MutationProposal {
  return {
    id: params.id,
    kind: params.kind,
    proposal: params.proposal,
    evidence: null,
    expectedGain: params.expectedGain ?? null,
    status: params.status ?? "pending",
    createdAt: Date.now(),
    resolvedAt: null,
  };
}

afterEach(() => {
  vi.useRealTimers();
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("buildPromotionStory", () => {
  it("surfaces recent promotions and candidate buckets from current store truth", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-03-18T16:00:00.000Z"));

    const brainRoot = makeTempDir("promotion-story-state-");
    const db = new DatabaseSync(join(brainRoot, "state.db"));
    runBrainMigrations(db);
    const store = new BrainStore(db, { brainRoot });

    const pack = store.insertPack({
      nodeCount: 3,
      edgeCount: 2,
      healthJson: JSON.stringify({ nodeCount: 3, edgeCount: 2, firedPerQuery: 1.5 }),
    });
    store.writePackSnapshot({
      version: pack.version,
      nodes: [],
      edges: [],
      seedWeights: [],
      metadata: {
        reason: "worker",
        workspaceRoot: "/tmp/demo-workspace",
      },
    });
    store.promotePack(pack.version);

    vi.advanceTimersByTime(1_000);
    const promoted = makeMutation({
      id: "bm_promoted",
      kind: "connect",
      proposal: { nodeA: "bn_a", nodeB: "bn_b", coFireCount: 4 },
      expectedGain: 0.2,
    });
    store.insertMutation(promoted);
    store.resolveMutation(promoted.id, "promoted");

    vi.advanceTimersByTime(1_000);
    const rejected = makeMutation({
      id: "bm_rejected",
      kind: "prune",
      proposal: { source: "bn_old", target: "bn_leaf", edgeKind: "learned" },
      expectedGain: 0.01,
    });
    store.insertMutation(rejected);
    store.resolveMutation(rejected.id, "rejected");

    vi.advanceTimersByTime(1_000);
    const pending = makeMutation({
      id: "bm_pending",
      kind: "inject",
      proposal: { nodeKind: "episode_anchor", content: "retry deployment after CI passes", firedNodes: ["bn_a", "bn_b"] },
      expectedGain: 0.05,
    });
    store.insertMutation(pending);

    store.setTrainingState("last_promotion_reason", "bundle evaluation promoted 1 candidate");
    store.setTrainingState("last_replay_failure_reason", "candidate score regressed");

    store.insertTrace({
      id: "bt_feedback",
      episodeId: "ep_feedback",
      packVersion: pack.version,
      queryText: "how do I open a pull request?",
      seedScores: [],
      trajectory: [],
      firedNodes: [],
      vetoedNodes: [],
      contextChars: 128,
      footer: "",
      routeTrace: null,
      createdAt: Date.now(),
    });
    const observation = store.insertObservation({
      episodeId: "ep_feedback",
      conversationId: 42,
      traceId: "bt_feedback",
      queryText: "how do I open a pull request?",
      retrievedContext: [],
      routeMetadata: {
        requestDigest: null,
        activePackId: `brain-pack-v${pack.version}`,
        routerIdentity: "brain-graph-traverse.v2",
        persistenceMode: "redacted",
        bindingMode: "trace_id",
        serveDecisionRecordId: null,
        selectionDigest: null,
        turnCompileEventId: null,
        decisionRecordedAt: null,
        activePackEventExportDigest: null,
        activePackGraphChecksum: null,
        activePackRouterChecksum: null,
        activePackBuiltAt: null,
        servedArtifact: null,
        candidateNodeIds: [],
        selectedNodeIds: [],
        selectedTraversalNodeIds: [],
        selectedPathNodeIds: [],
        selectedSeedNodeIds: [],
        sourceSummary: null,
        selectionMetadata: null,
      },
      assistantResponse: "Use `gh pr create`.",
      toolResults: [],
      status: "completed",
    });
    store.insertTraceSupervision({
      traceId: "bt_feedback",
      episodeId: "ep_feedback",
      conversationId: 42,
      source: "teacher",
      kind: "teacher_review",
      value: 0.78,
      confidence: 0.62,
      reason: "retrieved context matched the query",
      resolution: "promoted_to_label",
      metadata: {
        observationId: observation.id,
        bindingMode: "trace_id",
      },
    });

    const story = buildPromotionStory(store);

    expect(story.summary.currentPackVersion).toBe(pack.version);
    expect(story.summary.mutationBacklog).toEqual({
      pending: 1,
      validated: 0,
      promoted: 1,
      rejected: 1,
    });
    expect(story.currentPack).toMatchObject({
      version: pack.version,
      reason: "worker",
      metadata: {
        workspaceRoot: "/tmp/demo-workspace",
      },
    });
    expect(story.recentPromotions).toHaveLength(1);
    expect(story.candidates.pending[0]).toMatchObject({
      id: "bm_pending",
      status: "pending",
    });
    expect(story.candidates.pending[0]?.summary).toContain("inject episode_anchor");
    expect(story.candidates.promoted[0]).toMatchObject({
      id: "bm_promoted",
      status: "promoted",
    });
    expect(story.candidates.promoted[0]?.summary).toBe("connect bn_a -> bn_b (4 co-fires)");
    expect(story.candidates.rejected[0]).toMatchObject({
      id: "bm_rejected",
      status: "rejected",
    });
    expect(story.latestActivity).toMatchObject({
      type: "candidate_pending",
      candidateId: "bm_pending",
    });
    expect(story.integrations).toEqual({
      structuredVerdict: expect.objectContaining({
        verdictCounts: {
          helpful: 1,
          irrelevant: 0,
          harmful: 0,
        },
        coverage: expect.objectContaining({
          routeTraceCount: 1,
          observationCount: 1,
          completedObservationCount: 1,
          supervisedTraceCount: 1,
          unsupervisedTraceCount: 0,
        }),
        latest: expect.objectContaining({
          traceId: "bt_feedback",
          episodeId: "ep_feedback",
          observationId: observation.id,
          source: "teacher",
          verdict: "helpful",
          score: 0.78,
          confidence: 0.62,
          bindingMode: "trace_id",
        }),
      }),
      learningJournal: null,
    });
  });
});
