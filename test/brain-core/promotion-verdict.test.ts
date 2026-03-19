import { describe, expect, it } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import { evaluateBundle, type MutationBundle } from "../../src/brain-core/bundle-evaluator.js";
import { PackManager } from "../../src/brain-core/pack.js";
import type { Episode, MutationProposal } from "../../src/brain-core/types.js";

function hashQuery(queryText: string): string {
  let hash = 0;
  for (let i = 0; i < queryText.length; i++) {
    const char = queryText.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return `__query_${Math.abs(hash).toString(36).slice(0, 8)}`;
}

function makeEpisode(params: {
  id: string;
  queryText: string;
  firedNodes: string[];
  reward: number | null;
  rewardSource?: Episode["rewardSource"];
}): Episode {
  return {
    id: params.id,
    conversationId: 1,
    queryText: params.queryText,
    queryEmbedding: null,
    trajectory: [],
    firedNodes: params.firedNodes,
    vetoedNodes: [],
    contextChars: 0,
    reward: params.reward,
    rewardSource: params.rewardSource ?? "human",
    packVersion: 1,
    createdAt: Date.now(),
  };
}

function makeConnectMutation(id: string, nodeA: string, nodeB: string): MutationProposal {
  return {
    id,
    kind: "connect",
    proposal: { nodeA, nodeB },
    evidence: null,
    expectedGain: 0.2,
    status: "pending",
    createdAt: Date.now(),
    resolvedAt: null,
  };
}

describe("promotion verdicts", () => {
  it("returns a structured replay-gate verdict with metric details", () => {
    const graph = new BrainGraph();
    const packManager = new PackManager(
      {
        insertPack: () => { throw new Error("not used"); },
        promotePack: () => undefined,
        rollbackPack: () => undefined,
      },
      graph,
      { info: () => undefined, warn: () => undefined },
    );

    const verdict = packManager.replayGate(
      [makeEpisode({ id: "ep_metric", queryText: "metric gate", firedNodes: [], reward: 0.8 })],
      { minFiredPerQuery: 1, maxDormantPercent: 1, maxOrphanCount: 10 },
    );

    expect(verdict.passed).toBe(false);
    expect(verdict.reason.code).toBe("fired_per_query_below_min");
    expect(verdict.reason.details).toMatchObject({
      metric: "firedPerQuery",
      actual: 0,
      minimum: 1,
    });
  });

  it("promotes a bundle that establishes a positive score from a zero baseline", async () => {
    const graph = new BrainGraph();
    const queryText = "zero baseline query";
    const bundle: MutationBundle = {
      id: "mb_zero",
      mutationIds: ["mp_zero"],
      proposals: [makeConnectMutation("mp_zero", hashQuery(queryText), "node_a")],
      bundleSize: 1,
      status: "pending",
      baseScore: null,
      candidateScore: null,
      expectedGain: 0.2,
      rejectionReason: null,
      createdAt: Date.now(),
      resolvedAt: null,
    };

    const result = await evaluateBundle(bundle, graph, [
      makeEpisode({
        id: "ep_zero",
        queryText,
        firedNodes: ["node_a"],
        reward: 1,
      }),
    ]);

    expect(result.shouldPromote).toBe(true);
    expect(result.verdict.status).toBe("promoted");
    expect(result.verdict.reason.code).toBe("promoted");
    expect(result.verdict.baseScore).toBe(0);
    expect(result.verdict.candidateScore).toBeGreaterThan(0);
    expect(result.verdict.improvementRatio).toBeNull();
  });

  it("captures a structured no-qualifying-episodes rejection", async () => {
    const graph = new BrainGraph();
    const bundle: MutationBundle = {
      id: "mb_none",
      mutationIds: ["mp_none"],
      proposals: [makeConnectMutation("mp_none", "node_a", "node_b")],
      bundleSize: 1,
      status: "pending",
      baseScore: null,
      candidateScore: null,
      expectedGain: 0.2,
      rejectionReason: null,
      createdAt: Date.now(),
      resolvedAt: null,
    };

    const result = await evaluateBundle(bundle, graph, [
      makeEpisode({
        id: "ep_none",
        queryText: "below threshold",
        firedNodes: ["node_b"],
        reward: 0.1,
      }),
    ]);

    expect(result.shouldPromote).toBe(false);
    expect(result.verdict.status).toBe("rejected");
    expect(result.verdict.reason.code).toBe("no_qualifying_episodes");
    expect(result.verdict.reason.details).toMatchObject({
      episodeCount: 1,
      qualifyingEpisodeCount: 0,
    });
  });
});
