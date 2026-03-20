import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import { DEFAULT_BRAIN_CONFIG } from "../../src/brain-core/types.js";
import type { DecisionTrace, Episode, HealthMetrics, MutationProposal, ReplayGateVerdict } from "../../src/brain-core/types.js";
import { runBrainMigrations } from "../../src/brain-store/migrations.js";
import { BrainStore } from "../../src/brain-store/store.js";
import { BrainWorker } from "../../src/brain-worker/worker.js";

const tempDirs: string[] = [];

function makeEpisode(params: {
  id: string;
  conversationId: number;
  queryText?: string;
  firedNodes?: string[];
  reward?: number | null;
  rewardSource?: Episode["rewardSource"];
}): Episode {
  return {
    id: params.id,
    conversationId: params.conversationId,
    queryText: params.queryText ?? "test query",
    queryEmbedding: null,
    trajectory: [],
    firedNodes: params.firedNodes ?? [],
    vetoedNodes: [],
    contextChars: 0,
    reward: params.reward ?? null,
    rewardSource: params.rewardSource ?? null,
    packVersion: 1,
    createdAt: Date.now(),
  };
}

function makeTrace(params: {
  id: string;
  episodeId: string;
  conversationId: number;
  firedNodes?: string[];
}): DecisionTrace {
  const firedNodes = params.firedNodes ?? ["node_1"];
  return {
    id: params.id,
    episodeId: params.episodeId,
    packVersion: 1,
    queryText: "test query",
    seedScores: [],
    trajectory: [],
    firedNodes,
    vetoedNodes: [],
    contextChars: 0,
    footer: "trace footer",
    routeTrace: {
      requestDigest: `req_${params.id}`,
      conversationId: params.conversationId,
      activePackId: "brain-pack-v1",
      routerIdentity: "brain-graph-traverse.v1",
      candidateNodeIds: [...firedNodes],
      selectedNodeIds: [...firedNodes],
      selectedPathNodeIds: [...firedNodes],
      injectedNodeSummaries: [],
      sourceSummary: {
        injectedCount: firedNodes.length,
        kinds: {},
        trusts: {},
        sourceUris: [],
      },
      selectionMetadata: {
        traceSliceVersion: 1,
        queryChars: 10,
        budgetChars: 100,
        maxHops: 8,
        seedCount: 1,
        candidateCount: firedNodes.length,
        hopCount: firedNodes.length,
        firedCount: firedNodes.length,
        vetoedCount: 0,
        chosenSeedNodeId: null,
        routeSelectionMs: 1,
        embeddingMs: 1,
        totalQueryMs: 2,
        queryEmbeddingSource: "provided",
      },
    },
    createdAt: Date.now(),
  };
}

function makeHealthMetrics(overrides: Partial<HealthMetrics> = {}): HealthMetrics {
  return {
    nodeCount: 0,
    edgeCount: 0,
    nodesByKind: {
      chunk: 0,
      workflow: 0,
      correction: 0,
      toolcard: 0,
      episode_anchor: 0,
      summary_bridge: 0,
    },
    edgesByKind: {
      sibling: 0,
      semantic: 0,
      learned: 0,
      seed: 0,
      inhibitory: 0,
      bridge: 0,
    },
    firedPerQuery: 0,
    dormantPercent: 0,
    inhibitoryPercent: 0,
    orphanCount: 0,
    avgPathLength: 0,
    avgReward: 0,
    crossFileEdgePercent: 0,
    churn: 0,
    packVersion: 0,
    lastUpdateAt: Date.now(),
    totalEpisodes: 0,
    ...overrides,
  };
}

function makeReplayGateVerdict(params: {
  passed: boolean;
  code: ReplayGateVerdict["reason"]["code"];
  summary: string;
  details?: Record<string, unknown>;
  health?: HealthMetrics;
}): ReplayGateVerdict {
  return {
    passed: params.passed,
    reason: {
      code: params.code,
      summary: params.summary,
      details: params.details ?? {},
    },
    health: params.health ?? makeHealthMetrics(),
    evaluatedEpisodeCount: 1,
    humanPositiveEpisodeCount: 0,
    selfNegativeEpisodeCount: 0,
  };
}

function hashQuery(queryText: string): string {
  let hash = 0;
  for (let i = 0; i < queryText.length; i++) {
    const char = queryText.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return `__query_${Math.abs(hash).toString(36).slice(0, 8)}`;
}

function makeMutation(id: string, nodeA: string, nodeB: string): MutationProposal {
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

function setup(overrides: {
  replayGate?: () => ReplayGateVerdict;
  applyToCandidateGraph?: (graph: BrainGraph, proposal: MutationProposal) => void;
  applyMutation?: (proposal: MutationProposal) => void;
} = {}) {
  const dir = mkdtempSync(join(tmpdir(), "brain-worker-test-"));
  tempDirs.push(dir);
  const db = new DatabaseSync(join(dir, "test.db"));
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA foreign_keys = ON");
  runBrainMigrations(db);

  const store = new BrainStore(db);
  const graph = new BrainGraph();
  const applyMutation = vi.fn((proposal: MutationProposal) => {
    overrides.applyMutation?.(proposal);
  });
  const applyToCandidateGraph = vi.fn((candidateGraph: BrainGraph, proposal: MutationProposal) => {
    overrides.applyToCandidateGraph?.(candidateGraph, proposal);
  });
  const replayGate = vi.fn(() => overrides.replayGate?.() ?? makeReplayGateVerdict({
    passed: true,
    code: "all_gates_passed",
    summary: "all gates passed",
  }));
  const onPromotionReady = vi.fn(async () => undefined);
  const worker = new BrainWorker(
    store,
    graph,
    null,
    {
      proposeMutations: vi.fn(() => []),
      applyToCandidateGraph,
      applyMutation,
    } as never,
    {
      replayGate,
    } as never,
    {
      ...DEFAULT_BRAIN_CONFIG,
      mutationsEnabled: false,
    },
    {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
    },
    {
      onPromotionReady,
    },
  );

  return { store, worker, replayGate, applyMutation, onPromotionReady };
}

afterEach(() => {
  vi.restoreAllMocks();
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("BrainWorker evidence resolution", () => {
  it("keeps only the highest-trust pending evidence per episode in a worker cycle and materializes trace supervision", async () => {
    const { store, worker } = setup();
    store.insertEpisode(makeEpisode({ id: "ep_1", conversationId: 7 }));
    store.insertTrace(makeTrace({ id: "bt_ep_1", episodeId: "ep_1", conversationId: 7 }));
    store.insertEvidence({
      episodeId: "ep_1",
      conversationId: 7,
      source: "scanner",
      kind: "scanner_signal",
      value: 0.25,
      confidence: 0.55,
      reason: "scanner pattern",
    });
    store.insertEvidence({
      episodeId: "ep_1",
      conversationId: 7,
      source: "human",
      kind: "human_feedback",
      value: 0.8,
      confidence: 0.9,
      reason: "user confirmed",
    });

    await (worker as any).processEvidence();

    const pendingLabels = store.getPendingLabels();
    expect(pendingLabels).toHaveLength(1);
    expect(pendingLabels[0]?.source).toBe("human");
    expect(pendingLabels[0]?.value).toBe(0.8);

    const resolved = store.getResolvedLabelsForEpisode("ep_1", 10);
    expect(resolved).toHaveLength(2);
    expect(resolved.find((entry) => entry.source === "human")?.resolution).toBe("promoted_to_label");
    expect(resolved.find((entry) => entry.source === "scanner")?.resolution).toBe("discarded_lower_trust");

    const supervision = store.getTraceSupervision("bt_ep_1", 10);
    expect(supervision).toHaveLength(2);
    expect(supervision.find((entry) => entry.source === "human")?.resolution).toBe("promoted_to_label");
    expect(supervision.find((entry) => entry.source === "scanner")?.resolution).toBe("discarded_lower_trust");
    expect(supervision.find((entry) => entry.source === "human")?.metadata).toMatchObject({
      resolvedTraceId: "bt_ep_1",
      traceRequestDigest: "req_bt_ep_1",
      traceSelectedNodeIds: ["node_1"],
    });

    await (worker as any).processLabels();
    const episode = store.getEpisode("ep_1");
    expect(episode?.reward).toBe(0.8);
    expect(episode?.rewardSource).toBe("human");
  });

  it("collapses same-trust pending evidence to one promoted label using confidence", async () => {
    const { store, worker } = setup();
    store.insertEpisode(makeEpisode({ id: "ep_2", conversationId: 8 }));
    store.insertEvidence({
      episodeId: "ep_2",
      conversationId: 8,
      source: "self",
      kind: "self_result",
      value: -0.5,
      confidence: 0.4,
      reason: "weak failure signal",
    });
    store.insertEvidence({
      episodeId: "ep_2",
      conversationId: 8,
      source: "self",
      kind: "self_result",
      value: 0.5,
      confidence: 0.9,
      reason: "strong success signal",
    });

    await (worker as any).processEvidence();

    const pendingLabels = store.getPendingLabels();
    expect(pendingLabels).toHaveLength(1);
    expect(pendingLabels[0]?.source).toBe("self");
    expect(pendingLabels[0]?.value).toBe(0.5);
    expect(pendingLabels[0]?.confidence).toBe(0.9);

    const resolved = store.getResolvedLabelsForEpisode("ep_2", 10);
    expect(resolved).toHaveLength(2);
    expect(resolved.find((entry) => entry.value === 0.5)?.resolution).toBe("promoted_to_label");
    expect(resolved.find((entry) => entry.value === -0.5)?.resolution).toBe("discarded_duplicate");
  });

  it("prefers structured scanner evidence over heuristic scanner evidence when scanner signals conflict", async () => {
    const { store, worker } = setup();
    store.insertEpisode(makeEpisode({ id: "ep_2b", conversationId: 82 }));
    store.insertEvidence({
      episodeId: "ep_2b",
      conversationId: 82,
      source: "scanner",
      kind: "scanner_signal",
      value: -0.25,
      confidence: 0.95,
      reason: "heuristic scanner signal",
      metadata: { extractor: "scanner_heuristic" },
    });
    store.insertEvidence({
      episodeId: "ep_2b",
      conversationId: 82,
      source: "scanner",
      kind: "scanner_signal",
      value: 0.25,
      confidence: 0.6,
      reason: "structured guidance signal",
      metadata: { extractor: "structured_guidance_parts" },
    });

    await (worker as any).processEvidence();

    const pendingLabels = store.getPendingLabels();
    expect(pendingLabels).toHaveLength(1);
    expect(pendingLabels[0]?.source).toBe("scanner");
    expect(pendingLabels[0]?.value).toBe(0.25);
    expect(pendingLabels[0]?.confidence).toBe(0.6);

    const resolved = store.getResolvedLabelsForEpisode("ep_2b", 10);
    expect(resolved).toHaveLength(2);
    expect(resolved.find((entry) => entry.value === 0.25)?.resolution).toBe("promoted_to_label");
    expect(resolved.find((entry) => entry.value === -0.25)?.resolution).toBe("discarded_duplicate");
    expect(resolved.find((entry) => entry.value === -0.25)?.note).toContain("more-structured scanner evidence");
  });

  it("keeps the higher-confidence scanner label when same-value scanner evidence is only corroborating", async () => {
    const { store, worker } = setup();
    store.insertEpisode(makeEpisode({ id: "ep_2c", conversationId: 83 }));
    store.insertEvidence({
      episodeId: "ep_2c",
      conversationId: 83,
      source: "scanner",
      kind: "scanner_signal",
      value: 0.25,
      confidence: 0.95,
      reason: "high-confidence heuristic scanner signal",
      metadata: { extractor: "scanner_heuristic" },
    });
    store.insertEvidence({
      episodeId: "ep_2c",
      conversationId: 83,
      source: "scanner",
      kind: "scanner_signal",
      value: 0.25,
      confidence: 0.6,
      reason: "structured corroborating scanner signal",
      metadata: { extractor: "structured_guidance_parts" },
    });

    await (worker as any).processEvidence();

    const pendingLabels = store.getPendingLabels();
    expect(pendingLabels).toHaveLength(1);
    expect(pendingLabels[0]?.source).toBe("scanner");
    expect(pendingLabels[0]?.value).toBe(0.25);
    expect(pendingLabels[0]?.confidence).toBe(0.95);

    const resolved = store.getResolvedLabelsForEpisode("ep_2c", 10);
    expect(resolved).toHaveLength(2);
    expect(resolved.find((entry) => entry.confidence === 0.95)?.resolution).toBe("promoted_to_label");
    expect(resolved.find((entry) => entry.confidence === 0.6)?.resolution).toBe("discarded_duplicate");
    expect(resolved.find((entry) => entry.confidence === 0.6)?.note).toContain("matching scanner evidence already queued");
  });

  it("does not auto-override an existing equal-trust reward", async () => {
    const { store, worker } = setup();
    store.insertEpisode(makeEpisode({
      id: "ep_3",
      conversationId: 9,
      reward: 0.8,
      rewardSource: "human",
    }));
    store.insertEvidence({
      episodeId: "ep_3",
      conversationId: 9,
      source: "human",
      kind: "human_feedback",
      value: -0.8,
      confidence: 0.95,
      reason: "later conflicting human signal",
    });

    await (worker as any).processEvidence();

    expect(store.getPendingLabels()).toHaveLength(0);
    const resolved = store.getResolvedLabelsForEpisode("ep_3", 10);
    expect(resolved).toHaveLength(1);
    expect(resolved[0]?.resolution).toBe("discarded_duplicate");
    expect(resolved[0]?.note).toContain("equal-trust override");
  });

  it("promotes higher-trust evidence over an existing lower-trust reward", async () => {
    const { store, worker } = setup();
    store.insertEpisode(makeEpisode({
      id: "ep_4",
      conversationId: 10,
      reward: -0.5,
      rewardSource: "self",
    }));
    store.insertEvidence({
      episodeId: "ep_4",
      conversationId: 10,
      source: "human",
      kind: "human_feedback",
      value: 0.8,
      confidence: 0.95,
      reason: "user confirmed correct behavior",
    });

    await (worker as any).processEvidence();
    await (worker as any).processLabels();

    const episode = store.getEpisode("ep_4");
    expect(episode?.reward).toBe(0.8);
    expect(episode?.rewardSource).toBe("human");

    const resolved = store.getResolvedLabelsForEpisode("ep_4", 10);
    expect(resolved).toHaveLength(1);
    expect(resolved[0]?.resolution).toBe("promoted_to_label");
    expect(resolved[0]?.source).toBe("human");
  });
});

describe("BrainWorker promotion verdicts", () => {
  it("persists a structured replay-gate rejection verdict for legacy promotion", async () => {
    const gate = makeReplayGateVerdict({
      passed: false,
      code: "fired_per_query_below_min",
      summary: "firedPerQuery 0.25 < 1",
      details: {
        metric: "firedPerQuery",
        actual: 0.25,
        minimum: 1,
      },
      health: makeHealthMetrics({ firedPerQuery: 0.25 }),
    });
    const { store, worker, onPromotionReady } = setup({
      replayGate: () => gate,
    });
    const mutation = makeMutation("mp_legacy", "node_a", "node_b");
    store.insertMutation(mutation);

    await (worker as any).checkPromotionLegacy(
      [makeEpisode({ id: "ep_legacy", conversationId: 11, reward: 0.8 })],
      store.getMutationsByStatus("pending", 10),
    );

    const lastPromotionVerdict = store.getTrainingStateJson<{
      status: string;
      replayGate: ReplayGateVerdict | null;
    }>("last_promotion_verdict_json");
    expect(lastPromotionVerdict?.status).toBe("rejected");
    expect(lastPromotionVerdict?.replayGate?.reason.code).toBe("fired_per_query_below_min");
    expect(store.getTrainingState("last_replay_failure_reason")).toBe("firedPerQuery 0.25 < 1");
    expect(store.getTrainingStateJson<ReplayGateVerdict>("last_replay_gate_verdict_json")?.reason.details).toMatchObject({
      actual: 0.25,
      minimum: 1,
    });
    expect(store.getMutationsByStatus("rejected", 10)).toHaveLength(1);
    expect(onPromotionReady).not.toHaveBeenCalled();
  });

  it("stores structured bundle verdicts and forwards them to the promotion hook", async () => {
    const { store, worker, applyMutation, onPromotionReady } = setup({
      applyMutation: (proposal) => {
        store.resolveMutation(proposal.id, "promoted");
      },
    });
    const queryText = "bundle verdict query";
    const queryNodeId = hashQuery(queryText);
    store.insertEpisode(makeEpisode({
      id: "ep_bundle",
      conversationId: 12,
      queryText,
      firedNodes: ["node_1", "node_2", "node_3"],
      reward: 1,
      rewardSource: "human",
    }));
    for (const mutation of [
      makeMutation("mp_bundle_1", queryNodeId, "node_1"),
      makeMutation("mp_bundle_2", queryNodeId, "node_2"),
      makeMutation("mp_bundle_3", queryNodeId, "node_3"),
    ]) {
      store.insertMutation(mutation);
    }

    await (worker as any).checkPromotion();

    const bundles = store.getRecentMutationBundles(5);
    expect(bundles).toHaveLength(1);
    expect(bundles[0]?.status).toBe("promoted");
    expect(bundles[0]?.verdict?.reason.code).toBe("promoted");
    expect(bundles[0]?.verdict?.candidateScore).toBeGreaterThan(0);

    const lastPromotionVerdict = store.getTrainingStateJson<{
      mode: string;
      promotedBundleCount: number;
      promotedMutationCount: number;
      bundleVerdicts: Array<{ reason: { code: string } }>;
    }>("last_promotion_verdict_json");
    expect(lastPromotionVerdict?.mode).toBe("bundle");
    expect(lastPromotionVerdict?.promotedBundleCount).toBe(1);
    expect(lastPromotionVerdict?.promotedMutationCount).toBe(3);
    expect(lastPromotionVerdict?.bundleVerdicts[0]?.reason.code).toBe("promoted");
    expect(store.getTrainingStateJson("last_replay_gate_verdict_json")).toBeNull();
    expect(applyMutation).toHaveBeenCalledTimes(3);
    expect(onPromotionReady).toHaveBeenCalledTimes(1);
    expect(onPromotionReady.mock.calls[0]?.[0]?.promotionVerdict?.bundleVerdicts?.[0]?.reason?.code).toBe("promoted");
  });
});
