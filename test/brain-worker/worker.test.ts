import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import { DEFAULT_BRAIN_CONFIG } from "../../src/brain-core/types.js";
import type {
  BrainNode,
  DecisionTrace,
  Episode,
  HealthMetrics,
  MutationProposal,
  PolicyGradientCandidateUpdateArtifact,
  ReplayGateVerdict,
  TrajectoryStep,
} from "../../src/brain-core/types.js";
import { runBrainMigrations } from "../../src/brain-store/migrations.js";
import { BrainStore } from "../../src/brain-store/store.js";
import { BrainWorker } from "../../src/brain-worker/worker.js";

const tempDirs: string[] = [];

function makeEpisode(params: {
  id: string;
  conversationId: number;
  queryText?: string;
  trajectory?: TrajectoryStep[];
  firedNodes?: string[];
  reward?: number | null;
  rewardSource?: Episode["rewardSource"];
}): Episode {
  return {
    id: params.id,
    conversationId: params.conversationId,
    queryText: params.queryText ?? "test query",
    queryEmbedding: null,
    trajectory: params.trajectory ?? [],
    firedNodes: params.firedNodes ?? [],
    vetoedNodes: [],
    contextChars: 0,
    reward: params.reward ?? null,
    rewardSource: params.rewardSource ?? null,
    packVersion: 1,
    createdAt: Date.now(),
  };
}

function makeNode(id: string): BrainNode {
  return {
    id,
    kind: "chunk",
    content: `content for ${id}`,
    embedding: new Float32Array([1, 0, 0]),
    sourceUri: `docs/${id}.md`,
    trust: "human",
    tags: ["worker"],
    tokenCount: 32,
    metadata: {},
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function makeStep(targetId: string, probability = 0.6): TrajectoryStep {
  return {
    stateSnapshot: {
      currentNodeId: null,
      hopCount: 0,
      budgetRemaining: 1000,
      visitedCount: 0,
      firedCount: 0,
    },
    candidates: [
      { action: { type: "traverse", targetNodeId: targetId }, score: 1, probability },
      { action: { type: "stop" }, score: -1, probability: 1 - probability },
    ],
    chosenAction: { type: "traverse", targetNodeId: targetId },
    chosenActionProbability: probability,
    stopProbability: 1 - probability,
  };
}

function makeTrace(params: {
  id: string;
  episodeId: string;
  conversationId: number;
  firedNodes?: string[];
  queryText?: string;
  selectedNodeId?: string;
}): DecisionTrace {
  const queryText = params.queryText ?? "test query";
  const firedNodes = params.firedNodes ?? [params.selectedNodeId ?? "node_1"];
  const selectedNodeId = params.selectedNodeId ?? firedNodes[0] ?? "node_1";
  const injectedNodeSummaries = firedNodes.map((nodeId, index) => ({
    nodeId,
    kind: "workflow" as const,
    trust: "human" as const,
    sourceUri: index === 0 ? "PLAYBOOK.md" : null,
    tags: ["worker"],
    tokenCount: 12,
    contentPreview: `Use ${nodeId} for worker route guidance.`,
  }));
  return {
    id: params.id,
    episodeId: params.episodeId,
    packVersion: 1,
    queryText,
    seedScores: [],
    trajectory: [],
    firedNodes,
    vetoedNodes: [],
    contextChars: 64,
    footer: "trace footer",
    routeTrace: {
      requestDigest: hashQuery(queryText),
      conversationId: params.conversationId,
      activePackId: "brain-pack-v1",
      routerIdentity: "brain-graph-traverse.v1",
      candidateNodeIds: [...firedNodes],
      selectedNodeIds: [...firedNodes],
      selectedPathNodeIds: [...firedNodes],
      injectedNodeSummaries,
      sourceSummary: {
        injectedCount: injectedNodeSummaries.length,
        kinds: { workflow: injectedNodeSummaries.length },
        trusts: { human: injectedNodeSummaries.length },
        sourceUris: injectedNodeSummaries.flatMap((summary) => summary.sourceUri ? [summary.sourceUri] : []),
      },
      selectionMetadata: {
        traceSliceVersion: 1,
        queryChars: queryText.length,
        budgetChars: 4000,
        maxHops: 8,
        seedCount: 1,
        candidateCount: firedNodes.length,
        hopCount: firedNodes.length,
        firedCount: firedNodes.length,
        vetoedCount: 0,
        chosenSeedNodeId: selectedNodeId,
        routeSelectionMs: 8,
        embeddingMs: 3,
        totalQueryMs: 15,
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
  teacher?: unknown;
  config?: Partial<typeof DEFAULT_BRAIN_CONFIG>;
} = {}) {
  const dir = mkdtempSync(join(tmpdir(), "brain-worker-test-"));
  tempDirs.push(dir);
  const brainRoot = join(dir, "brain-root");
  const db = new DatabaseSync(join(dir, "test.db"));
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA foreign_keys = ON");
  runBrainMigrations(db);

  const store = new BrainStore(db, { brainRoot });
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
    (overrides.teacher ?? null) as never,
    {
      proposeMutations: vi.fn(() => []),
      applyToCandidateGraph,
      applyMutation,
    } as never,
    {
      replayGate,
      buildCandidate: vi.fn((health: HealthMetrics) => store.insertPack({
        nodeCount: health.nodeCount,
        edgeCount: health.edgeCount,
        healthJson: JSON.stringify(health),
      })),
    } as never,
    {
      ...DEFAULT_BRAIN_CONFIG,
      mutationsEnabled: false,
      ...(overrides.config ?? {}),
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

  return { store, worker, graph, brainRoot, replayGate, applyMutation, onPromotionReady };
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
      traceRequestDigest: hashQuery("test query"),
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

  it("stages teacher evidence from the latest teacher-eligible trace", async () => {
    const evaluateTrace = vi.fn(async (trace: DecisionTrace) => ({
      version: 1 as const,
      traceId: trace.id,
      episodeId: trace.episodeId,
      requestDigest: trace.routeTrace?.requestDigest ?? "",
      score: 0.7,
      reason: "selected route slice is directly relevant",
      input: {
        version: 1 as const,
        traceId: trace.id,
        episodeId: trace.episodeId,
        queryText: trace.queryText,
        routeDecision: {
          requestDigest: trace.routeTrace?.requestDigest ?? "",
          conversationId: trace.routeTrace?.conversationId ?? null,
          activePackId: trace.routeTrace?.activePackId ?? null,
          routerIdentity: trace.routeTrace?.routerIdentity ?? "brain-graph-traverse.v1",
          candidateNodeIds: [...(trace.routeTrace?.candidateNodeIds ?? [])],
          selectedNodeIds: [...(trace.routeTrace?.selectedNodeIds ?? [])],
          selectedPathNodeIds: [...(trace.routeTrace?.selectedPathNodeIds ?? [])],
          sourceSummary: {
            injectedCount: trace.routeTrace?.sourceSummary.injectedCount ?? 0,
            kinds: { ...(trace.routeTrace?.sourceSummary.kinds ?? {}) },
            trusts: { ...(trace.routeTrace?.sourceSummary.trusts ?? {}) },
            sourceUris: [...(trace.routeTrace?.sourceSummary.sourceUris ?? [])],
          },
          selectionMetadata: {
            ...(trace.routeTrace?.selectionMetadata ?? {
              traceSliceVersion: 1,
              queryChars: trace.queryText.length,
              budgetChars: 4000,
              maxHops: 8,
              seedCount: 1,
              candidateCount: 1,
              hopCount: 1,
              firedCount: 1,
              vetoedCount: 0,
              chosenSeedNodeId: null,
              routeSelectionMs: null,
              embeddingMs: null,
              totalQueryMs: null,
              queryEmbeddingSource: "provided" as const,
            }),
          },
        },
        selectedContext: trace.routeTrace?.injectedNodeSummaries ?? [],
      },
    }));
    const { store, worker } = setup({
      teacher: { evaluateTrace },
      config: { teacherEnabled: true },
    });

    store.insertEpisode(makeEpisode({ id: "ep_teacher", conversationId: 21, queryText: "how do I open a pull request?" }));
    store.insertTrace(makeTrace({
      id: "bt_teacher",
      episodeId: "ep_teacher",
      conversationId: 21,
      queryText: "how do I open a pull request?",
      selectedNodeId: "node_pr",
    }));

    await (worker as any).runTeacher();

    expect(evaluateTrace).toHaveBeenCalledTimes(1);
    const evidence = store.getPendingEvidence(10);
    expect(evidence).toHaveLength(1);
    expect(evidence[0]).toMatchObject({
      episodeId: "ep_teacher",
      conversationId: 21,
      source: "teacher",
      kind: "teacher_review",
      value: 0.7,
      reason: "selected route slice is directly relevant",
      metadata: {
        teacherLabel: {
          version: 1,
          traceId: "bt_teacher",
          requestDigest: hashQuery("how do I open a pull request?"),
          input: {
            routeDecision: {
              selectedNodeIds: ["node_pr"],
            },
          },
        },
      },
    });
  });

  it("skips teacher evaluation when no teacher-eligible trace exists", async () => {
    const evaluateTrace = vi.fn();
    const { store, worker } = setup({
      teacher: { evaluateTrace },
      config: { teacherEnabled: true },
    });

    store.insertEpisode(makeEpisode({ id: "ep_no_trace", conversationId: 22 }));

    await (worker as any).runTeacher();

    expect(evaluateTrace).not.toHaveBeenCalled();
    expect(store.getPendingEvidence()).toHaveLength(0);
  });

  it("materializes a candidate-pack PG update artifact from persisted trace supervision and traced teacher labels", async () => {
    const { store, worker, graph } = setup();

    graph.addNode(makeNode("node_human"));
    graph.addNode(makeNode("node_teacher"));
    graph.setSeedWeight("node_human", 0.2);
    graph.setSeedWeight("node_teacher", 0.1);

    store.insertEpisode(makeEpisode({
      id: "ep_human_pg",
      conversationId: 31,
      queryText: "human routed query",
      trajectory: [makeStep("node_human", 0.6)],
      firedNodes: ["node_human"],
    }));
    store.insertTrace(makeTrace({
      id: "bt_human_pg",
      episodeId: "ep_human_pg",
      conversationId: 31,
      queryText: "human routed query",
      selectedNodeId: "node_human",
    }));
    store.insertEvidence({
      episodeId: "ep_human_pg",
      conversationId: 31,
      source: "human",
      kind: "human_feedback",
      value: 0.8,
      confidence: 0.95,
      reason: "user confirmed the route",
    });

    store.insertEpisode(makeEpisode({
      id: "ep_teacher_pg",
      conversationId: 32,
      queryText: "teacher routed query",
      trajectory: [makeStep("node_teacher", 0.7)],
      firedNodes: ["node_teacher"],
    }));
    store.insertTrace(makeTrace({
      id: "bt_teacher_pg",
      episodeId: "ep_teacher_pg",
      conversationId: 32,
      queryText: "teacher routed query",
      selectedNodeId: "node_teacher",
    }));
    store.insertEvidence({
      episodeId: "ep_teacher_pg",
      conversationId: 32,
      source: "teacher",
      kind: "teacher_review",
      value: 0.6,
      confidence: 0.6,
      reason: "teacher verified the selected route",
      metadata: {
        teacherLabel: {
          version: 1,
          traceId: "bt_teacher_pg",
          episodeId: "ep_teacher_pg",
          requestDigest: hashQuery("teacher routed query"),
          input: {
            routeDecision: {
              selectedNodeIds: ["node_teacher"],
            },
          },
        },
      },
    });

    await (worker as any).processEvidence();
    await (worker as any).processLabels();
    await (worker as any).applyUpdates();

    const artifact = store.getTrainingStateJson<PolicyGradientCandidateUpdateArtifact>("last_pg_candidate_update_json");
    expect(artifact).not.toBeNull();
    expect(artifact).toMatchObject({
      version: 1,
      updateCount: 1,
      episodeCount: 2,
      traceCount: 2,
      supervisionCount: 2,
      teacherLabelCount: 1,
      rewardSources: {
        human: 1,
        teacher: 1,
      },
    });
    expect(artifact?.routeUpdateCount).toBeGreaterThan(0);
    expect(artifact?.seedUpdateCount).toBe(artifact?.routeUpdateCount);
    expect(artifact?.edgeUpdateCount).toBe(0);
    expect(artifact?.traceIds).toEqual(["bt_human_pg", "bt_teacher_pg"]);
    expect(artifact?.teacherTraceIds).toEqual(["bt_teacher_pg"]);
    expect(store.getCurrentPackVersion()).toBeNull();

    const candidatePackVersion = Number.parseInt(store.getTrainingState("last_pg_candidate_pack_version") ?? "", 10);
    expect(candidatePackVersion).toBe(artifact?.candidatePackVersion);

    const snapshot = store.readPackSnapshot(candidatePackVersion);
    expect(snapshot?.metadata).toMatchObject({
      reason: "pg_update_candidate",
      pgCandidateUpdate: {
        updateCount: 1,
        candidatePackVersion,
        supervisionCount: 2,
        teacherLabelCount: 1,
      },
    });

    expect(store.getEpisodesForUpdate(10)).toHaveLength(0);
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
    const promotionReadyCall = ((onPromotionReady.mock.calls as unknown) as Array<[{
      promotionVerdict?: { bundleVerdicts?: Array<{ reason?: { code?: string } }> };
    }]>) [0]?.[0];
    expect(promotionReadyCall?.promotionVerdict?.bundleVerdicts?.[0]?.reason?.code).toBe("promoted");
  });
});
