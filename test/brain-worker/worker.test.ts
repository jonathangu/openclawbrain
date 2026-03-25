import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import { DEFAULT_BRAIN_CONFIG } from "../../src/brain-core/types.js";
import type {
  BrainNode,
  BrainObservation,
  BrainObservationToolResult,
  DecisionTrace,
  Episode,
  HealthMetrics,
  MutationProposal,
  PolicyGradientCandidateUpdateArtifact,
  ReplayGateVerdict,
  TrajectoryExpansion,
} from "../../src/brain-core/types.js";
import { runBrainMigrations } from "../../src/brain-store/migrations.js";
import { BrainStore } from "../../src/brain-store/store.js";
import { BrainWorker } from "../../src/brain-worker/worker.js";

const tempDirs: string[] = [];

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
  conversationId: number;
  queryText?: string;
  trajectory?: TrajectoryExpansion[];
  firedNodes?: string[];
  reward?: number | null;
  rewardSource?: Episode["rewardSource"];
}): Episode {
  return {
    id: params.id,
    conversationId: params.conversationId,
    queryText: params.queryText ?? "test query",
    queryEmbedding: new Float32Array([1, 0, 0]),
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

function makeStep(targetId: string, probability = 0.6): TrajectoryExpansion {
  return {
    sourceNodeId: null,
    expansionIndex: 0,
    frontierBefore: [],
    frontierAfter: [targetId],
    budgetBefore: 1000,
    budgetAfter: 968,
    substeps: [
      {
        stateSnapshot: {
          sourceNodeId: null,
          expansionIndex: 0,
          selectionIndex: 0,
          budgetRemaining: 1000,
          initialBudget: 1000,
          reservedTokenCost: 0,
          maxHops: 8,
          frontierSize: 0,
          frontierNodeIds: [],
          visitedCount: 0,
          firedCount: 0,
        },
        candidates: [
          { action: { type: "traverse", targetNodeId: targetId }, score: 1, probability },
          { action: { type: "stop_local" }, score: -1, probability: 1 - probability },
        ],
        chosenAction: { type: "traverse", targetNodeId: targetId },
        chosenActionProbability: probability,
        stopProbability: 1 - probability,
      },
    ],
    selectedTargets: [targetId],
    acceptedTargets: [targetId],
    vetoedTargets: [],
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
      routerIdentity: "brain-graph-traverse.v2",
      candidateNodeIds: [...firedNodes],
      selectedNodeIds: [...firedNodes],
      selectedTraversalNodeIds: [...firedNodes],
      selectedPathNodeIds: [...firedNodes],
      selectedSeedNodeIds: [selectedNodeId],
      injectedNodeSummaries,
      sourceSummary: {
        injectedCount: injectedNodeSummaries.length,
        kinds: { workflow: injectedNodeSummaries.length },
        trusts: { human: injectedNodeSummaries.length },
        sourceUris: injectedNodeSummaries.flatMap((summary) => summary.sourceUri ? [summary.sourceUri] : []),
      },
      selectionMetadata: {
        traceSliceVersion: 2,
        queryChars: queryText.length,
        budgetChars: 4000,
        maxHops: 8,
        maxFanoutPerNode: 4,
        maxFrontierSize: 32,
        seedCount: 1,
        seedSelectionCount: 1,
        candidateCount: firedNodes.length,
        hopCount: firedNodes.length,
        expansionCount: firedNodes.length,
        selectionSubstepCount: firedNodes.length + 1,
        firedCount: firedNodes.length,
        vetoedCount: 0,
        chosenSeedNodeId: selectedNodeId,
        selectedSeedNodeIds: [selectedNodeId],
        routeSelectionMs: 8,
        embeddingMs: 3,
        totalQueryMs: 15,
        queryEmbeddingSource: "provided",
      },
    },
    createdAt: Date.now(),
  };
}

function makeObservation(params: {
  episodeId: string;
  conversationId: number;
  traceId: string;
  queryText?: string;
  assistantResponse?: string;
  toolResults?: BrainObservationToolResult[];
  followUpText?: string | null;
  createdAt?: number;
  updatedAt?: number;
}): Omit<BrainObservation, "id" | "phase1Score" | "phase2Score" | "finalScore" | "confidence" | "reason" | "teacherEvaluation" | "evaluatedAt" | "updatedAt"> & { updatedAt?: number } {
  const queryText = params.queryText ?? "test query";
  return {
    episodeId: params.episodeId,
    conversationId: params.conversationId,
    traceId: params.traceId,
    queryText,
    retrievedContext: [
      {
        nodeId: "node_1",
        kind: "workflow",
        trust: "human",
        sourceUri: "PLAYBOOK.md",
        tags: ["worker"],
        tokenCount: 12,
        contentPreview: "Use node_1 for worker route guidance.",
      },
    ],
    routeMetadata: {
      requestDigest: hashQuery(queryText),
      activePackId: "brain-pack-v1",
      routerIdentity: "brain-graph-traverse.v2",
      candidateNodeIds: ["node_1"],
      selectedNodeIds: ["node_1"],
      selectedTraversalNodeIds: ["node_1"],
      selectedPathNodeIds: ["node_1"],
      selectedSeedNodeIds: ["node_1"],
      sourceSummary: {
        injectedCount: 1,
        kinds: { workflow: 1 },
        trusts: { human: 1 },
        sourceUris: ["PLAYBOOK.md"],
      },
      selectionMetadata: {
        traceSliceVersion: 2,
        queryChars: queryText.length,
        budgetChars: 4000,
        maxHops: 8,
        maxFanoutPerNode: 4,
        maxFrontierSize: 32,
        seedCount: 1,
        seedSelectionCount: 1,
        candidateCount: 1,
        hopCount: 1,
        expansionCount: 1,
        selectionSubstepCount: 2,
        firedCount: 1,
        vetoedCount: 0,
        chosenSeedNodeId: "node_1",
        selectedSeedNodeIds: ["node_1"],
        routeSelectionMs: 8,
        embeddingMs: 3,
        totalQueryMs: 15,
        queryEmbeddingSource: "provided",
      },
    },
    assistantResponse: params.assistantResponse ?? "default answer",
    toolResults: params.toolResults ?? [],
    followUpText: params.followUpText ?? null,
    status: params.followUpText ? "pending_teacher" : "pending_followup",
    createdAt: params.createdAt ?? Date.now(),
    updatedAt: params.updatedAt ?? (params.createdAt ?? Date.now()),
  };
}

function makeTeacherReview(
  observation: BrainObservation,
  overrides: Partial<{
    retrievalRelevance: number;
    agentUsage: number;
    outcomeSupport: number;
    finalScore: number;
    confidence: number;
    reason: string;
  }> = {},
) {
  return {
    version: 2 as const,
    observationId: observation.id,
    episodeId: observation.episodeId,
    traceId: observation.traceId,
    retrievalRelevance: overrides.retrievalRelevance ?? 0.8,
    agentUsage: overrides.agentUsage ?? 0.4,
    outcomeSupport: overrides.outcomeSupport ?? 0.5,
    finalScore: overrides.finalScore ?? 0.55,
    confidence: overrides.confidence ?? 0.7,
    reason: overrides.reason ?? "teacher review",
    input: {
      version: 2 as const,
      observationId: observation.id,
      episodeId: observation.episodeId,
      traceId: observation.traceId,
      conversationId: observation.conversationId,
      queryText: observation.queryText,
      selectedContext: observation.retrievedContext,
      routeMetadata: observation.routeMetadata,
      assistantResponse: observation.assistantResponse,
      toolResults: observation.toolResults,
      nextUserTurn: observation.followUpText,
    },
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

describe("BrainWorker observation reward cutover", () => {
  it("scores good retrieval + bad generation through teacher-v2 and materializes supervision", async () => {
    const evaluateObservation = vi.fn(async (observation: BrainObservation) => makeTeacherReview(
      observation,
      {
        retrievalRelevance: 0.92,
        agentUsage: -0.4,
        outcomeSupport: -0.85,
        finalScore: -0.72,
        confidence: 0.74,
        reason: "retrieval was good but the answer contradicted the retrieved guidance",
      },
    ));
    const { store, worker } = setup({
      teacher: { evaluateObservation },
      config: { teacherEnabled: true },
    });
    store.insertEpisode(makeEpisode({
      id: "ep_bad_generation",
      conversationId: 7,
      queryText: "how do I open a pull request?",
      trajectory: [makeStep("node_1", 0.6)],
      firedNodes: ["node_1"],
    }));
    store.insertTrace(makeTrace({
      id: "bt_bad_generation",
      episodeId: "ep_bad_generation",
      conversationId: 7,
      queryText: "how do I open a pull request?",
      selectedNodeId: "node_1",
    }));
    store.insertObservation(makeObservation({
      episodeId: "ep_bad_generation",
      conversationId: 7,
      traceId: "bt_bad_generation",
      queryText: "how do I open a pull request?",
      assistantResponse: "Use `git push origin main` to open the pull request.",
      createdAt: Date.now() - 30_000,
    }));

    await (worker as any).evaluatePendingObservations();

    expect(evaluateObservation).toHaveBeenCalledTimes(1);
    const observation = store.getObservationForEpisode("ep_bad_generation");
    expect(observation).toMatchObject({
      status: "completed",
      phase1Score: 0.92,
      phase2Score: -0.85,
      finalScore: -0.72,
      confidence: 0.74,
    });
    expect(store.getPendingLabels()).toMatchObject([
      {
        source: "teacher",
        value: -0.72,
      },
    ]);

    await (worker as any).processLabels();
    expect(store.getEpisode("ep_bad_generation")).toMatchObject({
      reward: -0.72,
      rewardSource: "teacher",
    });

    const supervision = store.getTraceSupervision("bt_bad_generation", 10);
    expect(supervision).toHaveLength(1);
    expect(supervision[0]).toMatchObject({
      source: "teacher",
      kind: "teacher_review",
      value: -0.72,
      resolution: "promoted_to_label",
      metadata: expect.objectContaining({
        observationId: observation?.id,
        phase1Score: 0.92,
        phase2Score: -0.85,
        agentUsage: -0.4,
      }),
    });
  });

  it("scores good retrieval + tool failure without needing follow-up text", async () => {
    const evaluateObservation = vi.fn(async (observation: BrainObservation) => makeTeacherReview(
      observation,
      {
        retrievalRelevance: 0.88,
        agentUsage: 0.1,
        outcomeSupport: -0.91,
        finalScore: -0.58,
        confidence: 0.81,
        reason: "retrieval was relevant but the tool failure left the turn unsupported",
      },
    ));
    const { store, worker } = setup({
      teacher: { evaluateObservation },
      config: { teacherEnabled: true },
    });
    store.insertEpisode(makeEpisode({
      id: "ep_tool_failure",
      conversationId: 8,
      queryText: "run the test suite",
      trajectory: [makeStep("node_1", 0.7)],
      firedNodes: ["node_1"],
    }));
    store.insertTrace(makeTrace({
      id: "bt_tool_failure",
      episodeId: "ep_tool_failure",
      conversationId: 8,
      queryText: "run the test suite",
      selectedNodeId: "node_1",
    }));
    store.insertObservation(makeObservation({
      episodeId: "ep_tool_failure",
      conversationId: 8,
      traceId: "bt_tool_failure",
      queryText: "run the test suite",
      assistantResponse: "I ran the tests.",
      toolResults: [
        {
          sourceRole: "tool",
          toolCallId: "call_1",
          toolName: "bash",
          input: "{\"cmd\":\"pnpm test\"}",
          output: "{\"ok\":false,\"code\":\"ENOENT\",\"exitCode\":2}",
          isError: true,
          excerpt: "{\"ok\":false,\"code\":\"ENOENT\",\"exitCode\":2}",
        },
      ],
      createdAt: Date.now() - 30_000,
    }));

    await (worker as any).evaluatePendingObservations();
    await (worker as any).processLabels();

    expect(evaluateObservation).toHaveBeenCalledWith(
      expect.objectContaining({
        toolResults: [
          expect.objectContaining({
            toolName: "bash",
            isError: true,
          }),
        ],
      }),
    );
    expect(store.getEpisode("ep_tool_failure")).toMatchObject({
      reward: -0.58,
      rewardSource: "teacher",
    });
  });

  it("waits for maturity, then forwards ambiguous follow-up text when it arrives", async () => {
    const evaluateObservation = vi.fn(async (observation: BrainObservation) => makeTeacherReview(
      observation,
      {
        retrievalRelevance: 0.5,
        agentUsage: 0.1,
        outcomeSupport: 0.05,
        finalScore: 0.08,
        confidence: 0.2,
        reason: "follow-up was ambiguous so reward confidence stays low",
      },
    ));
    const { store, worker } = setup({
      teacher: { evaluateObservation },
      config: { teacherEnabled: true },
    });
    store.insertEpisode(makeEpisode({
      id: "ep_ambiguous",
      conversationId: 9,
      queryText: "what should I do next?",
    }));
    store.insertTrace(makeTrace({
      id: "bt_ambiguous",
      episodeId: "ep_ambiguous",
      conversationId: 9,
      queryText: "what should I do next?",
    }));
    store.insertObservation(makeObservation({
      episodeId: "ep_ambiguous",
      conversationId: 9,
      traceId: "bt_ambiguous",
      queryText: "what should I do next?",
      assistantResponse: "Try the deployment checklist.",
      createdAt: Date.now(),
    }));

    await (worker as any).evaluatePendingObservations();
    expect(evaluateObservation).not.toHaveBeenCalled();

    store.attachObservationFollowUp(9, "Thanks, maybe?");
    await (worker as any).evaluatePendingObservations();

    expect(evaluateObservation).toHaveBeenCalledTimes(1);
    expect(evaluateObservation).toHaveBeenCalledWith(
      expect.objectContaining({
        followUpText: "Thanks, maybe?",
      }),
    );
    expect(store.getObservationForEpisode("ep_ambiguous")).toMatchObject({
      status: "completed",
      confidence: 0.2,
      reason: "follow-up was ambiguous so reward confidence stays low",
    });
  });

  it("materializes a candidate-pack PG update artifact from human and teacher observation supervision", async () => {
    const evaluateObservation = vi.fn(async (observation: BrainObservation) => makeTeacherReview(
      observation,
      {
        retrievalRelevance: 0.7,
        agentUsage: 0.6,
        outcomeSupport: 0.5,
        finalScore: 0.6,
        confidence: 0.65,
        reason: "teacher verified the routed turn",
      },
    ));
    const { store, worker, graph } = setup({
      teacher: { evaluateObservation },
      config: { teacherEnabled: true },
    });

    graph.addNode(makeNode("node_human"));
    graph.addNode(makeNode("node_teacher"));
    graph.setSeedWeight("node_human", 0.2);
    graph.setSeedWeight("node_teacher", 0.1);

    store.insertEpisode(makeEpisode({
      id: "ep_human_pg",
      conversationId: 31,
      queryText: "stale retrieval query",
      trajectory: [makeStep("node_human", 0.6)],
      firedNodes: ["node_human"],
    }));
    store.insertTrace(makeTrace({
      id: "bt_human_pg",
      episodeId: "ep_human_pg",
      conversationId: 31,
      queryText: "stale retrieval query",
      selectedNodeId: "node_human",
    }));
    const humanLabel = store.insertLabel({
      episodeId: "ep_human_pg",
      source: "human",
      value: -0.5,
      reason: "user correction landed immediately",
    });
    store.insertTraceSupervision({
      traceId: "bt_human_pg",
      episodeId: "ep_human_pg",
      conversationId: 31,
      source: "human",
      kind: "teach_correction",
      value: -0.5,
      confidence: 1.0,
      reason: "user correction landed immediately",
      resolution: "promoted_to_label",
      labelId: humanLabel.id,
      metadata: {
        traceId: "bt_human_pg",
      },
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
    store.insertObservation(makeObservation({
      episodeId: "ep_teacher_pg",
      conversationId: 32,
      traceId: "bt_teacher_pg",
      queryText: "teacher routed query",
      assistantResponse: "Here is the routed answer.",
      createdAt: Date.now() - 30_000,
    }));

    await (worker as any).evaluatePendingObservations();
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
    expect(artifact?.traceIds).toEqual(["bt_human_pg", "bt_teacher_pg"]);
    expect(artifact?.teacherTraceIds).toEqual(["bt_teacher_pg"]);
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
