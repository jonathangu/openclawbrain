import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainGraph } from "../../src/brain-core/graph.js";
import type { BrainEdge, BrainNode } from "../../src/brain-core/types.js";
import { BrainAssemblerExtension } from "../../src/brain-runtime/assembler-extension.js";
import { BrainService } from "../../src/brain-runtime/service.js";
import type { LcmDependencies } from "../../src/types.js";

const tempDirs: string[] = [];

function deriveExpectedQueryBudgetChars(tokenBudget: number): number {
  return Math.max(256, Math.floor(tokenBudget * 4 * 0.3));
}

function makeTempDir(prefix: string): string {
  const dir = mkdtempSync(join(tmpdir(), prefix));
  tempDirs.push(dir);
  return dir;
}

function embed(text: string): Float32Array {
  const normalized = text.toLowerCase();
  if (normalized.includes("pull request") || normalized.includes("gh pr create")) {
    return new Float32Array([1, 0, 0]);
  }
  if (normalized.includes("deployment") || normalized.includes("ci")) {
    return new Float32Array([0, 1, 0]);
  }
  if (normalized.includes("codeword") || normalized.includes("hippo") || normalized.includes("giraffe")) {
    return new Float32Array([1, 0, 1]);
  }
  return new Float32Array([0.5, 0.5, 0]);
}

function makeRuntimeNode(id: string, embedding: Float32Array, tokenCount = 100): BrainNode {
  return {
    id,
    kind: "chunk",
    content: `node ${id}`,
    embedding,
    sourceUri: "test.md",
    trust: "scanner",
    tags: [],
    tokenCount,
    metadata: {},
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

function makeRuntimeEdge(
  source: string,
  target: string,
  kind: BrainEdge["kind"] = "learned",
  weight = 1,
): BrainEdge {
  return {
    source,
    target,
    kind,
    weight,
    prior: 1,
    metadata: {},
    decayedAt: Date.now(),
    createdAt: Date.now(),
  };
}

function createDeps(
  brainRoot: string,
  overrides?: Partial<NonNullable<LcmDependencies["config"]["brain"]>>,
): LcmDependencies {
  return {
    config: {
      enabled: true,
      databasePath: join(brainRoot, "lcm.db"),
      contextThreshold: 0.75,
      freshTailCount: 8,
      leafMinFanout: 8,
      condensedMinFanout: 4,
      condensedMinFanoutHard: 2,
      incrementalMaxDepth: 0,
      leafChunkTokens: 20000,
      leafTargetTokens: 1200,
      condensedTargetTokens: 2000,
      maxExpandTokens: 4000,
      largeFileTokenThreshold: 25000,
      largeFileSummaryProvider: "",
      largeFileSummaryModel: "",
      autocompactDisabled: false,
      timezone: "America/Los_Angeles",
      pruneHeartbeatOk: false,
      brain: {
        enabled: true,
        root: brainRoot,
        budgetFraction: 0.3,
        maxHops: 8,
        maxFanoutPerNode: 4,
        maxFrontierSize: 32,
        maxSeeds: 10,
        semanticThreshold: 0.1,
        servingTemperature: 0.1,
        learningTemperature: 1,
        learningRate: 0.01,
        baselineAlpha: 0.1,
        decayRate: 0.995,
        trainerIntervalMs: 10_000,
        workerMode: "in_process",
        workerHeartbeatTimeoutMs: 5_000,
        workerRestartDelayMs: 100,
        teacherEnabled: false,
        persistRawSurfaces: false,
        teacherProvider: "",
        teacherModel: "",
    autoUserCorrectionsEnabled: false,
    autoUserCorrectionsProvider: "",
    autoUserCorrectionsModel: "",
    autoUserCorrectionsMinConfidence: 0.8,
        mutationsEnabled: true,
        replayEpisodeCount: 100,
        minFiredPerQuery: 1,
        maxDormantPercent: 0.3,
        maxOrphanCount: 10,
        shadowMode: false,
        embeddingProvider: "openai",
        embeddingModel: "text-embedding-3-small",
        embeddingBaseUrl: "https://example.invalid/v1",
        ...overrides,
      },
    },
    complete: vi.fn(async () => ({ content: [{ type: "text", text: "{}" }] })),
    callGateway: vi.fn(async () => ({})),
    resolveModel: vi.fn(() => ({ provider: "openai", model: "gpt-5.4-mini" })),
    getApiKey: vi.fn(async () => "test-key"),
    requireApiKey: vi.fn(async () => "test-key"),
    parseAgentSessionKey: vi.fn(() => null),
    isSubagentSessionKey: vi.fn(() => false),
    normalizeAgentId: vi.fn((id?: string) => id ?? "main"),
    buildSubagentSystemPrompt: vi.fn(() => ""),
    readLatestAssistantReply: vi.fn(() => undefined),
    resolveAgentDir: vi.fn(() => brainRoot),
    resolveSessionIdFromSessionKey: vi.fn(async () => undefined),
    agentLaneSubagent: "subagent",
    log: {
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
      debug: vi.fn(),
    },
  };
}

async function waitFor(predicate: () => Promise<boolean> | boolean, timeoutMs = 3_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await predicate()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(`Condition not met within ${timeoutMs}ms`);
}

afterEach(() => {
  vi.restoreAllMocks();
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("BrainService", () => {
  it("initializes a workspace and serves query traces from the promoted pack", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-workspace-");
    const brainRoot = makeTempDir("openclawbrain-state-");
    writeFileSync(
      join(workspaceRoot, "PLAYBOOK.md"),
      "# Pull Requests\n\nUse gh pr create for pull request workflows.\n",
      "utf8",
    );

    const service = new BrainService({
      deps: createDeps(brainRoot),
    });

    const summary = await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });

    expect(summary).toContain("Brain initialized");
    expect(service.isInitialized()).toBe(true);

    const result = await service.query({
      conversationId: 42,
      agentIdentity: { agentId: "main", lane: "main" },
      queryText: "how do I open a pull request?",
      budgetChars: 4000,
      queryEmbedding: embed("gh pr create pull request"),
    });

    expect(result).not.toBeNull();
    expect(result?.episode.conversationId).toBe(42);

    const trace = await service.getTrace();
    expect(trace?.episodeId).toBe(result?.episode.id ?? null);
    expect(trace?.routeTrace).toMatchObject({
      persistenceMode: "redacted",
      conversationId: 42,
      activePackId: "brain-pack-v1",
      routerIdentity: "brain-graph-traverse.v2",
    });
    expect(trace?.routeTrace?.selectedNodeIds).toContain(result?.fired[0]?.nodeId ?? "");
    expect(trace?.routeTrace?.injectedNodeSummaries).toEqual([
      expect.objectContaining({
        nodeId: result?.fired[0]?.nodeId,
        kind: "chunk",
        provenanceRef: expect.stringMatching(/^prov_[a-f0-9]{16}$/),
        sourceUri: null,
        contentPreview: expect.stringContaining("[redacted source_content chars="),
      }),
    ]);
    expect(trace?.routeTrace?.selectionMetadata).toEqual(expect.objectContaining({
      traceSliceVersion: 4,
      budgetChars: 4000,
      maxHops: 8,
      firedCount: result?.fired.length,
      queryEmbeddingSource: "provided",
      chosenStopCount: expect.any(Number),
      forcedStopCount: expect.any(Number),
      droppedProposalCount: 0,
      droppedProposalReasons: null,
      compileReportSummary: expect.stringContaining("[brain compile]"),
      retryIdentity: expect.objectContaining({
        turnId: expect.stringMatching(/^turn_[a-f0-9]{16}$/),
        traceId: expect.stringMatching(/^rt_[a-f0-9]{16}$/),
      }),
      compileReport: expect.objectContaining({
        schemaVersion: 1,
        summary: expect.stringContaining("[brain compile]"),
        buckets: expect.objectContaining({
          selected: expect.any(Array),
          dropped: expect.any(Array),
        }),
      }),
    }));
    expect(trace?.routeTrace?.selectionMetadata?.decisionPointSnapshots?.length).toBeGreaterThan(0);
    expect(trace?.routeTrace?.selectionMetadata?.decisionPointSummary).toContain("[brain decision points]");
    expect(trace?.routeTrace?.requestDigest).toMatch(/^[a-f0-9]{16}$/);
    expect(trace?.routeTrace?.candidateNodeIds).toContain(result?.fired[0]?.nodeId ?? "");
    expect(trace?.routeTrace?.sourceSummary.kinds).toMatchObject({ chunk: 1 });
    expect(trace?.routeTrace?.sourceSummary.sourceUris).toEqual([]);
    expect(trace?.routeTrace?.sourceSummary.sourceRefs[0]).toMatch(/^prov_[a-f0-9]{16}$/);
    expect(trace?.routeTrace?.selectionMetadata.chosenStopCount).toBe(0);
    expect(trace?.routeTrace?.selectionMetadata.forcedStopCount ?? 0).toBeGreaterThan(0);
    expect(trace?.routeTrace?.selectionMetadata.branchOutcomeSummary).toEqual(expect.objectContaining({
      branchCount: expect.any(Number),
      continuingBranchCount: expect.any(Number),
      stoppedWithoutProgressCount: expect.any(Number),
      chosenStopBranchCount: expect.any(Number),
      forcedStopBranchCount: expect.any(Number),
      detail: expect.stringContaining("branches continued"),
    }));
    expect(trace?.routeTrace?.branchOutcomes[0]).toEqual(expect.objectContaining({
      sourceNodeId: null,
      continued: true,
      proof: expect.stringContaining("branch start continued"),
    }));
    const status = await service.status();
    expect(status.currentPackVersion).toBe(1);
    expect(status.routeTraceCount).toBe(1);
    expect(status.supervisionCount).toBe(0);
    expect(status.rawPersistenceEnabled).toBe(false);
    expect(status.modelTraceSurface).toBe("redacted");
    expect(status.teacherInputSurface).toBe("redacted");
    expect(status.lastTraceContextChars).toBe(result?.trace.contextChars ?? null);
    expect(status.lastTraceSelectionMetadata).toEqual(expect.objectContaining({
      budgetChars: 4000,
      totalQueryMs: expect.any(Number),
      queryEmbeddingSource: "provided",
      chosenStopCount: expect.any(Number),
      forcedStopCount: expect.any(Number),
      branchOutcomeSummary: expect.objectContaining({
        branchCount: expect.any(Number),
        continuingBranchCount: expect.any(Number),
        detail: expect.stringContaining("branches continued"),
      }),
      droppedProposalCount: 0,
      droppedProposalReasons: null,
      compileReportSummary: expect.stringContaining("[brain compile]"),
      retryIdentity: expect.objectContaining({
        turnId: expect.stringMatching(/^turn_[a-f0-9]{16}$/),
        traceId: expect.stringMatching(/^rt_[a-f0-9]{16}$/),
      }),
    }));
    expect(status.lastCompileReportSummary).toEqual(expect.stringContaining("[brain compile]"));
    expect(status.lastCompileReportSummary).toEqual(expect.stringContaining("q_budget=4000"));
  });

  it("reports recent decision histograms plus clip and fail-open rates", async () => {
    const brainRoot = makeTempDir("openclawbrain-decision-summary-state-");
    const service = new BrainService({
      deps: createDeps(brainRoot),
    });
    const graph = new BrainGraph();
    graph.addNode(makeRuntimeNode("a", new Float32Array([1, 0, 0])));
    (service as unknown as { servingGraph: BrainGraph }).servingGraph = graph;

    const servedFull = await service.query({
      conversationId: 1,
      queryText: "first question",
      budgetChars: 4000,
      queryEmbedding: embed("pull request"),
    });
    const servedClipped = await service.query({
      conversationId: 2,
      queryText: "second question",
      budgetChars: 4000,
      queryEmbedding: embed("pull request"),
    });
    const failOpen = await service.query({
      conversationId: 3,
      queryText: "third question",
      budgetChars: 4000,
      queryEmbedding: embed("pull request"),
    });

    expect(servedFull?.trace).toBeTruthy();
    expect(servedClipped?.trace).toBeTruthy();
    expect(failOpen?.trace).toBeTruthy();

    service.recordTraceSelectionMetadata(servedClipped?.trace ?? null, {
      contextClipped: true,
      brainDropReason: "injection_cap_clipped",
      fitStrategy: "structured_node_budget",
    });
    service.recordTraceSelectionMetadata(failOpen?.trace ?? null, {
      queryInterrupted: true,
      servedPartial: true,
      interruptionStage: "injection",
      interruptionReason: "deadline_before_injection",
      brainDropReason: "deadline_before_injection",
    });

    const status = await service.status();
    expect(status.recentDecisionSummary).toEqual(expect.objectContaining({
      windowSize: 25,
      sampleSize: 3,
      histograms: expect.objectContaining({
        decisionOutcome: expect.objectContaining({
          served_full: 1,
          served_clipped: 1,
          partial_fail_open: 1,
          partial_fail_open_clipped: 0,
          interrupted_without_partial: 0,
        }),
        brainDropReason: expect.objectContaining({
          none: 1,
          injection_cap_clipped: 1,
          deadline_before_injection: 1,
        }),
        interruptionStage: expect.objectContaining({
          none: 2,
          injection: 1,
        }),
        fitStrategy: expect.objectContaining({
          none: 2,
          structured_node_budget: 1,
        }),
        queryEmbeddingSource: expect.objectContaining({
          provided: 3,
        }),
      }),
      clipRate: {
        count: 1,
        rate: 1 / 3,
      },
      failOpenRate: {
        count: 1,
        rate: 1 / 3,
      },
      detail: "1/3 clipped and 1/3 fail-open or interrupted across the recent decision window",
      branchBehavior: expect.objectContaining({
        branchCount: expect.any(Number),
        continuingBranchCount: expect.any(Number),
        histograms: expect.objectContaining({
          stopTruth: expect.objectContaining({
            forced: expect.any(Number),
          }),
          terminationReason: expect.any(Object),
        }),
        detail: expect.stringContaining("recent branches continued"),
      }),
    }));
    expect(status.boundedAnytimeSummary).toEqual(expect.objectContaining({
      defaultDeadlinePosture: "off_by_default",
      configuredCompileDeadlineMs: null,
      latest: expect.objectContaining({
        deadline: expect.objectContaining({
          configuredMs: null,
          posture: "off_by_default",
          hit: false,
        }),
      }),
      recent: expect.objectContaining({
        sampleSize: 3,
        clipRate: expect.objectContaining({
          count: 1,
          rate: 1 / 3,
        }),
        failOpenRate: expect.objectContaining({
          count: 1,
          rate: 1 / 3,
        }),
      }),
    }));
    expect(["full", "partial", "fail_open"]).toContain(status.boundedAnytimeSummary.latest?.posture ?? null);
    expect(status.boundedAnytimeSummary.detail).toContain("deadline=off-by-default");
    expect(status.boundedAnytimeSummary.detail).toContain("recent 1/3 clipped and 1/3 fail-open or interrupted across the recent decision window");
  });

  it("persists trace-v3 dropped proposal reasons from traversal into stored traces", async () => {
    const brainRoot = makeTempDir("openclawbrain-trace-v3-state-");
    const service = new BrainService({
      deps: createDeps(brainRoot),
    });
    const graph = new BrainGraph();
    graph.addNode(makeRuntimeNode("a", new Float32Array([1, 0, 0])));
    graph.addNode(makeRuntimeNode("b", new Float32Array([0, 1, 0]), 80));
    graph.addEdge(makeRuntimeEdge("a", "b"));
    (service as unknown as { servingGraph: BrainGraph }).servingGraph = graph;
    vi.spyOn(Math, "random").mockReturnValue(0);

    const originalGetNode = graph.getNode.bind(graph);
    let bCalls = 0;
    vi.spyOn(graph, "getNode").mockImplementation((nodeId: string) => {
      const node = originalGetNode(nodeId);
      if (nodeId !== "b") {
        return node;
      }
      bCalls += 1;
      return bCalls === 3 ? undefined : node;
    });

    const result = await service.query({
      conversationId: 99,
      queryText: "trace-v3 missing target",
      budgetChars: 4000,
      queryEmbedding: new Float32Array([1, 0, 0]),
    });

    expect(result).not.toBeNull();
    expect(result?.trace.routeTrace?.selectionMetadata).toMatchObject({
      traceSliceVersion: 4,
      chosenStopCount: 0,
      forcedStopCount: 2,
      droppedProposalCount: 1,
      droppedProposalReasons: {
        missing_target_node: 1,
      },
    });
    const branchExpansion = result?.trace.trajectory.find((expansion) => expansion.sourceNodeId === "a");
    expect(branchExpansion).toMatchObject({
      terminationReason: "missing_target_node",
      proposalOutcomes: [
        expect.objectContaining({
          targetNodeId: "b",
          outcome: "dropped",
          reason: "missing_target_node",
        }),
      ],
    });

    const storedTrace = await service.getTrace(result?.trace.id);
    expect(storedTrace?.routeTrace?.selectionMetadata).toMatchObject({
      droppedProposalCount: 1,
      droppedProposalReasons: {
        missing_target_node: 1,
      },
    });
  });

  it("records structured interruption truth when the deadline expires before embedding starts", async () => {
    const brainRoot = makeTempDir("openclawbrain-interrupt-state-");
    const service = new BrainService({
      deps: createDeps(brainRoot),
    });
    const graph = new BrainGraph();
    graph.addNode(makeRuntimeNode("a", new Float32Array([1, 0, 0])));
    (service as unknown as { servingGraph: BrainGraph }).servingGraph = graph;

    const result = await service.query({
      conversationId: 77,
      queryText: "interrupt before embedding",
      budgetChars: 4000,
      deadlineAtMs: Date.now() - 1,
    });

    expect(result).toBeNull();
    expect(service.getLastQueryInterruption()).toEqual({
      interrupted: true,
      stage: "embedding",
      reason: "deadline_before_embedding",
      servedPartial: false,
    });
  });

  it("surfaces the active runtime controls in status", async () => {
    const brainRoot = makeTempDir("openclawbrain-controls-");
    const service = new BrainService({
      deps: createDeps(brainRoot, {
        maxCompileMs: 85,
        budgetFraction: 0.55,
        maxHops: 11,
        maxFanoutPerNode: 6,
        maxFrontierSize: 21,
        maxSeeds: 14,
        semanticThreshold: 0.61,
        shadowMode: true,
        workerMode: "child",
        workerHeartbeatTimeoutMs: 15_000,
        workerRestartDelayMs: 750,
      }),
    });

    const status = await service.status();
    expect(status).toEqual(expect.objectContaining({
      maxCompileMs: 85,
      budgetFraction: 0.55,
      maxHops: 11,
      maxFanoutPerNode: 6,
      maxFrontierSize: 21,
      maxSeeds: 14,
      semanticThreshold: 0.61,
      shadowMode: true,
      workerMode: "child",
      workerHeartbeatTimeoutMs: 15_000,
      workerRestartDelayMs: 750,
    }));
  });

  it("surfaces bounded assembly metrics in status", async () => {
    const queryBudgetChars = deriveExpectedQueryBudgetChars(4096);
    const brainRoot = makeTempDir("openclawbrain-status-");
    const service = new BrainService({
      deps: createDeps(brainRoot),
    });

    service.noteAssemblyDecision({
      mode: "use_brain",
      conversationId: 42,
      episodeId: "ep_1",
      traceId: "tr_1",
      footer: "[brain] used graph retrieval for this turn.",
      compileElapsedMs: 12,
      compileDeadlineMs: 20,
      compileDeadlineHit: false,
      brainDropReason: "injection_cap_clipped",
      brainDropStage: "injection",
      budgetFraction: 0.3,
      maxContextChars: 240,
      queryBudgetChars,
      injectedChars: 180,
      droppedChars: 72,
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: 3,
      fittedNodeCount: 2,
      droppedNodeCount: 1,
      fittingDropReasons: {
        omitted_for_max_context_chars: 1,
      },
      summaryRoutingMode: "expand_to_source",
      summaryMetadata: {
        totalCount: 3,
        maxDepth: 2,
        condensedCount: 1,
        episodeCount: 1,
        snapshotCount: 1,
        branchCount: 2,
        typedMemoryRefCount: 1,
        freshnessStateCounts: { fresh: 2, superseded: 1 },
        hasNonFreshSummaries: true,
        hasTruthConflict: true,
        latestRole: "support",
        items: [],
      },
    });

    const status = await service.status();
    expect(status.lastAssemblyDecision).toMatchObject({
      mode: "use_brain",
      conversationId: 42,
      traceId: "tr_1",
      compileElapsedMs: 12,
      compileDeadlineMs: 20,
      compileDeadlineHit: false,
      brainDropReason: "injection_cap_clipped",
      brainDropStage: "injection",
      budgetFraction: 0.3,
      maxContextChars: 240,
      queryBudgetChars,
      injectedChars: 180,
      droppedChars: 72,
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: 3,
      fittedNodeCount: 2,
      droppedNodeCount: 1,
      fittingDropReasons: {
        omitted_for_max_context_chars: 1,
      },
      summaryRoutingMode: "expand_to_source",
      summaryMetadata: expect.objectContaining({
        totalCount: 3,
        branchCount: 2,
        snapshotCount: 1,
        hasNonFreshSummaries: true,
      }),
    });
    expect(status.routeQuality).toMatchObject({
      summaryRoutingMode: "expand_to_source",
      compactHealth: {
        status: "healthy",
        count: 3,
        nonFreshCount: 1,
        branchCount: 2,
        snapshotCount: 1,
      },
    });
  });

  it("surfaces the canonical context-management model in status", async () => {
    const brainRoot = makeTempDir("openclawbrain-context-model-");
    const service = new BrainService({
      deps: createDeps(brainRoot),
    });

    const status = await service.status();

    expect(status.contextManagement).toMatchObject({
      model: "summary_spine_with_protected_fresh_tail",
      hotContext: {
        layout: "summary_spine_then_protected_fresh_tail",
        sourceOfTruth: "context_items",
        freshTailCount: 8,
      },
      freshness: {
        nonFreshStates: [
          "stale_source",
          "stale_branch",
          "stale_pack",
          "superseded",
          "tombstoned",
        ],
      },
      expansion: {
        maxExpandTokens: 4000,
        summaryRoutingModes: [
          "ignore",
          "summary_suffices",
          "expand_to_source",
          "prefer_typed_memory",
        ],
      },
      prefetch: {
        lifecycleStates: [
          "scheduled",
          "materialized",
          "hit",
          "miss",
          "stale",
          "invalidated",
          "dropped",
        ],
        keyDimensions: [
          "queryDigest",
          "activePackVersion",
          "budgetClass",
          "summaryRoutingMode",
          "kind",
        ],
      },
      budget: {
        controls: {
          freshTailCount: {
            env: "LCM_FRESH_TAIL_COUNT",
            value: 8,
          },
          maxExpandTokens: {
            env: "LCM_MAX_EXPAND_TOKENS",
            value: 4000,
          },
          learnedQueryBudgetFraction: {
            env: "OPENCLAWBRAIN_BUDGET_FRACTION",
            value: 0.3,
          },
        },
      },
    });
    expect((status.contextManagement as { operatorSummary?: string }).operatorSummary).toContain("protected fresh tail");
  });

  it("hydrates the persisted bounded assembly decision across service restart", async () => {
    const brainRoot = makeTempDir("openclawbrain-status-restart-");
    const first = new BrainService({
      deps: createDeps(brainRoot),
    });

    first.noteAssemblyDecision({
      mode: "use_brain",
      conversationId: 42,
      episodeId: "ep_1",
      traceId: "tr_1",
      footer: "[brain] used graph retrieval for this turn.",
      compileElapsedMs: 12,
      compileDeadlineMs: 20,
      compileDeadlineHit: false,
      brainDropReason: "injection_cap_clipped",
      brainDropStage: "injection",
      budgetFraction: 0.3,
      maxContextChars: 240,
      queryBudgetChars: deriveExpectedQueryBudgetChars(4096),
      injectedChars: 180,
      droppedChars: 72,
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: 3,
      fittedNodeCount: 2,
      droppedNodeCount: 1,
      fittingDropReasons: {
        omitted_for_max_context_chars: 1,
      },
    });

    const second = new BrainService({
      deps: createDeps(brainRoot),
    });
    const status = await second.status();
    expect(status.lastAssemblyDecision).toMatchObject({
      mode: "use_brain",
      conversationId: 42,
      traceId: "tr_1",
      budgetFraction: 0.3,
      queryBudgetChars: deriveExpectedQueryBudgetChars(4096),
      fitStrategy: "structured_node_budget",
      fittedNodeCount: 2,
      droppedNodeCount: 1,
      fittingDropReasons: {
        omitted_for_max_context_chars: 1,
      },
    });
  });

  it("reuses a prefetched traversal without duplicating persistence", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-prefetch-workspace-");
    const brainRoot = makeTempDir("openclawbrain-prefetch-state-");
    writeFileSync(
      join(workspaceRoot, "PLAYBOOK.md"),
      "# Pull Requests\n\nUse gh pr create for pull request workflows.\n",
      "utf8",
    );

    const service = new BrainService({
      deps: createDeps(brainRoot),
    });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });

    const store = service as unknown as {
      store: {
        getRecentEpisodes: (limit: number) => Array<{ id: string }>;
        countTraces: () => number;
      };
    };

    expect(store.store.getRecentEpisodes(10)).toHaveLength(0);
    expect(store.store.countTraces()).toBe(0);

    await service.schedulePrefetch({
      conversationId: 7,
      queryText: "How do I open a pull request?",
      budgetChars: 4000,
      summaryRoutingMode: "ignore",
      queryEmbedding: embed("gh pr create pull request"),
    });

    expect(store.store.getRecentEpisodes(10)).toHaveLength(0);
    expect(store.store.countTraces()).toBe(0);

    const prefetchedStatus = await service.status();
    expect(prefetchedStatus.lastPrefetchDecision).toMatchObject({
      state: "materialized",
      summaryRoutingMode: "ignore",
      budgetClass: "large",
    });

    const result = await service.query({
      conversationId: 7,
      queryText: "How do I open a pull request?",
      budgetChars: 4000,
      summaryRoutingMode: "ignore",
    });

    expect(result).not.toBeNull();
    expect(store.store.getRecentEpisodes(10)).toHaveLength(1);
    expect(store.store.countTraces()).toBe(1);

    const status = await service.status();
    expect(status.lastPrefetchDecision).toMatchObject({
      state: "hit",
      summaryRoutingMode: "ignore",
      budgetClass: "large",
      reusedNodeCount: expect.any(Number),
      reusedChars: expect.any(Number),
    });
    expect(status.recentPrefetchSummary).toEqual(expect.objectContaining({
      sampleSize: expect.any(Number),
      hitRate: expect.objectContaining({ count: expect.any(Number) }),
    }));
  });

  it("invalidates a prefetched traversal after pack promotion and falls open safely", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-prefetch-invalidated-workspace-");
    const brainRoot = makeTempDir("openclawbrain-prefetch-invalidated-state-");
    writeFileSync(
      join(workspaceRoot, "PLAYBOOK.md"),
      "# Pull Requests\n\nUse gh pr create for pull request workflows.\n",
      "utf8",
    );

    const service = new BrainService({
      deps: createDeps(brainRoot),
    });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });
    (service as unknown as { embeddingClient: (text: string) => Promise<Float32Array> }).embeddingClient = async (text: string) => embed(text);

    await service.schedulePrefetch({
      conversationId: 7,
      queryText: "How do I open a pull request?",
      budgetChars: 4000,
      summaryRoutingMode: "ignore",
      queryEmbedding: embed("gh pr create pull request"),
    });

    await service.teach({
      instruction: "Use gh pr create for pull request workflows and prefer the latest canonical command form.",
      conversationId: 7,
      kind: "correction",
    });

    const result = await service.query({
      conversationId: 7,
      queryText: "How do I open a pull request?",
      budgetChars: 4000,
      summaryRoutingMode: "ignore",
    });

    expect(result).not.toBeNull();

    const status = await service.status();
    const lastPrefetchDecision = status.lastPrefetchDecision as { invalidatedReason?: string | null } | null;
    expect(status.lastPrefetchDecision).toEqual(expect.objectContaining({
      state: expect.stringMatching(/^(stale|invalidated)$/),
      summaryRoutingMode: "ignore",
    }));
    expect(lastPrefetchDecision?.invalidatedReason ?? null).toMatch(/pack_version_changed|prefetch_key_mismatch|budget_class_changed|summary_routing_changed/);
  });

  it("invalidates prefetched traversals explicitly when budget class or summary routing changes", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-prefetch-route-change-workspace-");
    const brainRoot = makeTempDir("openclawbrain-prefetch-route-change-state-");
    writeFileSync(
      join(workspaceRoot, "PLAYBOOK.md"),
      "# Pull Requests\n\nUse gh pr create for pull request workflows.\n",
      "utf8",
    );

    const service = new BrainService({
      deps: createDeps(brainRoot),
    });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });

    await service.schedulePrefetch({
      conversationId: 7,
      queryText: "How do I open a pull request?",
      budgetChars: 4000,
      summaryRoutingMode: "ignore",
      queryEmbedding: embed("gh pr create pull request"),
    });

    await service.query({
      conversationId: 7,
      queryText: "How do I open a pull request?",
      budgetChars: 400,
      summaryRoutingMode: "ignore",
      queryEmbedding: embed("gh pr create pull request"),
    });

    expect(await service.status()).toEqual(expect.objectContaining({
      lastPrefetchDecision: expect.objectContaining({
        state: expect.stringMatching(/^(stale|invalidated)$/),
        invalidatedReason: "budget_class_changed",
        summaryRoutingMode: "ignore",
      }),
    }));

    await service.schedulePrefetch({
      conversationId: 7,
      queryText: "How do I open a pull request?",
      budgetChars: 400,
      summaryRoutingMode: "ignore",
      queryEmbedding: embed("gh pr create pull request"),
    });

    await service.query({
      conversationId: 7,
      queryText: "How do I open a pull request?",
      budgetChars: 400,
      summaryRoutingMode: "prefer_typed_memory",
      queryEmbedding: embed("gh pr create pull request"),
    });

    expect(await service.status()).toEqual(expect.objectContaining({
      lastPrefetchDecision: expect.objectContaining({
        state: expect.stringMatching(/^(stale|invalidated)$/),
        invalidatedReason: "summary_routing_changed",
        summaryRoutingMode: "ignore",
      }),
    }));
  });

  it("persists post-injection clip attribution through trace and observation metadata", async () => {
    const queryBudgetChars = deriveExpectedQueryBudgetChars(4096);
    const workspaceRoot = makeTempDir("openclawbrain-attribution-workspace-");
    const brainRoot = makeTempDir("openclawbrain-attribution-state-");
    writeFileSync(
      join(workspaceRoot, "PLAYBOOK.md"),
      "# Pull Requests\n\nUse gh pr create for pull request workflows and include the exact flags required by the repo automation so the compact route summary stays long enough to exercise clipping.\n",
      "utf8",
    );

    const service = new BrainService({
      deps: createDeps(brainRoot),
    });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });
    (service as unknown as { embeddingClient: (text: string) => Promise<Float32Array> }).embeddingClient = async (text: string) => embed(text);

    const extension = new BrainAssemblerExtension(service);
    const assembled = {
      messages: [{ role: "user", content: "live tail" }],
      estimatedTokens: 2,
      stats: {
        rawMessageCount: 1,
        summaryCount: 0,
        totalContextItems: 1,
      },
    };

    const result = await extension.augmentAssembly({
      conversationId: 42,
      tokenBudget: 4096,
      maxContextChars: 120,
      assembled,
      liveMessages: [{ role: "user", content: "How do I open a pull request?" }],
    });

    expect(result.brainDecision).toEqual(expect.objectContaining({
      mode: "use_brain",
      episodeId: expect.any(String),
      traceId: expect.any(String),
      compileElapsedMs: expect.any(Number),
      brainDropReason: "injection_cap_clipped",
      brainDropStage: "injection",
      budgetFraction: 0.3,
      maxContextChars: 120,
      queryBudgetChars,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: expect.any(Number),
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
    }));
    if ((result.brainDecision?.droppedNodeCount ?? 0) > 0) {
      expect(result.brainDecision?.fittingDropReasons).toEqual(expect.objectContaining({
        omitted_for_max_context_chars: expect.any(Number),
      }));
    } else {
      expect(result.brainDecision?.fittingDropReasons ?? null).toBeNull();
    }

    const trace = await service.getTrace(String(result.brainDecision?.traceId ?? ""));
    expect(trace?.routeTrace?.selectionMetadata).toMatchObject({
      budgetChars: queryBudgetChars,
      compileElapsedMs: expect.any(Number),
      brainDropReason: "injection_cap_clipped",
      brainDropStage: "injection",
      budgetFraction: 0.3,
      maxContextChars: 120,
      queryBudgetChars,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: expect.any(Number),
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
      compileReportSummary: expect.stringContaining("mode=use_brain"),
      compileReport: expect.objectContaining({
        schemaVersion: 1,
        summary: expect.stringContaining("mode=use_brain"),
        decision: expect.objectContaining({
          mode: "use_brain",
        }),
      }),
    });
    if ((trace?.routeTrace?.selectionMetadata.droppedNodeCount ?? 0) > 0) {
      expect(trace?.routeTrace?.selectionMetadata.fittingDropReasons).toEqual(expect.objectContaining({
        omitted_for_max_context_chars: expect.any(Number),
      }));
    } else {
      expect(trace?.routeTrace?.selectionMetadata.fittingDropReasons ?? null).toBeNull();
    }
    expect((trace?.routeTrace?.selectionMetadata.injectedChars ?? 0)).toBeLessThanOrEqual(120);
    expect(trace?.routeTrace?.selectionMetadata.droppedChars).toBeGreaterThan(0);

    await service.recordTurnObservation({
      episodeId: String(result.brainDecision?.episodeId ?? ""),
      assistantResponse: "Use `gh pr create` to open the pull request.",
      toolResults: [],
    });

    const observation = (
      service as unknown as {
        store: { getObservationForEpisode: (episodeId: string) => Record<string, unknown> | null };
      }
    ).store.getObservationForEpisode(String(result.brainDecision?.episodeId ?? ""));
    const observationSelectionMetadata = (observation as any)?.routeMetadata?.selectionMetadata ?? null;

    expect(observation).toMatchObject({
      traceId: expect.stringMatching(/^rt_[a-f0-9]{16}$/),
      routeMetadata: {
        selectionMetadata: {
          budgetChars: queryBudgetChars,
          compileElapsedMs: expect.any(Number),
          brainDropReason: "injection_cap_clipped",
          brainDropStage: "injection",
          budgetFraction: 0.3,
          maxContextChars: 120,
          injectedChars: expect.any(Number),
          droppedChars: expect.any(Number),
          contextClipped: true,
          fitStrategy: "structured_node_budget",
          retrievedNodeCount: expect.any(Number),
          fittedNodeCount: expect.any(Number),
          droppedNodeCount: expect.any(Number),
          compileReportSummary: expect.stringContaining("mode=use_brain"),
        },
      },
    });
    if ((observationSelectionMetadata?.droppedNodeCount ?? 0) > 0) {
      expect(observationSelectionMetadata?.fittingDropReasons).toEqual(expect.objectContaining({
        omitted_for_max_context_chars: expect.any(Number),
      }));
    } else {
      expect(observationSelectionMetadata?.fittingDropReasons ?? null).toBeNull();
    }

    const status = await service.status();
    const lastTraceSelectionMetadata = status.lastTraceSelectionMetadata as {
      droppedNodeCount?: number | null;
      fittingDropReasons?: unknown;
    } | null;
    expect(status.lastTraceSelectionMetadata).toEqual(expect.objectContaining({
      budgetChars: queryBudgetChars,
      compileElapsedMs: expect.any(Number),
      brainDropReason: "injection_cap_clipped",
      brainDropStage: "injection",
      budgetFraction: 0.3,
      maxContextChars: 120,
      queryBudgetChars,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: expect.any(Number),
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
    }));
    if ((lastTraceSelectionMetadata?.droppedNodeCount ?? 0) > 0) {
      expect(lastTraceSelectionMetadata?.fittingDropReasons).toEqual(expect.objectContaining({
        omitted_for_max_context_chars: expect.any(Number),
      }));
    } else {
      expect(lastTraceSelectionMetadata?.fittingDropReasons ?? null).toBeNull();
    }
  });

  it("persists partial-serve interruption truth through trace, observation, and status", async () => {
    const queryBudgetChars = deriveExpectedQueryBudgetChars(4096);
    const workspaceRoot = makeTempDir("openclawbrain-partial-serve-workspace-");
    const brainRoot = makeTempDir("openclawbrain-partial-serve-state-");
    writeFileSync(
      join(workspaceRoot, "PLAYBOOK.md"),
      "# Pull Requests\n\nUse gh pr create for pull request workflows and keep the response auditable when the compile path needs to fall back to a committed prefix.\n",
      "utf8",
    );

    const service = new BrainService({
      deps: createDeps(brainRoot, {
        maxCompileMs: 15,
      }),
    });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });
    (service as unknown as { embeddingClient: (text: string) => Promise<Float32Array> }).embeddingClient = async (text: string) => embed(text);

    const extension = new BrainAssemblerExtension(service);
    const assembled = {
      messages: [{ role: "user", content: "live tail" }],
      estimatedTokens: 2,
      stats: {
        rawMessageCount: 1,
        summaryCount: 0,
        totalContextItems: 1,
      },
    };

    let mockedNow = 100;
    vi.spyOn(Date, "now").mockImplementation(() => mockedNow++);

    const result = await extension.augmentAssembly({
      conversationId: 42,
      tokenBudget: 4096,
      maxContextChars: 240,
      assembled,
      liveMessages: [{ role: "user", content: "How do I open a pull request?" }],
    });
    expect(result.messages[0]).toMatchObject({
      role: "user",
      content: expect.stringContaining("[brain partial]"),
    });
    expect(result.messages.slice(1)).toEqual(assembled.messages);
    expect(result.brainDecision).toEqual(expect.objectContaining({
      mode: "partial_deadline_after_query",
      episodeId: expect.any(String),
      traceId: expect.any(String),
      compileElapsedMs: expect.any(Number),
      compileDeadlineMs: 15,
      compileDeadlineHit: true,
      brainDropReason: "deadline_after_query",
      brainDropStage: "query",
      queryInterrupted: false,
      interruptionStage: "query",
      interruptionReason: "soft_compile_deadline",
      servedPartial: true,
      maxContextChars: 240,
      queryBudgetChars,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: expect.any(Boolean),
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: expect.any(Number),
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
      chosenStopCount: expect.any(Number),
      forcedStopCount: expect.any(Number),
      droppedProposalCount: expect.any(Number),
      droppedProposalReasons: null,
    }));
    if ((result.brainDecision?.droppedNodeCount ?? 0) > 0) {
      expect(result.brainDecision?.fittingDropReasons).toEqual(expect.objectContaining({
        omitted_for_partial_serve: expect.any(Number),
      }));
    } else {
      expect(result.brainDecision?.fittingDropReasons ?? null).toBeNull();
    }

    const trace = await service.getTrace(String(result.brainDecision?.traceId ?? ""));
    expect(trace?.routeTrace?.selectionMetadata).toMatchObject({
      budgetChars: queryBudgetChars,
      compileElapsedMs: expect.any(Number),
      compileDeadlineMs: 15,
      compileDeadlineHit: true,
      brainDropReason: "deadline_after_query",
      brainDropStage: "query",
      queryInterrupted: false,
      interruptionStage: "query",
      interruptionReason: "soft_compile_deadline",
      servedPartial: true,
      maxContextChars: 240,
      queryBudgetChars,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: expect.any(Boolean),
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: expect.any(Number),
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
      chosenStopCount: expect.any(Number),
      forcedStopCount: expect.any(Number),
    });
    expect((trace?.routeTrace?.selectionMetadata.injectedChars ?? 0)).toBeGreaterThan(0);
    expect((trace?.routeTrace?.selectionMetadata?.forcedStopCount ?? 0)).toBeGreaterThan(0);
    if ((trace?.routeTrace?.selectionMetadata.droppedNodeCount ?? 0) > 0) {
      expect(trace?.routeTrace?.selectionMetadata.fittingDropReasons).toEqual(expect.objectContaining({
        omitted_for_partial_serve: expect.any(Number),
      }));
    } else {
      expect(trace?.routeTrace?.selectionMetadata.fittingDropReasons ?? null).toBeNull();
    }

    await service.recordTurnObservation({
      episodeId: String(result.brainDecision?.episodeId ?? ""),
      assistantResponse: "Use `gh pr create` to open the pull request.",
      toolResults: [],
    });

    const observation = (
      service as unknown as {
        store: { getObservationForEpisode: (episodeId: string) => Record<string, unknown> | null };
      }
    ).store.getObservationForEpisode(String(result.brainDecision?.episodeId ?? ""));

    expect(observation).toMatchObject({
      traceId: expect.stringMatching(/^rt_[a-f0-9]{16}$/),
      routeMetadata: {
        selectionMetadata: {
          budgetChars: queryBudgetChars,
          brainDropReason: "deadline_after_query",
          brainDropStage: "query",
          queryInterrupted: false,
          interruptionStage: "query",
          interruptionReason: "soft_compile_deadline",
          servedPartial: true,
          maxContextChars: 240,
          queryBudgetChars,
          injectedChars: expect.any(Number),
          droppedChars: expect.any(Number),
          contextClipped: expect.any(Boolean),
          fitStrategy: "structured_node_budget",
          retrievedNodeCount: expect.any(Number),
          fittedNodeCount: expect.any(Number),
          droppedNodeCount: expect.any(Number),
          chosenStopCount: expect.any(Number),
          forcedStopCount: expect.any(Number),
        },
      },
    });

    const status = await service.status();
    expect(status.lastAssemblyDecision).toEqual(expect.objectContaining({
      mode: "partial_deadline_after_query",
      traceId: result.brainDecision?.traceId,
      brainDropReason: "deadline_after_query",
      brainDropStage: "query",
      queryInterrupted: false,
      interruptionStage: "query",
      interruptionReason: "soft_compile_deadline",
      servedPartial: true,
      maxContextChars: 240,
      queryBudgetChars,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: expect.any(Boolean),
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: expect.any(Number),
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
    }));
    expect(status.lastTraceSelectionMetadata).toEqual(expect.objectContaining({
      budgetChars: queryBudgetChars,
      brainDropReason: "deadline_after_query",
      brainDropStage: "query",
      queryInterrupted: false,
      interruptionStage: "query",
      interruptionReason: "soft_compile_deadline",
      servedPartial: true,
      maxContextChars: 240,
      queryBudgetChars,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: expect.any(Boolean),
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: expect.any(Number),
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
      chosenStopCount: expect.any(Number),
      forcedStopCount: expect.any(Number),
    }));
  });

  it("records turn observations, attaches next-user follow-up, and surfaces teacher supervision in status", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-workspace-");
    const brainRoot = makeTempDir("openclawbrain-state-");
    writeFileSync(
      join(workspaceRoot, "PLAYBOOK.md"),
      "# Pull Requests\n\nUse gh pr create for pull request workflows.\n",
      "utf8",
    );

    const deps = createDeps(brainRoot, {
      teacherEnabled: true,
        persistRawSurfaces: false,
      teacherProvider: "openai",
      teacherModel: "gpt-5.4-mini",
    });
    deps.complete = vi.fn(async () => ({
      content: [{
        type: "text",
        text: "{\"retrieval_relevance\":0.9,\"agent_usage\":0.8,\"outcome_support\":0.85,\"final_score\":0.82,\"confidence\":0.67,\"reason\":\"selected context matched the query and the follow-up confirmed the outcome\"}",
      }],
    }));

    const service = new BrainService({
      deps,
    });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });

    const result = await service.query({
      conversationId: 42,
      agentIdentity: { agentId: "main", lane: "main" },
      queryText: "how do I open a pull request?",
      budgetChars: 4000,
      queryEmbedding: embed("gh pr create pull request"),
    });
    expect(result).not.toBeNull();

    await service.recordTurnObservation({
      episodeId: result?.episode.id,
      assistantResponse: "Use `gh pr create` to open the pull request.",
      toolResults: [],
    });
    await service.observeUserTurn({
      conversationId: 42,
      messageId: 99,
      episodeId: result?.episode.id,
      userText: "Perfect, that's exactly right!",
      recentMessages: [
        { role: "assistant", content: "Use `gh pr create` to open the pull request." },
        { role: "user", content: "how do I open a pull request?" },
      ],
      recentSummaries: [],
    });
    await ((service as unknown as { worker: { tick: () => Promise<void> } | null }).worker?.tick() ?? Promise.resolve());

    const trace = await service.getTrace(result?.trace.id);
    expect(trace?.supervision).toMatchObject([
      {
        traceId: result?.trace.id,
        episodeId: result?.episode.id,
        source: "teacher",
        kind: "teacher_review",
        value: 0.82,
        resolution: "promoted_to_label",
        labelId: expect.stringMatching(/^bl_/),
        evidenceId: expect.stringMatching(/^be_/),
        metadata: expect.objectContaining({
          observationId: expect.stringMatching(/^bo_/),
          resolvedTraceId: result?.trace.id,
          phase1Score: 0.9,
          phase2Score: 0.85,
          agentUsage: 0.8,
        }),
      },
    ]);

    const status = await service.status();
    expect(status.routeTraceCount).toBe(1);
    expect(status.supervisionCount).toBe(1);
    expect(status.pendingObservations).toBe(0);
    expect(status.routeOutcomeTruth).toMatchObject({
      coverage: {
        routeTraceCount: 1,
        observationCount: 1,
        followUpCount: 1,
        confirmationCount: 1,
      },
      activation: {
        totalServedCount: 1,
        learnedNontrivialCount: 1,
        learnedNontrivialOutcomeCount: 1,
        learnedNontrivialResolvedCount: 1,
        learnedNontrivialCorrectionLikeCount: 0,
      },
      outcomes: {
        resolved: 1,
        reask: 0,
      },
      resolutions: {
        completed: 1,
        unresolved: 0,
        totalRetryCount: 0,
      },
      metrics: {
        activationPrecision: {
          value: 1,
          numerator: 1,
          denominator: 1,
        },
        retryRate: {
          value: 0,
          numerator: 0,
          denominator: 1,
        },
      },
      latest: {
        routeServed: expect.objectContaining({
          contract: "ocb.route_served.v1",
          episode_id: result?.episode.id,
        }),
        turnOutcome: expect.objectContaining({
          contract: "ocb.turn_outcome.v1",
          outcome_class: "resolved",
        }),
        episodeResolution: expect.objectContaining({
          contract: "ocb.episode_resolution.v1",
          resolution_class: "completed",
          resolved: true,
        }),
      },
    });
    expect(status.attributionTruth).toMatchObject({
      contract: "openclawbrain_attribution_truth.v1",
      visible: true,
      primaryState: "ambiguous",
      activeStates: ["ambiguous"],
      counts: {
        observationCount: 1,
        evaluatedCount: 1,
        completedWithoutEvaluationCount: 0,
        matchedCount: 0,
        ambiguousCount: 1,
        unmatchedCount: 0,
        pendingCount: 0,
        pendingFollowupCount: 0,
        pendingTeacherCount: 0,
        readyCount: 0,
        delayedCount: 0,
        budgetDeferredCount: 0,
        sparseReadyCount: 0,
        richReadyCount: 0,
      },
      latest: {
        ambiguous: {
          observationId: expect.stringMatching(/^bo_/),
          episodeId: result?.episode.id,
          traceId: expect.stringMatching(/^rt_[a-f0-9]{16}$/),
          attributionQuality: "fallback",
          feedbackRichness: "followup_only",
        },
      },
    });
    expect(status.observationAttribution).toMatchObject({
      totalObservationCount: 1,
      completedObservationCount: 1,
      completedWithoutEvaluationCount: 0,
      teacherEvaluationCount: 1,
      nonExactCount: 1,
      bindingModes: {
        exact_decision_id: 0,
        exact_selection_digest: 0,
        turn_compile_event_id: 0,
        trace_id: 1,
        legacy_heuristic: 0,
        unbound: 0,
      },
      attributionQuality: {
        exact: 0,
        fallback: 1,
        unbound: 0,
      },
      latestNonExact: {
        observationId: expect.stringMatching(/^bo_/),
        episodeId: result?.episode.id,
        traceId: expect.stringMatching(/^rt_[a-f0-9]{16}$/),
        bindingMode: "trace_id",
        attributionQuality: "fallback",
        feedbackRichness: "followup_only",
        confidence: 0.67,
      },
      latestAmbiguous: {
        observationId: expect.stringMatching(/^bo_/),
        episodeId: result?.episode.id,
        traceId: expect.stringMatching(/^rt_[a-f0-9]{16}$/),
        bindingMode: "trace_id",
        attributionQuality: "fallback",
        feedbackRichness: "followup_only",
        confidence: 0.67,
      },
      latestUnmatched: null,
      detail: "teacher attribution counts: evaluated=1/1, exact=0, fallback=1, unbound=0, pending_followup=0, pending_teacher=0, completed_without_evaluation=0",
    });
    expect(status.teacherTruth).toMatchObject({
      queue: {
        pendingCount: 0,
        readyCount: 0,
        delayedCount: 0,
        budgetDeferredCount: 0,
        sparseReadyCount: 0,
      },
      lastEvaluationCycle: {
        evaluatedCount: 1,
        skippedCount: 0,
        exactAttributionCount: 0,
        fallbackAttributionCount: 1,
        unboundAttributionCount: 0,
        decisions: [
          expect.objectContaining({
            episodeId: result?.episode.id,
            decision: "evaluated",
            feedbackRichness: "followup_only",
            attributionQuality: "fallback",
          }),
        ],
      },
      lastUpdateCycle: {
        appliedEpisodeCount: 0,
        skippedEpisodeCount: 1,
        decisions: [
          expect.objectContaining({
            episodeId: result?.episode.id,
            status: "skipped",
            reason: "reward matched the running baseline, so no policy delta was emitted",
            routeUpdateCount: 0,
            attributionQuality: "fallback",
            feedbackRichness: "followup_only",
          }),
        ],
      },
    });
    expect(status.contextFeedback).toMatchObject({
      verdictCounts: {
        helpful: 1,
        irrelevant: 0,
        harmful: 0,
      },
      coverage: {
        routeTraceCount: 1,
        identifiedRouteTraceCount: 1,
        unidentifiedRouteTraceCount: 0,
        agentIdentityCoverage: 1,
        observationCount: 1,
        completedObservationCount: 1,
        supervisedTraceCount: 1,
        unsupervisedTraceCount: 0,
        observationCoverage: 1,
        supervisionCoverage: 1,
        pendingFollowupCount: 0,
        pendingTeacherCount: 0,
      },
      latest: {
        traceId: result?.trace.id,
        episodeId: result?.episode.id,
        observationId: expect.stringMatching(/^bo_/),
        agentIdentity: { agentId: "main", lane: "main" },
        source: "teacher",
        verdict: "helpful",
        score: 0.82,
        confidence: 0.67,
        bindingMode: "trace_id",
      },
      focus: {
        action: "monitor",
        detail: "feedback loop is closed on every traced route",
      },
      agents: [
        expect.objectContaining({
          agentIdentity: { agentId: "main", lane: "main" },
          routeTraceCount: 1,
          supervisedTraceCount: 1,
          unsupervisedTraceCount: 0,
          supervisionCoverage: 1,
          verdictCounts: {
            helpful: 1,
            irrelevant: 0,
            harmful: 0,
          },
        }),
      ],
    });
    expect(status.contextUsefulness).toMatchObject({
      verdictCounts: {
        helpful: 0,
        irrelevant: 1,
        harmful: 0,
      },
      coverage: {
        observationCount: 1,
        readyObservationCount: 1,
        scoredObservationCount: 1,
        completedObservationCount: 1,
      },
      latest: {
        observationId: expect.stringMatching(/^bo_/),
        episodeId: result?.episode.id,
        verdict: "irrelevant",
      },
    });
    expect(status.learningHealth).toMatchObject({
      status: "learning_backed_by_feedback",
      signals: {
        routeTraceCount: 1,
        supervisedTraceCount: 1,
        helpfulCount: 1,
        harmfulCount: 0,
      },
    });
    expect((status.contextFeedback as { detail?: string }).detail).toContain("1 helpful");
    expect((status.promotionStory as {
      integrations?: {
        structuredVerdict?: {
          verdictCounts?: { helpful?: number; irrelevant?: number; harmful?: number };
          coverage?: { supervisionCoverage?: number };
        } | null;
      };
    }).integrations?.structuredVerdict).toMatchObject({
      verdictCounts: {
        helpful: 1,
        irrelevant: 0,
        harmful: 0,
      },
      coverage: {
        supervisionCoverage: 1,
      },
    });
  });

  it("surfaces delayed attribution truth before follow-up or the teacher delay window completes", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-delayed-attribution-workspace-");
    const brainRoot = makeTempDir("openclawbrain-delayed-attribution-state-");
    writeFileSync(
      join(workspaceRoot, "PLAYBOOK.md"),
      "# Pull Requests\n\nUse gh pr create for pull request workflows.\n",
      "utf8",
    );

    const service = new BrainService({
      deps: createDeps(brainRoot, {
        teacherEnabled: true,
      }),
    });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });

    const result = await service.query({
      conversationId: 7,
      agentIdentity: { agentId: "main", lane: "main" },
      queryText: "how do I open a pull request?",
      budgetChars: 4000,
      queryEmbedding: embed("gh pr create pull request"),
    });
    expect(result).not.toBeNull();

    await service.recordTurnObservation({
      episodeId: result?.episode.id,
      assistantResponse: "Use `gh pr create` to open the pull request.",
      toolResults: [],
    });

    const status = await service.status();
    expect(status.pendingObservations).toBe(1);
    expect(status.attributionTruth).toMatchObject({
      contract: "openclawbrain_attribution_truth.v1",
      visible: true,
      primaryState: "delayed",
      activeStates: ["delayed"],
      counts: {
        observationCount: 1,
        evaluatedCount: 0,
        completedWithoutEvaluationCount: 0,
        matchedCount: 0,
        ambiguousCount: 0,
        unmatchedCount: 0,
        pendingCount: 1,
        pendingFollowupCount: 1,
        pendingTeacherCount: 0,
        readyCount: 0,
        delayedCount: 1,
        budgetDeferredCount: 0,
        sparseReadyCount: 0,
        richReadyCount: 0,
      },
      latest: {
        followupPending: {
          observationId: expect.stringMatching(/^bo_/),
          episodeId: result?.episode.id,
          traceId: expect.stringMatching(/^rt_[a-f0-9]{16}$/),
          status: "pending_followup",
        },
        delayed: {
          observationId: expect.stringMatching(/^bo_/),
          episodeId: result?.episode.id,
          traceId: expect.stringMatching(/^rt_[a-f0-9]{16}$/),
          status: "pending_followup",
        },
      },
    });
  });

  it("replays pending observations after a process restart", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-restart-workspace-");
    const brainRoot = makeTempDir("openclawbrain-restart-state-");
    writeFileSync(
      join(workspaceRoot, "PLAYBOOK.md"),
      "# Pull Requests\n\nUse gh pr create for pull request workflows.\n",
      "utf8",
    );

    const deps = createDeps(brainRoot, {
      teacherEnabled: true,
        persistRawSurfaces: false,
      teacherProvider: "openai",
      teacherModel: "gpt-5.4-mini",
    });
    deps.complete = vi.fn(async () => ({
      content: [{
        type: "text",
        text: "{\"retrieval_relevance\":0.88,\"agent_usage\":0.72,\"outcome_support\":0.8,\"final_score\":0.79,\"confidence\":0.61,\"reason\":\"persisted observation survived restart and still looked good\"}",
      }],
    }));

    const first = new BrainService({ deps });
    await first.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });

    const result = await first.query({
      conversationId: 77,
      queryText: "how do I open a pull request?",
      budgetChars: 4000,
      queryEmbedding: embed("gh pr create pull request"),
    });
    expect(result).not.toBeNull();

    await first.recordTurnObservation({
      episodeId: result?.episode.id,
      assistantResponse: "Use `gh pr create` to open the pull request.",
      toolResults: [],
    });
    await first.observeUserTurn({
      conversationId: 77,
      messageId: 200,
      episodeId: result?.episode.id,
      userText: "That worked.",
      recentMessages: [
        { role: "assistant", content: "Use `gh pr create` to open the pull request." },
        { role: "user", content: "how do I open a pull request?" },
      ],
      recentSummaries: [],
    });

    const restartDeps = createDeps(brainRoot, {
      teacherEnabled: true,
        persistRawSurfaces: false,
      teacherProvider: "openai",
      teacherModel: "gpt-5.4-mini",
    });
    restartDeps.complete = deps.complete;
    const second = new BrainService({ deps: restartDeps });

    await ((second as unknown as { worker: { tick: () => Promise<void> } | null }).worker?.tick() ?? Promise.resolve());

    const trace = await second.getTrace(result?.trace.id);
    expect(trace?.supervision).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          source: "teacher",
          value: 0.79,
        }),
      ]),
    );
    expect((await second.status()).pendingObservations).toBe(0);
  });

  it("surfaces the latest candidate-pack PG update artifact in runtime status", async () => {
    const brainRoot = makeTempDir("openclawbrain-state-");
    const service = new BrainService({
      deps: createDeps(brainRoot),
    });
    const store = (service as unknown as {
      store: {
        setTrainingStateJson: (key: string, value: unknown | null) => void;
        setTrainingState: (key: string, value: string | number) => void;
      };
    }).store;

    store.setTrainingStateJson("last_pg_candidate_update_json", {
      version: 2,
      updateCount: 2,
      candidatePackVersion: 9,
      currentPackVersion: 3,
      generatedAt: 123456789,
      episodeIds: ["ep_1", "ep_2"],
      traceIds: ["bt_1", "bt_2"],
      observationIds: ["bo_2"],
      supervisionIds: ["ts_1", "ts_2"],
      teacherTraceIds: ["bt_2"],
      rewardSources: { human: 1, scanner: 0, teacher: 1, self: 0 },
      episodeCount: 2,
      traceCount: 2,
      observationCount: 1,
      supervisionCount: 2,
      teacherLabelCount: 1,
      routeUpdateCount: 3,
      seedUpdateCount: 2,
      stopLocalUpdateCount: 0,
      edgeUpdateCount: 1,
      baselineBefore: 0,
      baselineAfter: 0.12,
      episodeUpdates: [
        {
          episodeId: "ep_2",
          observationIds: ["bo_2"],
          traceIds: ["bt_2"],
          supervisionIds: ["ts_2"],
          reward: 0.79,
          rewardSource: "teacher",
          attributionQuality: "exact",
          feedbackRichness: "followup_only",
          updateReason: "teacher teacher_review 0.79 (confidence 0.80, exact_decision_id attribution, followup only) updated 1 route weight(s)",
          baselineBefore: 0.04,
          baselineAfter: 0.12,
          advantage: 0.75,
          routeUpdateCount: 1,
          seedUpdateCount: 1,
          stopLocalUpdateCount: 0,
          edgeUpdateCount: 0,
          supervision: [
            {
              supervisionId: "ts_2",
              traceId: "bt_2",
              source: "teacher",
              kind: "teacher_review",
              value: 0.79,
              confidence: 0.8,
              reason: "teacher verified the route",
              labelId: "bl_2",
              evidenceId: "be_2",
              observationId: "bo_2",
              teacherTraceId: "bt_2",
              serveDecisionRecordId: "decision-2",
              selectionDigest: "selection-2",
              turnCompileEventId: "compile-2",
              activePackGraphChecksum: "graph-2",
              bindingMode: "exact_decision_id",
              attributionQuality: "exact",
              feedbackRichness: "followup_only",
              traceRequestDigest: "digest-2",
              traceSelectedNodeIds: ["node_2"],
              traceSelectedPathNodeIds: ["node_2"],
            },
          ],
          routeUpdates: [
            {
              updateKey: "seed→node_2",
              kind: "seed",
              sourceNodeId: "__START__",
              targetNodeId: "node_2",
              delta: 0.01,
              previousWeight: 0.1,
              nextWeight: 0.11,
              contributionCount: 1,
              contributions: [
                {
                  updateKey: "seed→node_2",
                  kind: "seed",
                  sourceNodeId: "__START__",
                  targetNodeId: "node_2",
                  expansionIndex: 0,
                  selectionIndex: 0,
                  chosenActionProbability: 0.6,
                  delta: 0.01,
                  stopTruth: null,
                  stopReason: null,
                },
              ],
            },
          ],
        },
      ],
    });
    store.setTrainingState("last_pg_candidate_pack_version", 9);

    const status = await service.status();
    expect(status.lastPgCandidatePackVersion).toBe(9);
    expect(status.lastPgCandidateUpdate).toMatchObject({
      version: 2,
      updateCount: 2,
      candidatePackVersion: 9,
      teacherLabelCount: 1,
      observationIds: ["bo_2"],
      traceIds: ["bt_1", "bt_2"],
      episodeUpdates: [
        expect.objectContaining({
          episodeId: "ep_2",
          attributionQuality: "exact",
          feedbackRichness: "followup_only",
          observationIds: ["bo_2"],
          updateReason: "teacher teacher_review 0.79 (confidence 0.80, exact_decision_id attribution, followup only) updated 1 route weight(s)",
          supervision: [
            expect.objectContaining({
              observationId: "bo_2",
              selectionDigest: "selection-2",
              attributionQuality: "exact",
              feedbackRichness: "followup_only",
            }),
          ],
        }),
      ],
    });
  });

  it("fails open when teacher resolution has no model and reports that truth in status", async () => {
    const brainRoot = makeTempDir("openclawbrain-state-");
    const deps = createDeps(brainRoot, {
      teacherEnabled: true,
        persistRawSurfaces: false,
      teacherProvider: "",
      teacherModel: "",
    });
    deps.resolveModel = vi.fn(() => {
      throw new Error("No model configured for LCM summarization.");
    });

    const service = new BrainService({ deps });
    const status = await service.status();

    expect(status.teacherEnabled).toBe(true);
    expect(status.teacherConfigured).toBe(false);
    expect(status.teacherConfigError).toBe("No model configured for LCM summarization.");
    expect(deps.log.warn).toHaveBeenCalledWith(
      "[brain] Teacher disabled: No model configured for LCM summarization.",
    );
  });

  it("surfaces structured promotion verdicts and recent bundle records in runtime status", async () => {
    const brainRoot = makeTempDir("openclawbrain-state-");
    const service = new BrainService({
      deps: createDeps(brainRoot),
    });
    const store = (service as unknown as {
      store: {
        setTrainingStateJson: (key: string, value: unknown | null) => void;
        setTrainingState: (key: string, value: string) => void;
        insertMutationBundle: (params: {
          id: string;
          mutationIds: string[];
          bundleSize: number;
          status: "promoted";
          expectedGain: number;
          createdAt: number;
          baseScore: number;
          candidateScore: number;
          rejectionReason?: string | null;
          verdict: Record<string, unknown>;
          resolvedAt: number;
        }) => void;
      };
    }).store;

    store.setTrainingState("last_promotion_reason", "candidate graph promoted after bundle evaluation");
    store.setTrainingState("last_replay_failure_reason", "");
    store.setTrainingStateJson("last_promotion_verdict_json", {
      mode: "bundle",
      status: "promoted",
      promotedBundleCount: 1,
    });
    store.setTrainingStateJson("last_replay_gate_verdict_json", {
      passed: true,
      reason: { code: "all_gates_passed", summary: "all gates passed", details: {} },
    });
    store.insertMutationBundle({
      id: "mb_status",
      mutationIds: ["mp_1", "mp_2"],
      bundleSize: 2,
      status: "promoted",
      expectedGain: 0.4,
      createdAt: Date.now(),
      baseScore: 0.2,
      candidateScore: 0.5,
      verdict: {
        bundleId: "mb_status",
        mutationIds: ["mp_1", "mp_2"],
        bundleSize: 2,
        status: "promoted",
        baseScore: 0.2,
        candidateScore: 0.5,
        expectedGain: 0.4,
        evaluatedEpisodeCount: 3,
        qualifyingEpisodeCount: 2,
        improvementRatio: 2.5,
        reason: { code: "promoted", summary: "candidate improved replay score", details: {} },
        createdAt: Date.now(),
        resolvedAt: Date.now(),
      },
      resolvedAt: Date.now(),
    });

    const status = await service.status();

    expect(status.lastPromotionVerdict).toMatchObject({
      mode: "bundle",
      status: "promoted",
      promotedBundleCount: 1,
    });
    expect(status.lastReplayGateVerdict).toMatchObject({
      passed: true,
      reason: { code: "all_gates_passed" },
    });
    expect(status.recentMutationBundles).toMatchObject([
      {
        id: "mb_status",
        status: "promoted",
        verdict: {
          reason: { code: "promoted" },
        },
      },
    ]);
    expect(status.learningHealth).toMatchObject({
      status: "changing_without_feedback",
      signals: {
        routeTraceCount: 0,
        supervisedTraceCount: 0,
        recentBundleCount: 1,
        promotedBundleCount: 1,
      },
    });
  });

  it("teaches a correction against the active conversation, labels only matching episodes, and retrieves it immediately", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-workspace-");
    const brainRoot = makeTempDir("openclawbrain-state-");
    writeFileSync(
      join(workspaceRoot, "DEPLOY.md"),
      "# Deploy\n\nCheck CI logs before retrying a deployment.\n",
      "utf8",
    );

    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ data: [{ embedding: Array.from(embed("deployment ci")) }] }),
    }));
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const service = new BrainService({
      deps: createDeps(brainRoot),
    });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });

    const targetResult = await service.query({
      conversationId: 7,
      agentIdentity: { agentId: "main", lane: "main" },
      queryText: "deployment failed",
      budgetChars: 4000,
      queryEmbedding: embed("deployment ci"),
    });
    await service.query({
      conversationId: 99,
      agentIdentity: { agentId: "reviewer", lane: "subagent" },
      queryText: "deployment failed elsewhere",
      budgetChars: 4000,
      queryEmbedding: embed("deployment ci"),
    });

    const taught = await service.teach({
      instruction: "For deployment errors, inspect CI logs before retrying.",
      conversationId: 7,
      kind: "correction",
    });

    expect(taught.nodeId).toMatch(/^bn_/);
    expect(taught.packVersion).toBeGreaterThanOrEqual(2);

    const status = await service.status();
    expect(status.pendingLabels).toBe(1);
    expect(status.currentPackVersion).toBe(taught.packVersion);
    expect((status.currentPackMetadata as { reason?: string; taughtNodeId?: string } | null)?.reason).toBe("teach");
    expect((status.currentPackMetadata as { reason?: string; taughtNodeId?: string } | null)?.taughtNodeId).toBe(taught.nodeId);
    expect((status.promotionStory as {
      currentPack?: { reason?: string; metadata?: { taughtNodeId?: string } };
      recentPromotions?: Array<{ reason?: string }>;
    }).currentPack?.reason).toBe("teach");
    expect((status.promotionStory as {
      currentPack?: { reason?: string; metadata?: { taughtNodeId?: string } };
      recentPromotions?: Array<{ reason?: string }>;
    }).currentPack?.metadata?.taughtNodeId).toBe(taught.nodeId);

    const privateService = (service as unknown as {
      store: { getPendingLabels: () => Array<{ source: string; value: number }> };
      worker: { tick: () => Promise<void> } | null;
    });
    expect(privateService.store.getPendingLabels()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          source: "human",
          value: -0.5,
        }),
      ]),
    );

    const correctedTrace = await service.getTrace(targetResult?.trace.id);
    expect(correctedTrace?.supervision).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: "teach_correction",
          source: "human",
          resolution: "promoted_to_label",
        }),
      ]),
    );
    await (privateService.worker?.tick() ?? Promise.resolve());

    const retrieved = await service.query({
      conversationId: 7,
      agentIdentity: { agentId: "main", lane: "main" },
      queryText: "deployment failed again",
      budgetChars: 4000,
      queryEmbedding: embed("deployment ci"),
    });

    expect(retrieved).not.toBeNull();
    expect(retrieved?.episode.packVersion).toBe(taught.packVersion);
    expect(retrieved?.fired.some((node) => node.kind === "correction" && node.content.includes("inspect CI logs before retrying"))).toBe(true);

    const trace = await service.getTrace();
    expect(trace?.firedNodes).toContain(taught.nodeId);

    const feedbackStatus = await service.status();
    expect(feedbackStatus.contextFeedback).toMatchObject({
      verdictCounts: {
        helpful: 0,
        irrelevant: 0,
        harmful: 1,
      },
      coverage: {
        routeTraceCount: 3,
        identifiedRouteTraceCount: 3,
        unidentifiedRouteTraceCount: 0,
        agentIdentityCoverage: 1,
        observationCount: 0,
        completedObservationCount: 0,
        supervisedTraceCount: 1,
        unsupervisedTraceCount: 2,
        observationCoverage: 0,
        supervisionCoverage: 1 / 3,
        pendingFollowupCount: 0,
        pendingTeacherCount: 0,
      },
      latest: {
        source: "human",
        verdict: "harmful",
        score: -0.5,
        confidence: 1,
        traceId: targetResult?.trace.id,
        episodeId: targetResult?.episode.id,
        observationId: null,
        agentIdentity: { agentId: "main", lane: "main" },
        bindingMode: null,
      },
      focus: {
        action: "review_harmful_context",
      },
      agents: expect.arrayContaining([
        expect.objectContaining({
          agentIdentity: { agentId: "main", lane: "main" },
          routeTraceCount: 2,
          supervisedTraceCount: 1,
          unsupervisedTraceCount: 1,
          verdictCounts: {
            helpful: 0,
            irrelevant: 0,
            harmful: 1,
          },
        }),
        expect.objectContaining({
          agentIdentity: { agentId: "reviewer", lane: "subagent" },
          routeTraceCount: 1,
          supervisedTraceCount: 0,
          unsupervisedTraceCount: 1,
        }),
      ]),
    });
    expect(feedbackStatus.learningHealth).toMatchObject({
      status: "review_harmful_context",
      signals: {
        routeTraceCount: 3,
        supervisedTraceCount: 1,
        harmfulCount: 1,
      },
    });
    expect((feedbackStatus.promotionStory as {
      integrations?: {
        structuredVerdict?: {
          verdictCounts?: { harmful?: number };
          focus?: { action?: string };
        } | null;
      };
    }).integrations?.structuredVerdict).toMatchObject({
      verdictCounts: {
        harmful: 1,
      },
      focus: {
        action: "review_harmful_context",
      },
    });
  });

  it("commits fast explicit user corrections immediately from recent context", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-codeword-workspace-");
    const brainRoot = makeTempDir("openclawbrain-codeword-state-");
    writeFileSync(
      join(workspaceRoot, "CODEWORD.md"),
      "# Demo\n\nThe codeword is hippo.\n",
      "utf8",
    );

    const fetchMock = vi.fn(async (_input: unknown, init?: { body?: unknown }) => {
      const rawBody = typeof init?.body === "string" ? init.body : "{}";
      const parsed = JSON.parse(rawBody) as { input?: string | string[] };
      const input = Array.isArray(parsed.input) ? parsed.input[0] : parsed.input ?? "";
      return {
        ok: true,
        json: async () => ({ data: [{ embedding: Array.from(embed(String(input))) }] }),
      };
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const service = new BrainService({ deps: createDeps(brainRoot) });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });

    await service.teachUserCorrection({
      canonicalInstruction: "The codeword is hippo.",
      sourceQuote: "the codeword is hippo",
      sourceMessageId: 1,
      conversationId: 17,
      via: "demo_seed",
    });

    await service.query({
      conversationId: 17,
      queryText: "what's the codeword?",
      budgetChars: 4000,
      queryEmbedding: embed("what's the codeword?"),
    });

    await service.observeUserTurn({
      conversationId: 17,
      messageId: 3,
      userText: "wrong, the codeword is giraffe",
      recentMessages: [
        { role: "assistant", content: "The codeword is hippo." },
        { role: "user", content: "what's the codeword?" },
      ],
      recentSummaries: [],
    });

    const status = await service.status();
    expect(status.currentPackVersion).toBeGreaterThanOrEqual(3);

    const matchingNode = (service as unknown as {
      store: { getAllNodes: () => Array<{ metadata?: Record<string, unknown>; content: string }> };
    }).store.getAllNodes().find((node) => node.content.includes("The codeword is giraffe."));

    expect(matchingNode?.metadata).toMatchObject({
      sourceAuthority: "user_explicit",
      sourceMessageId: 3,
      via: "brain_auto_user_correction_fast",
      proposalLane: "fast_deterministic",
    });
  });

  it("prefers the observed episode id when attaching follow-up text and auto-correction supervision", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-episode-attribution-workspace-");
    const brainRoot = makeTempDir("openclawbrain-episode-attribution-state-");
    writeFileSync(
      join(workspaceRoot, "CODEWORD.md"),
      "# Demo\n\nThe codeword is hippo.\n",
      "utf8",
    );

    const fetchMock = vi.fn(async (_input: unknown, init?: { body?: unknown }) => {
      const rawBody = typeof init?.body === "string" ? init.body : "{}";
      const parsed = JSON.parse(rawBody) as { input?: string | string[] };
      const input = Array.isArray(parsed.input) ? parsed.input[0] : parsed.input ?? "";
      return {
        ok: true,
        json: async () => ({ data: [{ embedding: Array.from(embed(String(input))) }] }),
      };
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const service = new BrainService({ deps: createDeps(brainRoot) });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });

    const first = await service.query({
      conversationId: 23,
      queryText: "what's the codeword?",
      budgetChars: 4000,
      queryEmbedding: embed("what's the codeword?"),
    });
    await service.recordTurnObservation({
      episodeId: first?.episode.id,
      assistantResponse: "The codeword is hippo.",
      toolResults: [],
    });

    const second = await service.query({
      conversationId: 23,
      queryText: "what's the codeword again?",
      budgetChars: 4000,
      queryEmbedding: embed("what's the codeword again?"),
    });
    await service.recordTurnObservation({
      episodeId: second?.episode.id,
      assistantResponse: "The codeword is hippo.",
      toolResults: [],
    });

    await service.observeUserTurn({
      conversationId: 23,
      messageId: 5,
      episodeId: first?.episode.id,
      userText: "wrong, the codeword is giraffe",
      recentMessages: [
        { role: "assistant", content: "The codeword is hippo." },
        { role: "user", content: "what's the codeword?" },
      ],
      recentSummaries: [],
    });

    const privateService = service as unknown as {
      store: {
        getObservationForEpisode: (episodeId: string) => { followUpText: string | null; status: string } | null;
      };
    };
    expect(privateService.store.getObservationForEpisode(first?.episode.id ?? "")).toMatchObject({
      followUpText: expect.stringContaining("[redacted follow_up chars="),
      status: "pending_teacher",
    });
    expect(privateService.store.getObservationForEpisode(second?.episode.id ?? "")).toMatchObject({
      followUpText: null,
      status: "pending_followup",
    });

    const firstTrace = await service.getTrace(first?.trace.id);
    const secondTrace = await service.getTrace(second?.trace.id);
    expect(firstTrace?.supervision).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: "teach_correction",
          episodeId: first?.episode.id,
          metadata: expect.objectContaining({
            correctedEpisodeId: first?.episode.id,
            episodeAttributionMode: "explicit_episode",
            episodeAttributionRequestedId: first?.episode.id,
          }),
        }),
      ]),
    );
    expect(secondTrace?.supervision ?? []).toHaveLength(0);
  });

  it("queues async user-correction proposals off-path and commits high-confidence results", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-async-codeword-workspace-");
    const brainRoot = makeTempDir("openclawbrain-async-codeword-state-");
    writeFileSync(
      join(workspaceRoot, "CODEWORD.md"),
      "# Demo\n\nThe codeword is hippo.\n",
      "utf8",
    );

    const fetchMock = vi.fn(async (_input: unknown, init?: { body?: unknown }) => {
      const rawBody = typeof init?.body === "string" ? init.body : "{}";
      const parsed = JSON.parse(rawBody) as { input?: string | string[] };
      const input = Array.isArray(parsed.input) ? parsed.input[0] : parsed.input ?? "";
      return {
        ok: true,
        json: async () => ({ data: [{ embedding: Array.from(embed(String(input))) }] }),
      };
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const deps = createDeps(brainRoot, {
      autoUserCorrectionsEnabled: true,
      autoUserCorrectionsProvider: "openai",
      autoUserCorrectionsModel: "gpt-5.4-mini",
      autoUserCorrectionsMinConfidence: 0.75,
    });
    deps.complete = vi.fn(async () => ({
      content: [{ type: "text", text: JSON.stringify({
        kind: "explicit_correction",
        canonicalInstruction: "The codeword is giraffe.",
        confidence: 0.93,
        reason: "latest user turn explicitly corrected the codeword",
      }) }],
    }));

    const service = new BrainService({ deps });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });

    await service.observeUserTurn({
      conversationId: 17,
      messageId: 4,
      episodeId: "ep_async_1",
      userText: "no, use the new one",
      recentMessages: [
        { role: "assistant", content: "The codeword is hippo." },
        { role: "user", content: "what's the codeword?" },
      ],
      recentSummaries: [
        {
          summaryId: "sum_1",
          kind: "leaf",
          depth: 1,
          content: "The user asked about the codeword and the assistant answered hippo.",
        },
      ],
    });

    await waitFor(async () => {
      const nodes = (service as unknown as {
        store: { getAllNodes: () => Array<{ content: string }> };
      }).store.getAllNodes();
      return nodes.some((node) => node.content.includes("The codeword is giraffe."));
    });

    const status = await service.status();
    expect(status.pendingUserObservationCount).toBe(0);
    expect(deps.complete).toHaveBeenCalledTimes(1);
  });

  it("runs the learner in a supervised child process and reports heartbeat truth", async () => {
    const brainRoot = makeTempDir("openclawbrain-state-");
    const service = new BrainService({
      deps: createDeps(brainRoot, {
        workerMode: "child",
        trainerIntervalMs: 200,
        workerHeartbeatTimeoutMs: 5_000,
        workerRestartDelayMs: 100,
      }),
    });

    try {
      service.startWorker();
      await waitFor(async () => {
        const status = await service.status();
        return Boolean(status.workerPid) && status.workerMode === "child" && status.workerHealthy === true;
      });

      const status = await service.status();
      expect(status.workerMode).toBe("child");
      expect(status.workerStatus).toBe("running");
      expect(status.workerPid).toEqual(expect.any(Number));
      expect(status.workerHealthy).toBe(true);
    } finally {
      service.stopWorker();
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  });

  it("keeps serving from the last promoted pack when the child worker dies", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-workspace-");
    const brainRoot = makeTempDir("openclawbrain-state-");
    writeFileSync(
      join(workspaceRoot, "PLAYBOOK.md"),
      "# Pull Requests\n\nUse gh pr create for pull request workflows.\n",
      "utf8",
    );

    const service = new BrainService({
      deps: createDeps(brainRoot, {
        workerMode: "child",
        trainerIntervalMs: 200,
        workerHeartbeatTimeoutMs: 150,
        workerRestartDelayMs: 5_000,
      }),
    });

    try {
      await service.init({
        workspaceRoot,
        embedFn: async (text) => embed(text),
      });
      service.startWorker();
      await waitFor(async () => {
        const status = await service.status();
        return Boolean(status.workerPid) && status.workerHealthy === true;
      });

      const beforeCrash = await service.query({
        conversationId: 42,
        queryText: "how do I open a pull request?",
        budgetChars: 4000,
        queryEmbedding: embed("gh pr create pull request"),
      });
      expect(beforeCrash).not.toBeNull();

      const childPid = (await service.status()).workerPid as number;
      process.kill(childPid, "SIGKILL");
      await waitFor(async () => Boolean((await service.status()).workerLastExit), 1_500);
      await new Promise((resolve) => setTimeout(resolve, 250));

      const statusAfterCrash = await service.status();
      expect(statusAfterCrash.workerMode).toBe("child");
      expect(statusAfterCrash.workerHealthy).toBe(false);
      expect(statusAfterCrash.currentPackVersion).toBe(1);
      expect(statusAfterCrash.workerLastExit).toEqual(expect.objectContaining({
        signal: "SIGKILL",
      }));

      const afterCrash = await service.query({
        conversationId: 42,
        queryText: "how do I open a pull request again?",
        budgetChars: 4000,
        queryEmbedding: embed("gh pr create pull request"),
      });
      expect(afterCrash).not.toBeNull();
      expect(afterCrash?.episode.packVersion).toBe(1);
      expect(afterCrash?.fired.some((node) => node.content.includes("gh pr create"))).toBe(true);
    } finally {
      service.stopWorker();
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  });

  it("records worker restart accounting after a crash and restart", async () => {
    const brainRoot = makeTempDir("openclawbrain-state-");
    const service = new BrainService({
      deps: createDeps(brainRoot, {
        workerMode: "child",
        trainerIntervalMs: 200,
        workerHeartbeatTimeoutMs: 5_000,
        workerRestartDelayMs: 100,
      }),
    });

    try {
      service.startWorker();
      await waitFor(async () => Boolean((await service.status()).workerPid));
      const firstPid = (await service.status()).workerPid as number;
      process.kill(firstPid, "SIGKILL");
      await waitFor(async () => {
        const status = await service.status();
        return status.workerRestartCount === 1
          && status.workerLastRestartAt !== null
          && status.workerPid !== null
          && status.workerPid !== firstPid
          && status.workerHealthy === true;
      }, 5_000);
    } finally {
      service.stopWorker();
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  });

  it("records reload acknowledgements from the child worker", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-workspace-");
    const brainRoot = makeTempDir("openclawbrain-state-");
    writeFileSync(join(workspaceRoot, "PLAYBOOK.md"), "# Pull Requests\n\nUse gh pr create.\n", "utf8");

    const service = new BrainService({
      deps: createDeps(brainRoot, {
        workerMode: "child",
        trainerIntervalMs: 200,
        workerHeartbeatTimeoutMs: 5_000,
        workerRestartDelayMs: 100,
      }),
    });

    try {
      service.startWorker();
      await waitFor(async () => Boolean((await service.status()).workerPid));
      await service.init({
        workspaceRoot,
        embedFn: async (text) => embed(text),
      });
      await waitFor(async () => {
        const status = await service.status();
        return Boolean(status.workerLastReloadRequestedAt) && Boolean(status.workerLastReloadAckAt);
      });

      const status = await service.status();
      expect(status.workerLastReloadRequestedAt).toEqual(expect.any(Number));
      expect(status.workerLastReloadAckAt).toEqual(expect.any(Number));
      expect(Number(status.workerLastReloadAckAt)).toBeGreaterThanOrEqual(Number(status.workerLastReloadRequestedAt));
    } finally {
      service.stopWorker();
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  });

  it("ignores a stale worker lease and starts a fresh child worker", async () => {
    const brainRoot = makeTempDir("openclawbrain-state-");
    writeFileSync(
      join(brainRoot, "worker-lease.json"),
      JSON.stringify({
        pid: 999999,
        startedAt: Date.now() - 10_000,
        heartbeatAt: Date.now() - 10_000,
        status: "running",
      }),
      "utf8",
    );

    const service = new BrainService({
      deps: createDeps(brainRoot, {
        workerMode: "child",
        trainerIntervalMs: 200,
        workerHeartbeatTimeoutMs: 150,
        workerRestartDelayMs: 100,
      }),
    });

    try {
      service.startWorker();
      await waitFor(async () => {
        const status = await service.status();
        return Boolean(status.workerPid) && status.workerHealthy === true;
      });
      expect((await service.status()).workerLastFatalError).toBeNull();
    } finally {
      service.stopWorker();
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  });

  it("refuses a second live child worker on the same brain root", async () => {
    const brainRoot = makeTempDir("openclawbrain-state-");
    const primary = new BrainService({
      deps: createDeps(brainRoot, {
        workerMode: "child",
        trainerIntervalMs: 200,
        workerHeartbeatTimeoutMs: 5_000,
        workerRestartDelayMs: 500,
      }),
    });
    const secondary = new BrainService({
      deps: createDeps(brainRoot, {
        workerMode: "child",
        trainerIntervalMs: 200,
        workerHeartbeatTimeoutMs: 5_000,
        workerRestartDelayMs: 500,
      }),
    });

    try {
      primary.startWorker();
      await waitFor(async () => Boolean((await primary.status()).workerPid));

      secondary.startWorker();
      await waitFor(async () => {
        const status = await secondary.status();
        return status.workerLastFatalError === `worker lease already held by pid ${(await primary.status()).workerPid}`;
      }, 3_000);

      const primaryStatus = await primary.status();
      const secondaryStatus = await secondary.status();
      expect(primaryStatus.workerHealthy).toBe(true);
      expect(secondaryStatus.workerPid).toBe(primaryStatus.workerPid);
      expect(secondaryStatus.workerLastFatalError).toContain("worker lease already held by pid");
    } finally {
      secondary.stopWorker();
      primary.stopWorker();
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  });
});
