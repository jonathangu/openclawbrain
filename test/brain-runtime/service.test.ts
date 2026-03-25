import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainService } from "../../src/brain-runtime/service.js";
import type { LcmDependencies } from "../../src/types.js";

const tempDirs: string[] = [];

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
      queryText: "how do I open a pull request?",
      budgetChars: 4000,
      queryEmbedding: embed("gh pr create pull request"),
    });

    expect(result).not.toBeNull();
    expect(result?.episode.conversationId).toBe(42);

    const trace = await service.getTrace();
    expect(trace?.episodeId).toBe(result?.episode.id ?? null);
    expect(trace?.routeTrace).toMatchObject({
      conversationId: 42,
      activePackId: "brain-pack-v1",
      routerIdentity: "brain-graph-traverse.v2",
      selectedNodeIds: [result?.fired[0]?.nodeId],
      injectedNodeSummaries: [
        expect.objectContaining({
          nodeId: result?.fired[0]?.nodeId,
          kind: "chunk",
          sourceUri: "PLAYBOOK.md",
          contentPreview: expect.stringContaining("Use gh pr create"),
        }),
      ],
      selectionMetadata: expect.objectContaining({
        traceSliceVersion: 2,
        budgetChars: 4000,
        maxHops: 8,
        firedCount: result?.fired.length,
        queryEmbeddingSource: "provided",
      }),
    });
    expect(trace?.routeTrace?.requestDigest).toMatch(/^[a-f0-9]{16}$/);
    expect(trace?.routeTrace?.candidateNodeIds).toContain(result?.fired[0]?.nodeId ?? "");
    expect(trace?.routeTrace?.sourceSummary.kinds).toMatchObject({ chunk: 1 });
    expect(trace?.routeTrace?.sourceSummary.sourceUris[0]).toContain("PLAYBOOK.md");
    const status = await service.status();
    expect(status.currentPackVersion).toBe(1);
    expect(status.routeTraceCount).toBe(1);
    expect(status.supervisionCount).toBe(0);
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
      version: 1,
      updateCount: 2,
      candidatePackVersion: 9,
      currentPackVersion: 3,
      generatedAt: 123456789,
      episodeIds: ["ep_1", "ep_2"],
      traceIds: ["bt_1", "bt_2"],
      supervisionIds: ["ts_1", "ts_2"],
      teacherTraceIds: ["bt_2"],
      rewardSources: { human: 1, scanner: 0, teacher: 1, self: 0 },
      episodeCount: 2,
      traceCount: 2,
      supervisionCount: 2,
      teacherLabelCount: 1,
      routeUpdateCount: 3,
      seedUpdateCount: 2,
      edgeUpdateCount: 1,
      baselineBefore: 0,
      baselineAfter: 0.12,
    });
    store.setTrainingState("last_pg_candidate_pack_version", 9);

    const status = await service.status();
    expect(status.lastPgCandidatePackVersion).toBe(9);
    expect(status.lastPgCandidateUpdate).toMatchObject({
      updateCount: 2,
      candidatePackVersion: 9,
      teacherLabelCount: 1,
      traceIds: ["bt_1", "bt_2"],
    });
  });

  it("fails open when teacher resolution has no model and reports that truth in status", async () => {
    const brainRoot = makeTempDir("openclawbrain-state-");
    const deps = createDeps(brainRoot, {
      teacherEnabled: true,
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
      queryText: "deployment failed",
      budgetChars: 4000,
      queryEmbedding: embed("deployment ci"),
    });
    await service.query({
      conversationId: 99,
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
      queryText: "deployment failed again",
      budgetChars: 4000,
      queryEmbedding: embed("deployment ci"),
    });

    expect(retrieved).not.toBeNull();
    expect(retrieved?.episode.packVersion).toBe(taught.packVersion);
    expect(retrieved?.fired.some((node) => node.kind === "correction" && node.content.includes("inspect CI logs before retrying"))).toBe(true);

    const trace = await service.getTrace();
    expect(trace?.firedNodes).toContain(taught.nodeId);
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
      followUpText: "wrong, the codeword is giraffe",
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
