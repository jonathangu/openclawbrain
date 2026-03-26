import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it, vi } from "vitest";
import { materializeTeacherLabelInput } from "../../src/brain-core/teacher.js";
import type { BrainObservation } from "../../src/brain-core/types.js";
import { BrainAssemblerExtension } from "../../src/brain-runtime/assembler-extension.js";
import { BrainService } from "../../src/brain-runtime/service.js";
import { runBrainMigrations } from "../../src/brain-store/migrations.js";
import { BrainStore } from "../../src/brain-store/store.js";
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
  return new Float32Array([0.5, 0.5, 0]);
}

function createDeps(brainRoot: string): LcmDependencies {
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

afterEach(() => {
  vi.restoreAllMocks();
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("BrainService observations", () => {
  it("persists the durable turn snapshot used by teacher-v2", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-observation-workspace-");
    const brainRoot = makeTempDir("openclawbrain-observation-state-");
    writeFileSync(
      join(workspaceRoot, "PLAYBOOK.md"),
      "# Pull Requests\n\nUse gh pr create for pull request workflows.\n",
      "utf8",
    );

    const service = new BrainService({ deps: createDeps(brainRoot) });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });

    const result = await service.query({
      conversationId: 51,
      queryText: "how do I open a pull request?",
      budgetChars: 4000,
      queryEmbedding: embed("gh pr create pull request"),
    });

    await service.recordTurnObservation({
      episodeId: result?.episode.id,
      assistantResponse: "Use `gh pr create` to open the pull request.",
      toolResults: [{
        sourceRole: "tool",
        toolCallId: "call_1",
        toolName: "bash",
        input: "{\"cmd\":\"gh pr create\"}",
        output: "{\"ok\":true}",
        isError: false,
        excerpt: "{\"ok\":true}",
      }],
    });

    const observation = (
      service as unknown as {
        store: { getObservationForEpisode: (episodeId: string) => BrainObservation | null };
      }
    ).store.getObservationForEpisode(result?.episode.id ?? "");

    expect(observation).toMatchObject({
      traceId: result?.trace.id,
      queryText: "how do I open a pull request?",
      assistantResponse: "Use `gh pr create` to open the pull request.",
      toolResults: [
        expect.objectContaining({
          toolName: "bash",
          output: "{\"ok\":true}",
        }),
      ],
      status: "pending_followup",
      routeMetadata: {
        selectionMetadata: {
          traceSliceVersion: 3,
          chosenStopCount: 0,
          forcedStopCount: expect.any(Number),
          droppedProposalCount: 0,
          droppedProposalReasons: null,
        },
      },
    });
    const teacherInput = materializeTeacherLabelInput(observation!);
    expect(teacherInput?.routeMetadata.selectionMetadata).toMatchObject({
      traceSliceVersion: 3,
      chosenStopCount: 0,
      forcedStopCount: expect.any(Number),
      droppedProposalCount: 0,
      droppedProposalReasons: null,
    });
    expect((teacherInput?.routeMetadata.selectionMetadata?.forcedStopCount ?? 0)).toBeGreaterThan(0);
  });

  it("persists exact provenance columns from the runtime truth path into observations and teacher input", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-provenance-observation-workspace-");
    const brainRoot = makeTempDir("openclawbrain-provenance-observation-state-");
    writeFileSync(
      join(workspaceRoot, "PLAYBOOK.md"),
      "# Pull Requests\n\nUse gh pr create for pull request workflows.\n",
      "utf8",
    );

    const service = new BrainService({ deps: createDeps(brainRoot) });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });

    const result = await service.query({
      conversationId: 71,
      queryText: "how do I open a pull request?",
      budgetChars: 4000,
      queryEmbedding: embed("gh pr create pull request"),
    });
    expect(result).not.toBeNull();

    service.noteAssemblyDecision({
      mode: "use_brain",
      conversationId: 71,
      episodeId: result?.episode.id ?? null,
      traceId: result?.trace.id ?? null,
      footer: result?.trace.footer ?? null,
      serveDecisionRecordId: "decision-observation-1",
      selectionDigest: "selection-observation-1",
      turnCompileEventId: "evt-compile-observation-1",
      decisionRecordedAt: "2026-03-25T01:02:03.000Z",
      activePackId: "brain-pack-v1",
      activePackEventExportDigest: "export-digest-1",
      activePackGraphChecksum: "graph-checksum-1",
      activePackRouterChecksum: "router-checksum-1",
      activePackBuiltAt: "2026-03-25T01:00:00.000Z",
      servedArtifact: {
        kind: "runtime_compile_v1",
        traceId: result?.trace.id ?? null,
      },
    });

    await service.recordTurnObservation({
      episodeId: result?.episode.id,
      assistantResponse: "Use `gh pr create` to open the pull request.",
      toolResults: [],
    });

    const observation = (
      service as unknown as {
        store: { getObservationForEpisode: (episodeId: string) => BrainObservation | null };
      }
    ).store.getObservationForEpisode(result?.episode.id ?? "");
    expect(observation).not.toBeNull();
    expect(observation?.routeMetadata).toMatchObject({
      bindingMode: "exact_decision_id",
      serveDecisionRecordId: "decision-observation-1",
      selectionDigest: "selection-observation-1",
      turnCompileEventId: "evt-compile-observation-1",
      decisionRecordedAt: "2026-03-25T01:02:03.000Z",
      activePackId: "brain-pack-v1",
      activePackEventExportDigest: "export-digest-1",
      activePackGraphChecksum: "graph-checksum-1",
      activePackRouterChecksum: "router-checksum-1",
      activePackBuiltAt: "2026-03-25T01:00:00.000Z",
      servedArtifact: {
        kind: "runtime_compile_v1",
        traceId: result?.trace.id,
      },
    });

    const teacherInput = materializeTeacherLabelInput(observation!);
    expect(teacherInput?.routeMetadata).toMatchObject({
      bindingMode: "exact_decision_id",
      serveDecisionRecordId: "decision-observation-1",
      selectionDigest: "selection-observation-1",
      turnCompileEventId: "evt-compile-observation-1",
      activePackGraphChecksum: "graph-checksum-1",
      servedArtifact: {
        kind: "runtime_compile_v1",
        traceId: result?.trace.id,
      },
    });
  });

  it("treats explicit provenance columns as authoritative over stale JSON payloads", () => {
    const brainRoot = makeTempDir("openclawbrain-observation-authority-");
    const db = new DatabaseSync(join(brainRoot, "state.db"));
    db.exec("PRAGMA journal_mode = WAL");
    db.exec("PRAGMA foreign_keys = ON");
    runBrainMigrations(db);

    const store = new BrainStore(db, { brainRoot });
    const createdAt = Date.now();
    const inserted = store.insertObservation({
      episodeId: "ep_authoritative",
      conversationId: 81,
      traceId: "trace_authoritative",
      queryText: "how do I open a pull request?",
      retrievedContext: [],
      routeMetadata: {
        requestDigest: "digest-authoritative",
        activePackId: "brain-pack-column",
        routerIdentity: "brain-graph-traverse.v2",
        bindingMode: "exact_decision_id",
        serveDecisionRecordId: "decision-column",
        selectionDigest: "selection-column",
        turnCompileEventId: "compile-column",
        decisionRecordedAt: "2026-03-25T01:02:03.000Z",
        activePackEventExportDigest: "export-column",
        activePackGraphChecksum: "graph-column",
        activePackRouterChecksum: "router-column",
        activePackBuiltAt: "2026-03-25T01:00:00.000Z",
        servedArtifact: {
          kind: "runtime_compile_v1",
          traceId: "trace_authoritative",
        },
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
      createdAt,
      updatedAt: createdAt,
    });

    db.prepare(`
      UPDATE brain_observations
      SET route_metadata_json = ?,
          teacher_evaluation_json = ?
      WHERE id = ?
    `).run(
      JSON.stringify({
        requestDigest: "digest-json",
        activePackId: "brain-pack-json",
        routerIdentity: "brain-graph-traverse.v2",
        bindingMode: "legacy_heuristic",
        serveDecisionRecordId: "decision-json",
        selectionDigest: "selection-json",
        turnCompileEventId: "compile-json",
        decisionRecordedAt: "2026-03-24T01:02:03.000Z",
        activePackEventExportDigest: "export-json",
        activePackGraphChecksum: "graph-json",
        activePackRouterChecksum: "router-json",
        activePackBuiltAt: "2026-03-24T01:00:00.000Z",
        servedArtifact: {
          kind: "runtime_compile_v1",
          traceId: "trace_json",
        },
        candidateNodeIds: [],
        selectedNodeIds: [],
        selectedTraversalNodeIds: [],
        selectedPathNodeIds: [],
        selectedSeedNodeIds: [],
        sourceSummary: null,
        selectionMetadata: null,
      }),
      JSON.stringify({
        version: 2,
        observationId: inserted.id,
        episodeId: "ep_authoritative",
        traceId: "trace_authoritative",
        serveDecisionRecordId: "decision-json",
        selectionDigest: "selection-json",
        turnCompileEventId: "compile-json",
        decisionRecordedAt: "2026-03-24T01:02:03.000Z",
        activePackId: "brain-pack-json",
        activePackEventExportDigest: "export-json",
        activePackGraphChecksum: "graph-json",
        activePackRouterChecksum: "router-json",
        activePackBuiltAt: "2026-03-24T01:00:00.000Z",
        bindingMode: "legacy_heuristic",
        retrievalRelevance: 0.8,
        agentUsage: 0.4,
        outcomeSupport: 0.6,
        finalScore: 0.6,
        confidence: 0.7,
        reason: "stale JSON teacher evaluation",
      }),
      inserted.id,
    );

    const observation = store.getObservationForEpisode("ep_authoritative");
    expect(observation?.routeMetadata).toMatchObject({
      bindingMode: "exact_decision_id",
      serveDecisionRecordId: "decision-column",
      selectionDigest: "selection-column",
      turnCompileEventId: "compile-column",
      activePackId: "brain-pack-column",
      activePackEventExportDigest: "export-column",
      activePackGraphChecksum: "graph-column",
      activePackRouterChecksum: "router-column",
      activePackBuiltAt: "2026-03-25T01:00:00.000Z",
    });
    expect(observation?.teacherEvaluation).toMatchObject({
      serveDecisionRecordId: "decision-column",
      selectionDigest: "selection-column",
      turnCompileEventId: "compile-column",
      activePackId: "brain-pack-column",
      activePackEventExportDigest: "export-column",
      activePackGraphChecksum: "graph-column",
      activePackRouterChecksum: "router-column",
      activePackBuiltAt: "2026-03-25T01:00:00.000Z",
      bindingMode: "exact_decision_id",
      reason: "stale JSON teacher evaluation",
    });
  });

  it("reads legacy JSON-only observation provenance when exact columns are absent", () => {
    const brainRoot = makeTempDir("openclawbrain-observation-legacy-");
    const db = new DatabaseSync(join(brainRoot, "state.db"));
    db.exec("PRAGMA journal_mode = WAL");
    db.exec("PRAGMA foreign_keys = ON");
    runBrainMigrations(db);

    const store = new BrainStore(db, { brainRoot });
    const createdAt = Date.now();
    db.prepare(`
      INSERT INTO brain_observations (
        id,
        episode_id,
        conversation_id,
        trace_id,
        query_text,
        retrieved_context_json,
        route_metadata_json,
        assistant_response,
        tool_results_json,
        status,
        created_at,
        updated_at
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      "bo_legacy",
      "ep_legacy",
      82,
      "trace_legacy",
      "how do I open a pull request?",
      "[]",
      JSON.stringify({
        requestDigest: "digest-legacy",
        activePackId: "brain-pack-legacy",
        routerIdentity: "brain-graph-traverse.v2",
        serveDecisionRecordId: null,
        selectionDigest: "selection-legacy",
        turnCompileEventId: "compile-legacy",
        decisionRecordedAt: "2026-03-24T01:02:03.000Z",
        activePackEventExportDigest: "export-legacy",
        activePackGraphChecksum: "graph-legacy",
        activePackRouterChecksum: "router-legacy",
        activePackBuiltAt: "2026-03-24T01:00:00.000Z",
        servedArtifact: {
          kind: "runtime_compile_v1",
          traceId: "trace_legacy",
        },
        candidateNodeIds: [],
        selectedNodeIds: [],
        selectedTraversalNodeIds: [],
        selectedPathNodeIds: [],
        selectedSeedNodeIds: [],
        sourceSummary: null,
        selectionMetadata: null,
      }),
      "Use `gh pr create`.",
      "[]",
      "pending_followup",
      createdAt,
      createdAt,
    );

    const observation = store.getObservationForEpisode("ep_legacy");
    expect(observation?.routeMetadata).toMatchObject({
      bindingMode: "exact_selection_digest",
      selectionDigest: "selection-legacy",
      turnCompileEventId: "compile-legacy",
      activePackId: "brain-pack-legacy",
      activePackEventExportDigest: "export-legacy",
      activePackGraphChecksum: "graph-legacy",
      activePackRouterChecksum: "router-legacy",
      activePackBuiltAt: "2026-03-24T01:00:00.000Z",
      servedArtifact: {
        kind: "runtime_compile_v1",
        traceId: "trace_legacy",
      },
    });

    const teacherInput = materializeTeacherLabelInput(observation!);
    expect(teacherInput?.routeMetadata).toMatchObject({
      bindingMode: "exact_selection_digest",
      selectionDigest: "selection-legacy",
      activePackGraphChecksum: "graph-legacy",
    });
  });

  it("persists clipped assembly attribution through trace, observation, and teacher input", async () => {
    const queryBudgetChars = deriveExpectedQueryBudgetChars(4096);
    const workspaceRoot = makeTempDir("openclawbrain-clipped-observation-workspace-");
    const brainRoot = makeTempDir("openclawbrain-clipped-observation-state-");
    writeFileSync(
      join(workspaceRoot, "PLAYBOOK.md"),
      [
        "# Pull Requests",
        "",
        "Use gh pr create for pull request workflows.",
        "Include the exact base branch, reviewer expectations, rollout notes, and verification steps.",
        "Keep the explanation operator-auditable with enough detail to exceed compact clip thresholds.",
        "Repeat the operator-proof guidance so the formatted brain block is long enough to clip in tests.",
        "Use gh pr create for pull request workflows and include the exact validation evidence.",
      ].join("\n"),
      "utf8",
    );

    const service = new BrainService({ deps: createDeps(brainRoot) });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });
    (service as unknown as { embeddingClient: (text: string) => Promise<Float32Array> }).embeddingClient = async (text: string) => embed(text);

    const extension = new BrainAssemblerExtension(service);
    const assembly = await extension.augmentAssembly({
      conversationId: 51,
      tokenBudget: 4096,
      maxContextChars: 240,
      assembled: {
        messages: [{ role: "user", content: "live tail" }],
        estimatedTokens: 2,
        stats: {
          rawMessageCount: 1,
          summaryCount: 0,
          totalContextItems: 1,
        },
      },
      liveMessages: [{ role: "user", content: "How do I open a pull request?" }],
    });

    const traceId = assembly.brainDecision?.traceId ?? "";
    const episodeId = assembly.brainDecision?.episodeId ?? "";
    const trace = await service.getTrace(traceId);
    expect(trace?.routeTrace?.selectionMetadata).toMatchObject({
      compileElapsedMs: expect.any(Number),
      brainDropReason: "injection_cap_clipped",
      brainDropStage: "injection",
      budgetFraction: 0.3,
      servedPartial: true,
      maxContextChars: 240,
      queryBudgetChars,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: expect.any(Number),
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
    });
    if ((trace?.routeTrace?.selectionMetadata.droppedNodeCount ?? 0) > 0) {
      expect(trace?.routeTrace?.selectionMetadata.fittingDropReasons).toEqual(expect.objectContaining({
        omitted_for_max_context_chars: expect.any(Number),
      }));
    } else {
      expect(trace?.routeTrace?.selectionMetadata.fittingDropReasons ?? null).toBeNull();
    }

    await service.recordTurnObservation({
      episodeId,
      assistantResponse: "Use `gh pr create` to open the pull request.",
      toolResults: [],
    });

    const observation = (
      service as unknown as {
        store: { getObservationForEpisode: (episodeId: string) => BrainObservation | null };
      }
    ).store.getObservationForEpisode(episodeId);

    expect(observation).not.toBeNull();
    expect(observation?.routeMetadata.selectionMetadata).toMatchObject({
      compileElapsedMs: expect.any(Number),
      brainDropReason: "injection_cap_clipped",
      brainDropStage: "injection",
      budgetFraction: 0.3,
      servedPartial: true,
      maxContextChars: 240,
      queryBudgetChars,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: expect.any(Number),
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
    });
    if ((observation?.routeMetadata.selectionMetadata?.droppedNodeCount ?? 0) > 0) {
      expect(observation?.routeMetadata.selectionMetadata?.fittingDropReasons).toEqual(expect.objectContaining({
        omitted_for_max_context_chars: expect.any(Number),
      }));
    } else {
      expect(observation?.routeMetadata.selectionMetadata?.fittingDropReasons ?? null).toBeNull();
    }
    const teacherInput = materializeTeacherLabelInput(observation!);
    expect(teacherInput?.routeMetadata.selectionMetadata).toMatchObject({
      compileElapsedMs: expect.any(Number),
      brainDropReason: "injection_cap_clipped",
      brainDropStage: "injection",
      budgetFraction: 0.3,
      servedPartial: true,
      maxContextChars: 240,
      queryBudgetChars,
      injectedChars: expect.any(Number),
      droppedChars: expect.any(Number),
      contextClipped: true,
      fitStrategy: "structured_node_budget",
      retrievedNodeCount: expect.any(Number),
      fittedNodeCount: expect.any(Number),
      droppedNodeCount: expect.any(Number),
    });
    if ((teacherInput?.routeMetadata.selectionMetadata?.droppedNodeCount ?? 0) > 0) {
      expect(teacherInput?.routeMetadata.selectionMetadata?.fittingDropReasons).toEqual(expect.objectContaining({
        omitted_for_max_context_chars: expect.any(Number),
      }));
    } else {
      expect(teacherInput?.routeMetadata.selectionMetadata?.fittingDropReasons ?? null).toBeNull();
    }
  });

  it("persists deadline interruption truth through observation and teacher metadata", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-interruption-observation-workspace-");
    const brainRoot = makeTempDir("openclawbrain-interruption-observation-state-");
    writeFileSync(
      join(workspaceRoot, "PLAYBOOK.md"),
      "# Pull Requests\n\nUse gh pr create for pull request workflows.\n",
      "utf8",
    );

    const service = new BrainService({ deps: createDeps(brainRoot) });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });

    const result = await service.query({
      conversationId: 53,
      queryText: "how do I open a pull request?",
      budgetChars: 4000,
      queryEmbedding: embed("gh pr create pull request"),
    });
    expect(result).not.toBeNull();

    service.recordTraceSelectionMetadata(result?.trace, {
      compileElapsedMs: 12,
      compileDeadlineMs: 10,
      compileDeadlineHit: true,
      brainDropReason: "deadline_after_query",
      brainDropStage: "query",
      queryInterrupted: true,
      interruptionStage: "query",
      interruptionReason: "deadline_after_query",
      servedPartial: false,
    });

    await service.recordTurnObservation({
      episodeId: result?.episode.id,
      assistantResponse: "Use `gh pr create` to open the pull request.",
      toolResults: [],
    });

    const observation = (
      service as unknown as {
        store: { getObservationForEpisode: (episodeId: string) => BrainObservation | null };
      }
    ).store.getObservationForEpisode(result?.episode.id ?? "");

    expect(observation?.routeMetadata.selectionMetadata).toMatchObject({
      compileElapsedMs: 12,
      compileDeadlineMs: 10,
      compileDeadlineHit: true,
      brainDropReason: "deadline_after_query",
      brainDropStage: "query",
      queryInterrupted: true,
      interruptionStage: "query",
      interruptionReason: "deadline_after_query",
      servedPartial: false,
    });
    const teacherInput = materializeTeacherLabelInput(observation!);
    expect(teacherInput?.routeMetadata.selectionMetadata).toMatchObject({
      compileElapsedMs: 12,
      compileDeadlineMs: 10,
      compileDeadlineHit: true,
      brainDropReason: "deadline_after_query",
      brainDropStage: "query",
      queryInterrupted: true,
      interruptionStage: "query",
      interruptionReason: "deadline_after_query",
      servedPartial: false,
    });
  });

  it("keeps system scaffolding out of follow-up attachment", async () => {
    const workspaceRoot = makeTempDir("openclawbrain-system-followup-workspace-");
    const brainRoot = makeTempDir("openclawbrain-system-followup-state-");
    writeFileSync(
      join(workspaceRoot, "PLAYBOOK.md"),
      "# Pull Requests\n\nUse gh pr create for pull request workflows.\n",
      "utf8",
    );

    const service = new BrainService({ deps: createDeps(brainRoot) });
    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });

    const result = await service.query({
      conversationId: 52,
      queryText: "how do I open a pull request?",
      budgetChars: 4000,
      queryEmbedding: embed("gh pr create pull request"),
    });
    await service.recordTurnObservation({
      episodeId: result?.episode.id,
      assistantResponse: "Use `gh pr create` to open the pull request.",
      toolResults: [],
    });

    await service.observeUserTurn({
      conversationId: 52,
      messageId: 400,
      episodeId: result?.episode.id,
      userText: "NO_REPLY",
      recentMessages: [],
      recentSummaries: [],
    });

    const observation = (
      service as unknown as {
        store: { getObservationForEpisode: (episodeId: string) => Record<string, unknown> | null };
      }
    ).store.getObservationForEpisode(result?.episode.id ?? "");

    expect(observation).toMatchObject({
      followUpText: null,
      status: "pending_followup",
    });
  });
});
