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
        store: { getObservationForEpisode: (episodeId: string) => Record<string, unknown> | null };
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
