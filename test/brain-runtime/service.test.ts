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
    const status = await service.status();
    expect(status.currentPackVersion).toBe(1);
  });

  it("teaches a correction against the active conversation and only labels matching episodes", async () => {
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

    await service.query({
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
});
