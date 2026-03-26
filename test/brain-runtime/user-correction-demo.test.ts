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

  if (normalized.includes("giraffe")) {
    return new Float32Array([1, 0, 1]);
  }
  if (normalized.includes("hippo")) {
    return new Float32Array([1, 0, 0]);
  }
  if (normalized.includes("codeword")) {
    return new Float32Array([1, 0, 0.5]);
  }

  return new Float32Array([0.2, 0.2, 0.2]);
}

function createDeps(
  brainRoot: string,
  overrides?: Partial<LcmDependencies["config"]["brain"]>,
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
      timezone: "America/Denver",
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

afterEach(() => {
  vi.restoreAllMocks();
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("explicit user correction demo", () => {
  it("stores provenance and makes giraffe win for the codeword demo", async () => {
    vi.spyOn(Math, "random").mockReturnValue(0);

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
        json: async () => ({
          data: [{ embedding: Array.from(embed(String(input))) }],
        }),
      };
    });

    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const service = new BrainService({
      deps: createDeps(brainRoot),
    });

    await service.init({
      workspaceRoot,
      embedFn: async (text) => embed(text),
    });

    const initial = await service.teach({
      instruction: "The codeword is hippo.",
      conversationId: 17,
      kind: "correction",
      tags: ["demo", "codeword"],
      metadata: {
        sourceAuthority: "user_explicit",
        sourceQuote: "codeword is hippo",
        sourceMessageId: 1,
        via: "demo_seed",
      },
      via: "demo_seed",
    });

    expect(initial.packVersion).toBeGreaterThanOrEqual(2);

    const before = await service.query({
      conversationId: 17,
      queryText: "what's the codeword?",
      budgetChars: 4000,
      queryEmbedding: embed("what's the codeword?"),
    });

    expect(before).not.toBeNull();

    const corrected = await service.teach({
      instruction: "The codeword is giraffe.",
      conversationId: 17,
      kind: "correction",
      tags: ["demo", "codeword"],
      metadata: {
        sourceAuthority: "user_explicit",
        sourceQuote: "wrong, it changed to giraffe",
        sourceMessageId: 3,
        via: "brain_teach_user_correction",
      },
      via: "brain_teach_user_correction",
    });

    const after = await service.query({
      conversationId: 17,
      queryText: "what's the codeword?",
      budgetChars: 4000,
      queryEmbedding: embed("what's the codeword?"),
    });

    expect(after).not.toBeNull();
    expect(
      after?.fired.some(
        (node) => node.kind === "correction" && node.content.includes("giraffe"),
      ),
    ).toBe(true);

    const matchingNode = (
      service as unknown as {
        store: {
          getAllNodes: () => Array<{
            metadata?: Record<string, unknown>;
            id: string;
          }>;
        };
      }
    ).store.getAllNodes().find(
      (row) => row.id === corrected.nodeId,
    );

    expect(matchingNode?.metadata).toMatchObject({
      sourceAuthority: "user_explicit",
      sourceQuote: "wrong, it changed to giraffe",
      sourceMessageId: 3,
      via: "brain_teach_user_correction",
    });
  });
});
