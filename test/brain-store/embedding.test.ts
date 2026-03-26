import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createEmbeddingClient,
  resolveEmbeddingAuthMode,
  resolveEmbeddingBaseUrl,
} from "../../src/brain-store/embedding.js";
import type { OpenClawBrainRuntimeConfig } from "../../src/db/config.js";

function makeConfig(overrides?: Partial<OpenClawBrainRuntimeConfig>): OpenClawBrainRuntimeConfig {
  return {
    enabled: true,
    root: "/tmp/openclawbrain-test",
    budgetFraction: 0.3,
    maxHops: 8,
    maxFanoutPerNode: 4,
    maxFrontierSize: 32,
    maxSeeds: 10,
    semanticThreshold: 0.7,
    servingTemperature: 0.1,
    learningTemperature: 1,
    learningRate: 0.01,
    baselineAlpha: 0.1,
    decayRate: 0.995,
    trainerIntervalMs: 30_000,
    workerMode: "child",
    workerHeartbeatTimeoutMs: 90_000,
    workerRestartDelayMs: 5_000,
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
    embeddingBaseUrl: "",
    ...overrides,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  delete process.env.OPENAI_API_KEY;
  delete process.env.OPENCLAWBRAIN_EMBEDDING_API_KEY;
});

describe("brain embedding client", () => {
  it("returns null when the embedding model is unset", () => {
    const warn = vi.fn();
    const client = createEmbeddingClient({
      config: makeConfig({ embeddingModel: "" }),
      log: { warn },
    });

    expect(client).toBeNull();
    expect(warn).toHaveBeenCalledWith(
      "[brain] Embedding model is unset; learned retrieval is disabled until init/configuration is complete",
    );
  });

  it("defaults ollama embeddings to the local OpenAI-compatible endpoint without auth", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ data: [{ embedding: [0.1, 0.2, 0.3] }] }),
    }));
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const config = makeConfig({
      embeddingProvider: "ollama",
      embeddingModel: "bge-large:latest",
      embeddingBaseUrl: "",
    });
    const getApiKey = vi.fn(async () => "should-not-be-used");
    const client = createEmbeddingClient({ config, getApiKey });

    expect(resolveEmbeddingBaseUrl(config)).toBe("http://127.0.0.1:11434/v1");
    expect(resolveEmbeddingAuthMode(config)).toBe("none");

    const embedding = Array.from((await client?.("hello world")) ?? []);
    expect(embedding).toHaveLength(3);
    expect(embedding[0]).toBeCloseTo(0.1);
    expect(embedding[1]).toBeCloseTo(0.2);
    expect(embedding[2]).toBeCloseTo(0.3);
    expect(getApiKey).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:11434/v1/embeddings",
      expect.objectContaining({
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
      }),
    );
  });

  it("treats an explicit localhost base URL as local and omits auth headers", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ data: [{ embedding: [1, 2, 3] }] }),
    }));
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const getApiKey = vi.fn(async () => "should-not-be-used");
    const client = createEmbeddingClient({
      config: makeConfig({
        embeddingProvider: "openai",
        embeddingModel: "bge-large:latest",
        embeddingBaseUrl: "http://localhost:11434/v1",
      }),
      getApiKey,
    });

    await client?.("hello localhost");

    expect(getApiKey).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:11434/v1/embeddings",
      expect.objectContaining({
        headers: {
          "content-type": "application/json",
        },
      }),
    );
  });

  it("uses OpenAI auth for remote endpoints by default", async () => {
    process.env.OPENAI_API_KEY = "remote-openai-key";
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ data: [{ embedding: [0.4, 0.5] }] }),
    }));
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const config = makeConfig({
      embeddingProvider: "openai",
      embeddingModel: "text-embedding-3-small",
      embeddingBaseUrl: "https://api.openai.com/v1",
    });
    const client = createEmbeddingClient({ config });

    expect(resolveEmbeddingAuthMode(config)).toBe("api_key");
    await client?.("remote call");

    expect(fetchMock).toHaveBeenCalledWith(
      "https://api.openai.com/v1/embeddings",
      expect.objectContaining({
        headers: {
          "content-type": "application/json",
          authorization: "Bearer remote-openai-key",
        },
      }),
    );
  });

  it("prefers OPENCLAWBRAIN_EMBEDDING_API_KEY when present", async () => {
    process.env.OPENCLAWBRAIN_EMBEDDING_API_KEY = "embedding-proxy-key";
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ data: [{ embedding: [9, 8, 7] }] }),
    }));
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    const getApiKey = vi.fn(async () => "other-key");
    const client = createEmbeddingClient({
      config: makeConfig({
        embeddingProvider: "ollama",
        embeddingModel: "bge-large:latest",
        embeddingBaseUrl: "https://embedding-proxy.example/v1",
      }),
      getApiKey,
    });

    await client?.("proxy call");

    expect(getApiKey).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledWith(
      "https://embedding-proxy.example/v1/embeddings",
      expect.objectContaining({
        headers: {
          "content-type": "application/json",
          authorization: "Bearer embedding-proxy-key",
        },
      }),
    );
  });
});
