import type { OpenClawBrainRuntimeConfig } from "../db/config.js";

export type BrainEmbeddingFn = (text: string) => Promise<Float32Array>;

export type BrainEmbeddingOptions = {
  config: OpenClawBrainRuntimeConfig;
  getApiKey?: (provider: string, model: string) => Promise<string | undefined>;
  log?: {
    debug?: (msg: string) => void;
    warn?: (msg: string) => void;
  };
};

function normalizeProvider(provider: string): string {
  return provider.trim().toLowerCase();
}

function resolveBaseUrl(config: OpenClawBrainRuntimeConfig): string {
  const explicit = config.embeddingBaseUrl.trim();
  if (explicit) {
    return explicit.replace(/\/+$/, "");
  }

  const provider = normalizeProvider(config.embeddingProvider);
  if (provider === "openai" || provider === "openai-resp") {
    return "https://api.openai.com/v1";
  }

  throw new Error(`Unsupported embedding provider "${config.embeddingProvider}"`);
}

async function resolveApiKey(
  config: OpenClawBrainRuntimeConfig,
  getApiKey?: (provider: string, model: string) => Promise<string | undefined>,
): Promise<string> {
  if (getApiKey) {
    const key = await getApiKey(config.embeddingProvider, config.embeddingModel);
    if (key) {
      return key;
    }
  }

  if (normalizeProvider(config.embeddingProvider) === "openai" || normalizeProvider(config.embeddingProvider) === "openai-resp") {
    const envKey = process.env.OPENAI_API_KEY?.trim();
    if (envKey) {
      return envKey;
    }
  }

  throw new Error(`Missing API key for embedding provider "${config.embeddingProvider}"`);
}

export function hasEmbeddingConfiguration(config: OpenClawBrainRuntimeConfig): boolean {
  return config.embeddingModel.trim().length > 0;
}

export function createEmbeddingClient(options: BrainEmbeddingOptions): BrainEmbeddingFn | null {
  const { config, getApiKey, log } = options;
  if (!hasEmbeddingConfiguration(config)) {
    log?.warn?.("[brain] Embedding model is unset; learned retrieval is disabled until init/configuration is complete");
    return null;
  }

  return async (text: string): Promise<Float32Array> => {
    const apiKey = await resolveApiKey(config, getApiKey);
    const baseUrl = resolveBaseUrl(config);
    const response = await fetch(`${baseUrl}/embeddings`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: config.embeddingModel,
        input: text,
      }),
    });

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`Embedding request failed (${response.status}): ${body.slice(0, 200)}`);
    }

    const payload = await response.json() as {
      data?: Array<{ embedding?: number[] }>;
    };
    const embedding = payload.data?.[0]?.embedding;
    if (!Array.isArray(embedding) || embedding.length === 0) {
      throw new Error("Embedding response did not include a vector");
    }

    return new Float32Array(embedding.map((value) => Number(value)));
  };
}
