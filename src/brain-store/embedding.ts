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

export type EmbeddingAuthMode = "none" | "api_key";

export type EmbeddingConfigSummary = {
  baseUrl: string;
  authMode: EmbeddingAuthMode | "unknown";
  error: string | null;
};

function normalizeProvider(provider: string): string {
  return provider.trim().toLowerCase();
}

function trimTrailingSlashes(value: string): string {
  return value.replace(/\/+$/, "");
}

function resolveExplicitBaseUrl(config: OpenClawBrainRuntimeConfig): string | null {
  const explicit = config.embeddingBaseUrl.trim();
  return explicit ? trimTrailingSlashes(explicit) : null;
}

export function resolveEmbeddingBaseUrl(config: OpenClawBrainRuntimeConfig): string {
  const explicit = resolveExplicitBaseUrl(config);
  if (explicit) {
    return explicit;
  }

  const provider = normalizeProvider(config.embeddingProvider);
  if (provider === "openai" || provider === "openai-resp") {
    return "https://api.openai.com/v1";
  }
  if (provider === "ollama") {
    return "http://127.0.0.1:11434/v1";
  }

  throw new Error(`Unsupported embedding provider "${config.embeddingProvider}"`);
}

function isLoopbackHostname(hostname: string): boolean {
  const normalized = hostname.trim().toLowerCase();
  return normalized === "localhost"
    || normalized === "127.0.0.1"
    || normalized === "::1"
    || normalized === "[::1]";
}

export function isLocalEmbeddingBaseUrl(baseUrl: string): boolean {
  try {
    const parsed = new URL(baseUrl);
    return isLoopbackHostname(parsed.hostname);
  } catch {
    return false;
  }
}

export function resolveEmbeddingAuthMode(config: OpenClawBrainRuntimeConfig): EmbeddingAuthMode {
  const explicitApiKey = process.env.OPENCLAWBRAIN_EMBEDDING_API_KEY?.trim();
  if (explicitApiKey) {
    return "api_key";
  }

  const provider = normalizeProvider(config.embeddingProvider);
  if (provider === "ollama") {
    return "none";
  }

  const baseUrl = resolveEmbeddingBaseUrl(config);
  if (isLocalEmbeddingBaseUrl(baseUrl)) {
    return "none";
  }

  return "api_key";
}

export function describeEmbeddingConfig(config: OpenClawBrainRuntimeConfig): EmbeddingConfigSummary {
  try {
    return {
      baseUrl: resolveEmbeddingBaseUrl(config),
      authMode: resolveEmbeddingAuthMode(config),
      error: null,
    };
  } catch (error) {
    return {
      baseUrl: config.embeddingBaseUrl.trim(),
      authMode: "unknown",
      error: (error as Error).message,
    };
  }
}

async function resolveApiKey(
  config: OpenClawBrainRuntimeConfig,
  getApiKey?: (provider: string, model: string) => Promise<string | undefined>,
): Promise<string | undefined> {
  const explicitEmbeddingKey = process.env.OPENCLAWBRAIN_EMBEDDING_API_KEY?.trim();
  if (explicitEmbeddingKey) {
    return explicitEmbeddingKey;
  }

  if (resolveEmbeddingAuthMode(config) === "none") {
    return undefined;
  }

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
    const baseUrl = resolveEmbeddingBaseUrl(config);
    const headers: Record<string, string> = {
      "content-type": "application/json",
    };
    if (apiKey) {
      headers.authorization = `Bearer ${apiKey}`;
    }

    const response = await fetch(`${baseUrl}/embeddings`, {
      method: "POST",
      headers,
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
