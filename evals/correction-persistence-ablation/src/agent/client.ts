import { Agent, fetch as undiciFetch } from "undici";

export interface ChatTurn {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  content: string;
  response_tokens: number;
  latency_ms: number;
}

export interface AgentClient {
  chat(turns: ChatTurn[]): Promise<ChatResponse>;
}

export class OllamaClient implements AgentClient {
  private dispatcher: Agent;

  constructor(
    private model: string = "qwen2.5:32b-instruct",
    private host: string = "http://localhost:11434",
    private timeoutMs: number = 120_000,
    private maxRetries: number = 2,
    private maxOutputTokens: number = 128,
  ) {
    this.dispatcher = new Agent({
      headersTimeout: this.timeoutMs,
      bodyTimeout: this.timeoutMs,
      connect: { timeout: Math.min(this.timeoutMs, 30_000) },
    });
  }

  async chat(turns: ChatTurn[]): Promise<ChatResponse> {
    const t0 = Date.now();
    let attempt = 0;
    let lastError: unknown;

    while (attempt <= this.maxRetries) {
      try {
        const res = await undiciFetch(`${this.host}/api/chat`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: this.model,
            messages: turns,
            stream: false,
            keep_alive: "30m",
            options: { temperature: 0.2, num_predict: this.maxOutputTokens },
          }),
          signal: AbortSignal.timeout(this.timeoutMs),
          dispatcher: this.dispatcher,
        });
        if (!res.ok) {
          throw new Error(`Ollama error ${res.status}: ${await res.text()}`);
        }
        const body = (await res.json()) as {
          message: { content: string };
          eval_count?: number;
        };
        return {
          content: body.message.content,
          response_tokens: body.eval_count ?? estimateTokens(body.message.content),
          latency_ms: Date.now() - t0,
        };
      } catch (error) {
        lastError = error;
        if (attempt >= this.maxRetries || !isRetryableOllamaError(error)) {
          throw error;
        }
        await sleep(1_500 * (attempt + 1));
        attempt += 1;
      }
    }

    throw lastError instanceof Error ? lastError : new Error(String(lastError));
  }
}

function isRetryableOllamaError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const cause = (error as Error & { cause?: { code?: string } }).cause;
  return (
    error.name === "TimeoutError" ||
    error.message.includes("fetch failed") ||
    cause?.code === "UND_ERR_HEADERS_TIMEOUT" ||
    cause?.code === "UND_ERR_BODY_TIMEOUT"
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function estimateTokens(text: string): number {
  if (!text.trim()) return 0;
  return Math.ceil(text.split(/\s+/).filter(Boolean).length * 1.3);
}
