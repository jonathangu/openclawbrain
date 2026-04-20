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
  constructor(
    private model: string = "gemma4:31b-q4_k_m",
    private host: string = "http://localhost:11434",
    private timeoutMs: number = 120_000,
  ) {}

  async chat(turns: ChatTurn[]): Promise<ChatResponse> {
    const t0 = Date.now();
    const res = await fetch(`${this.host}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: this.model,
        messages: turns,
        stream: false,
        options: { temperature: 0.2 },
      }),
      signal: AbortSignal.timeout(this.timeoutMs),
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
  }
}

export function estimateTokens(text: string): number {
  if (!text.trim()) return 0;
  return Math.ceil(text.split(/\s+/).filter(Boolean).length * 1.3);
}
