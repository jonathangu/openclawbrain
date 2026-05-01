export interface JsonLlmCall<TOutput = unknown> {
  task: string;
  model: string;
  systemPrompt: string;
  input: unknown;
  schema?: unknown;
  temperature?: number;
  maxTokens?: number;
  timeoutMs?: number;
  metadata?: Record<string, unknown>;
}

export interface LlmClient {
  runJson<TOutput = unknown>(call: JsonLlmCall<TOutput>): Promise<unknown>;
}

export interface FakeLlmClientOptions {
  handler?: (call: JsonLlmCall<any>, attempt: number) => unknown | Promise<unknown>;
  responses?: unknown[];
}

export class FakeLlmClient implements LlmClient {
  private handler?: FakeLlmClientOptions['handler'];
  private responses: unknown[];
  private attempts = 0;

  constructor(options: FakeLlmClientOptions = {}) {
    this.handler = options.handler;
    this.responses = [...(options.responses ?? [])];
  }

  async runJson<TOutput = unknown>(call: JsonLlmCall<TOutput>): Promise<unknown> {
    this.attempts += 1;
    if (this.handler) return this.handler(call, this.attempts);
    if (this.responses.length === 0) throw new Error('FakeLlmClient has no queued response');
    return this.responses.shift();
  }
}

export interface OpenAICompatibleLlmClientOptions {
  baseUrl: string;
  path?: string;
  fetchImpl?: typeof fetch;
  headers?: Record<string, string>;
}

export class OpenAICompatibleLlmClient implements LlmClient {
  private baseUrl: string;
  private path: string;
  private fetchImpl: typeof fetch;
  private headers: Record<string, string>;

  constructor(options: OpenAICompatibleLlmClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, '');
    this.path = options.path ?? '/chat/completions';
    this.fetchImpl = options.fetchImpl ?? fetch;
    this.headers = options.headers ?? {};
  }

  async runJson<TOutput = unknown>(call: JsonLlmCall<TOutput>): Promise<unknown> {
    const response = await this.fetchImpl(`${this.baseUrl}${this.path}`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...this.headers,
      },
      body: JSON.stringify({
        model: call.model,
        temperature: call.temperature,
        max_tokens: call.maxTokens,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: call.systemPrompt },
          {
            role: 'user',
            content: JSON.stringify({
              task: call.task,
              input: call.input,
              schema: call.schema ?? null,
            }),
          },
        ],
      }),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => '');
      throw new Error(`OpenAI-compatible JSON call failed: ${response.status} ${response.statusText} ${text}`.trim());
    }

    const payload = await response.json();
    const content = payload?.choices?.[0]?.message?.content;
    if (typeof content !== 'string') {
      throw new Error('OpenAI-compatible JSON call returned no message content');
    }
    return content;
  }
}
