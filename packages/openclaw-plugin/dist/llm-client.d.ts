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
export declare class FakeLlmClient implements LlmClient {
    private handler?;
    private responses;
    private attempts;
    constructor(options?: FakeLlmClientOptions);
    runJson<TOutput = unknown>(call: JsonLlmCall<TOutput>): Promise<unknown>;
}
export interface OpenAICompatibleLlmClientOptions {
    baseUrl: string;
    path?: string;
    fetchImpl?: typeof fetch;
    headers?: Record<string, string>;
}
export declare class OpenAICompatibleLlmClient implements LlmClient {
    private baseUrl;
    private path;
    private fetchImpl;
    private headers;
    constructor(options: OpenAICompatibleLlmClientOptions);
    runJson<TOutput = unknown>(call: JsonLlmCall<TOutput>): Promise<unknown>;
}
