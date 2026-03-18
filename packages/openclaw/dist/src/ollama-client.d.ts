export declare const DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/api";
export declare const DEFAULT_OLLAMA_TIMEOUT_MS = 30000;
export interface OllamaChatMessage {
    role: string;
    content: string;
}
export interface OllamaClientOptions {
    baseURL?: string;
    timeoutMs?: number;
    fetch?: OllamaFetch;
}
export interface OllamaRequestOptions {
    timeoutMs?: number;
}
export interface OllamaFetchRequestInit {
    method?: string;
    headers?: Record<string, string>;
    body?: string;
    signal?: AbortSignal;
}
export interface OllamaFetchResponse {
    ok: boolean;
    status: number;
    statusText: string;
    text(): Promise<string>;
}
export type OllamaFetch = (input: string, init?: OllamaFetchRequestInit) => Promise<OllamaFetchResponse>;
export declare class OllamaClientError extends Error {
    readonly endpoint: string;
    readonly status: number | null;
    readonly cause: unknown;
    constructor(message: string, options: {
        endpoint: string;
        status?: number | null;
        cause?: unknown;
    });
}
export declare class OllamaClient {
    #private;
    readonly baseURL: string;
    readonly timeoutMs: number;
    constructor(options?: OllamaClientOptions);
    chat(model: string, messages: readonly OllamaChatMessage[], options?: OllamaRequestOptions): Promise<string>;
    embed(model: string, input: string, options?: OllamaRequestOptions): Promise<number[]>;
}
export declare function createOllamaClient(options?: OllamaClientOptions): OllamaClient;
