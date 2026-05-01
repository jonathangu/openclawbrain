import type { JsonLlmCall, LlmClient } from './llm-client.js';
export declare class JsonTimeoutError extends Error {
    constructor(message?: string);
}
export declare class JsonParseError extends Error {
    rawText?: string;
    constructor(message: string, rawText?: string);
}
export declare class JsonValidationError extends Error {
    constructor(message: string);
}
export type ValidationResult<T> = {
    ok: true;
    value: T;
} | {
    ok: false;
    error: string;
};
export type JsonValidator<T> = (value: unknown) => ValidationResult<T>;
export interface JsonRunAudit {
    task: string;
    model: string;
    inputHash: string;
    redactedInputSummary: string;
    startedAt: string;
    finishedAt: string;
    latencyMs: number;
    attempts: number;
    timeoutMs: number;
    validationStatus: 'valid' | 'invalid' | 'repaired' | 'fallback';
    validationError?: string;
    parseError?: string;
    repaired: boolean;
    fallbackUsed: boolean;
}
export interface JsonRunResult<T> {
    output: T;
    audit: JsonRunAudit;
    rawOutput: unknown;
}
export interface RunJsonWithValidationOptions<T> {
    client: LlmClient;
    call: JsonLlmCall<T>;
    validate: JsonValidator<T>;
    repair?: (value: unknown, error: string, audit: JsonRunAudit) => unknown | Promise<unknown>;
    fallback?: (error: Error, audit: JsonRunAudit) => T | Promise<T>;
    timeoutMs?: number;
    maxAttempts?: number;
}
export declare function runJsonWithValidation<T>(options: RunJsonWithValidationOptions<T>): Promise<JsonRunResult<T>>;
export declare function validateWithGuard<T>(value: unknown, guard: (value: unknown) => value is T, error?: string): ValidationResult<T>;
export declare function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T>;
export declare function normalizeJsonCandidate(value: unknown): unknown;
