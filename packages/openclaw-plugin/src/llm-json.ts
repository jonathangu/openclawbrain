import type { JsonLlmCall, LlmClient } from './llm-client.js';
import { hashText, redactText } from './redact.js';

export class JsonTimeoutError extends Error {
  constructor(message = 'LLM JSON call timed out') {
    super(message);
    this.name = 'JsonTimeoutError';
  }
}

export class JsonParseError extends Error {
  rawText?: string;

  constructor(message: string, rawText?: string) {
    super(message);
    this.name = 'JsonParseError';
    this.rawText = rawText;
  }
}

export class JsonValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'JsonValidationError';
  }
}

export type ValidationResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: string };

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

export async function runJsonWithValidation<T>(options: RunJsonWithValidationOptions<T>): Promise<JsonRunResult<T>> {
  const timeoutMs = Math.max(1, options.timeoutMs ?? options.call.timeoutMs ?? 1500);
  const maxAttempts = Math.max(1, options.maxAttempts ?? 1);
  const started = Date.now();
  const startedAt = new Date(started).toISOString();
  let lastError: Error | null = null;
  let lastRaw: unknown;
  let repaired = false;
  let validationStatus: JsonRunAudit['validationStatus'] = 'invalid';
  let validationError = '';
  let parseError = '';

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      const raw = await withTimeout(options.client.runJson(options.call), timeoutMs);
      lastRaw = raw;
      const parsed = normalizeJsonCandidate(raw);
      const valid = options.validate(parsed);
      if (valid.ok) {
        return {
          output: valid.value,
          rawOutput: raw,
          audit: buildAudit({
            call: options.call,
            startedAt,
            started,
            attempts: attempt,
            timeoutMs,
            validationStatus: repaired ? 'repaired' : 'valid',
            validationError,
            parseError,
            repaired,
            fallbackUsed: false,
          }),
        };
      }

      validationError = valid.error;
      lastError = new JsonValidationError(valid.error);

      if (options.repair) {
        const repairedValue = await options.repair(parsed, valid.error, buildAudit({
          call: options.call,
          startedAt,
          started,
          attempts: attempt,
          timeoutMs,
          validationStatus: 'invalid',
          validationError,
          parseError,
          repaired: false,
          fallbackUsed: false,
        }));
        const reparsed = normalizeJsonCandidate(repairedValue);
        const repairedValid = options.validate(reparsed);
        if (repairedValid.ok) {
          repaired = true;
          return {
            output: repairedValid.value,
            rawOutput: repairedValue,
            audit: buildAudit({
              call: options.call,
              startedAt,
              started,
              attempts: attempt,
              timeoutMs,
              validationStatus: 'repaired',
              validationError,
              parseError,
              repaired: true,
              fallbackUsed: false,
            }),
          };
        }
        validationError = repairedValid.error;
        lastError = new JsonValidationError(`repair failed validation: ${repairedValid.error}`);
      }
    } catch (error: any) {
      lastError = error instanceof Error ? error : new Error(String(error ?? 'unknown llm json error'));
      if (lastError instanceof JsonParseError) parseError = lastError.message;
      if (lastError instanceof JsonValidationError) validationError = lastError.message;
      if (lastError instanceof JsonTimeoutError) validationStatus = 'fallback';
    }
  }

  if (options.fallback && lastError) {
    return {
      output: await options.fallback(lastError, buildAudit({
        call: options.call,
        startedAt,
        started,
        attempts: maxAttempts,
        timeoutMs,
        validationStatus: 'fallback',
        validationError,
        parseError,
        repaired,
        fallbackUsed: true,
      })),
      rawOutput: lastRaw,
      audit: buildAudit({
        call: options.call,
        startedAt,
        started,
        attempts: maxAttempts,
        timeoutMs,
        validationStatus: 'fallback',
        validationError,
        parseError,
        repaired,
        fallbackUsed: true,
      }),
    };
  }

  throw lastError ?? new JsonValidationError('LLM JSON call failed without a captured error');
}

export function validateWithGuard<T>(value: unknown, guard: (value: unknown) => value is T, error = 'schema guard rejected value'): ValidationResult<T> {
  return guard(value)
    ? { ok: true, value }
    : { ok: false, error };
}

export async function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  return Promise.race<T>([
    promise,
    new Promise<T>((_, reject) => setTimeout(() => reject(new JsonTimeoutError()), timeoutMs)),
  ]);
}

export function normalizeJsonCandidate(value: unknown): unknown {
  if (typeof value === 'string') {
    const text = value.trim();
    try {
      return JSON.parse(text);
    } catch (error: any) {
      throw new JsonParseError(`invalid JSON string: ${error?.message ?? 'parse failed'}`, text);
    }
  }
  return value;
}

function buildAudit(args: {
  call: JsonLlmCall<any>;
  startedAt: string;
  started: number;
  attempts: number;
  timeoutMs: number;
  validationStatus: JsonRunAudit['validationStatus'];
  validationError?: string;
  parseError?: string;
  repaired: boolean;
  fallbackUsed: boolean;
}): JsonRunAudit {
  const finishedAt = new Date().toISOString();
  return {
    task: args.call.task,
    model: args.call.model,
    inputHash: hashText(JSON.stringify(args.call.input ?? null)),
    redactedInputSummary: redactText(JSON.stringify(args.call.input ?? null), 500),
    startedAt: args.startedAt,
    finishedAt,
    latencyMs: Date.now() - args.started,
    attempts: args.attempts,
    timeoutMs: args.timeoutMs,
    validationStatus: args.validationStatus,
    validationError: args.validationError || undefined,
    parseError: args.parseError || undefined,
    repaired: args.repaired,
    fallbackUsed: args.fallbackUsed,
  };
}
