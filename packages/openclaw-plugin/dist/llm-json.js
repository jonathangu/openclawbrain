import { hashText, redactText } from './redact.js';
export class JsonTimeoutError extends Error {
    constructor(message = 'LLM JSON call timed out') {
        super(message);
        this.name = 'JsonTimeoutError';
    }
}
export class JsonParseError extends Error {
    rawText;
    constructor(message, rawText) {
        super(message);
        this.name = 'JsonParseError';
        this.rawText = rawText;
    }
}
export class JsonValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = 'JsonValidationError';
    }
}
export async function runJsonWithValidation(options) {
    const timeoutMs = Math.max(1, options.timeoutMs ?? options.call.timeoutMs ?? 1500);
    const maxAttempts = Math.max(1, options.maxAttempts ?? 1);
    const started = Date.now();
    const startedAt = new Date(started).toISOString();
    let lastError = null;
    let lastRaw;
    let repaired = false;
    let validationStatus = 'invalid';
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
        }
        catch (error) {
            lastError = error instanceof Error ? error : new Error(String(error ?? 'unknown llm json error'));
            if (lastError instanceof JsonParseError)
                parseError = lastError.message;
            if (lastError instanceof JsonValidationError)
                validationError = lastError.message;
            if (lastError instanceof JsonTimeoutError)
                validationStatus = 'fallback';
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
export function validateWithGuard(value, guard, error = 'schema guard rejected value') {
    return guard(value)
        ? { ok: true, value }
        : { ok: false, error };
}
export async function withTimeout(promise, timeoutMs) {
    return Promise.race([
        promise,
        new Promise((_, reject) => setTimeout(() => reject(new JsonTimeoutError()), timeoutMs)),
    ]);
}
export function normalizeJsonCandidate(value) {
    if (typeof value === 'string') {
        const text = value.trim();
        try {
            return JSON.parse(text);
        }
        catch (error) {
            throw new JsonParseError(`invalid JSON string: ${error?.message ?? 'parse failed'}`, text);
        }
    }
    return value;
}
function buildAudit(args) {
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
