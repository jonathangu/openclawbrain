import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
const OPENCLAWBRAIN_TEACHER_BASE_URL_ENV = "OPENCLAWBRAIN_TEACHER_BASE_URL";
const OPENCLAWBRAIN_EMBEDDER_BASE_URL_ENV = "OPENCLAWBRAIN_EMBEDDER_BASE_URL";
const OPENCLAWBRAIN_TEACHER_PROVIDER_ENV = "OPENCLAWBRAIN_TEACHER_PROVIDER";
const OPENCLAWBRAIN_TEACHER_MODEL_ENV = "OPENCLAWBRAIN_TEACHER_MODEL";
const OPENCLAWBRAIN_TEACHER_TIMEOUT_MS_ENV = "OPENCLAWBRAIN_TEACHER_TIMEOUT_MS";
const OPENCLAWBRAIN_TEACHER_MAX_PROMPT_CHARS_ENV = "OPENCLAWBRAIN_TEACHER_MAX_PROMPT_CHARS";
const OPENCLAWBRAIN_TEACHER_MAX_RESPONSE_CHARS_ENV = "OPENCLAWBRAIN_TEACHER_MAX_RESPONSE_CHARS";
const OPENCLAWBRAIN_TEACHER_MAX_OUTPUT_TOKENS_ENV = "OPENCLAWBRAIN_TEACHER_MAX_OUTPUT_TOKENS";
const OPENCLAWBRAIN_TEACHER_MAX_ARTIFACTS_ENV = "OPENCLAWBRAIN_TEACHER_MAX_ARTIFACTS";
const OPENCLAWBRAIN_TEACHER_MAX_INTERACTIONS_ENV = "OPENCLAWBRAIN_TEACHER_MAX_INTERACTIONS";
const OPENCLAWBRAIN_EMBEDDER_PROVIDER_ENV = "OPENCLAWBRAIN_EMBEDDER_PROVIDER";
const OPENCLAWBRAIN_EMBEDDER_MODEL_ENV = "OPENCLAWBRAIN_EMBEDDER_MODEL";
const DEFAULT_BASE_URL = "http://127.0.0.1:11434";
const DEFAULT_TEACHER_PROVIDER = "heuristic";
const DEFAULT_TEACHER_MODEL = "qwen3.5:9b";
const DEFAULT_EMBEDDER_PROVIDER = "ollama";
const DEFAULT_EMBEDDER_MODEL = "bge-large";
const OPENCLAWBRAIN_PROVIDER_DEFAULTS_CONTRACT = "openclawbrain_provider_defaults.v1";
const OPENCLAWBRAIN_PROVIDER_DEFAULTS_FILE = "provider-defaults.json";
const ALLOWED_TEACHER_PROVIDERS = ["heuristic", "ollama", "off"];
const ALLOWED_EMBEDDER_PROVIDERS = ["keywords", "ollama", "off"];
export function resolveOpenClawBrainProviderDefaultsPath(activationRoot) {
    return path.resolve(activationRoot, OPENCLAWBRAIN_PROVIDER_DEFAULTS_FILE);
}
export function readOpenClawBrainProviderDefaults(activationRoot) {
    const warnings = [];
    const defaultsPath = resolveOpenClawBrainProviderDefaultsPath(activationRoot);
    if (!existsSync(defaultsPath)) {
        return {
            defaults: null,
            warnings
        };
    }
    let parsed;
    try {
        parsed = JSON.parse(readFileSync(defaultsPath, "utf8"));
    }
    catch (error) {
        warnings.push(`provider defaults file is unreadable; ignoring ${defaultsPath}: ${describeUnknownError(error)}`);
        return {
            defaults: null,
            warnings
        };
    }
    if (!isRecord(parsed) || parsed.contract !== OPENCLAWBRAIN_PROVIDER_DEFAULTS_CONTRACT) {
        warnings.push(`provider defaults file must use contract ${OPENCLAWBRAIN_PROVIDER_DEFAULTS_CONTRACT}; ignoring ${defaultsPath}`);
        return {
            defaults: null,
            warnings
        };
    }
    return {
        defaults: parsed,
        warnings
    };
}
export function resolveOpenClawBrainProviderConfigNotes(config) {
    const notes = [
        `teacher_base_url=${config.teacherBaseUrl}`,
        `embedder_base_url=${config.embedderBaseUrl}`,
        `teacher_provider=${config.teacher.provider}`,
        `teacher_model=${config.teacher.model}`,
        `embedder_provider=${config.embedder.provider}`,
        `embedder_model=${config.embedder.model}`
    ];
    const optionalTeacherNotes = [
        ["teacher_timeout_ms", config.teacher.timeoutMs],
        ["teacher_max_prompt_chars", config.teacher.maxPromptChars],
        ["teacher_max_response_chars", config.teacher.maxResponseChars],
        ["teacher_max_output_tokens", config.teacher.maxOutputTokens],
        ["teacher_max_artifacts", config.teacher.maxArtifactsPerExport],
        ["teacher_max_interactions", config.teacher.maxInteractionsPerExport]
    ];
    for (const [name, value] of optionalTeacherNotes) {
        if (value !== undefined) {
            notes.push(`${name}=${value}`);
        }
    }
    return notes;
}
export function readOpenClawBrainProviderConfig(env = process.env) {
    return readOpenClawBrainProviderConfigFromSources({ env });
}
export function readOpenClawBrainProviderConfigFromSources(input = {}) {
    const env = input.env ?? process.env;
    const warnings = [];
    const defaultsResult = input.defaults !== undefined
        ? { defaults: input.defaults, warnings: [] }
        : input.activationRoot === undefined || input.activationRoot === null
            ? { defaults: null, warnings: [] }
            : readOpenClawBrainProviderDefaults(input.activationRoot);
    warnings.push(...defaultsResult.warnings);
    const resolvedDefaults = resolveProviderDefaults(defaultsResult.defaults, warnings);
    return {
        teacherBaseUrl: readBaseUrlEnv(env, OPENCLAWBRAIN_TEACHER_BASE_URL_ENV, resolvedDefaults.teacherBaseUrl, warnings),
        embedderBaseUrl: readBaseUrlEnv(env, OPENCLAWBRAIN_EMBEDDER_BASE_URL_ENV, resolvedDefaults.embedderBaseUrl, warnings),
        teacher: {
            provider: readProviderEnv(env, OPENCLAWBRAIN_TEACHER_PROVIDER_ENV, ALLOWED_TEACHER_PROVIDERS, resolvedDefaults.teacherProvider, warnings),
            model: readModelEnv(env, OPENCLAWBRAIN_TEACHER_MODEL_ENV, resolvedDefaults.teacherModel ?? DEFAULT_TEACHER_MODEL, warnings),
            ...readTeacherBudgetEnv(env, warnings, resolvedDefaults)
        },
        embedder: {
            provider: readProviderEnv(env, OPENCLAWBRAIN_EMBEDDER_PROVIDER_ENV, ALLOWED_EMBEDDER_PROVIDERS, resolvedDefaults.embedderProvider, warnings),
            model: readModelEnv(env, OPENCLAWBRAIN_EMBEDDER_MODEL_ENV, resolvedDefaults.embedderModel ?? DEFAULT_EMBEDDER_MODEL, warnings)
        },
        warnings
    };
}
function readBaseUrlEnv(env, name, fallback, warnings) {
    const raw = env[name];
    if (raw === undefined) {
        return fallback;
    }
    const value = normalizeOptionalEnvString(raw);
    if (value === undefined) {
        warnings.push(`${name} must be a valid http(s) URL when set; using default ${fallback}`);
        return fallback;
    }
    let parsed;
    try {
        parsed = new URL(value);
    }
    catch {
        warnings.push(`${name} must be a valid http(s) URL when set; using default ${fallback}`);
        return fallback;
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
        warnings.push(`${name} must use http or https when set; using default ${fallback}`);
        return fallback;
    }
    return parsed.toString().replace(/\/+$/, "");
}
function readProviderEnv(env, name, allowedValues, fallback, warnings) {
    const raw = env[name];
    if (raw === undefined) {
        return fallback;
    }
    const value = normalizeOptionalEnvString(raw);
    if (value === undefined) {
        warnings.push(`${name} must be one of ${allowedValues.join("|")}; using default ${fallback}`);
        return fallback;
    }
    if (allowedValues.includes(value)) {
        return value;
    }
    warnings.push(`${name} must be one of ${allowedValues.join("|")}; using default ${fallback}`);
    return fallback;
}
function readModelEnv(env, name, fallback, warnings) {
    const raw = env[name];
    if (raw === undefined) {
        return fallback;
    }
    const value = normalizeOptionalEnvString(raw);
    if (value !== undefined) {
        return value;
    }
    warnings.push(`${name} must be a non-empty string when set; using default ${fallback}`);
    return fallback;
}
function readTeacherBudgetEnv(env, warnings, defaults) {
    return {
        ...readOptionalPositiveIntegerEnv(env, OPENCLAWBRAIN_TEACHER_TIMEOUT_MS_ENV, "timeoutMs", warnings, defaults.teacherTimeoutMs),
        ...readOptionalPositiveIntegerEnv(env, OPENCLAWBRAIN_TEACHER_MAX_PROMPT_CHARS_ENV, "maxPromptChars", warnings, defaults.teacherMaxPromptChars),
        ...readOptionalPositiveIntegerEnv(env, OPENCLAWBRAIN_TEACHER_MAX_RESPONSE_CHARS_ENV, "maxResponseChars", warnings, defaults.teacherMaxResponseChars),
        ...readOptionalPositiveIntegerEnv(env, OPENCLAWBRAIN_TEACHER_MAX_OUTPUT_TOKENS_ENV, "maxOutputTokens", warnings, defaults.teacherMaxOutputTokens),
        ...readOptionalPositiveIntegerEnv(env, OPENCLAWBRAIN_TEACHER_MAX_ARTIFACTS_ENV, "maxArtifactsPerExport", warnings, defaults.teacherMaxArtifactsPerExport),
        ...readOptionalPositiveIntegerEnv(env, OPENCLAWBRAIN_TEACHER_MAX_INTERACTIONS_ENV, "maxInteractionsPerExport", warnings, defaults.teacherMaxInteractionsPerExport)
    };
}
function readOptionalPositiveIntegerEnv(env, name, propertyName, warnings, fallback) {
    const raw = env[name];
    if (raw === undefined) {
        if (fallback === undefined) {
            return {};
        }
        return {
            [propertyName]: fallback
        };
    }
    const value = normalizeOptionalEnvString(raw);
    if (value === undefined || /^[0-9]+$/u.test(value) === false) {
        warnings.push(`${name} must be a positive integer when set; leaving it unset`);
        return {};
    }
    const parsed = Number.parseInt(value, 10);
    if (!Number.isInteger(parsed) || parsed <= 0) {
        warnings.push(`${name} must be a positive integer when set; leaving it unset`);
        return {};
    }
    return {
        [propertyName]: parsed
    };
}
function normalizeOptionalEnvString(value) {
    const normalized = value?.trim();
    if (normalized === undefined || normalized.length === 0) {
        return undefined;
    }
    return normalized;
}
function resolveProviderDefaults(defaults, warnings) {
    if (defaults === null) {
        return {
            teacherBaseUrl: DEFAULT_BASE_URL,
            embedderBaseUrl: DEFAULT_BASE_URL,
            teacherProvider: DEFAULT_TEACHER_PROVIDER,
            teacherModel: null,
            teacherTimeoutMs: undefined,
            teacherMaxPromptChars: undefined,
            teacherMaxResponseChars: undefined,
            teacherMaxOutputTokens: undefined,
            teacherMaxArtifactsPerExport: undefined,
            teacherMaxInteractionsPerExport: undefined,
            embedderProvider: DEFAULT_EMBEDDER_PROVIDER,
            embedderModel: DEFAULT_EMBEDDER_MODEL
        };
    }
    const teacherDefaults = isRecord(defaults.teacher) ? defaults.teacher : null;
    const embedderDefaults = isRecord(defaults.embedder) ? defaults.embedder : null;
    const teacherProvider = readDefaultProvider(teacherDefaults?.provider, "teacher.provider", ALLOWED_TEACHER_PROVIDERS, DEFAULT_TEACHER_PROVIDER, warnings);
    const embedderProvider = readDefaultProvider(embedderDefaults?.provider, "embedder.provider", ALLOWED_EMBEDDER_PROVIDERS, DEFAULT_EMBEDDER_PROVIDER, warnings);
    return {
        teacherBaseUrl: readDefaultBaseUrl(defaults.teacherBaseUrl, "teacherBaseUrl", warnings),
        embedderBaseUrl: readDefaultBaseUrl(defaults.embedderBaseUrl, "embedderBaseUrl", warnings),
        teacherProvider,
        teacherModel: teacherProvider === "ollama"
            ? readDefaultModel(teacherDefaults?.model, "teacher.model", warnings)
            : null,
        teacherTimeoutMs: readDefaultPositiveInteger(teacherDefaults?.timeoutMs, "teacher.timeoutMs", warnings),
        teacherMaxPromptChars: readDefaultPositiveInteger(teacherDefaults?.maxPromptChars, "teacher.maxPromptChars", warnings),
        teacherMaxResponseChars: readDefaultPositiveInteger(teacherDefaults?.maxResponseChars, "teacher.maxResponseChars", warnings),
        teacherMaxOutputTokens: readDefaultPositiveInteger(teacherDefaults?.maxOutputTokens, "teacher.maxOutputTokens", warnings),
        teacherMaxArtifactsPerExport: readDefaultPositiveInteger(teacherDefaults?.maxArtifactsPerExport, "teacher.maxArtifactsPerExport", warnings),
        teacherMaxInteractionsPerExport: readDefaultPositiveInteger(teacherDefaults?.maxInteractionsPerExport, "teacher.maxInteractionsPerExport", warnings),
        embedderProvider,
        embedderModel: embedderProvider === "off"
            ? null
            : readDefaultModel(embedderDefaults?.model, "embedder.model", warnings) ?? DEFAULT_EMBEDDER_MODEL
    };
}
function readDefaultBaseUrl(value, fieldName, warnings) {
    if (typeof value !== "string") {
        return DEFAULT_BASE_URL;
    }
    const normalized = normalizeOptionalEnvString(value);
    if (normalized === undefined) {
        warnings.push(`provider defaults ${fieldName} must be a valid http(s) URL; using ${DEFAULT_BASE_URL}`);
        return DEFAULT_BASE_URL;
    }
    let parsed;
    try {
        parsed = new URL(normalized);
    }
    catch {
        warnings.push(`provider defaults ${fieldName} must be a valid http(s) URL; using ${DEFAULT_BASE_URL}`);
        return DEFAULT_BASE_URL;
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
        warnings.push(`provider defaults ${fieldName} must use http or https; using ${DEFAULT_BASE_URL}`);
        return DEFAULT_BASE_URL;
    }
    return parsed.toString().replace(/\/+$/, "");
}
function readDefaultProvider(value, fieldName, allowedValues, fallback, warnings) {
    if (typeof value === "string" && allowedValues.includes(value)) {
        return value;
    }
    if (value !== undefined) {
        warnings.push(`provider defaults ${fieldName} must be one of ${allowedValues.join("|")}; using ${fallback}`);
    }
    return fallback;
}
function readDefaultModel(value, fieldName, warnings) {
    if (value === null || value === undefined) {
        return null;
    }
    if (typeof value === "string") {
        const normalized = normalizeOptionalEnvString(value);
        if (normalized !== undefined) {
            return normalized;
        }
    }
    warnings.push(`provider defaults ${fieldName} must be a non-empty string when set; leaving it unset`);
    return null;
}
function readDefaultPositiveInteger(value, fieldName, warnings) {
    if (value === undefined || value === null) {
        return undefined;
    }
    if (typeof value === "number" && Number.isInteger(value) && value > 0) {
        return value;
    }
    warnings.push(`provider defaults ${fieldName} must be a positive integer when set; leaving it unset`);
    return undefined;
}
function isRecord(value) {
    return typeof value === "object" && value !== null;
}
function describeUnknownError(error) {
    return error instanceof Error ? error.message : String(error);
}
//# sourceMappingURL=provider-config.js.map