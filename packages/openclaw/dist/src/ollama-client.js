const DEFAULT_BASE_URL = "http://127.0.0.1:11434/api";
const DEFAULT_TIMEOUT_MS = 30_000;
export const DEFAULT_OLLAMA_BASE_URL = DEFAULT_BASE_URL;
export const DEFAULT_OLLAMA_TIMEOUT_MS = DEFAULT_TIMEOUT_MS;
export class OllamaClientError extends Error {
    endpoint;
    status;
    cause;
    constructor(message, options) {
        super(message);
        this.name = "OllamaClientError";
        this.endpoint = options.endpoint;
        this.status = options.status ?? null;
        this.cause = options.cause;
    }
}
export class OllamaClient {
    baseURL;
    timeoutMs;
    #fetch;
    constructor(options = {}) {
        this.baseURL = normalizeBaseURL(options.baseURL ?? DEFAULT_BASE_URL);
        this.timeoutMs = normalizeTimeoutMs(options.timeoutMs ?? DEFAULT_TIMEOUT_MS, "timeoutMs", "initialization");
        this.#fetch = resolveFetch(options.fetch);
    }
    async chat(model, messages, options = {}) {
        const normalizedModel = requireNonEmptyString(model, "model", "chat");
        const normalizedMessages = normalizeMessages(messages);
        const response = await this.#post("chat", {
            model: normalizedModel,
            messages: normalizedMessages,
            stream: false
        }, options);
        return readChatText(response);
    }
    async embed(model, input, options = {}) {
        const normalizedModel = requireNonEmptyString(model, "model", "embed");
        const normalizedInput = requireNonEmptyString(input, "input", "embed");
        const response = await this.#post("embed", {
            model: normalizedModel,
            input: normalizedInput
        }, options);
        return readEmbeddingVector(response);
    }
    async #post(endpoint, body, options) {
        const timeoutMs = normalizeTimeoutMs(options.timeoutMs ?? this.timeoutMs, "timeoutMs", endpoint);
        const url = `${this.baseURL}/${endpoint}`;
        const controller = new AbortController();
        const timeout = setTimeout(() => {
            controller.abort();
        }, timeoutMs);
        let response;
        try {
            response = await this.#fetch(url, {
                method: "POST",
                headers: {
                    accept: "application/json",
                    "content-type": "application/json"
                },
                body: JSON.stringify(body),
                signal: controller.signal
            });
        }
        catch (error) {
            if (controller.signal.aborted) {
                throw new OllamaClientError(`Ollama request to ${url} timed out after ${timeoutMs}ms.`, { endpoint: endpoint, cause: error });
            }
            throw new OllamaClientError(`Ollama request to ${url} failed: ${describeUnknownError(error)}.`, { endpoint: endpoint, cause: error });
        }
        finally {
            clearTimeout(timeout);
        }
        const text = await readResponseText(response, endpoint);
        if (!response.ok) {
            throw new OllamaClientError(formatHttpErrorMessage(url, response.status, response.statusText, text), { endpoint: endpoint, status: response.status });
        }
        if (text.trim().length === 0) {
            throw new OllamaClientError(`Ollama response from ${url} was empty.`, { endpoint: endpoint, status: response.status });
        }
        try {
            return JSON.parse(text);
        }
        catch (error) {
            throw new OllamaClientError(`Ollama response from ${url} was not valid JSON.`, { endpoint: endpoint, status: response.status, cause: error });
        }
    }
}
export function createOllamaClient(options = {}) {
    return new OllamaClient(options);
}
function resolveFetch(fetchFn) {
    if (typeof fetchFn === "function") {
        return fetchFn;
    }
    const globalFetch = globalThis.fetch;
    if (typeof globalFetch === "function") {
        return globalFetch;
    }
    throw new OllamaClientError("Ollama client requires a fetch implementation.", {
        endpoint: "initialization"
    });
}
function normalizeBaseURL(baseURL) {
    const normalized = requireNonEmptyString(baseURL, "baseURL", "initialization").replace(/\/+$/, "");
    let parsed;
    try {
        parsed = new URL(normalized);
    }
    catch (error) {
        throw new OllamaClientError(`Invalid Ollama baseURL: ${describeUnknownError(error)}.`, { endpoint: "initialization", cause: error });
    }
    if (parsed.pathname === "" || parsed.pathname === "/") {
        parsed.pathname = "/api";
    }
    parsed.pathname = parsed.pathname.replace(/\/+$/, "") || "/api";
    return parsed.toString().replace(/\/+$/, "");
}
function normalizeMessages(messages) {
    if (!Array.isArray(messages) || messages.length === 0) {
        throw new OllamaClientError("messages must contain at least one chat message.", {
            endpoint: "chat"
        });
    }
    return messages.map((message, index) => {
        if (!isRecord(message)) {
            throw new OllamaClientError(`messages[${index}] must be an object with role and content strings.`, {
                endpoint: "chat"
            });
        }
        return {
            role: requireNonEmptyString(message.role, `messages[${index}].role`, "chat"),
            content: requireString(message.content, `messages[${index}].content`, "chat")
        };
    });
}
function normalizeTimeoutMs(timeoutMs, fieldName, endpoint) {
    if (!Number.isInteger(timeoutMs) || timeoutMs <= 0) {
        throw new OllamaClientError(`${fieldName} must be a positive integer number of milliseconds.`, {
            endpoint
        });
    }
    return timeoutMs;
}
function readChatText(value) {
    if (!isRecord(value) || !isRecord(value.message) || typeof value.message.content !== "string") {
        throw new OllamaClientError("Ollama chat response did not include message.content.", {
            endpoint: "chat"
        });
    }
    return value.message.content;
}
function readEmbeddingVector(value) {
    if (!isRecord(value) || !Array.isArray(value.embeddings) || value.embeddings.length === 0) {
        throw new OllamaClientError("Ollama embed response did not include embeddings[0].", {
            endpoint: "embed"
        });
    }
    const [firstEmbedding] = value.embeddings;
    if (!Array.isArray(firstEmbedding) || firstEmbedding.some((item) => typeof item !== "number" || !Number.isFinite(item))) {
        throw new OllamaClientError("Ollama embed response embeddings[0] was not a numeric vector.", {
            endpoint: "embed"
        });
    }
    return [...firstEmbedding];
}
async function readResponseText(response, endpoint) {
    try {
        return await response.text();
    }
    catch (error) {
        throw new OllamaClientError("Failed to read Ollama response body.", {
            endpoint,
            status: response.status,
            cause: error
        });
    }
}
function formatHttpErrorMessage(url, status, statusText, text) {
    const details = summarizeErrorBody(text);
    const suffix = details === null ? "." : `: ${details}`;
    const normalizedStatusText = statusText.trim().length === 0 ? "" : ` ${statusText}`;
    return `Ollama request to ${url} failed with HTTP ${status}${normalizedStatusText}${suffix}`;
}
function summarizeErrorBody(text) {
    const normalized = text.trim();
    if (normalized.length === 0) {
        return null;
    }
    try {
        const parsed = JSON.parse(normalized);
        if (isRecord(parsed)) {
            if (typeof parsed.error === "string" && parsed.error.trim().length > 0) {
                return parsed.error.trim();
            }
            if (typeof parsed.message === "string" && parsed.message.trim().length > 0) {
                return parsed.message.trim();
            }
        }
    }
    catch {
        // Preserve plain-text bodies when the upstream error is not JSON.
    }
    return normalized.replace(/\s+/g, " ").slice(0, 240);
}
function requireNonEmptyString(value, fieldName, endpoint) {
    const normalized = requireString(value, fieldName, endpoint).trim();
    if (normalized.length === 0) {
        throw new OllamaClientError(`${fieldName} must be a non-empty string.`, {
            endpoint
        });
    }
    return normalized;
}
function requireString(value, fieldName, endpoint) {
    if (typeof value !== "string") {
        throw new OllamaClientError(`${fieldName} must be a string.`, {
            endpoint
        });
    }
    return value;
}
function describeUnknownError(error) {
    if (error instanceof Error && error.message.trim().length > 0) {
        return error.message;
    }
    return String(error);
}
function isRecord(value) {
    return typeof value === "object" && value !== null;
}
