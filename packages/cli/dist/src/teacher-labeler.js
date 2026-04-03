import { CONTRACT_IDS, checksumJsonPayload, validateTeacherSupervisionArtifact } from "@openclawbrain/contracts";
import { buildNormalizedEventDedupId } from "@openclawbrain/event-export";
import { createServeTimeDecisionMatcher } from "./teacher-decision-match.js";
const FEEDBACK_KINDS = new Set(["correction", "teaching", "approval", "suppression"]);
const OLLAMA_ROUTE_DECISION_STREAM = "openclaw/learning-spine/serve-time-route-decisions";
const DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434";
const DEFAULT_OLLAMA_MODEL = "qwen3.5:9b";
const DEFAULT_OLLAMA_TIMEOUT_MS = 2_000;
const DEFAULT_OLLAMA_MAX_PROMPT_CHARS = 4_000;
const DEFAULT_OLLAMA_MAX_RESPONSE_CHARS = 4_000;
const DEFAULT_OLLAMA_MAX_OUTPUT_TOKENS = 192;
const DEFAULT_OLLAMA_MAX_ARTIFACTS_PER_EXPORT = 2;
const DEFAULT_OLLAMA_MAX_INTERACTIONS = 2;
const DEFAULT_OLLAMA_MAX_USER_MESSAGE_CHARS = 480;
const DEFAULT_OLLAMA_MAX_CONTEXT_IDS = 10;
function normalizeOptionalString(value) {
    if (typeof value !== "string") {
        return undefined;
    }
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : undefined;
}
function normalizePositiveInteger(value, fieldName, fallback) {
    const resolved = value ?? fallback;
    if (!Number.isInteger(resolved) || resolved <= 0) {
        throw new Error(`${fieldName} must be a positive integer`);
    }
    return resolved;
}
function normalizeBaseUrl(value) {
    const normalized = normalizeOptionalString(value) ?? DEFAULT_OLLAMA_BASE_URL;
    return normalized.replace(/\/+$/u, "");
}
function truncate(value, maxChars) {
    if (value.length <= maxChars) {
        return value;
    }
    if (maxChars <= 1) {
        return value.slice(0, Math.max(0, maxChars));
    }
    return `${value.slice(0, Math.max(0, maxChars - 3))}...`;
}
function normalizeContent(value) {
    return value.replace(/\s+/gu, " ").trim();
}
function uniqueStrings(values) {
    return [...new Set(values)];
}
function newestIsoTimestamp(values) {
    return [...values].sort((left, right) => Date.parse(right) - Date.parse(left))[0] ?? "1970-01-01T00:00:00.000Z";
}
function slugifyTeacherIdentityFragment(value) {
    return value
        .toLowerCase()
        .replace(/[^a-z0-9]+/gu, "-")
        .replace(/^-+|-+$/gu, "");
}
function buildTeacherPrincipal(sessionId, teacherIdentity) {
    return {
        teacherIdentity,
        teacherRole: "assistant",
        teacherAuthority: "background",
        priorityClass: "low",
        principalScope: {
            kind: "session",
            sessionId,
            scopeKey: `session:${sessionId}|teacher:${teacherIdentity}`
        }
    };
}
function isFeedbackKind(value) {
    return FEEDBACK_KINDS.has(value);
}
function extractJsonCandidate(value) {
    const trimmed = value.trim();
    if (trimmed.length === 0) {
        return null;
    }
    const firstObjectStart = trimmed.indexOf("{");
    const lastObjectEnd = trimmed.lastIndexOf("}");
    if (firstObjectStart >= 0 && lastObjectEnd > firstObjectStart) {
        return trimmed.slice(firstObjectStart, lastObjectEnd + 1);
    }
    const firstArrayStart = trimmed.indexOf("[");
    const lastArrayEnd = trimmed.lastIndexOf("]");
    if (firstArrayStart >= 0 && lastArrayEnd > firstArrayStart) {
        return trimmed.slice(firstArrayStart, lastArrayEnd + 1);
    }
    return null;
}
function parseTeacherLabelerResponse(response) {
    const candidate = extractJsonCandidate(response);
    if (candidate === null) {
        return [];
    }
    try {
        const parsed = JSON.parse(candidate);
        if (Array.isArray(parsed)) {
            return parsed;
        }
        if (parsed !== null && typeof parsed === "object" && Array.isArray(parsed.labels)) {
            return parsed.labels;
        }
    }
    catch {
        return [];
    }
    return [];
}
function buildPromptPayload(candidates, config) {
    return {
        task: "teacher_supervision_labeling",
        maxLabels: config.maxArtifactsPerExport,
        allowedKinds: ["correction", "teaching", "approval", "suppression"],
        interactions: candidates.map((candidate) => ({
            interactionEventId: candidate.interaction.eventId,
            createdAt: candidate.interaction.createdAt,
            userMessage: truncate(candidate.userMessage, config.maxUserMessageChars),
            routeDecision: {
                usedLearnedRouteFn: candidate.decision.usedLearnedRouteFn,
                fallbackReason: candidate.decision.fallbackReason,
                selectedContextCount: candidate.decision.actualBudget.selectedCount,
                kernelContextCount: candidate.decision.kernelContextCount,
                brainContextCount: candidate.decision.brainContextCount,
                chosenContextIds: candidate.decision.chosenContextIds.slice(0, config.maxContextIdsPerDecision),
                selectedBrainContextIds: candidate.decision.selectedBrainContextIds.slice(0, config.maxContextIdsPerDecision),
                selectedKernelContextIds: candidate.decision.selectedKernelContextIds.slice(0, config.maxContextIdsPerDecision)
            }
        }))
    };
}
function buildPrompt(candidates, config) {
    const payload = JSON.stringify(buildPromptPayload(candidates, config));
    return [
        "You create background teacher supervision artifacts for OpenClawBrain.",
        "Return JSON only.",
        'Use this shape: {"labels":[{"interactionEventId":"...","kind":"correction|teaching|approval|suppression","content":"..."}]}.',
        `Emit at most ${config.maxArtifactsPerExport} labels.`,
        "Only emit concise, reusable supervision grounded in the provided interaction and route decision.",
        "Skip empty, hedged, or duplicate labels.",
        payload
    ].join("\n");
}
function isTeacherLabelerCandidateInteraction(interaction) {
    return interaction.kind === "memory_compiled" || interaction.kind === "message_delivered";
}
function collectCandidates(input, config) {
    const decisions = [...(input.serveTimeDecisions ?? [])].sort((left, right) => Date.parse(right.recordedAt) - Date.parse(left.recordedAt));
    const matchServeTimeDecision = createServeTimeDecisionMatcher(decisions);
    return input.normalizedEventExport.interactionEvents
        .filter((interaction) => isTeacherLabelerCandidateInteraction(interaction))
        .sort((left, right) => Date.parse(right.createdAt) - Date.parse(left.createdAt))
        .map((interaction) => {
        const decision = matchServeTimeDecision(interaction);
        const userMessage = normalizeOptionalString(decision?.userMessage);
        if (decision === null || userMessage === undefined) {
            return null;
        }
        return {
            interaction,
            decision,
            userMessage
        };
    })
        .filter((candidate) => candidate !== null)
        .slice(0, config.maxInteractionsPerExport);
}
function fitCandidatesToPromptBudget(candidates, config) {
    const accepted = [];
    for (const candidate of candidates) {
        const nextAccepted = [...accepted, candidate];
        if (buildPrompt(nextAccepted, config).length > config.maxPromptChars) {
            break;
        }
        accepted.push(candidate);
    }
    return accepted;
}
function dedupExistingArtifactIds(artifacts) {
    return new Set((artifacts ?? []).map((artifact) => artifact.dedupId));
}
function buildGeneratedArtifact(input) {
    const newestSourceCreatedAt = newestIsoTimestamp([
        input.candidate.interaction.createdAt,
        input.candidate.decision.recordedAt,
        input.candidate.decision.turnCreatedAt ?? input.candidate.decision.recordedAt
    ].filter((value) => typeof value === "string" && value.length > 0));
    const ageMs = Math.max(0, Date.parse(input.observedAt) - Date.parse(newestSourceCreatedAt));
    const dedupId = checksumJsonPayload({
        provider: "ollama",
        teacherIdentity: input.teacherIdentity,
        interactionDedupId: buildNormalizedEventDedupId(input.candidate.interaction),
        kind: input.kind,
        content: normalizeContent(input.content)
    });
    const sourceEventIds = uniqueStrings([input.candidate.interaction.eventId, input.candidate.decision.turnCompileEventId ?? input.candidate.interaction.eventId].filter((value) => typeof value === "string" && value.length > 0));
    const artifact = {
        contract: CONTRACT_IDS.teacherSupervisionArtifact,
        artifactId: `teacher-${dedupId}`,
        dedupId,
        kind: input.kind,
        createdAt: input.observedAt,
        source: {
            runtimeOwner: "openclaw",
            sessionId: input.candidate.interaction.sessionId,
            channel: input.candidate.interaction.channel,
            sourceStreams: uniqueStrings([input.candidate.interaction.source.stream, OLLAMA_ROUTE_DECISION_STREAM]),
            eventRange: {
                start: input.eventRange.start,
                end: input.eventRange.end,
                count: input.eventRange.count
            },
            eventExportDigest: input.eventExportDigest
        },
        sourceEventIds,
        relatedInteractionId: input.candidate.interaction.eventId,
        principal: buildTeacherPrincipal(input.candidate.interaction.sessionId, input.teacherIdentity),
        content: normalizeContent(input.content),
        freshness: {
            status: ageMs <= input.staleAfterMs ? "fresh" : "stale",
            observedAt: input.observedAt,
            newestSourceCreatedAt,
            ageMs,
            staleAfterMs: input.staleAfterMs
        }
    };
    const validationErrors = validateTeacherSupervisionArtifact(artifact);
    if (validationErrors.length > 0) {
        throw new Error(`ollama teacher artifact is invalid: ${validationErrors.join("; ")}`);
    }
    return artifact;
}
function normalizeOllamaTeacherLabelerConfig(config) {
    const model = normalizeOptionalString(config.model) ?? DEFAULT_OLLAMA_MODEL;
    const teacherIdentity = normalizeOptionalString(config.teacherIdentity) ?? `openclaw/teacher/ollama/${slugifyTeacherIdentityFragment(model)}`;
    return {
        provider: "ollama",
        baseUrl: normalizeBaseUrl(config.baseUrl),
        model,
        timeoutMs: normalizePositiveInteger(config.timeoutMs, "teacherLabeler.timeoutMs", DEFAULT_OLLAMA_TIMEOUT_MS),
        maxPromptChars: normalizePositiveInteger(config.maxPromptChars, "teacherLabeler.maxPromptChars", DEFAULT_OLLAMA_MAX_PROMPT_CHARS),
        maxResponseChars: normalizePositiveInteger(config.maxResponseChars, "teacherLabeler.maxResponseChars", DEFAULT_OLLAMA_MAX_RESPONSE_CHARS),
        maxOutputTokens: normalizePositiveInteger(config.maxOutputTokens, "teacherLabeler.maxOutputTokens", DEFAULT_OLLAMA_MAX_OUTPUT_TOKENS),
        maxArtifactsPerExport: normalizePositiveInteger(config.maxArtifactsPerExport, "teacherLabeler.maxArtifactsPerExport", DEFAULT_OLLAMA_MAX_ARTIFACTS_PER_EXPORT),
        maxInteractionsPerExport: normalizePositiveInteger(config.maxInteractionsPerExport, "teacherLabeler.maxInteractionsPerExport", DEFAULT_OLLAMA_MAX_INTERACTIONS),
        maxUserMessageChars: normalizePositiveInteger(config.maxUserMessageChars, "teacherLabeler.maxUserMessageChars", DEFAULT_OLLAMA_MAX_USER_MESSAGE_CHARS),
        maxContextIdsPerDecision: normalizePositiveInteger(config.maxContextIdsPerDecision, "teacherLabeler.maxContextIdsPerDecision", DEFAULT_OLLAMA_MAX_CONTEXT_IDS),
        teacherIdentity,
        client: config.client ?? createHttpOllamaTeacherLabelerClient(normalizeBaseUrl(config.baseUrl))
    };
}
export function summarizeTeacherLabelerOpportunity(input, config) {
    const normalized = config === undefined || config === null || config.provider === "none"
        ? {
            enabled: false,
            maxPromptChars: DEFAULT_OLLAMA_MAX_PROMPT_CHARS,
            maxArtifactsPerExport: DEFAULT_OLLAMA_MAX_ARTIFACTS_PER_EXPORT,
            maxInteractionsPerExport: DEFAULT_OLLAMA_MAX_INTERACTIONS,
            maxUserMessageChars: DEFAULT_OLLAMA_MAX_USER_MESSAGE_CHARS,
            maxContextIdsPerDecision: DEFAULT_OLLAMA_MAX_CONTEXT_IDS
        }
        : {
            enabled: true,
            ...normalizeOllamaTeacherLabelerConfig(config)
        };
    const candidates = collectCandidates(input, normalized);
    if (candidates.length === 0) {
        return {
            enabled: normalized.enabled,
            candidateCount: 0,
            budgetedCandidateCount: 0,
            status: normalized.enabled ? "skipped" : "disabled",
            detail: "no_matching_interaction_text"
        };
    }
    const budgetedCandidates = fitCandidatesToPromptBudget(candidates, normalized);
    if (budgetedCandidates.length === 0) {
        return {
            enabled: normalized.enabled,
            candidateCount: candidates.length,
            budgetedCandidateCount: 0,
            status: normalized.enabled ? "skipped" : "disabled",
            detail: "prompt_budget_exhausted"
        };
    }
    return {
        enabled: normalized.enabled,
        candidateCount: candidates.length,
        budgetedCandidateCount: budgetedCandidates.length,
        status: normalized.enabled ? "ready" : "disabled",
        detail: `candidates=${budgetedCandidates.length}`
    };
}
class HttpOllamaTeacherLabelerClient {
    baseUrl;
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }
    async generate(input) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), input.timeoutMs);
        try {
            const response = await fetch(`${this.baseUrl}/api/generate`, {
                method: "POST",
                headers: {
                    "content-type": "application/json"
                },
                body: JSON.stringify({
                    model: input.model,
                    prompt: input.prompt,
                    stream: false,
                    options: {
                        num_predict: input.maxOutputTokens,
                        temperature: 0
                    }
                }),
                signal: controller.signal
            });
            if (!response.ok) {
                throw new Error(`ollama generate failed with status ${response.status}`);
            }
            const parsed = (await response.json());
            if (typeof parsed.error === "string" && parsed.error.trim().length > 0) {
                throw new Error(parsed.error);
            }
            if (typeof parsed.response !== "string") {
                throw new Error("ollama generate response field is missing");
            }
            return {
                response: parsed.response
            };
        }
        finally {
            clearTimeout(timeout);
        }
    }
}
export function createHttpOllamaTeacherLabelerClient(baseUrl = DEFAULT_OLLAMA_BASE_URL) {
    return new HttpOllamaTeacherLabelerClient(normalizeBaseUrl(baseUrl));
}
export function createOllamaTeacherLabeler(config) {
    const normalized = normalizeOllamaTeacherLabelerConfig(config);
    return {
        async label(input) {
            const candidates = collectCandidates(input, normalized);
            if (candidates.length === 0) {
                return {
                    artifacts: [],
                    status: "skipped",
                    detail: "no_matching_interaction_text"
                };
            }
            const budgetedCandidates = fitCandidatesToPromptBudget(candidates, normalized);
            if (budgetedCandidates.length === 0) {
                return {
                    artifacts: [],
                    status: "skipped",
                    detail: "prompt_budget_exhausted"
                };
            }
            const prompt = buildPrompt(budgetedCandidates, normalized);
            try {
                const generated = await normalized.client.generate({
                    model: normalized.model,
                    prompt,
                    maxOutputTokens: normalized.maxOutputTokens,
                    timeoutMs: normalized.timeoutMs
                });
                if (generated.response.length > normalized.maxResponseChars) {
                    return {
                        artifacts: [],
                        status: "fail_open",
                        detail: "response_budget_exhausted"
                    };
                }
                const parsedLabels = parseTeacherLabelerResponse(generated.response);
                if (parsedLabels.length === 0) {
                    return {
                        artifacts: [],
                        status: "skipped",
                        detail: "no_labels_emitted"
                    };
                }
                const candidateByInteractionId = new Map(budgetedCandidates.map((candidate) => [candidate.interaction.eventId, candidate]));
                const seenDedupIds = dedupExistingArtifactIds(input.existingArtifacts);
                const nextArtifacts = [];
                for (const label of parsedLabels) {
                    if (nextArtifacts.length >= normalized.maxArtifactsPerExport) {
                        break;
                    }
                    const interactionEventId = typeof label.interactionEventId === "string" && label.interactionEventId.trim().length > 0
                        ? label.interactionEventId.trim()
                        : null;
                    const kind = typeof label.kind === "string" && isFeedbackKind(label.kind) ? label.kind : null;
                    const content = typeof label.content === "string" && normalizeContent(label.content).length > 0
                        ? normalizeContent(label.content)
                        : null;
                    if (interactionEventId === null || kind === null || content === null) {
                        continue;
                    }
                    const candidate = candidateByInteractionId.get(interactionEventId);
                    if (candidate === undefined) {
                        continue;
                    }
                    const artifact = buildGeneratedArtifact({
                        candidate,
                        kind,
                        content,
                        observedAt: input.observedAt,
                        staleAfterMs: input.staleAfterMs,
                        eventRange: input.normalizedEventExport.range,
                        eventExportDigest: input.normalizedEventExport.provenance.exportDigest,
                        teacherIdentity: normalized.teacherIdentity
                    });
                    if (seenDedupIds.has(artifact.dedupId)) {
                        continue;
                    }
                    seenDedupIds.add(artifact.dedupId);
                    nextArtifacts.push(artifact);
                }
                return {
                    artifacts: nextArtifacts,
                    status: "ok",
                    detail: `candidates=${budgetedCandidates.length};labels=${nextArtifacts.length}`
                };
            }
            catch (error) {
                return {
                    artifacts: [],
                    status: "fail_open",
                    detail: error instanceof Error ? error.message : String(error)
                };
            }
        }
    };
}
export function createTeacherLabeler(config) {
    if (config === undefined || config === null || config.provider === "none") {
        return null;
    }
    return createOllamaTeacherLabeler(config);
}
//# sourceMappingURL=teacher-labeler.js.map
