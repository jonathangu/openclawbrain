import { buildServedArtifactProof, CONTRACT_IDS, checksumJsonPayload, describeRetrievalSemanticClass, validateRuntimeCompileExpectation, validateRuntimeCompileRequest, validateRuntimeCompileResponse, validateRuntimeCompileTargetExpectation } from "@openclawbrain/contracts";
import { describePackCompileTarget, describePackInitHandoff, inspectActivationState, loadPack, loadPackFromActivation } from "@openclawbrain/pack-format";
export const DEFAULT_OLLAMA_EMBEDDING_MODEL = "bge-large";
const DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434";
const STRONG_DIAGNOSTIC_REQUEST_TOKENS = new Set([
    "debug",
    "debugging",
    "diagnostic",
    "diagnostics",
    "headers",
    "instrumentation",
    "latency",
    "log",
    "logging",
    "logs",
    "observability",
    "probe",
    "probes",
    "replay",
    "replays",
    "residue",
    "trace",
    "traces",
    "transport"
]);
const WEAK_DIAGNOSTIC_REQUEST_TOKENS = new Set(["delivery", "event", "events", "export", "exports", "session", "sessions", "status"]);
const OBSERVABILITY_BLOCK_TOKENS = new Set([
    "diagnostic",
    "diagnostics",
    "latency",
    "log",
    "logging",
    "logs",
    "observability",
    "probe",
    "proof",
    "replay",
    "replays",
    "trace",
    "traces",
    "transport"
]);
const INSTRUCTIONAL_SCAFFOLDING_TOKENS = new Set([
    "ask",
    "copy",
    "forward",
    "paste",
    "prompt",
    "prompts",
    "question",
    "questions",
    "questionnaire",
    "reply",
    "respond",
    "send",
    "template"
]);
const INSTRUCTIONAL_SCAFFOLDING_TARGET_TOKENS = new Set([
    "agent",
    "assistant",
    "bot",
    "codex",
    "eagle",
    "llm",
    "model",
    "tern"
]);
const INSTRUCTIONAL_SCAFFOLDING_ROLE_HINTS = new Set(["feedback", "teacher_supervision", "label_surface"]);
const INSTRUCTIONAL_SCAFFOLDING_PATTERNS = [
    { label: "send_questions_to_agent", pattern: /\bsend (?:these|the following)?\s*(?:questions|prompts?) to\b/i },
    { label: "use_prompt_template", pattern: /\b(?:use|copy|paste|forward|send) (?:this|these|the following) (?:prompt|prompts|template|questionnaire)\b/i },
    { label: "reply_with_questions", pattern: /\b(?:reply|respond) with (?:these|the following)?\s*(?:questions|prompts?)\b/i },
    { label: "ask_agent", pattern: /\bask (?:the )?(?:agent|assistant|bot|codex|eagle|llm|model|tern)\b/i }
];
const INSTRUCTIONAL_REQUEST_PATTERNS = [
    { signal: "request:what_should_i_ask", pattern: /\bwhat\s+(?:should|could)\s+i\s+ask\b/i },
    { signal: "request:which_questions", pattern: /\bwhich\s+(?:questions|prompts?)\b/i },
    { signal: "request:questionnaire_agent", pattern: /\b(?:questionnaire|questions|prompts?)\b.*\b(?:eagle|tern|assistant|agent|model|codex)\b/i },
    { signal: "request:handoff_agent", pattern: /\b(?:send|forward|route|handoff)\b.*\b(?:eagle|tern|assistant|agent|model|codex)\b/i }
];
const TRANSPORT_RESIDUE_TOKENS = new Set([
    "channel",
    "delivery",
    "exec",
    "headers",
    "latency",
    "log",
    "logs",
    "messageid",
    "sessionid",
    "stderr",
    "stdout",
    "stream",
    "trace",
    "transport",
    "untrusted"
]);
const OPERATOR_OBSERVABILITY_FILE_ROLES = new Set(["operator_observability", "ops_recipe", "session_replay_proof"]);
const NORMAL_OBSERVABILITY_DOWNRANK = 4;
const NON_INSTRUCTIONAL_SCAFFOLD_DOWNRANK = 8;
const UNCERTAIN_TRANSPORT_DOWNRANK = 5;
const UNCERTAIN_OBSERVABILITY_DOWNRANK = 2;
const FAIL_OPEN_TRANSPORT_DOWNRANK = 7;
const DIAGNOSTIC_OBSERVABILITY_BOOST = 3;
const DIAGNOSTIC_TRANSPORT_BOOST = 1.5;
function normalizeTokens(value) {
    return [...new Set(value.toLowerCase().split(/[^a-z0-9]+/u).filter((token) => token.length >= 2))];
}
function intersectTokens(tokens, vocabulary) {
    const hits = [];
    for (const token of tokens) {
        if (vocabulary.has(token) && !hits.includes(token)) {
            hits.push(token);
        }
    }
    return hits;
}
function compareIsoDates(left, right) {
    return Date.parse(left) - Date.parse(right);
}
function noteValue(notes, prefix) {
    const match = notes.find((note) => note.startsWith(prefix));
    return match === undefined ? null : match.slice(prefix.length);
}
function estimateTokenCount(value) {
    return normalizeTokens(value).length;
}
function clampScore(value, min, max) {
    return Math.max(min, Math.min(max, value));
}
function requestText(request) {
    return [request.userMessage, ...(request.runtimeHints ?? [])].join(" ").trim();
}
function requestTokens(request) {
    return normalizeTokens(requestText(request));
}
function classifyRetrievalIntent(request) {
    const tokens = requestTokens(request);
    const strongSignals = intersectTokens(tokens, STRONG_DIAGNOSTIC_REQUEST_TOKENS).map((token) => `strong:${token}`);
    const weakSignals = intersectTokens(tokens, WEAK_DIAGNOSTIC_REQUEST_TOKENS).map((token) => `weak:${token}`);
    if (strongSignals.length > 0 || weakSignals.length >= 2) {
        return {
            intent: "diagnostic_observability",
            signals: [...strongSignals, ...weakSignals].slice(0, 6)
        };
    }
    if (weakSignals.length === 1) {
        return {
            intent: "uncertain",
            signals: weakSignals
        };
    }
    return {
        intent: "normal_semantic",
        signals: []
    };
}
function classifyInstructionalRequest(request) {
    const text = requestText(request);
    const tokens = requestTokens(request);
    const scaffoldHits = intersectTokens(tokens, INSTRUCTIONAL_SCAFFOLDING_TOKENS).map((token) => `scaffold:${token}`);
    const targetHits = intersectTokens(tokens, INSTRUCTIONAL_SCAFFOLDING_TARGET_TOKENS).map((token) => `target:${token}`);
    const signals = INSTRUCTIONAL_REQUEST_PATTERNS.filter((entry) => entry.pattern.test(text)).map((entry) => entry.signal);
    if (signals.length > 0) {
        return {
            allowScaffolding: true,
            signals
        };
    }
    if (scaffoldHits.length > 0 && targetHits.length > 0) {
        return {
            allowScaffolding: true,
            signals: [...scaffoldHits.slice(0, 2), ...targetHits.slice(0, 2)]
        };
    }
    return {
        allowScaffolding: false,
        signals: []
    };
}
function trimTrailingSlash(value) {
    return value.endsWith("/") ? value.slice(0, -1) : value;
}
function isFiniteVector(value) {
    return value.length > 0 && value.every((candidate) => Number.isFinite(candidate));
}
function vectorMagnitude(value) {
    return Math.sqrt(value.reduce((sum, candidate) => sum + candidate * candidate, 0));
}
function cosineSimilarity(left, right) {
    if (left.length !== right.length || left.length === 0) {
        return null;
    }
    let dot = 0;
    for (let index = 0; index < left.length; index += 1) {
        const leftValue = left[index];
        const rightValue = right[index];
        if (leftValue === undefined || rightValue === undefined) {
            return null;
        }
        dot += leftValue * rightValue;
    }
    const leftMagnitude = vectorMagnitude(left);
    const rightMagnitude = vectorMagnitude(right);
    if (leftMagnitude === 0 || rightMagnitude === 0) {
        return null;
    }
    return dot / (leftMagnitude * rightMagnitude);
}
function normalizeTextEmbeddingResult(value, fallbackModel) {
    if (value === undefined || !isFiniteVector(value.values)) {
        return null;
    }
    const model = value.model.length > 0 ? value.model : fallbackModel;
    if (model.length === 0) {
        return null;
    }
    return {
        model,
        values: [...value.values]
    };
}
async function embedCompileRequest(embedder, request) {
    const [embedding] = await embedder.embed([requestText(request)]);
    return normalizeTextEmbeddingResult(embedding, embedder.model);
}
export function createOllamaEmbedder(options = {}) {
    const model = options.model ?? DEFAULT_OLLAMA_EMBEDDING_MODEL;
    const baseUrl = trimTrailingSlash(options.baseUrl ?? DEFAULT_OLLAMA_BASE_URL);
    const fetchImpl = options.fetchImpl ?? fetch;
    return {
        model,
        async embed(input) {
            if (input.length === 0) {
                return [];
            }
            const response = await fetchImpl(`${baseUrl}/api/embed`, {
                method: "POST",
                headers: {
                    "content-type": "application/json",
                    ...(options.headers ?? {})
                },
                body: JSON.stringify({
                    model,
                    input
                })
            });
            if (!response.ok) {
                throw new Error(`Ollama embed failed: ${response.status} ${response.statusText}`);
            }
            const payload = (await response.json());
            const embeddings = Array.isArray(payload.embeddings)
                ? payload.embeddings
                : Array.isArray(payload.embedding)
                    ? [payload.embedding]
                    : null;
            if (embeddings === null || embeddings.length !== input.length) {
                throw new Error("Ollama embed returned an unexpected embedding payload");
            }
            const resolvedModel = typeof payload.model === "string" && payload.model.length > 0 ? payload.model : model;
            return embeddings.map((values) => {
                if (!isFiniteVector(values)) {
                    throw new Error("Ollama embed returned a non-finite embedding vector");
                }
                return {
                    model: resolvedModel,
                    values: [...values]
                };
            });
        }
    };
}
function buildKeywordWeights(block, vectorEntry) {
    const weights = new Map();
    function assignWeight(keyword, weight) {
        for (const token of normalizeTokens(keyword)) {
            weights.set(token, Math.max(weights.get(token) ?? 0, weight));
        }
    }
    for (const keyword of block.keywords) {
        assignWeight(keyword, 3);
    }
    if (vectorEntry !== undefined) {
        for (const keyword of vectorEntry.keywords) {
            assignWeight(keyword, 4);
        }
        for (const [keyword, weight] of Object.entries(vectorEntry.weights ?? {})) {
            const numericWeight = Number.isFinite(weight) ? weight : 0;
            assignWeight(keyword, numericWeight);
        }
    }
    return weights;
}
const STOP_ACTION_ID = "__STOP__";
const STOP_ACTION_UPDATE_SEPARATOR = "::";
function buildStopActionUpdateBlockId(sourceBlockId) {
    return `${sourceBlockId}${STOP_ACTION_UPDATE_SEPARATOR}${STOP_ACTION_ID}`;
}
function routerPolicyUpdateForBlock(pack, blockId) {
    return pack.router?.policyUpdates.find((update) => update.blockId === blockId) ?? null;
}
function routerStopPolicyUpdateForSource(pack, sourceBlockId) {
    return routerPolicyUpdateForBlock(pack, buildStopActionUpdateBlockId(sourceBlockId));
}
function routerLearnedPrior(update) {
    if (update === null) {
        return 0;
    }
    return clampScore(update.delta * 0.45, -3, 4) +
        clampScore(Math.sign(update.delta || 0) * update.evidenceCount * 0.2, -1, 1.5) +
        clampScore(update.rewardSum * 0.1, -1.5, 1.5);
}
function graphWalkNeighborActionScore(edge, targetEntry) {
    return edge.weight + targetEntry.score * 0.2 + targetEntry.matchedTokens.length * 1.5 + targetEntry.priority * 0.1;
}
function describeGraphWalkStopDecision(pack, sourceBlockId, nextNeighbor) {
    if (nextNeighbor === undefined) {
        return {
            shouldStop: true,
            learnedStopPolicyActive: false
        };
    }
    const stopPolicyUpdate = routerStopPolicyUpdateForSource(pack, sourceBlockId);
    const stopScore = routerLearnedPrior(stopPolicyUpdate);
    return {
        shouldStop: stopScore >= graphWalkNeighborActionScore(nextNeighbor.edge, nextNeighbor.targetEntry),
        learnedStopPolicyActive: stopPolicyUpdate !== null
    };
}
function routerDeltaSummary(pack) {
    if (pack.router === null || pack.router.policyUpdates.length === 0) {
        return undefined;
    }
    return pack.router.policyUpdates
        .slice(0, 3)
        .map((update) => `${update.blockId}:${update.delta}`)
        .join(",");
}
function resolveRoutingChannels(block, vectorEntry) {
    if (block.routing !== undefined) {
        return [...block.routing.channels];
    }
    const channels = new Set();
    if ((block.edges?.length ?? 0) > 0 || (block.state?.traversalBias ?? 0) > 0 || block.learning.role === "structural") {
        channels.add("graph");
    }
    if (["interaction", "feedback", "teacher_supervision", "label_surface"].includes(block.learning.role)) {
        channels.add("short_term");
    }
    if (vectorEntry !== undefined || block.keywords.length > 0 || ["background_expectation", "boot_default", "workspace"].includes(block.learning.role)) {
        channels.add("vector");
    }
    return channels.size === 0 ? ["vector"] : [...channels];
}
function routerFeatureKeysForBlock(block, vectorEntry, routingChannels) {
    const features = new Set();
    features.add(`feature:role:${block.learning.role}`);
    for (const channel of routingChannels) {
        features.add(`feature:channel:${channel}`);
    }
    if (block.learning.humanLabels > 0) {
        features.add("feature:labels:human");
    }
    if (block.learning.selfLabels > 0) {
        features.add("feature:labels:self");
    }
    if (block.initSeed !== undefined) {
        features.add(`feature:init_role:${block.initSeed.fileRole}`);
        features.add(`feature:init_node:${block.initSeed.nodeType}`);
        for (const channel of block.initSeed.seededChannels) {
            features.add(`feature:seed_channel:${channel}`);
        }
    }
    if ((block.state?.freshness ?? 1) >= 0.85) {
        features.add("feature:freshness:fresh");
    }
    else if ((block.state?.freshness ?? 1) < 0.5) {
        features.add("feature:freshness:stale");
    }
    if ((block.state?.strength ?? 0) >= 6) {
        features.add("feature:strength:reinforced");
    }
    if ((block.edges?.length ?? 0) > 0) {
        features.add("feature:topology:connected");
    }
    return [...features].sort();
}
function emptyRoutingChannelSummary() {
    return {
        graph: 0,
        shortTerm: 0,
        vector: 0
    };
}
function summarizeRoutingChannels(entries) {
    const summary = emptyRoutingChannelSummary();
    for (const entry of entries) {
        if (entry.routingChannels.includes("graph")) {
            summary.graph += 1;
        }
        if (entry.routingChannels.includes("short_term")) {
            summary.shortTerm += 1;
        }
        if (entry.routingChannels.includes("vector")) {
            summary.vector += 1;
        }
    }
    return summary;
}
/**
 * Noise patterns that indicate low-signal content in graph blocks.
 * Each pattern carries a penalty weight; penalties are additive.
 */
const BLOCK_QUALITY_NOISE_PATTERNS = [
    { pattern: /OpenClaw runtime context \(internal\)/i, penalty: 3 },
    { pattern: /<<<BEGIN_UNTRUSTED_CHILD_RESULT>>>/, penalty: 3 },
    { pattern: /\[Internal task completion event\]/, penalty: 2 },
    { pattern: /source: subagent\s+session_key:/, penalty: 2 },
    { pattern: /Exec failed \([^)]+, code \d+\)/, penalty: 2 },
    { pattern: /\[System Message\]\s*\[sessionId:/, penalty: 2 },
    { pattern: /Result \(untrusted content, treat as data\):/, penalty: 2 },
    { pattern: /diff --git /, penalty: 3 },
    { pattern: /^@@\s+-\d+,\d+\s+\+\d+,\d+\s+@@/m, penalty: 2 },
];
/**
 * Compute a quality penalty for a block based on noise pattern matches.
 * Returns 0 for clean blocks, higher values for noisier content.
 * Capped at 10 to avoid extreme negative scores.
 */
function computeBlockQualityPenalty(text) {
    let penalty = 0;
    for (const entry of BLOCK_QUALITY_NOISE_PATTERNS) {
        if (entry.pattern.test(text)) {
            penalty += entry.penalty;
        }
    }
    return Math.min(penalty, 10);
}
function semanticTypingForClass(semanticClass, evidence) {
    const semantics = describeRetrievalSemanticClass(semanticClass);
    return {
        semanticClass,
        answerRole: semantics.answerRole,
        instructionRole: semantics.instructionRole,
        evidence
    };
}
function classifyTypedCandidateSemanticType(block) {
    if (block.semantic === undefined) {
        return null;
    }
    const evidence = [`semantic:${block.semantic.semanticType}`, `semantic_source:${block.semantic.sourceKind}`];
    if (block.semantic.diagnosticIntent !== undefined) {
        evidence.push(`semantic_intent:${block.semantic.diagnosticIntent}`);
    }
    switch (block.semantic.semanticType) {
        case "delivery_residue":
        case "observability_residue":
            return semanticTypingForClass("transport_residue", evidence);
        case "instructional_scaffolding":
            return semanticTypingForClass("instructional_scaffolding", evidence);
        case "memory_candidate":
        case "teacher_signal":
        case "control_signal":
            return semanticTypingForClass("answer_bearing", evidence);
    }
}
function classifyInstructionalScaffolding(block, text, textTokens, evidence) {
    const patternHits = INSTRUCTIONAL_SCAFFOLDING_PATTERNS.filter((entry) => entry.pattern.test(text)).map((entry) => entry.label);
    const scaffoldHits = intersectTokens(textTokens, INSTRUCTIONAL_SCAFFOLDING_TOKENS);
    const targetHits = intersectTokens(textTokens, INSTRUCTIONAL_SCAFFOLDING_TARGET_TOKENS);
    const imperativeLead = /^(?:please\s+)?(?:ask|copy|forward|paste|reply|respond|send|use)\b/i.test(block.text.trim());
    const roleHint = INSTRUCTIONAL_SCAFFOLDING_ROLE_HINTS.has(block.learning.role);
    const templateHint = scaffoldHits.includes("prompt") || scaffoldHits.includes("prompts") || scaffoldHits.includes("questionnaire") || scaffoldHits.includes("template");
    if (!((patternHits.length > 0 && (imperativeLead || roleHint || templateHint)) ||
        (imperativeLead && scaffoldHits.length >= 2 && targetHits.length >= 1) ||
        ((roleHint || templateHint) && scaffoldHits.length >= 3 && targetHits.length >= 1))) {
        return null;
    }
    return semanticTypingForClass("instructional_scaffolding", [
        ...evidence,
        `role:${block.learning.role}`,
        ...(roleHint ? ["role:instructional_surface"] : []),
        ...(imperativeLead ? ["imperative:leading_instruction"] : []),
        ...(templateHint ? ["scaffold:container_hint"] : []),
        ...patternHits.map((hit) => `scaffold_pattern:${hit}`),
        ...scaffoldHits.slice(0, 4).map((hit) => `scaffold:${hit}`),
        ...targetHits.slice(0, 2).map((hit) => `target:${hit}`)
    ]);
}
function classifyCandidateSemanticType(block) {
    const typed = classifyTypedCandidateSemanticType(block);
    const sourceTokens = normalizeTokens(block.source);
    const combinedText = [
        block.source,
        block.text,
        block.keywords.join(" "),
        block.semantic?.semanticType ?? "",
        block.semantic?.sourceKind ?? "",
        block.semantic?.diagnosticIntent ?? "",
        block.init?.sourceKind ?? "",
        block.init?.nodeKind ?? "",
        block.init?.fileRole?.role ?? "",
        block.init?.fileRole?.path ?? ""
    ].join(" ");
    const normalizedBlockTokens = normalizeTokens(combinedText);
    const blockTokens = new Set(normalizedBlockTokens);
    const observabilityHits = intersectTokens(blockTokens, OBSERVABILITY_BLOCK_TOKENS);
    const transportHits = intersectTokens(blockTokens, TRANSPORT_RESIDUE_TOKENS);
    const sourceResidueHits = intersectTokens(sourceTokens, TRANSPORT_RESIDUE_TOKENS);
    const qualityPenalty = computeBlockQualityPenalty(block.text);
    const evidence = [...(typed?.evidence ?? [])];
    const fileRole = block.init?.fileRole?.role ?? null;
    const sourceKind = block.init?.sourceKind ?? null;
    if (typed !== null && typed.semanticClass === "transport_residue") {
        return typed;
    }
    if (typed !== null && typed.answerRole === "answer_bearing") {
        const scaffolding = classifyInstructionalScaffolding(block, combinedText, normalizedBlockTokens, evidence);
        if (scaffolding !== null) {
            return scaffolding;
        }
        return typed;
    }
    if (typed !== null) {
        return typed;
    }
    const scaffoldFromUntyped = classifyInstructionalScaffolding(block, combinedText, normalizedBlockTokens, evidence);
    if (scaffoldFromUntyped !== null) {
        return scaffoldFromUntyped;
    }
    if (fileRole !== null) {
        evidence.push(`file_role:${fileRole}`);
    }
    if (sourceKind !== null) {
        evidence.push(`source_kind:${sourceKind}`);
    }
    if (OPERATOR_OBSERVABILITY_FILE_ROLES.has(fileRole ?? "")) {
        return semanticTypingForClass("observability_support", evidence);
    }
    if (qualityPenalty >= 3) {
        return semanticTypingForClass("transport_residue", [...evidence, `quality_penalty:${qualityPenalty}`]);
    }
    if ((sourceKind === "event" || sourceKind === "event_export") && (transportHits.length >= 2 || observabilityHits.length >= 2)) {
        return semanticTypingForClass("transport_residue", [
            ...evidence,
            ...transportHits.slice(0, 3).map((token) => `transport:${token}`),
            ...observabilityHits.slice(0, 2).map((token) => `obs:${token}`)
        ]);
    }
    if ((block.learning.role === "interaction" || block.learning.role === "feedback") &&
        sourceResidueHits.length > 0 &&
        transportHits.length >= 2) {
        return semanticTypingForClass("transport_residue", [
            ...evidence,
            `role:${block.learning.role}`,
            ...sourceResidueHits.slice(0, 2).map((token) => `source:${token}`),
            ...transportHits.slice(0, 2).map((token) => `transport:${token}`)
        ]);
    }
    if (observabilityHits.length >= 2 && block.init?.fileRole?.audience !== "runtime") {
        return semanticTypingForClass("observability_support", [...evidence, ...observabilityHits.slice(0, 4).map((token) => `obs:${token}`)]);
    }
    return semanticTypingForClass("answer_bearing", evidence);
}
function baseBlockScore(block) {
    const strength = block.state?.strength ?? block.priority;
    const traversalBias = block.state?.traversalBias ?? 0;
    const freshness = block.state?.freshness ?? 1;
    return strength + traversalBias * 0.5 + freshness;
}
function applyRouterFeaturePrior(block, vectorEntry, routingChannels, routerTokenWeights, channelScores) {
    let sharedFeatureScore = 0;
    for (const featureKey of routerFeatureKeysForBlock(block, vectorEntry, routingChannels)) {
        const weight = routerTokenWeights[featureKey];
        if (weight === undefined) {
            continue;
        }
        const normalizedWeight = clampScore(weight * 0.35, -1.5, 1.5);
        if (featureKey === "feature:channel:graph") {
            channelScores.graph += normalizedWeight;
            continue;
        }
        if (featureKey === "feature:channel:short_term") {
            channelScores.shortTerm += normalizedWeight;
            continue;
        }
        if (featureKey === "feature:channel:vector") {
            channelScores.vector += normalizedWeight;
            continue;
        }
        sharedFeatureScore += normalizedWeight;
    }
    if (sharedFeatureScore === 0) {
        return;
    }
    const distributed = sharedFeatureScore / Math.max(1, routingChannels.length);
    for (const channel of routingChannels) {
        if (channel === "graph") {
            channelScores.graph += distributed;
        }
        else if (channel === "short_term") {
            channelScores.shortTerm += distributed;
        }
        else {
            channelScores.vector += distributed;
        }
    }
}
function blockTokenCount(block) {
    return block.tokenCount ?? estimateTokenCount(block.text);
}
function buildContextBlock(block) {
    return {
        id: block.id,
        source: block.source,
        text: block.text,
        tokenCount: blockTokenCount(block),
        ...(block.compactedFrom !== undefined ? { compactedFrom: [...block.compactedFrom] } : {})
    };
}
function flattenContextIds(block) {
    return block.compactedFrom ?? [block.id];
}
function flattenRankedContextIds(block) {
    return block.compactedFrom ?? [block.blockId];
}
function buildSelectedContextBlock(entry) {
    return buildContextBlock({
        id: entry.blockId,
        source: entry.source,
        text: entry.text,
        tokenCount: entry.tokenCount,
        ...(entry.compactedFrom !== undefined ? { compactedFrom: entry.compactedFrom } : {})
    });
}
function normalizeLearnedRouteSelectionOverrideResult(value) {
    const selectedBlockIds = Array.isArray(value?.selectedBlockIds)
        ? [...new Set(value.selectedBlockIds
                .filter((entry) => typeof entry === "string")
                .map((entry) => entry.trim())
                .filter((entry) => entry.length > 0))]
        : [];
    const routerIdentity = typeof value?.routerIdentity === "string" && value.routerIdentity.trim().length > 0
        ? value.routerIdentity.trim()
        : null;
    const evidenceSource = typeof value?.evidenceSource === "string" && value.evidenceSource.trim().length > 0
        ? value.evidenceSource.trim()
        : "replay_candidate_override";
    const selectionMode = value?.selectionMode === "graph_walk_score_boost"
        ? "graph_walk_score_boost"
        : "direct_select";
    const scoreBoostsByBlockId = value?.scoreBoostsByBlockId && typeof value.scoreBoostsByBlockId === "object"
        ? Object.fromEntries(Object.entries(value.scoreBoostsByBlockId)
            .filter(([blockId, boost]) => typeof blockId === "string"
                && blockId.trim().length > 0
                && typeof boost === "number"
                && Number.isFinite(boost))
            .map(([blockId, boost]) => [blockId.trim(), boost]))
        : {};
    return {
        selectedBlockIds,
        routerIdentity,
        evidenceSource,
        selectionMode,
        scoreBoostsByBlockId
    };
}
function sortRankedContextEntries(left, right) {
    if (right.matchedTokens.length !== left.matchedTokens.length) {
        return right.matchedTokens.length - left.matchedTokens.length;
    }
    if (right.score !== left.score) {
        return right.score - left.score;
    }
    if (right.priority !== left.priority) {
        return right.priority - left.priority;
    }
    return left.packOrder - right.packOrder;
}
function applyLearnedRouteScoreBoosts(ranked, scoreBoostsByBlockId) {
    const boosted = ranked.map((entry) => {
        const boost = scoreBoostsByBlockId[entry.blockId] ?? 0;
        if (!Number.isFinite(boost) || boost === 0) {
            return entry;
        }
        return {
            ...entry,
            score: entry.score + boost
        };
    });
    boosted.sort(sortRankedContextEntries);
    return boosted;
}
function selectContextBlocksByLearnedRouteOverride(ranked, maxBlocks, selectedBlockIds) {
    if (maxBlocks === 0 || selectedBlockIds.length === 0) {
        return {
            selected: [],
            matchedSelectedCount: 0,
            overlapPrunedCount: 0,
            overlapPrunedBlockIds: [],
            graphWalkSeedBlockIds: [],
            graphWalkHopCount: 0,
            graphWalkLearnedStopPolicyDecisionCount: 0
        };
    }
    const rankedById = new Map(ranked.map((entry) => [entry.blockId, entry]));
    const selected = [];
    const coveredIds = new Set();
    const overlapPrunedBlockIds = [];
    let matchedSelectedCount = 0;
    let overlapPrunedCount = 0;
    const trySelect = (entry) => {
        if (selected.length >= maxBlocks) {
            return;
        }
        const coverageIds = flattenRankedContextIds(entry);
        if (coverageIds.some((coverageId) => coveredIds.has(coverageId))) {
            overlapPrunedCount += 1;
            overlapPrunedBlockIds.push(entry.blockId);
            return;
        }
        selected.push(buildSelectedContextBlock(entry));
        for (const coverageId of coverageIds) {
            coveredIds.add(coverageId);
        }
        if (entry.matchedTokens.length > 0) {
            matchedSelectedCount += 1;
        }
    };
    for (const blockId of selectedBlockIds) {
        const entry = rankedById.get(blockId);
        if (entry === undefined) {
            continue;
        }
        trySelect(entry);
    }
    return {
        selected,
        matchedSelectedCount,
        overlapPrunedCount,
        overlapPrunedBlockIds,
        graphWalkSeedBlockIds: [],
        graphWalkHopCount: 0,
        graphWalkLearnedStopPolicyDecisionCount: 0
    };
}
function selectContextBlocks(ranked, maxBlocks) {
    if (maxBlocks === 0) {
        return {
            selected: [],
            matchedSelectedCount: 0,
            overlapPrunedCount: 0,
            overlapPrunedBlockIds: [],
            graphWalkSeedBlockIds: [],
            graphWalkHopCount: 0,
            graphWalkLearnedStopPolicyDecisionCount: 0
        };
    }
    const selected = [];
    const coveredIds = new Set();
    const overlapPrunedBlockIds = [];
    let matchedSelectedCount = 0;
    let overlapPrunedCount = 0;
    const trySelect = (entry, matchedTier) => {
        if (selected.length >= maxBlocks) {
            return;
        }
        const coverageIds = flattenRankedContextIds(entry);
        if (coverageIds.some((coverageId) => coveredIds.has(coverageId))) {
            overlapPrunedCount += 1;
            overlapPrunedBlockIds.push(entry.blockId);
            return;
        }
        selected.push(buildSelectedContextBlock(entry));
        for (const coverageId of coverageIds) {
            coveredIds.add(coverageId);
        }
        if (matchedTier) {
            matchedSelectedCount += 1;
        }
    };
    const matched = ranked.filter((entry) => entry.matchedTokens.length > 0);
    const unmatched = ranked.filter((entry) => entry.matchedTokens.length === 0);
    for (const entry of matched) {
        trySelect(entry, true);
    }
    for (const entry of unmatched) {
        trySelect(entry, false);
    }
    return {
        selected,
        matchedSelectedCount,
        overlapPrunedCount,
        overlapPrunedBlockIds,
        graphWalkSeedBlockIds: [],
        graphWalkHopCount: 0,
        graphWalkLearnedStopPolicyDecisionCount: 0
    };
}
function selectContextBlocksByGraphWalk(pack, ranked, maxBlocks) {
    if (maxBlocks === 0) {
        return {
            selected: [],
            matchedSelectedCount: 0,
            overlapPrunedCount: 0,
            overlapPrunedBlockIds: [],
            graphWalkSeedBlockIds: [],
            graphWalkHopCount: 0,
            graphWalkLearnedStopPolicyDecisionCount: 0
        };
    }
    const selected = [];
    const coveredIds = new Set();
    const overlapPrunedBlockIds = [];
    const rankedById = new Map(ranked.map((entry) => [entry.blockId, entry]));
    const blocksById = new Map(pack.graph.blocks.map((block) => [block.id, block]));
    const graphWalkSeedBlockIds = [];
    let matchedSelectedCount = 0;
    let overlapPrunedCount = 0;
    let graphWalkHopCount = 0;
    let graphWalkLearnedStopPolicyDecisionCount = 0;
    const trySelect = (entry, matchedTier) => {
        if (selected.length >= maxBlocks) {
            return false;
        }
        const coverageIds = flattenRankedContextIds(entry);
        if (coverageIds.some((coverageId) => coveredIds.has(coverageId))) {
            overlapPrunedCount += 1;
            overlapPrunedBlockIds.push(entry.blockId);
            return false;
        }
        selected.push(buildSelectedContextBlock(entry));
        for (const coverageId of coverageIds) {
            coveredIds.add(coverageId);
        }
        if (matchedTier) {
            matchedSelectedCount += 1;
        }
        return true;
    };
    const walkFromSeed = (seed) => {
        const visited = new Set([seed.blockId]);
        const frontier = [seed.blockId];
        while (frontier.length > 0 && selected.length < maxBlocks) {
            const currentBlockId = frontier[frontier.length - 1];
            if (currentBlockId === undefined) {
                break;
            }
            const currentBlock = blocksById.get(currentBlockId);
            if (currentBlock === undefined || (currentBlock.edges?.length ?? 0) === 0) {
                frontier.pop();
                continue;
            }
            const nextNeighbor = (currentBlock.edges ?? [])
                .filter((edge) => !visited.has(edge.targetBlockId))
                .map((edge) => {
                const targetEntry = rankedById.get(edge.targetBlockId);
                return targetEntry === undefined ? null : { edge, targetEntry };
            })
                .filter((entry) => entry !== null)
                .sort((left, right) => {
                if (right.targetEntry.matchedTokens.length !== left.targetEntry.matchedTokens.length) {
                    return right.targetEntry.matchedTokens.length - left.targetEntry.matchedTokens.length;
                }
                if (right.edge.weight !== left.edge.weight) {
                    return right.edge.weight - left.edge.weight;
                }
                if (right.targetEntry.score !== left.targetEntry.score) {
                    return right.targetEntry.score - left.targetEntry.score;
                }
                if (right.targetEntry.priority !== left.targetEntry.priority) {
                    return right.targetEntry.priority - left.targetEntry.priority;
                }
                return left.targetEntry.packOrder - right.targetEntry.packOrder;
            })[0];
            const stopDecision = describeGraphWalkStopDecision(pack, currentBlockId, nextNeighbor);
            if (stopDecision.learnedStopPolicyActive) {
                graphWalkLearnedStopPolicyDecisionCount += 1;
            }
            if (stopDecision.shouldStop) {
                frontier.pop();
                continue;
            }
            visited.add(nextNeighbor.targetEntry.blockId);
            frontier.push(nextNeighbor.targetEntry.blockId);
            if (trySelect(nextNeighbor.targetEntry, nextNeighbor.targetEntry.matchedTokens.length > 0)) {
                graphWalkHopCount += 1;
            }
        }
    };
    for (const entry of ranked) {
        if (selected.length >= maxBlocks) {
            break;
        }
        if (!trySelect(entry, entry.matchedTokens.length > 0)) {
            continue;
        }
        graphWalkSeedBlockIds.push(entry.blockId);
        walkFromSeed(entry);
    }
    return {
        selected,
        matchedSelectedCount,
        overlapPrunedCount,
        overlapPrunedBlockIds,
        graphWalkSeedBlockIds,
        graphWalkHopCount,
        graphWalkLearnedStopPolicyDecisionCount
    };
}
function totalCharCount(blocks) {
    return blocks.reduce((sum, block) => sum + block.text.length, 0);
}
function totalTokenCount(blocks) {
    return blocks.reduce((sum, block) => sum + blockTokenCount(block), 0);
}
function mergeDiagnosticNotes(existing, additions) {
    const seen = new Set();
    const merged = [];
    for (const note of [...existing, ...additions]) {
        if (note === undefined || note.length === 0 || seen.has(note)) {
            continue;
        }
        seen.add(note);
        merged.push(note);
    }
    return merged;
}
function annotateCompileResponse(response, additions) {
    return {
        ...response,
        diagnostics: {
            ...response.diagnostics,
            notes: mergeDiagnosticNotes(response.diagnostics.notes, additions)
        }
    };
}
function truncateText(value, maxChars) {
    if (maxChars <= 0) {
        return "";
    }
    if (value.length <= maxChars) {
        return value;
    }
    if (maxChars === 1) {
        return value.slice(0, 1);
    }
    return `${value.slice(0, maxChars - 1).trimEnd()}…`;
}
function buildCompactedText(blocks, maxChars) {
    const prefix = "Compacted pack-backed context: ";
    const separator = " | ";
    const available = Math.max(0, maxChars - prefix.length);
    const perBlockBudget = Math.max(20, Math.floor((available - separator.length * Math.max(0, blocks.length - 1)) / Math.max(1, blocks.length)));
    const parts = blocks.map((block) => `${block.id}: ${truncateText(block.text, perBlockBudget)}`);
    return truncateText(`${prefix}${parts.join(separator)}`, maxChars);
}
function mergeContextBlocks(blocks, maxChars) {
    const compactedFrom = [...new Set(blocks.flatMap((block) => flattenContextIds(block)))];
    const sources = [...new Set(blocks.map((block) => block.source))];
    return buildContextBlock({
        id: `compact:${compactedFrom.join("+")}`,
        source: sources.length === 1 ? `compact:${sources[0]}` : `compact:${sources.join("|")}`,
        text: buildCompactedText(blocks, maxChars),
        tokenCount: totalTokenCount(blocks),
        compactedFrom
    });
}
function fitContextToCharBudget(blocks, maxChars, compactionMode) {
    const normalized = blocks.map((block) => buildContextBlock(block));
    if (maxChars === undefined || totalCharCount(normalized) <= maxChars) {
        return {
            blocks: normalized,
            compactionApplied: false
        };
    }
    if (maxChars <= 0 || normalized.length === 0) {
        return {
            blocks: [],
            compactionApplied: false
        };
    }
    if (compactionMode === "none") {
        const fitted = [];
        let remaining = maxChars;
        let modified = false;
        for (const block of normalized) {
            if (remaining <= 0) {
                break;
            }
            if (block.text.length <= remaining) {
                fitted.push(block);
                remaining -= block.text.length;
                continue;
            }
            const truncated = buildContextBlock({
                ...block,
                text: truncateText(block.text, remaining)
            });
            if (truncated.text.length > 0) {
                fitted.push(truncated);
                modified = truncated.text !== block.text;
            }
            break;
        }
        return {
            blocks: fitted,
            compactionApplied: modified
        };
    }
    if (normalized.length === 1) {
        const block = normalized[0];
        const truncated = buildContextBlock({
            ...block,
            text: truncateText(block.text, maxChars)
        });
        return {
            blocks: truncated.text.length === 0 ? [] : [truncated],
            compactionApplied: truncated.text !== block.text
        };
    }
    const head = normalized[0];
    if (head === undefined) {
        return {
            blocks: [],
            compactionApplied: false
        };
    }
    const tail = normalized.slice(1);
    if (head.text.length < maxChars) {
        const tailBudget = maxChars - head.text.length;
        const compactedTail = mergeContextBlocks(tail, tailBudget);
        if (compactedTail.text.length > 0 && totalCharCount([head, compactedTail]) <= maxChars) {
            return {
                blocks: [head, compactedTail],
                compactionApplied: true
            };
        }
    }
    return {
        blocks: [mergeContextBlocks(normalized, maxChars)],
        compactionApplied: true
    };
}
export function determineRouteMode(pack, requested) {
    return pack.manifest.routePolicy === "requires_learned_routing" ? "learned" : requested;
}
function isStrictlyFresherTarget(candidate, active) {
    return (compareIsoDates(candidate.builtAt, active.builtAt) > 0 ||
        candidate.eventRange.end > active.eventRange.end ||
        candidate.eventRange.count > active.eventRange.count ||
        candidate.workspaceSnapshot !== active.workspaceSnapshot ||
        (candidate.workspaceRevision ?? null) !== (active.workspaceRevision ?? null) ||
        (candidate.eventExportDigest ?? null) !== (active.eventExportDigest ?? null));
}
function assertRequestPackExpectation(pack, request) {
    if (request.activePackId !== undefined && request.activePackId !== pack.manifest.packId) {
        throw new Error(`Compile request activePackId ${request.activePackId} does not match loaded pack ${pack.manifest.packId}`);
    }
}
export function loadPackForCompile(rootDir) {
    return loadPack(rootDir);
}
function resolveActivationCompileExpectation(options) {
    if (Object.prototype.hasOwnProperty.call(options, "expectation")) {
        throw new Error("Activation compile options expectation has been removed; use expectedTarget");
    }
    return options.expectedTarget;
}
function assertActivationCompileSafety(rootDir, slot, options) {
    if (slot !== "candidate" || options.requirePromotionSafe === false) {
        return;
    }
    const inspection = inspectActivationState(rootDir);
    if (!inspection.promotion.allowed) {
        throw new Error(`Candidate compile blocked: ${inspection.promotion.findings.join("; ")}`);
    }
}
function activationFreshnessNotes(rootDir, slot, target) {
    if (slot !== "active") {
        return [];
    }
    const inspection = inspectActivationState(rootDir);
    const candidate = inspection.candidate;
    if (candidate === null) {
        return [];
    }
    if (!inspection.promotion.allowed) {
        return [`candidate_rejected=${candidate.packId}:${inspection.promotion.findings.join(" | ")}`];
    }
    const candidateTarget = {
        packId: candidate.packId,
        routePolicy: candidate.routePolicy,
        routerIdentity: candidate.routerIdentity,
        workspaceSnapshot: candidate.workspaceSnapshot,
        workspaceRevision: candidate.workspaceRevision,
        eventRange: {
            start: candidate.eventRange.start,
            end: candidate.eventRange.end,
            count: candidate.eventRange.count
        },
        eventExportDigest: candidate.eventExportDigest,
        builtAt: candidate.builtAt
    };
    if (!isStrictlyFresherTarget(candidateTarget, target)) {
        return [];
    }
    return [`stale_route_warning=active pack ${target.packId} is behind promotion-ready candidate ${candidate.packId}`];
}
export function resolveActivationCompileTarget(rootDir, options = {}) {
    const slot = options.slot ?? "active";
    const pack = options.pack ?? loadPackFromActivation(rootDir, slot, {
        requireActivationReady: options.requireActivationReady !== false
    });
    if (pack === null) {
        throw new Error(`Activation slot ${slot} is empty`);
    }
    const target = options.target ?? describePackCompileTarget(pack);
    const expectation = resolveActivationCompileExpectation(options);
    if (expectation !== undefined) {
        const expectationErrors = validateRuntimeCompileExpectation(expectation);
        if (expectationErrors.length > 0) {
            throw new Error(`Invalid compile expectation: ${expectationErrors.join("; ")}`);
        }
        const compatibilityErrors = validateRuntimeCompileTargetExpectation(target, expectation);
        if (compatibilityErrors.length > 0) {
            throw new Error(`Activation compile target mismatch: ${compatibilityErrors.join("; ")}`);
        }
    }
    assertActivationCompileSafety(rootDir, slot, options);
    return {
        slot,
        pack,
        target
    };
}
export function loadPackForActivationCompile(rootDir, options = {}) {
    return resolveActivationCompileTarget(rootDir, options).pack;
}
function rankContextBlocksInternal(pack, request, options = {}) {
    const tokens = requestTokens(request);
    const vectorsByBlockId = new Map(pack.vectors.entries.map((entry) => [entry.blockId, entry]));
    const semantic = {
        queryEmbeddingModel: options.queryEmbedding?.model ?? null,
        comparedCount: 0,
        boostedCount: 0
    };
    const direct = pack.graph.blocks
        .map((block, packOrder) => {
        const vectorEntry = vectorsByBlockId.get(block.id);
        const weights = buildKeywordWeights(block, vectorEntry);
        const routingChannels = resolveRoutingChannels(block, vectorEntry);
        const semanticTyping = classifyCandidateSemanticType(block);
        const textTokens = new Set(normalizeTokens(`${block.source} ${block.text}`));
        const matchedTokens = [];
        let semanticSimilarity;
        const channelScores = {
            graph: routingChannels.includes("graph") ? baseBlockScore(block) + (block.routing?.graphBias ?? 0) : 0,
            shortTerm: routingChannels.includes("short_term")
                ? (block.state?.freshness ?? 1) * 2 + block.learning.humanLabels + block.learning.selfLabels * 0.5 + (block.routing?.shortTermBias ?? 0)
                : 0,
            vector: routingChannels.includes("vector") ? block.priority * 0.5 + (block.routing?.vectorBias ?? 0) : 0
        };
        const scoreTotal = () => channelScores.graph + channelScores.shortTerm + channelScores.vector;
        for (const token of tokens) {
            const weightedScore = weights.get(token);
            if (weightedScore !== undefined) {
                matchedTokens.push(token);
                channelScores.vector += weightedScore;
                if (routingChannels.includes("short_term")) {
                    channelScores.shortTerm += 0.5;
                }
                continue;
            }
            if (textTokens.has(token)) {
                matchedTokens.push(token);
                if (routingChannels.includes("vector")) {
                    channelScores.vector += 1;
                }
                else if (routingChannels.includes("short_term")) {
                    channelScores.shortTerm += 1;
                }
            }
        }
        if (routingChannels.includes("vector") &&
            options.queryEmbedding !== undefined &&
            options.queryEmbedding !== null &&
            vectorEntry?.embedding !== undefined &&
            vectorEntry.embedding.model === options.queryEmbedding.model) {
            const similarity = cosineSimilarity(options.queryEmbedding.values, vectorEntry.embedding.values);
            if (similarity !== null) {
                semantic.comparedCount += 1;
                semanticSimilarity = similarity;
                if (similarity > 0) {
                    semantic.boostedCount += 1;
                }
                channelScores.vector += clampScore(similarity, -1, 1) * 4;
            }
        }
        if (matchedTokens.length > 0 && vectorEntry !== undefined && routingChannels.includes("vector")) {
            channelScores.vector += vectorEntry.boost;
        }
        if ((block.state?.freshness ?? 1) < 0.4) {
            if (routingChannels.includes("short_term")) {
                channelScores.shortTerm -= 1;
            }
            else {
                channelScores.graph -= 1;
            }
        }
        // Noise heuristics are fallback evidence only; the typed gate classifies first.
        const qualityPenalty = computeBlockQualityPenalty(block.text);
        if (qualityPenalty > 0) {
            channelScores.graph -= qualityPenalty;
            channelScores.shortTerm -= qualityPenalty;
            channelScores.vector -= qualityPenalty;
        }
        return {
            blockId: block.id,
            source: block.source,
            text: block.text,
            channelScores,
            routingChannels,
            matchedTokens,
            ...(matchedTokens.length > 0 ? { directMatchedTokens: [...matchedTokens] } : {}),
            ...(semanticSimilarity !== undefined ? { semanticSimilarity } : {}),
            priority: block.priority,
            score: scoreTotal(),
            traversalScore: 0,
            tokenCount: block.tokenCount ?? estimateTokenCount(block.text),
            ...(block.compactedFrom !== undefined ? { compactedFrom: [...block.compactedFrom] } : {}),
            packOrder,
            candidateSemanticClass: semanticTyping.semanticClass,
            candidateSemanticEvidence: semanticTyping.evidence,
            routerPolicyUpdate: routerPolicyUpdateForBlock(pack, block.id)
        };
    });
    const gated = gateRankedCandidates(direct, request);
    for (const entry of gated.candidates) {
        const block = pack.graph.blocks[entry.packOrder];
        if (block === undefined || entry.routerPolicyUpdate === null) {
            continue;
        }
        const vectorEntry = vectorsByBlockId.get(entry.blockId);
        for (const token of tokens) {
            const routerWeight = entry.routerPolicyUpdate.tokenWeights[token];
            if (routerWeight === undefined) {
                continue;
            }
            if (!entry.matchedTokens.includes(token)) {
                entry.matchedTokens.push(token);
            }
            entry.channelScores.vector += routerWeight;
        }
        const learnedPrior = clampScore(entry.routerPolicyUpdate.delta * 0.45, -3, 4) +
            clampScore(Math.sign(entry.routerPolicyUpdate.delta || 0) * entry.routerPolicyUpdate.evidenceCount * 0.2, -1, 1.5) +
            clampScore(entry.routerPolicyUpdate.rewardSum * 0.1, -1.5, 1.5);
        if (entry.routingChannels.includes("short_term") && !entry.routingChannels.includes("graph")) {
            entry.channelScores.shortTerm += learnedPrior;
        }
        else if (entry.routingChannels.includes("vector")) {
            entry.channelScores.vector += learnedPrior;
        }
        else if (entry.routingChannels.includes("graph")) {
            entry.channelScores.graph += learnedPrior;
        }
        applyRouterFeaturePrior(block, vectorEntry, entry.routingChannels, entry.routerPolicyUpdate.tokenWeights, entry.channelScores);
        entry.score = entry.channelScores.graph + entry.channelScores.shortTerm + entry.channelScores.vector;
    }
    const rankedById = new Map(gated.candidates.map((entry) => [entry.blockId, entry]));
    const blocksById = new Map(pack.graph.blocks.map((block) => [block.id, block]));
    for (const entry of gated.candidates) {
        if (entry.matchedTokens.length === 0) {
            continue;
        }
        const sourceBlock = blocksById.get(entry.blockId);
        if (sourceBlock === undefined) {
            continue;
        }
        for (const edge of sourceBlock.edges ?? []) {
            const targetEntry = rankedById.get(edge.targetBlockId);
            const targetBlock = blocksById.get(edge.targetBlockId);
            if (targetEntry === undefined || targetBlock === undefined) {
                continue;
            }
            const traversalScore = edge.weight + (sourceBlock.state?.traversalBias ?? 0) * 0.5 + (targetBlock.state?.freshness ?? 1);
            if (targetEntry.routingChannels.includes("graph")) {
                targetEntry.channelScores.graph += traversalScore + (targetBlock.routing?.graphBias ?? 0);
            }
            else {
                targetEntry.channelScores.vector += traversalScore * 0.5;
            }
            targetEntry.score = targetEntry.channelScores.graph + targetEntry.channelScores.shortTerm + targetEntry.channelScores.vector;
            targetEntry.traversalScore = (targetEntry.traversalScore ?? 0) + traversalScore;
            for (const token of entry.matchedTokens.slice(0, 2)) {
                if (!targetEntry.matchedTokens.includes(token)) {
                    targetEntry.matchedTokens.push(token);
                }
            }
        }
    }
    const ranked = gated.candidates.sort((left, right) => {
        if (right.matchedTokens.length !== left.matchedTokens.length) {
            return right.matchedTokens.length - left.matchedTokens.length;
        }
        if (left.matchedTokens.length === 0 && right.matchedTokens.length === 0) {
            if (right.priority !== left.priority) {
                return right.priority - left.priority;
            }
        }
        if (right.score !== left.score) {
            return right.score - left.score;
        }
        if (right.priority !== left.priority) {
            return right.priority - left.priority;
        }
        return left.packOrder - right.packOrder;
    });
    return { ranked, semantic, gating: gated.summary };
}
export function rankContextBlocks(pack, request) {
    return rankContextBlocksInternal(pack, request).ranked;
}
export async function rankContextBlocksWithEmbedder(pack, request, embedder) {
    try {
        const queryEmbedding = await embedCompileRequest(embedder, request);
        if (queryEmbedding === null) {
            return rankContextBlocks(pack, request);
        }
        return rankContextBlocksInternal(pack, request, { queryEmbedding }).ranked;
    }
    catch {
        return rankContextBlocks(pack, request);
    }
}
function applyScoreDelta(entry, delta) {
    if (delta === 0) {
        return;
    }
    const channels = entry.routingChannels.length === 0 ? ["vector"] : entry.routingChannels;
    const perChannel = delta / channels.length;
    for (const channel of channels) {
        if (channel === "graph") {
            entry.channelScores.graph += perChannel;
        }
        else if (channel === "short_term") {
            entry.channelScores.shortTerm += perChannel;
        }
        else {
            entry.channelScores.vector += perChannel;
        }
    }
    entry.score = entry.channelScores.graph + entry.channelScores.shortTerm + entry.channelScores.vector;
}
function applyFailOpenFilter(candidates, semanticClass, downrank) {
    const retained = candidates.filter((candidate) => candidate.candidateSemanticClass !== semanticClass);
    const filtered = candidates.filter((candidate) => candidate.candidateSemanticClass === semanticClass);
    if (filtered.length === 0) {
        return {
            candidates,
            filteredCount: 0,
            retainedFailOpenCount: 0,
            failOpenApplied: false
        };
    }
    const matchedBeforeFilter = candidates.some((candidate) => candidate.matchedTokens.length > 0);
    const matchedAfterFilter = retained.some((candidate) => candidate.matchedTokens.length > 0);
    const failOpenApplied = retained.length === 0 || (matchedBeforeFilter && !matchedAfterFilter);
    if (failOpenApplied) {
        for (const candidate of filtered) {
            applyScoreDelta(candidate, -downrank);
        }
    }
    return {
        candidates: failOpenApplied ? [...retained, ...filtered] : retained,
        filteredCount: failOpenApplied ? 0 : filtered.length,
        retainedFailOpenCount: failOpenApplied ? filtered.length : 0,
        failOpenApplied
    };
}
function gateRankedCandidates(candidates, request) {
    const intent = classifyRetrievalIntent(request);
    const instructionalRequest = classifyInstructionalRequest(request);
    // TODO(openclawbrain): when suppression artifacts are materialized as first-class blocks,
    // fold them into this gate before router priors instead of inferring residue only from
    // source typing plus fallback noise signals.
    let workingCandidates = candidates;
    let instructionalFilteredCount = 0;
    let instructionalRetainedFailOpenCount = 0;
    let instructionalDownrankedCount = 0;
    if (!instructionalRequest.allowScaffolding) {
        const scaffolding = applyFailOpenFilter(workingCandidates, "instructional_scaffolding", NON_INSTRUCTIONAL_SCAFFOLD_DOWNRANK);
        workingCandidates = scaffolding.candidates;
        instructionalFilteredCount = scaffolding.filteredCount;
        instructionalRetainedFailOpenCount = scaffolding.retainedFailOpenCount;
        instructionalDownrankedCount = scaffolding.retainedFailOpenCount;
    }
    if (intent.intent === "diagnostic_observability") {
        let diagnosticBoostedCount = 0;
        for (const candidate of workingCandidates) {
            if (candidate.candidateSemanticClass === "observability_support") {
                applyScoreDelta(candidate, DIAGNOSTIC_OBSERVABILITY_BOOST);
                diagnosticBoostedCount += 1;
            }
            else if (candidate.candidateSemanticClass === "transport_residue") {
                applyScoreDelta(candidate, DIAGNOSTIC_TRANSPORT_BOOST);
                diagnosticBoostedCount += 1;
            }
        }
        return {
            candidates: workingCandidates,
            summary: {
                intent: intent.intent,
                mode: "diagnostic_carve_out",
                signals: [...intent.signals, ...instructionalRequest.signals],
                transportFilteredCount: 0,
                transportRetainedFailOpenCount: 0,
                instructionalFilteredCount,
                instructionalRetainedFailOpenCount,
                instructionalDownrankedCount,
                observabilityDownrankedCount: 0,
                diagnosticBoostedCount
            }
        };
    }
    if (intent.intent === "uncertain") {
        let observabilityDownrankedCount = 0;
        for (const candidate of workingCandidates) {
            if (candidate.candidateSemanticClass === "transport_residue") {
                applyScoreDelta(candidate, -UNCERTAIN_TRANSPORT_DOWNRANK);
                observabilityDownrankedCount += 1;
            }
            else if (candidate.candidateSemanticClass === "observability_support") {
                applyScoreDelta(candidate, -UNCERTAIN_OBSERVABILITY_DOWNRANK);
                observabilityDownrankedCount += 1;
            }
        }
        return {
            candidates: workingCandidates,
            summary: {
                intent: intent.intent,
                mode: "uncertain_penalty",
                signals: [...intent.signals, ...instructionalRequest.signals],
                transportFilteredCount: 0,
                transportRetainedFailOpenCount: 0,
                instructionalFilteredCount,
                instructionalRetainedFailOpenCount,
                instructionalDownrankedCount,
                observabilityDownrankedCount,
                diagnosticBoostedCount: 0
            }
        };
    }
    const transport = applyFailOpenFilter(workingCandidates, "transport_residue", FAIL_OPEN_TRANSPORT_DOWNRANK);
    const effective = transport.candidates;
    let observabilityDownrankedCount = 0;
    for (const candidate of effective) {
        if (candidate.candidateSemanticClass === "observability_support") {
            applyScoreDelta(candidate, -NORMAL_OBSERVABILITY_DOWNRANK);
            observabilityDownrankedCount += 1;
        }
    }
    return {
        candidates: effective,
        summary: {
            intent: intent.intent,
            mode: transport.failOpenApplied ? "normal_fail_open" : "normal_filter",
            signals: [...intent.signals, ...instructionalRequest.signals],
            transportFilteredCount: transport.filteredCount,
            transportRetainedFailOpenCount: transport.retainedFailOpenCount,
            instructionalFilteredCount,
            instructionalRetainedFailOpenCount,
            instructionalDownrankedCount,
            observabilityDownrankedCount,
            diagnosticBoostedCount: 0
        }
    };
}
const SEED_ROLES = new Set(["boot_default", "workspace", "structural"]);
function buildStructuralSignals(ranked, selectedContext, selection) {
    const selectedCoverageIds = new Set(selectedContext.flatMap((block) => flattenContextIds(block)));
    const traversalActivatedBlockIds = ranked.filter((entry) => (entry.traversalScore ?? 0) > 0).map((entry) => entry.blockId);
    const overlapPrunedBlockIds = [...selection.overlapPrunedBlockIds];
    const overlapPrunedSet = new Set(overlapPrunedBlockIds);
    const candidates = ranked.map((entry, index) => {
        const selected = flattenRankedContextIds(entry).some((coverageId) => selectedCoverageIds.has(coverageId));
        return {
            blockId: entry.blockId,
            rank: index + 1,
            score: entry.score,
            selected,
            selectedBy: selected ? (entry.matchedTokens.length > 0 ? "token_match" : "priority_fallback") : null,
            matchedTokens: [...entry.matchedTokens],
            directMatchedTokens: [...(entry.directMatchedTokens ?? [])],
            traversalActivated: (entry.traversalScore ?? 0) > 0,
            traversalScore: entry.traversalScore ?? 0,
            overlapPruned: overlapPrunedSet.has(entry.blockId),
            compactedFrom: [...(entry.compactedFrom ?? [])]
        };
    });
    const selectedBlockIds = candidates.filter((candidate) => candidate.selected).map((candidate) => candidate.blockId);
    const selectedMatchedCount = candidates.filter((candidate) => candidate.selectedBy === "token_match").length;
    const selectedPriorityFallbackCount = candidates.filter((candidate) => candidate.selectedBy === "priority_fallback").length;
    return {
        matchedCandidateCount: ranked.filter((entry) => entry.matchedTokens.length > 0).length,
        selectedMatchedCount,
        selectedPriorityFallbackCount,
        overlapPrunedCount: selection.overlapPrunedCount,
        traversalActivatedCount: traversalActivatedBlockIds.length,
        selectedBlockIds,
        overlapPrunedBlockIds,
        traversalActivatedBlockIds,
        candidates
    };
}
function compileRuntimeCore(packOrRoot, request, options = {}, rankOptions = {}) {
    const requestErrors = validateRuntimeCompileRequest(request);
    if (requestErrors.length > 0) {
        throw new Error(`Invalid compile request: ${requestErrors.join("; ")}`);
    }
    const pack = typeof packOrRoot === "string" ? loadPackForCompile(packOrRoot) : packOrRoot;
    assertRequestPackExpectation(pack, request);
    const modeEffective = determineRouteMode(pack, request.modeRequested);
    const learnedRouteSelectionOverride = modeEffective === "learned" &&
        typeof options._learnedRouteSelectionOverride?.select === "function"
        ? options._learnedRouteSelectionOverride
        : null;
    const usedLearnedRouteFn = modeEffective === "learned" && learnedRouteSelectionOverride === null;
    if (usedLearnedRouteFn && pack.router === null) {
        throw new Error("learned-routing pack cannot compile without a router artifact");
    }
    const ranking = rankContextBlocksInternal(pack, request, rankOptions);
    const ranked = ranking.ranked;
    const maxBlocks = Math.max(0, request.maxContextBlocks);
    const matched = ranked.filter((entry) => entry.matchedTokens.length > 0);
    const selectionMode = options.selectionMode ?? "flat_rank_v1";
    const learnedRouteSelectionOverrideResult = learnedRouteSelectionOverride === null
        ? null
        : normalizeLearnedRouteSelectionOverrideResult(learnedRouteSelectionOverride.select({
            request,
            ranked,
            maxBlocks
        }));
    const rankedWithOverrideBoosts = learnedRouteSelectionOverrideResult !== null
        && learnedRouteSelectionOverrideResult.selectionMode === "graph_walk_score_boost"
        ? applyLearnedRouteScoreBoosts(ranked, learnedRouteSelectionOverrideResult.scoreBoostsByBlockId)
        : ranked;
    const selection = learnedRouteSelectionOverrideResult !== null
        ? learnedRouteSelectionOverrideResult.selectionMode === "graph_walk_score_boost"
            ? selectionMode === "graph_walk_v1"
                ? selectContextBlocksByGraphWalk(pack, rankedWithOverrideBoosts, maxBlocks)
                : selectContextBlocks(rankedWithOverrideBoosts, maxBlocks)
            : selectContextBlocksByLearnedRouteOverride(ranked, maxBlocks, learnedRouteSelectionOverrideResult.selectedBlockIds)
        : selectionMode === "graph_walk_v1"
            ? selectContextBlocksByGraphWalk(pack, ranked, maxBlocks)
            : selectContextBlocks(ranked, maxBlocks);
    const selected = selection.selected;
    const selectedBlockIds = new Set(selected.map((block) => block.id));
    const compactionMode = request.compactionMode ?? "native";
    const charsBefore = totalCharCount(selected);
    const fitted = fitContextToCharBudget(selected, request.maxContextChars, compactionMode);
    const selectedContext = fitted.blocks;
    const traversalActivatedCount = ranked.filter((entry) => (entry.traversalScore ?? 0) > 0).length;
    const learnedRoutePolicyUpdateCandidateCount = ranked.filter((entry) => entry.routerPolicyUpdate !== null).length;
    const learnedRoutePolicyUpdateSelectedCount = ranked.filter((entry) => selectedBlockIds.has(entry.blockId) && entry.routerPolicyUpdate !== null).length;
    const learnedRouteStopPolicyDecisionCount = selection.graphWalkLearnedStopPolicyDecisionCount ?? 0;
    const learnedRouteEvidence = learnedRouteSelectionOverrideResult !== null
        ? learnedRouteSelectionOverrideResult.evidenceSource
        : usedLearnedRouteFn
            ? "learned_route_fn"
            : learnedRoutePolicyUpdateSelectedCount > 0
                ? "router_policy_update_selected"
                : learnedRouteStopPolicyDecisionCount > 0
                    ? "graph_walk_stop_policy"
                    : learnedRoutePolicyUpdateCandidateCount > 0
                        ? "router_policy_update_candidate_only"
                        : "none";
    const candidateRoutingChannels = summarizeRoutingChannels(ranked);
    const selectedRoutingChannels = summarizeRoutingChannels(ranked.filter((entry) => selectedContext.some((selectedBlock) => selectedBlock.id === entry.blockId)));
    const structuralSignals = buildStructuralSignals(ranked, selectedContext, selection);
    const initHandoff = describePackInitHandoff(pack);
    const notes = [];
    notes.push(`selected_context_ids=${selectedContext.map((block) => block.id).join(",")}`);
    notes.push(`candidate_gate_intent=${ranking.gating.intent}`);
    notes.push(`candidate_gate_mode=${ranking.gating.mode}`);
    notes.push(`candidate_gate_transport_filtered=${ranking.gating.transportFilteredCount}`);
    notes.push(`candidate_gate_transport_retained_fail_open=${ranking.gating.transportRetainedFailOpenCount}`);
    notes.push(`candidate_gate_instructional_filtered=${ranking.gating.instructionalFilteredCount}`);
    notes.push(`candidate_gate_instructional_retained_fail_open=${ranking.gating.instructionalRetainedFailOpenCount}`);
    notes.push(`candidate_gate_instructional_downranked=${ranking.gating.instructionalDownrankedCount}`);
    notes.push(`candidate_gate_observability_downranked=${ranking.gating.observabilityDownrankedCount}`);
    notes.push(`candidate_gate_diagnostic_boosted=${ranking.gating.diagnosticBoostedCount}`);
    if (ranking.gating.signals.length > 0) {
        notes.push(`candidate_gate_signals=${ranking.gating.signals.join("|")}`);
    }
    notes.push(matched.length > 0 ? `selection_mode=token_match(${requestTokens(request).join(",")})` : "selection_mode=priority_fallback");
    notes.push(matched.length > 0 && selection.matchedSelectedCount < maxBlocks && selection.selected.length > selection.matchedSelectedCount
        ? "selection_tiers=token_match+priority_fallback"
        : matched.length > 0
            ? "selection_tiers=token_match_only"
            : "selection_tiers=priority_fallback_only");
    notes.push("selection_strategy=pack_route_fn_selection_v1");
    if (learnedRouteSelectionOverrideResult !== null) {
        notes.push(`replay_learned_route_override=${learnedRouteSelectionOverrideResult.evidenceSource}`);
        if (learnedRouteSelectionOverrideResult.routerIdentity !== null) {
            notes.push(`replay_learned_route_override_router_identity=${learnedRouteSelectionOverrideResult.routerIdentity}`);
        }
    }
    if (selectionMode === "graph_walk_v1" && learnedRouteSelectionOverrideResult === null) {
        notes.push("selection_graph_walk=graph_walk_v1");
        notes.push(`selection_graph_walk_seed_count=${selection.graphWalkSeedBlockIds.length}`);
        notes.push(`selection_graph_walk_hops=${selection.graphWalkHopCount}`);
        if (selection.graphWalkSeedBlockIds.length > 0) {
            notes.push(`selection_graph_walk_seeds=${selection.graphWalkSeedBlockIds.join(",")}`);
        }
    }
    if (modeEffective !== request.modeRequested) {
        notes.push(`learned_required_enforced=requested_${request.modeRequested}->${modeEffective}`);
    }
    if (selection.overlapPrunedCount > 0) {
        notes.push(`selection_compaction_deduped=${selection.overlapPrunedCount}`);
    }
    if (traversalActivatedCount > 0) {
        notes.push(`graph_traversal_activated=${traversalActivatedCount}`);
    }
    notes.push(`learned_route_evidence=${learnedRouteEvidence}`);
    notes.push(`learned_route_policy_update_candidates=${learnedRoutePolicyUpdateCandidateCount}`);
    notes.push(`learned_route_policy_update_selected=${learnedRoutePolicyUpdateSelectedCount}`);
    notes.push(`learned_route_stop_policy_decisions=${learnedRouteStopPolicyDecisionCount}`);
    notes.push(`pack_graph_blocks=${pack.graph.blocks.length}`);
    notes.push(`runtime_plasticity_source=${pack.manifest.graphDynamics.runtimePlasticitySource}`);
    if (pack.graph.evolution !== undefined) {
        notes.push(`graph_evolution=split:${pack.graph.evolution.structuralOps.split},merge:${pack.graph.evolution.structuralOps.merge},prune:${pack.graph.evolution.structuralOps.prune},connect:${pack.graph.evolution.structuralOps.connect}`);
        if (pack.graph.evolution.strongestBlockId !== null) {
            notes.push(`graph_strongest_block=${pack.graph.evolution.strongestBlockId}`);
        }
    }
    if (request.maxContextChars !== undefined) {
        notes.push(`max_context_chars=${request.maxContextChars}`);
    }
    if (fitted.compactionApplied) {
        notes.push("native_structural_compaction=applied");
    }
    notes.push(`routing_channels_candidates=graph:${candidateRoutingChannels.graph},short_term:${candidateRoutingChannels.shortTerm},vector:${candidateRoutingChannels.vector}`);
    notes.push(`routing_channels_selected=graph:${selectedRoutingChannels.graph},short_term:${selectedRoutingChannels.shortTerm},vector:${selectedRoutingChannels.vector}`);
    if (ranking.semantic.queryEmbeddingModel !== null) {
        notes.push(`semantic_vector_query_model=${ranking.semantic.queryEmbeddingModel}`);
        notes.push(`semantic_vector_compared=${ranking.semantic.comparedCount}`);
        notes.push(`semantic_vector_boosted=${ranking.semantic.boostedCount}`);
        if (ranking.semantic.comparedCount === 0) {
            notes.push("semantic_vector_fallback=keyword_weights_only");
        }
    }
    if (initHandoff.initMode !== null) {
        notes.push(`init_mode=${initHandoff.initMode}`);
    }
    notes.push(`seed_state_visible=${initHandoff.seedStateVisible}`);
    notes.push(`seed_state_block_count=${initHandoff.seedBlockCount}`);
    if (initHandoff.seedSources.length > 0) {
        notes.push(`seed_sources=${initHandoff.seedSources.join("|")}`);
    }
    if (initHandoff.seedRoles.length > 0) {
        notes.push(`seed_roles=${initHandoff.seedRoles.join("|")}`);
    }
    notes.push(`handoff_state=${initHandoff.handoffState}`);
    notes.push(`pg_route_authoritative=${initHandoff.pgRouteAuthoritative}`);
    if (initHandoff.learnedRouteUpdateCount !== null) {
        notes.push(`learned_route_update_count=${initHandoff.learnedRouteUpdateCount}`);
    }
    if (pack.router !== null && learnedRouteEvidence !== "none" && learnedRouteSelectionOverrideResult === null) {
        notes.push(`router_strategy=${pack.router.strategy}`);
        notes.push(`router_update_method=${pack.router.training.method}`);
        notes.push(`router_refresh_status=${pack.router.training.status}`);
        notes.push(`router_update_mechanism=${pack.router.training.objective.updateMechanism}`);
        notes.push(`router_update_version=${pack.router.training.objective.updateVersion}`);
        notes.push(`router_objective=${pack.router.training.objective.objective}`);
        notes.push(`router_pg_trace_source=${pack.router.training.objective.profile.traceSource}`);
        notes.push(`router_pg_action_space=${pack.router.training.objective.profile.actionSpace}`);
        notes.push(`router_pg_target_construction=${pack.router.training.objective.profile.targetConstruction}`);
        notes.push(`router_pg_reward_signal=${pack.router.training.objective.profile.rewardSignal}`);
        notes.push(`router_pg_baseline=${pack.router.training.objective.profile.baseline}`);
        notes.push(`router_pg_off_policy_correction=${pack.router.training.objective.profile.offPolicyCorrection}`);
        notes.push(`router_pg_update_cadence=${pack.router.training.objective.profile.updateCadence}`);
        notes.push(`router_objective_checksum=${pack.router.training.objective.objectiveChecksum}`);
        notes.push(`router_update_count=${pack.router.training.updateCount}`);
        notes.push(`router_route_trace_count=${pack.router.training.routeTraceCount}`);
        notes.push(`router_supervision_count=${pack.router.training.supervisionCount}`);
        notes.push(`router_collected_labels_total=${pack.router.training.collectedLabels.total}`);
        notes.push(`router_collected_labels_human_feedback=${pack.router.training.collectedLabels.humanFeedback}`);
        notes.push(`router_collected_labels_operator_override=${pack.router.training.collectedLabels.operatorOverride}`);
        notes.push(`router_collected_labels_self_memory=${pack.router.training.collectedLabels.selfMemory}`);
        if (pack.router.training.eventExportDigest !== null) {
            notes.push(`router_event_export_digest=${pack.router.training.eventExportDigest}`);
        }
        if (pack.manifest.payloadChecksums.router !== null) {
            notes.push(`router_checksum=${pack.manifest.payloadChecksums.router}`);
        }
        notes.push(`router_weights_checksum=${pack.router.training.weightsChecksum}`);
        notes.push(`router_freshness_checksum=${pack.router.training.freshnessChecksum}`);
        const topDeltas = routerDeltaSummary(pack);
        if (topDeltas !== undefined) {
            notes.push(`router_top_deltas=${topDeltas}`);
        }
        if (pack.router.training.noOpReason !== null) {
            notes.push(`router_noop_warning=${pack.router.training.noOpReason}`);
        }
        notes.push("router_update_source=promoted_pack_only");
    }
    notes.push("brain_boundary=promoted_pack_compile_only");
    const response = {
        contract: CONTRACT_IDS.runtimeCompile,
        packId: pack.manifest.packId,
        selectedContext,
        diagnostics: {
            modeRequested: request.modeRequested,
            modeEffective,
            usedLearnedRouteFn,
            routerIdentity: learnedRouteSelectionOverrideResult?.routerIdentity ?? pack.router?.routerIdentity ?? null,
            servedArtifact: buildServedArtifactProof(describePackCompileTarget(pack), pack.manifest.routeArtifact),
            candidateCount: ranked.length,
            selectedCount: selectedContext.length,
            selectedCharCount: totalCharCount(selectedContext),
            selectedTokenCount: totalTokenCount(selectedContext),
            selectionStrategy: "pack_route_fn_selection_v1",
            selectionDigest: checksumJsonPayload({
                packId: pack.manifest.packId,
                selectedContext: selectedContext.map((block) => ({
                    id: block.id,
                    source: block.source,
                    text: block.text,
                    tokenCount: block.tokenCount ?? null,
                    compactedFrom: block.compactedFrom ?? null
                }))
            }),
            structuralSignals,
            compactionMode,
            compactionApplied: fitted.compactionApplied,
            routingChannels: {
                candidates: candidateRoutingChannels,
                selected: selectedRoutingChannels
            },
            notes
        }
    };
    const responseErrors = validateRuntimeCompileResponse(response);
    if (responseErrors.length > 0) {
        throw new Error(`Invalid compile response: ${responseErrors.join("; ")}`);
    }
    const compactedBlocks = selectedContext.filter((b) => (b.compactedFrom?.length ?? 0) > 0).length;
    const seedBlockCount = pack.graph.blocks.filter((b) => SEED_ROLES.has(b.learning.role)).length;
    const economicsLog = {
        packId: pack.manifest.packId,
        plasticitySource: pack.manifest.graphDynamics.runtimePlasticitySource,
        requestedBudget: {
            maxBlocks,
            maxChars: request.maxContextChars ?? null
        },
        selectionCounts: {
            candidates: ranked.length,
            selected: selectedContext.length,
            overlapDropped: selection.overlapPrunedCount,
            compactedBlocks
        },
        usedBudget: {
            chars: totalCharCount(selectedContext),
            tokens: totalTokenCount(selectedContext),
            blocks: selectedContext.length
        },
        packBlockCounts: {
            total: pack.graph.blocks.length,
            seed: seedBlockCount,
            brain: pack.graph.blocks.length - seedBlockCount
        },
        routingChannels: {
            candidates: { graph: candidateRoutingChannels.graph, shortTerm: candidateRoutingChannels.shortTerm, vector: candidateRoutingChannels.vector },
            selected: { graph: selectedRoutingChannels.graph, shortTerm: selectedRoutingChannels.shortTerm, vector: selectedRoutingChannels.vector }
        },
        compaction: {
            applied: fitted.compactionApplied,
            mode: compactionMode,
            charsBefore: request.maxContextChars !== undefined ? charsBefore : null,
            charsAfter: request.maxContextChars !== undefined ? totalCharCount(selectedContext) : null
        }
    };
    return { response, economicsLog };
}
export function compileRuntime(packOrRoot, request, options = {}) {
    return compileRuntimeCore(packOrRoot, request, options).response;
}
export async function compileRuntimeWithEmbedder(packOrRoot, request, embedder, options = {}) {
    try {
        const queryEmbedding = await embedCompileRequest(embedder, request);
        if (queryEmbedding === null) {
            return annotateCompileResponse(compileRuntime(packOrRoot, request, options), ["semantic_vector_fallback=empty_query_embedding"]);
        }
        return compileRuntimeCore(packOrRoot, request, options, { queryEmbedding }).response;
    }
    catch {
        return annotateCompileResponse(compileRuntime(packOrRoot, request, options), ["semantic_vector_fallback=embedder_error"]);
    }
}
function finalizeActivationCompileResult(rootDir, resolved, response, economicsLog) {
    const freshnessNotes = activationFreshnessNotes(rootDir, resolved.slot, resolved.target);
    const targetNotes = [
        `activation_slot=${resolved.slot}`,
        `target_pack_id=${resolved.target.packId}`,
        `target_route_policy=${resolved.target.routePolicy}`,
        `target_workspace_snapshot=${resolved.target.workspaceSnapshot}`,
        resolved.target.workspaceRevision === null ? undefined : `target_workspace_revision=${resolved.target.workspaceRevision}`,
        `target_event_range=${resolved.target.eventRange.start}-${resolved.target.eventRange.end}#${resolved.target.eventRange.count}`,
        resolved.target.eventExportDigest === null ? undefined : `target_event_export_digest=${resolved.target.eventExportDigest}`,
        `target_built_at=${resolved.target.builtAt}`,
        resolved.target.routerIdentity === null ? undefined : `target_router_identity=${resolved.target.routerIdentity}`,
        resolved.pack.manifest.payloadChecksums.router === null ? undefined : `target_router_checksum=${resolved.pack.manifest.payloadChecksums.router}`
    ];
    const resolvedResponse = {
        ...response,
        diagnostics: {
            ...response.diagnostics,
            servedArtifact: buildServedArtifactProof(resolved.target, resolved.pack.manifest.routeArtifact),
            notes: mergeDiagnosticNotes(response.diagnostics.notes, [...freshnessNotes, ...targetNotes])
        }
    };
    const responseErrors = validateRuntimeCompileResponse(resolvedResponse);
    if (responseErrors.length > 0) {
        throw new Error(`Invalid compile response: ${responseErrors.join("; ")}`);
    }
    return {
        ...resolvedResponse,
        slot: resolved.slot,
        target: resolved.target,
        response: resolvedResponse,
        economicsLog
    };
}
export function compileRuntimeFromActivation(rootDir, request, options = {}) {
    const resolved = resolveActivationCompileTarget(rootDir, options);
    const compiledRequest = request.activePackId === undefined && resolved.slot === "active"
        ? {
            ...request,
            activePackId: resolved.target.packId
        }
        : request;
    const { response, economicsLog } = compileRuntimeCore(resolved.pack, compiledRequest, options);
    return finalizeActivationCompileResult(rootDir, resolved, response, economicsLog);
}
export async function compileRuntimeFromActivationWithEmbedder(rootDir, request, embedder, options = {}) {
    const resolved = resolveActivationCompileTarget(rootDir, options);
    const compiledRequest = request.activePackId === undefined && resolved.slot === "active"
        ? {
            ...request,
            activePackId: resolved.target.packId
        }
        : request;
    try {
        const queryEmbedding = await embedCompileRequest(embedder, compiledRequest);
        if (queryEmbedding === null) {
            const fallback = compileRuntimeFromActivation(rootDir, request, options);
            return {
                ...annotateCompileResponse(fallback, ["semantic_vector_fallback=empty_query_embedding"]),
                slot: fallback.slot,
                target: fallback.target,
                response: annotateCompileResponse(fallback.response, ["semantic_vector_fallback=empty_query_embedding"]),
                economicsLog: fallback.economicsLog
            };
        }
        const { response, economicsLog } = compileRuntimeCore(resolved.pack, compiledRequest, options, { queryEmbedding });
        return finalizeActivationCompileResult(rootDir, resolved, response, economicsLog);
    }
    catch {
        const fallback = compileRuntimeFromActivation(rootDir, request, options);
        return {
            ...annotateCompileResponse(fallback, ["semantic_vector_fallback=embedder_error"]),
            slot: fallback.slot,
            target: fallback.target,
            response: annotateCompileResponse(fallback.response, ["semantic_vector_fallback=embedder_error"]),
            economicsLog: fallback.economicsLog
        };
    }
}
export function describeCompileFallbackUsage(response) {
    const selectionModeValue = noteValue(response.diagnostics.notes, "selection_mode=");
    const selectionTiersValue = noteValue(response.diagnostics.notes, "selection_tiers=");
    const selectionMode = selectionModeValue === null
        ? null
        : selectionModeValue.startsWith("token_match(")
            ? "token_match"
            : selectionModeValue === "priority_fallback"
                ? "priority_fallback"
                : null;
    const selectionTiers = selectionTiersValue === "token_match+priority_fallback" ||
        selectionTiersValue === "token_match_only" ||
        selectionTiersValue === "priority_fallback_only"
        ? selectionTiersValue
        : null;
    return {
        packId: response.packId,
        modeRequested: response.diagnostics.modeRequested,
        modeEffective: response.diagnostics.modeEffective,
        usedLearnedRouteFn: response.diagnostics.usedLearnedRouteFn,
        routerIdentity: response.diagnostics.routerIdentity,
        selectionDigest: response.diagnostics.selectionDigest,
        selectionMode,
        selectionTiers,
        priorityFallbackUsed: selectionMode === "priority_fallback" ||
            selectionTiers === "token_match+priority_fallback" ||
            selectionTiers === "priority_fallback_only",
        notes: response.diagnostics.notes.filter((note) => note.startsWith("selection_mode=") || note.startsWith("selection_tiers="))
    };
}
