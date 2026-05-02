import { hashText } from './redact.js';
import { detectCaptureIntent, detectRetrievalIntent } from './capture-intent.js';
export class RouteCache {
    cache = new Map();
    get(fingerprint) {
        const key = fingerprintKey(fingerprint);
        const hit = this.cache.get(key);
        if (!hit)
            return null;
        if (Date.parse(hit.expiresAt) < Date.now()) {
            this.cache.delete(key);
            return null;
        }
        return hit;
    }
    set(fingerprint, plan) {
        this.cache.set(fingerprintKey(fingerprint), plan);
    }
    invalidate(predicate) {
        if (!predicate) {
            this.cache.clear();
            return;
        }
        for (const [key, value] of this.cache.entries()) {
            if (predicate(key, value))
                this.cache.delete(key);
        }
    }
}
export class RouteFn {
    config;
    cache;
    store;
    constructor(options) {
        this.config = options.config;
        this.cache = options.cache ?? new RouteCache();
        this.store = options.store;
    }
    fingerprint(packet) {
        const message = packet.latestUserMessageRedacted.toLowerCase();
        const captureIntent = detectCaptureIntent(packet);
        const retrievalIntent = detectRetrievalIntent(packet);
        return {
            agentId: packet.agentId,
            scopeKey: packet.sessionId || packet.sessionKey || undefined,
            taskTypeHint: String(packet.metadata.turnType || ''),
            topicKeys: extractTopicKeys(message),
            explicitMemoryReference: /\b(as before|like i said|same as last time|we discussed before|remember)\b/i.test(packet.latestUserMessageRedacted),
            explicitCorrectionCue: /\b(actually|instead|no,|don't|do not|wrong|use .* instead)\b/i.test(packet.latestUserMessageRedacted),
            captureIntent: captureIntent.intent,
            retrievalIntent: retrievalIntent.intent,
        };
    }
    plan(packet) {
        const fingerprint = this.fingerprint(packet);
        const cached = this.cache.get(fingerprint);
        const turnFrame = turnFrameFromPacket(packet);
        const captureIntent = detectCaptureIntent(packet);
        const retrievalIntent = detectRetrievalIntent(packet);
        if (cached) {
            return {
                route: cached.route,
                confidence: cached.confidence,
                turnFrame,
                retrievalPlan: cached.retrievalPlan,
                injectionPlan: cached.injectionPlan,
                shouldRetrieve: retrievalIntent.shouldRetrieve,
                enqueueCapture: captureIntent.shouldConsiderCapture,
                retrievalIntent: cached.retrievalIntent ?? retrievalIntent,
                captureIntent: cached.captureIntent ?? captureIntent,
                latencyReason: 'cached route plan',
                policySnapshotId: this.store?.getActivePolicySnapshot?.(packet.agentId)?.id,
            };
        }
        const policySnapshot = this.loadPolicySnapshot(packet);
        const plan = heuristicRoutePlan(packet, turnFrame, this.config, policySnapshot, retrievalIntent, captureIntent);
        this.cache.set(fingerprint, {
            route: plan.route,
            retrievalPlan: plan.retrievalPlan,
            injectionPlan: plan.injectionPlan,
            confidence: plan.confidence,
            retrievalIntent: plan.retrievalIntent,
            captureIntent: plan.captureIntent,
            expiresAt: new Date(Date.now() + 5 * 60 * 1000).toISOString(),
        });
        return plan;
    }
    loadPolicySnapshot(packet) {
        if (!this.store)
            return null;
        try {
            return this.store.getActivePolicySnapshot(packet.agentId);
        }
        catch {
            return null;
        }
    }
}
function heuristicRoutePlan(packet, turnFrame, config, policySnapshot, retrievalIntent, captureIntent) {
    const message = packet.latestUserMessageRedacted.toLowerCase();
    const explicitCorrectionCue = /\b(actually|instead|wrong|no,)\b/i.test(packet.latestUserMessageRedacted);
    const planningLike = /\b(plan|design|architecture|file-by-file|implementation)\b/.test(message);
    let route = 'no_memory';
    let confidence = Math.max(retrievalIntent.confidence, captureIntent.confidence);
    if (explicitCorrectionCue && retrievalIntent.shouldRetrieve) {
        route = 'high_confidence_correction_only';
        confidence = Math.max(confidence, 0.9);
    }
    else if (retrievalIntent.shouldRetrieve && captureIntent.shouldConsiderCapture) {
        route = 'retrieve_and_distill';
    }
    else if (retrievalIntent.shouldRetrieve) {
        route = 'retrieve_memory';
        confidence = Math.max(confidence, planningLike ? 0.82 : 0.72);
    }
    else if (captureIntent.shouldConsiderCapture) {
        route = 'capture_only';
        confidence = Math.max(confidence, captureIntent.confidence);
    }
    const policyBoost = policySnapshot ? applyPolicySnapshot(packet, turnFrame, policySnapshot) : null;
    if (policyBoost && policyBoost.route && !explicitCorrectionCue && retrievalIntent.intent !== 'no_retrieval') {
        route = policyBoost.route;
        confidence = Math.max(confidence, policyBoost.confidence);
    }
    const heuristicQueries = buildQueries(packet, retrievalIntent);
    const policyQueries = policyBoost?.queries ?? [];
    const allQueries = [...new Set([...heuristicQueries, ...policyQueries])];
    const heuristicMemoryTypes = memoryTypesForTurn(route, retrievalIntent, captureIntent, message, planningLike);
    const policyMemoryTypes = policyBoost?.memoryTypes ?? [];
    const allMemoryTypes = [...new Set([...heuristicMemoryTypes, ...policyMemoryTypes])];
    const retrievalPlan = {
        queries: allQueries,
        memoryTypes: allMemoryTypes,
        requiredTags: [],
        excludedTags: retrievalIntent.includeRecallRules ? [] : ['recall_value'],
        graphDepth: planningLike || policyBoost ? 1 : 0,
        maxCandidates: config.routing.maxCandidateMemories,
    };
    const injectionPlan = {
        maxItems: config.routing.maxInjectedMemories,
        maxChars: config.routing.maxInjectedChars,
        preferredFormat: explicitCorrectionCue ? 'rules' : planningLike ? 'bullets' : retrievalIntent.intent === 'recall_value_request' ? 'rules' : 'none',
    };
    return {
        route,
        confidence,
        turnFrame,
        retrievalPlan,
        injectionPlan,
        shouldRetrieve: retrievalIntent.shouldRetrieve,
        enqueueCapture: captureIntent.shouldConsiderCapture,
        retrievalIntent,
        captureIntent,
        latencyReason: policySnapshot ? 'heuristic with policy snapshot' : 'heuristic uncached route',
        policySnapshotId: policySnapshot?.id,
    };
}
function applyPolicySnapshot(packet, turnFrame, policySnapshot) {
    const boost = { route: null, confidence: 0, memoryTypes: [], queries: [] };
    if (!policySnapshot?.policyText)
        return boost;
    const policy = String(policySnapshot.policyText).toLowerCase();
    const taskType = turnFrame.taskType;
    const taskTypeLine = policy.split('\n').find(line => line.includes(taskType));
    if (!taskTypeLine)
        return boost;
    if (/retrieve|memory|pull/.test(taskTypeLine) && taskTypeLine.includes(taskType)) {
        boost.route = 'retrieve_memory';
        boost.confidence = 0.78;
    }
    if (/no memory|prefer no|skip memory/.test(taskTypeLine)) {
        boost.route = 'no_memory';
        boost.confidence = 0.7;
    }
    const typeMatches = taskTypeLine.match(/\b(correction|preference|workflow|context|project_fact|tool_convention|routing_rule|agent_assignment|recall_rule|outcome)\b/gi);
    if (typeMatches)
        boost.memoryTypes = [...new Set(typeMatches.map(t => t.toLowerCase()))];
    if (/planning/.test(taskType))
        boost.queries.push('implementation planning architecture preferences workflow');
    if (/coding/.test(taskType) && /install|dependency|package/.test(policy))
        boost.queries.push('package manager correction workflow repo setup');
    return boost;
}
function memoryTypesForTurn(route, retrievalIntent, captureIntent, lower, planningLike) {
    if (retrievalIntent.intent === 'recall_value_request')
        return ['recall_rule'];
    if (route === 'high_confidence_correction_only')
        return ['correction'];
    const types = new Set(retrievalIntent.memoryTypes);
    if (planningLike)
        ['correction', 'preference', 'workflow', 'project_fact', 'tool_convention', 'routing_rule', 'agent_assignment', 'outcome', 'context'].forEach((t) => types.add(t));
    if (/\b(install|dependency|dependencies|pnpm|npm|yarn|build|test|setup)\b/.test(lower))
        ['correction', 'workflow', 'tool_convention'].forEach((t) => types.add(t));
    if (captureIntent.intent === 'routing_rule')
        types.add('routing_rule');
    if (captureIntent.intent === 'recall_rule')
        types.add('recall_rule');
    if (types.size === 0)
        ['preference', 'context'].forEach((t) => types.add(t));
    return [...types];
}
function buildQueries(packet, retrievalIntent) {
    const text = packet.latestUserMessageRedacted.trim();
    const lower = text.toLowerCase();
    const queries = [retrievalIntent?.query || text];
    if (/\b(plan|architecture|implementation|file-by-file)\b/.test(lower))
        queries.push('implementation planning architecture preferences workflow');
    if (/\b(install|dependency|dependencies|pnpm|npm|yarn|build|test|setup)\b/.test(lower))
        queries.push('package manager correction workflow repo setup');
    if (/\b(actually|instead|wrong|no,)\b/.test(lower))
        queries.push('recent correction preference update');
    if (/\b(codeword|phrase|recall)\b/.test(lower))
        queries.push('recall rule codeword phrase');
    return [...new Set(queries.map((q) => q.trim()).filter(Boolean))];
}
function turnFrameFromPacket(packet) {
    const message = packet.latestUserMessageRedacted;
    const lower = message.toLowerCase();
    const taskType = /\b(debug|error|failing|broken)\b/.test(lower)
        ? 'debugging'
        : /\b(plan|design|architecture|implementation)\b/.test(lower)
            ? 'planning'
            : /\b(code|build|test|install|dependency|repo|file)\b/.test(lower)
                ? 'coding'
                : /\b(write|draft|summarize)\b/.test(lower)
                    ? 'writing'
                    : /\b(actually|instead|wrong|no,)\b/.test(lower)
                        ? 'correction'
                        : /\b(i prefer|remember that|going forward|from now on)\b/.test(lower)
                            ? 'preference_update'
                            : 'other';
    return {
        summary: message.slice(0, 240),
        userGoal: message.slice(0, 240),
        taskType,
        activeObjects: extractTopicKeys(lower).slice(0, 6).map((value) => ({ kind: 'concept', value })),
        impliedNeeds: [
            /\b(plan|architecture|implementation)\b/.test(lower) ? 'Need prior architecture context' : '',
            /\b(pnpm|npm|yarn|install|dependency)\b/.test(lower) ? 'Need package-manager corrections' : '',
            /\b(if i ask|when i ask|codeword|phrase)\b/.test(lower) ? 'Need recall-rule handling' : '',
        ].filter(Boolean),
        memoryQuestions: [/\bremember|before|same as last time|codeword|phrase\b/.test(lower) ? 'What prior context or recall rule is being referenced?' : ''].filter(Boolean),
        constraints: [],
        routeHints: {
            likelyNeedsCorrections: /\b(actually|instead|wrong|pnpm|npm)\b/.test(lower),
            likelyNeedsPreferences: /\b(file-by-file|concrete|plan|prefer)\b/.test(lower),
            likelyNeedsWorkflow: /\b(install|build|test|setup|going forward|always|never)\b/.test(lower),
            likelyNeedsProjectContext: /\b(openclawbrain|repo|architecture|implementation|project)\b/.test(lower),
        },
    };
}
function extractTopicKeys(message) {
    const matches = message.match(/[a-z][a-z0-9_-]{2,}/g) || [];
    const stop = new Set(['this', 'that', 'with', 'from', 'have', 'like', 'said', 'same', 'last', 'time', 'what', 'when', 'then', 'them', 'your', 'into']);
    return [...new Set(matches.filter((word) => !stop.has(word)).slice(0, 12))];
}
function fingerprintKey(fingerprint) {
    return hashText(JSON.stringify(fingerprint));
}
