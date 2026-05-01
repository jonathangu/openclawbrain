import { hashText } from './redact.js';
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
    constructor(options) {
        this.config = options.config;
        this.cache = options.cache ?? new RouteCache();
    }
    fingerprint(packet) {
        const message = packet.latestUserMessage.toLowerCase();
        return {
            agentId: packet.agentId,
            scopeKey: packet.sessionId || packet.sessionKey || undefined,
            taskTypeHint: String(packet.metadata.turnType || ''),
            topicKeys: extractTopicKeys(message),
            explicitMemoryReference: /\b(as before|like i said|same as last time|we discussed before|remember)\b/i.test(packet.latestUserMessage),
            explicitCorrectionCue: /\b(actually|instead|no,|don't|do not|wrong|use .* instead)\b/i.test(packet.latestUserMessage),
        };
    }
    plan(packet) {
        const fingerprint = this.fingerprint(packet);
        const cached = this.cache.get(fingerprint);
        const turnFrame = turnFrameFromPacket(packet);
        if (cached) {
            return {
                route: cached.route,
                confidence: cached.confidence,
                turnFrame,
                retrievalPlan: cached.retrievalPlan,
                injectionPlan: cached.injectionPlan,
                shouldRetrieve: cached.route === 'retrieve_memory' || cached.route === 'retrieve_and_distill' || cached.route === 'high_confidence_correction_only',
                enqueueCapture: fingerprint.explicitCorrectionCue,
                latencyReason: 'cached route plan',
            };
        }
        const plan = heuristicRoutePlan(packet, turnFrame, this.config);
        this.cache.set(fingerprint, {
            route: plan.route,
            retrievalPlan: plan.retrievalPlan,
            injectionPlan: plan.injectionPlan,
            confidence: plan.confidence,
            expiresAt: new Date(Date.now() + 5 * 60 * 1000).toISOString(),
        });
        return plan;
    }
}
function heuristicRoutePlan(packet, turnFrame, config) {
    const message = packet.latestUserMessage.toLowerCase();
    const explicitCorrectionCue = /\b(actually|instead|wrong|no,)\b/i.test(packet.latestUserMessage);
    const explicitMemoryReference = /\b(as before|same as last time|remember|we discussed before)\b/i.test(packet.latestUserMessage);
    const installLike = /\b(install|dependency|dependencies|pnpm|npm|yarn|build|test|setup)\b/.test(message);
    const planningLike = /\b(plan|design|architecture|file-by-file|implementation)\b/.test(message);
    let route = 'no_memory';
    let confidence = 0.55;
    if (explicitCorrectionCue) {
        route = 'high_confidence_correction_only';
        confidence = 0.9;
    }
    else if (planningLike || explicitMemoryReference || installLike) {
        route = 'retrieve_memory';
        confidence = planningLike ? 0.82 : 0.72;
    }
    const retrievalPlan = {
        queries: buildQueries(packet),
        memoryTypes: route === 'high_confidence_correction_only'
            ? ['correction']
            : planningLike
                ? ['correction', 'preference', 'workflow', 'context']
                : installLike
                    ? ['correction', 'workflow']
                    : ['preference', 'context'],
        requiredTags: [],
        excludedTags: [],
        graphDepth: planningLike ? 1 : 0,
        maxCandidates: config.routing.maxCandidateMemories,
    };
    const injectionPlan = {
        maxItems: config.routing.maxInjectedMemories,
        maxChars: config.routing.maxInjectedChars,
        preferredFormat: explicitCorrectionCue ? 'rules' : planningLike ? 'bullets' : 'none',
    };
    return {
        route,
        confidence,
        turnFrame,
        retrievalPlan,
        injectionPlan,
        shouldRetrieve: route !== 'no_memory',
        enqueueCapture: explicitCorrectionCue,
        latencyReason: 'heuristic uncached route',
    };
}
function buildQueries(packet) {
    const text = packet.latestUserMessage.trim();
    const lower = text.toLowerCase();
    const queries = [text];
    if (/\b(plan|architecture|implementation|file-by-file)\b/.test(lower))
        queries.push('implementation planning architecture preferences workflow');
    if (/\b(install|dependency|dependencies|pnpm|npm|yarn|build|test|setup)\b/.test(lower))
        queries.push('package manager correction workflow repo setup');
    if (/\b(actually|instead|wrong|no,)\b/.test(lower))
        queries.push('recent correction preference update');
    return [...new Set(queries.map((q) => q.trim()).filter(Boolean))];
}
function turnFrameFromPacket(packet) {
    const message = packet.latestUserMessage;
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
                        : 'other';
    return {
        summary: message.slice(0, 240),
        userGoal: message.slice(0, 240),
        taskType,
        activeObjects: extractTopicKeys(lower).slice(0, 6).map((value) => ({ kind: 'concept', value })),
        impliedNeeds: [
            /\b(plan|architecture|implementation)\b/.test(lower) ? 'Need prior architecture context' : '',
            /\b(pnpm|npm|yarn|install|dependency)\b/.test(lower) ? 'Need package-manager corrections' : '',
        ].filter(Boolean),
        memoryQuestions: [/\bremember|before|same as last time\b/.test(lower) ? 'What prior context is being referenced?' : ''].filter(Boolean),
        constraints: [],
        routeHints: {
            likelyNeedsCorrections: /\b(actually|instead|wrong|pnpm|npm)\b/.test(lower),
            likelyNeedsPreferences: /\b(file-by-file|concrete|plan)\b/.test(lower),
            likelyNeedsWorkflow: /\b(install|build|test|setup)\b/.test(lower),
            likelyNeedsProjectContext: /\b(openclawbrain|repo|architecture|implementation)\b/.test(lower),
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
