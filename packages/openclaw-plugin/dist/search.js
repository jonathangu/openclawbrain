import { MemoryStore } from './memory-store.js';
import { safeString } from './redact.js';
import { isAgentAllowed } from './config.js';
import { defaultScopeContext, memoryInScope } from './scope.js';
export function buildMemoryPromptSupplement() {
    return () => [
        'OpenClawBrain local memory supplement is available for scoped corrections, preferences, workflows, and context when prior decisions matter.',
    ];
}
export function buildMemoryCorpusSupplement(config) {
    return {
        search: async ({ query, maxResults }) => {
            const agentId = defaultAgentId(config);
            const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
            try {
                return store.searchMemories(query, agentId, { limit: maxResults ?? 10, scopeContext: defaultScopeContext(agentId) }).map((memory) => searchResultFromMemory(memory));
            }
            finally {
                store.close();
            }
        },
        get: async ({ lookup }) => {
            const agentId = defaultAgentId(config);
            const memoryId = extractMemoryId(lookup);
            const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
            try {
                const memory = store.getMemory(memoryId);
                if (!memory || memory.deletedAt || memory.supersededBy || !memoryInScope(memory, defaultScopeContext(agentId)))
                    return null;
                return {
                    corpus: 'openclawbrain',
                    path: memoryPath(memory),
                    title: `${memory.type}: ${memory.normalizedKey}`,
                    kind: memory.type,
                    content: renderMemory(memory),
                    fromLine: 1,
                    lineCount: renderMemory(memory).split(/\n/).length,
                    id: memory.id,
                    provenanceLabel: 'OpenClawBrain local memory graph',
                    updatedAt: memory.updatedAt,
                };
            }
            finally {
                store.close();
            }
        },
    };
}
export function searchPayload(config, agentId, query, limit = 10) {
    if (!isAgentAllowed(config, agentId))
        return forbiddenAgentPayload(agentId);
    const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
    try {
        const memories = store.searchMemories(query, agentId, { limit, scopeContext: defaultScopeContext(agentId) });
        return {
            ok: true,
            agentId,
            query,
            limit,
            results: memories.map((memory) => ({
                id: memory.id,
                type: memory.type,
                normalizedKey: memory.normalizedKey,
                score: Number((memory.importance * memory.confidence).toFixed(3)),
                content: memory.content,
                tags: memory.tags,
                updatedAt: memory.updatedAt,
            })),
        };
    }
    finally {
        store.close();
    }
}
export function graphPayload(config, agentId, limit = 20) {
    if (!isAgentAllowed(config, agentId))
        return forbiddenAgentPayload(agentId);
    const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
    try {
        const nodes = store.listMemories(agentId, { limit, scopeContext: defaultScopeContext(agentId) });
        const nodeIds = new Set(nodes.map((node) => node.id));
        const edges = dedupeEdges(nodes.flatMap((node) => store.getEdges(node.id))).filter((edge) => nodeIds.has(edge.fromId) && nodeIds.has(edge.toId));
        return {
            ok: true,
            agentId,
            counts: {
                nodes: store.countMemories(agentId),
                edges: store.countEdgesForAgent(agentId),
            },
            nodes: nodes.map((node) => ({
                id: node.id,
                type: node.type,
                normalizedKey: node.normalizedKey,
                content: node.content,
                importance: node.importance,
                confidence: node.confidence,
                supersededBy: node.supersededBy || null,
                validity: store.getMemoryValidity(node.id),
                authorityEvents: store.listMemoryAuthorityEvents(agentId, 5, node.id).map((event) => ({
                    eventType: event.eventType,
                    source: event.source,
                    reason: event.reason || null,
                    createdAt: event.createdAt,
                })),
            })),
            edges,
        };
    }
    finally {
        store.close();
    }
}
export function learnPayload(config, agentId, limit = 20) {
    if (!isAgentAllowed(config, agentId))
        return forbiddenAgentPayload(agentId);
    const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
    try {
        return {
            ok: true,
            agentId,
            activePolicySnapshot: store.getActivePolicySnapshot(agentId),
            examples: store.getRouteExamples(agentId, limit),
            policySnapshots: store.listPolicySnapshots(agentId, limit),
        };
    }
    finally {
        store.close();
    }
}
export function auditPayload(config, agentId, limit = 20) {
    if (!isAgentAllowed(config, agentId))
        return forbiddenAgentPayload(agentId);
    const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
    try {
        const rows = store.listCaptureAudit(agentId, limit);
        return {
            ok: true,
            agentId,
            limit: Math.min(200, Math.max(1, Number(limit || 20))),
            captureOpportunityRate: rows.length ? Number((rows.filter((row) => row.captureIntent?.shouldConsiderCapture).length / rows.length).toFixed(3)) : 0,
            storageAcceptanceRate: rows.length ? Number((rows.reduce((sum, row) => sum + row.storedCount, 0) / Math.max(1, rows.reduce((sum, row) => sum + row.candidateCount, 0))).toFixed(3)) : 0,
            rejectionDistribution: rejectionDistribution(rows),
            rows: rows.map(renderCaptureAuditRow),
        };
    }
    finally {
        store.close();
    }
}
export function explainLastPayload(config, agentId, turnId) {
    if (!isAgentAllowed(config, agentId))
        return forbiddenAgentPayload(agentId);
    const store = new MemoryStore({ activationRoot: config.activationRoot, agentId });
    try {
        const rows = store.listCaptureAudit(agentId, 200);
        const row = turnId ? rows.find((candidate) => candidate.turnId === turnId) : rows[0];
        if (!row)
            return { ok: false, agentId, reason: 'no_capture_audit_rows' };
        const considered = row.captureIntent?.shouldConsiderCapture === true;
        const stored = row.storedCount > 0;
        const routeDecision = store.getRecentRouteDecisions(agentId, 200).find((decision) => turnId ? decision.turnId === turnId : decision.turnId === row.turnId) || null;
        const graphSnapshot = routeDecision ? store.getRouteGraphSnapshot(routeDecision.id) : null;
        const authorityEvents = routeDecision
            ? store.listMemoryAuthorityEvents(agentId, 200).filter((event) => event.routeId === routeDecision.id)
            : [];
        const teacherRun = routeDecision ? store.listRouteTeacherRuns(agentId, 200).find((run) => run.routeDecisionId === routeDecision.id) || null : null;
        const counterfactuals = routeDecision ? store.listRouteCounterfactuals(agentId, routeDecision.id, 20) : [];
        const activePolicy = store.getActivePolicySnapshotV3(agentId) || store.getActivePolicySnapshotV2(agentId);
        const matchedRule = routeDecision?.policyRuleId && activePolicy ? activePolicy.rules.find((rule) => rule.id === routeDecision.policyRuleId) || null : null;
        return {
            ok: true,
            agentId,
            turnId: row.turnId || null,
            createdAt: row.createdAt,
            summary: stored
                ? 'I considered it for memory and stored at least one distilled candidate.'
                : considered
                    ? 'I considered it for memory, but did not store it.'
                    : 'I did not consider it for durable memory.',
            retrieval: {
                intent: row.retrievalIntent?.intent || 'unknown',
                shouldRetrieve: row.retrievalIntent?.shouldRetrieve === true,
                includeRecallRules: row.retrievalIntent?.includeRecallRules === true,
            },
            route: routeDecision ? {
                id: routeDecision.id,
                route: routeDecision.route,
                confidence: routeDecision.confidence,
                latencyTier: routeDecision.latencyTier,
                policySnapshotId: routeDecision.policySnapshotId || null,
                policyRuleId: routeDecision.policyRuleId || null,
                reasonCode: routeDecision.reasonCode || null,
                candidateCount: routeDecision.candidateCount ?? null,
                selectedMemoryIds: routeDecision.selectedMemoryIds,
                omittedMemoryIds: routeDecision.omittedMemoryIds,
            } : null,
            policy: activePolicy ? {
                activeSnapshotId: activePolicy.id,
                ruleCount: activePolicy.rules.length,
                matchedRule: matchedRule ? {
                    id: matchedRule.id,
                    route: matchedRule.route,
                    memoryTypes: matchedRule.memoryTypes,
                    queries: matchedRule.queries,
                    graphDepth: matchedRule.graphDepth,
                    confidence: matchedRule.confidence,
                    evidenceIds: matchedRule.evidenceIds,
                    reason: matchedRule.reason || null,
                    stats: matchedRule.stats || null,
                } : null,
            } : null,
            graphSnapshot: graphSnapshot ? {
                id: graphSnapshot.id,
                candidateMemoryIds: graphSnapshot.candidateMemoryIds,
                graphStats: graphSnapshot.graphStats,
            } : null,
            authority: authorityEvents.map((event) => ({
                memoryId: event.memoryId,
                eventType: event.eventType,
                reason: event.reason || null,
                createdAt: event.createdAt,
            })),
            teacher: teacherRun ? {
                id: teacherRun.id,
                verdict: teacherRun.verdict,
                teacherRoute: teacherRun.teacherRoute,
                teacherMemoryIds: teacherRun.teacherMemoryIds,
                confidence: teacherRun.confidence,
                validated: teacherRun.validated,
                rationale: teacherRun.rationale,
            } : null,
            counterfactualSummary: counterfactuals.map((cf) => ({
                kind: cf.kind,
                memoryIds: cf.memoryIds,
                estimatedOutcome: cf.estimatedOutcome,
                confidence: cf.confidence,
            })),
            capture: {
                signalFound: considered,
                intent: row.captureIntent?.intent || 'unknown',
                confidence: row.captureIntent?.confidence ?? null,
                reason: row.captureIntent?.reason || null,
                matchedSignals: row.captureIntent?.matchedSignals || [],
            },
            distiller: {
                ran: row.distillerRan,
                model: row.distillerModel || null,
                latencyMs: row.distillerLatencyMs || null,
                fallbackRan: row.fallbackRan,
            },
            storage: {
                candidateCount: row.candidateCount,
                storedCount: row.storedCount,
                rejectedCount: row.rejectedCount,
                reasons: row.rejectionReasons,
                safeCandidatePreview: row.safeCandidatePreview || null,
            },
        };
    }
    finally {
        store.close();
    }
}
function defaultAgentId(config) {
    return safeString(config?.scopes?.agents?.[0] ?? 'main') || 'main';
}
function forbiddenAgentPayload(agentId) {
    return { ok: false, agentId, reason: 'agent_not_allowed' };
}
export function extractMemoryId(lookup) {
    const value = safeString(lookup);
    const match = value.match(/([0-9a-f]{8}-[0-9a-f-]{27,})/i);
    return match ? match[1] : value.replace(/^memory\//, '').replace(/\.md$/, '');
}
function searchResultFromMemory(memory) {
    return {
        corpus: 'openclawbrain',
        path: memoryPath(memory),
        title: `${memory.type}: ${memory.normalizedKey}`,
        kind: memory.type,
        score: Number((memory.importance * memory.confidence).toFixed(3)),
        snippet: memory.content,
        id: memory.id,
        startLine: 1,
        endLine: renderMemory(memory).split(/\n/).length,
        citation: `${memoryPath(memory)}#L1-L${renderMemory(memory).split(/\n/).length}`,
        source: 'openclawbrain',
        provenanceLabel: 'OpenClawBrain local memory graph',
        updatedAt: memory.updatedAt,
    };
}
export function renderMemory(memory) {
    return [
        `# ${memory.type}: ${memory.normalizedKey}`,
        '',
        memory.content,
        '',
        `- scope: ${memory.scopeKind}${memory.scopeKey ? `:${memory.scopeKey}` : ''}`,
        `- tags: ${memory.tags.join(', ')}`,
        `- confidence: ${memory.confidence}`,
        `- importance: ${memory.importance}`,
    ].join('\n');
}
function renderCaptureAuditRow(row) {
    return {
        id: row.id,
        turnId: row.turnId || null,
        sessionId: row.sessionId || null,
        createdAt: row.createdAt,
        retrievalIntent: row.retrievalIntent?.intent || 'unknown',
        shouldRetrieve: row.retrievalIntent?.shouldRetrieve === true,
        captureIntent: row.captureIntent?.intent || 'unknown',
        shouldConsiderCapture: row.captureIntent?.shouldConsiderCapture === true,
        captureJobCreated: row.captureJobCreated,
        distillerRan: row.distillerRan,
        fallbackRan: row.fallbackRan,
        candidateCount: row.candidateCount,
        storedCount: row.storedCount,
        rejectedCount: row.rejectedCount,
        rejectionReasons: row.rejectionReasons,
        safeCandidatePreview: row.safeCandidatePreview || null,
    };
}
function rejectionDistribution(rows) {
    const counts = {};
    for (const row of rows) {
        for (const reason of row.rejectionReasons || []) {
            counts[reason] = (counts[reason] || 0) + 1;
        }
    }
    return counts;
}
export function memoryPath(memory) {
    return `memory/${memory.id}.md`;
}
function dedupeEdges(edges) {
    const seen = new Set();
    return edges.filter((edge) => {
        const key = `${edge.id}:${edge.fromId}:${edge.toId}:${edge.relation}`;
        if (seen.has(key))
            return false;
        seen.add(key);
        return true;
    });
}
