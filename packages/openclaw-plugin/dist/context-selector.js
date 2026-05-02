import { clipText } from './redact.js';
import { filterMemoriesForScope, scopeContextFromPacket } from './scope.js';
export class ContextSelector {
    config;
    constructor(config) {
        this.config = config;
    }
    select(input) {
        const { packet, plan, store } = input;
        const scopeContext = scopeContextFromPacket(packet);
        let candidates = filterMemoriesForScope([...input.candidates], scopeContext);
        if (plan.retrievalPlan.graphDepth > 0 && store) {
            const expanded = new Set(candidates.map(c => c.id));
            for (const candidate of candidates.slice(0, 5)) {
                for (const connected of store.getConnectedMemories(candidate.id, plan.retrievalPlan.graphDepth, packet.agentId, scopeContext)) {
                    if (!filterMemoriesForScope([connected], scopeContext).length)
                        continue;
                    if (!expanded.has(connected.id)) {
                        expanded.add(connected.id);
                        candidates.push(connected);
                    }
                }
            }
        }
        const ranked = rankCandidates(packet, plan, candidates);
        const selected = [];
        const omitted = [];
        const lines = [];
        let usedChars = 0;
        for (const item of ranked) {
            if (selected.length >= plan.injectionPlan.maxItems) {
                omitted.push({ memoryId: item.memory.id, reason: 'budget' });
                continue;
            }
            if (item.memory.supersededBy || item.memory.deletedAt) {
                omitted.push({ memoryId: item.memory.id, reason: 'superseded' });
                continue;
            }
            if (item.memory.confidence < this.config.routing.minRouteConfidence * 0.5) {
                omitted.push({ memoryId: item.memory.id, reason: 'low_confidence' });
                continue;
            }
            if (item.memory.type === 'recall_rule' && plan.retrievalIntent?.intent !== 'recall_value_request') {
                omitted.push({ memoryId: item.memory.id, reason: 'would_pollute_prompt' });
                continue;
            }
            const line = formatMemoryLine(item.memory, item.reason, plan.retrievalIntent?.intent === 'recall_value_request');
            if (usedChars + line.length + 1 > plan.injectionPlan.maxChars) {
                omitted.push({ memoryId: item.memory.id, reason: 'budget' });
                continue;
            }
            lines.push(line);
            usedChars += line.length + 1;
            selected.push({
                memoryId: item.memory.id,
                reason: item.reason,
                useHow: item.useHow,
                confidence: item.score,
            });
        }
        const distilledContext = lines.length > 0
            ? clipText(`Relevant memory:\n${lines.map((line) => `- ${line}`).join('\n')}`, plan.injectionPlan.maxChars)
            : '';
        return {
            shouldInject: lines.length > 0,
            confidence: selected.length > 0 ? Math.max(...selected.map((item) => item.confidence)) : 0,
            selectedMemoryIds: selected.map((item) => item.memoryId),
            distilledContext,
            selected,
            omitted,
            audit: {
                promptBudgetUsedChars: distilledContext.length,
                risk: selected.some((item) => item.useHow === 'must_follow') ? 'medium' : 'low',
            },
        };
    }
}
function rankCandidates(packet, plan, candidates) {
    const lower = packet.latestUserMessageRedacted.toLowerCase();
    return candidates.map((memory) => {
        let score = memory.importance * 0.4 + memory.confidence * 0.4 + memory.freshness * 0.2;
        let reason = 'supporting_context';
        let useHow = 'consider';
        if (memory.type === 'correction') {
            score += 0.25;
            reason = 'directly_relevant_correction';
            useHow = 'must_follow';
        }
        if (memory.type === 'preference') {
            score += 0.15;
            reason = 'matching_user_preference';
            useHow = 'prefer';
        }
        if (memory.type === 'workflow') {
            score += 0.12;
            reason = 'repo_workflow';
            useHow = 'consider';
        }
        if (memory.type === 'routing_rule') {
            score += 0.22;
            reason = 'tool_guidance';
            useHow = 'must_follow';
        }
        if (memory.type === 'tool_convention') {
            score += 0.18;
            reason = 'tool_guidance';
            useHow = 'consider';
        }
        if (memory.type === 'agent_assignment') {
            score += 0.2;
            reason = 'tool_guidance';
            useHow = 'must_follow';
        }
        if (memory.type === 'project_fact') {
            score += 0.12;
            reason = 'supporting_context';
            useHow = 'consider';
        }
        if (memory.type === 'outcome') {
            score += 0.1;
            reason = 'supporting_context';
            useHow = 'consider';
        }
        if (memory.type === 'recall_rule') {
            if (/\b(codeword|phrase|answer|what is|what's|tell me|give me)\b/.test(lower)) {
                score += 0.35;
                reason = 'directly_relevant_correction';
                useHow = 'must_follow';
            }
            else {
                score -= 1;
            }
        }
        if (/\b(pnpm|npm|yarn|install|dependency|dependencies|build|test)\b/.test(lower) && /pnpm|npm|yarn|install|dependency|build|test/i.test(memory.content)) {
            score += 0.25;
        }
        if (/\b(plan|architecture|implementation|file-by-file)\b/.test(lower) && /plan|architecture|implementation|file-by-file/i.test(memory.content)) {
            score += 0.2;
        }
        return { memory, score, reason, useHow };
    }).sort((a, b) => b.score - a.score);
}
function formatMemoryLine(memory, reason, allowRecallValue = false) {
    if (memory.type === 'recall_rule') {
        if (allowRecallValue && memory.positive)
            return `Recall rule: ${memory.content} Authorized answer: ${memory.positive}`;
        return `Recall rule: ${memory.content}`;
    }
    switch (reason) {
        case 'directly_relevant_correction':
            return `Must follow: ${memory.content}`;
        case 'matching_user_preference':
            return `User preference: ${memory.content}`;
        case 'repo_workflow':
            return `Workflow: ${memory.content}`;
        case 'tool_guidance':
            if (memory.type === 'routing_rule')
                return `Routing rule: ${memory.content}`;
            if (memory.type === 'agent_assignment')
                return `Agent assignment: ${memory.content}`;
            return `Tool convention: ${memory.content}`;
        case 'contradiction_resolution':
            return `Latest correction: ${memory.content}`;
        default:
            if (memory.type === 'project_fact')
                return `Project fact: ${memory.content}`;
            if (memory.type === 'outcome')
                return `Prior outcome: ${memory.content}`;
            return memory.content;
    }
}
