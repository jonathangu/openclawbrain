import { redactText } from './redact.js';
export class MemoryOperationApplier {
    store;
    config;
    constructor(options) {
        this.store = options.store;
        this.config = options.config;
    }
    applyDistillation(distillation, packet) {
        if (!distillation.shouldStore && distillation.injectionFeedback.length === 0) {
            return { memoryIds: [], storedCandidates: 0, resolvedInjections: 0 };
        }
        const memoryIds = [];
        let storedCandidates = 0;
        let resolvedInjections = 0;
        this.store.transaction(() => {
            for (const candidate of distillation.memoryCandidates) {
                if (candidate.confidence < this.config.capture.minConfidence)
                    continue;
                const node = this.upsertCandidate(candidate, packet);
                memoryIds.push(node.id);
                storedCandidates += 1;
                for (const contradiction of candidate.contradictions) {
                    if (contradiction.action === 'supersede_existing' && contradiction.existingMemoryId) {
                        this.store.supersedeMemory(contradiction.existingMemoryId, node.id);
                    }
                }
            }
            for (const feedback of distillation.injectionFeedback) {
                this.store.resolveInjectionOutcome(feedback.injectionId, feedback.outcome, redactText(feedback.evidence, 300));
                resolvedInjections += 1;
            }
            this.store.insertProofEvent({
                agentId: packet.agentId,
                kind: 'llm_feedback_distillation_applied',
                sourceHook: packet.sourceHook,
                turnId: packet.turnId,
                sessionId: packet.sessionId,
                runId: packet.runId,
                rawTranscriptStored: false,
                payload: {
                    confidence: distillation.confidence,
                    feedbackType: distillation.feedbackType,
                    memoryCount: memoryIds.length,
                    resolvedInjections,
                },
            });
        });
        return { memoryIds, storedCandidates, resolvedInjections };
    }
    upsertCandidate(candidate, packet) {
        const existing = this.store.findMemoryByNormalizedKey(packet.agentId, candidate.normalizedKey, candidate.scope.kind, candidate.scope.key);
        if (existing) {
            const mergedTags = [...new Set([...(existing.tags || []), ...(candidate.tags || [])])];
            return this.store.updateMemory(existing.id, {
                content: redactText(candidate.distilledText, 2000),
                positive: candidate.positive ? redactText(candidate.positive, 500) : undefined,
                negative: candidate.negative ? redactText(candidate.negative, 500) : undefined,
                tags: mergedTags,
                importance: Math.max(existing.importance, candidate.importanceHint),
                confidence: Math.max(existing.confidence, candidate.confidence),
                captureCount: existing.captureCount + 1,
                lastSeenAt: new Date().toISOString(),
                sourceHook: packet.sourceHook,
                sourceTurnId: packet.turnId,
                sourceSessionId: packet.sessionId,
            }) || existing;
        }
        return this.store.insertMemory({
            agentId: packet.agentId,
            type: candidate.type,
            content: redactText(candidate.distilledText, 2000),
            positive: candidate.positive ? redactText(candidate.positive, 500) : undefined,
            negative: candidate.negative ? redactText(candidate.negative, 500) : undefined,
            scopeKind: candidate.scope.kind,
            scopeKey: candidate.scope.key,
            normalizedKey: candidate.normalizedKey,
            tags: candidate.tags,
            importance: candidate.importanceHint,
            freshness: 1,
            confidence: candidate.confidence,
            useCount: 0,
            usefulCount: 0,
            captureCount: 1,
            distilledByModel: this.config.llm.feedbackModel || this.config.llm.plannerModel || undefined,
            distillerPromptVersion: 'feedback-distiller-v1',
            distillationConfidence: candidate.confidence,
            evidenceKind: packet.sourceHook,
            evidenceHash: String(packet.metadata.promptHash || ''),
            sourceHook: packet.sourceHook,
            sourceTurnId: packet.turnId,
            sourceSessionId: packet.sessionId,
        });
    }
}
