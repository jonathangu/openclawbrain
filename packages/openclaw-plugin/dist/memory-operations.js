import { redactText, shortHash } from './redact.js';
import { captureStoreThreshold, classifySensitiveValue } from './capture-intent.js';
import { scopeContextFromPacket } from './scope.js';
export class MemoryOperationApplier {
    store;
    config;
    constructor(options) {
        this.store = options.store;
        this.config = options.config;
    }
    applyDistillation(distillation, packet, context = {}) {
        if (!distillation.shouldStore && distillation.injectionFeedback.length === 0) {
            const deletedOrSuppressed = distillation.feedbackType === 'delete_or_suppress' ? this.applyDeleteOrSuppress(packet) : 0;
            return { memoryIds: [], storedCandidates: 0, rejectedCandidates: 0, rejectionReasons: distillation.audit.rejectionReasons ?? [], resolvedInjections: 0, deletedOrSuppressed };
        }
        const memoryIds = [];
        let storedCandidates = 0;
        let rejectedCandidates = 0;
        const rejectionReasons = [];
        let resolvedInjections = 0;
        this.store.transaction(() => {
            for (const candidate of distillation.memoryCandidates) {
                const minConfidence = context.captureIntent
                    ? captureStoreThreshold(context.captureIntent.intent)
                    : this.config.capture.minConfidence;
                if (candidate.confidence < minConfidence) {
                    rejectedCandidates += 1;
                    rejectionReasons.push('distiller_low_confidence');
                    continue;
                }
                if (!this.isSafeToStore(candidate)) {
                    rejectedCandidates += 1;
                    rejectionReasons.push(candidate.type === 'recall_rule' ? 'recall_rule_missing_explicit_authorization' : 'sensitive_secret_blocked');
                    continue;
                }
                if (this.isBlockedByTombstone(candidate, packet)) {
                    rejectedCandidates += 1;
                    rejectionReasons.push('tombstoned_memory_blocked');
                    continue;
                }
                const node = this.upsertCandidate(candidate, packet);
                memoryIds.push(node.id);
                storedCandidates += 1;
                for (const contradiction of candidate.contradictions) {
                    if (contradiction.action === 'supersede_existing' && contradiction.existingMemoryId) {
                        this.store.supersedeMemory(contradiction.existingMemoryId, node.id);
                    }
                }
            }
            const allowedFeedback = new Set((packet.recentInjections || []).map((item) => `${item.injectionId}:${item.memoryId}`));
            const hasPreciseCorrelation = Boolean(packet.runId && packet.turnId);
            for (const feedback of distillation.injectionFeedback) {
                if (!hasPreciseCorrelation || !allowedFeedback.has(`${feedback.injectionId}:${feedback.memoryId}`)) {
                    rejectionReasons.push('injection_feedback_scope_mismatch');
                    continue;
                }
                const changed = this.store.resolveInjectionOutcome(feedback.injectionId, feedback.outcome, redactText(feedback.evidence, 300), {
                    agentId: packet.agentId,
                    runId: packet.runId || undefined,
                    turnId: packet.turnId || undefined,
                    sessionId: packet.sessionId || undefined,
                });
                if (changed > 0)
                    resolvedInjections += changed;
                else
                    rejectionReasons.push('scope_mismatch_or_missing_injection');
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
        return { memoryIds, storedCandidates, rejectedCandidates, rejectionReasons: [...new Set(rejectionReasons)], resolvedInjections, deletedOrSuppressed: 0 };
    }
    applyDeleteOrSuppress(packet) {
        const query = deletionQuery(packet.latestUserMessageRedacted || '');
        const scopeContext = scopeContextFromPacket(packet);
        const matches = this.store.searchMemories(query, packet.agentId, { limit: 10, scopeContext });
        const finalMatches = matches.length > 0
            ? matches
            : this.store.listMemories(packet.agentId, { limit: 50, scopeContext }).filter((memory) => deletionQueryMatchesMemory(query, memory));
        const tombstone = shouldCreateTombstone(packet.latestUserMessageRedacted || '');
        for (const memory of finalMatches) {
            if (tombstone) {
                this.store.tombstoneMemory(memory.id, {
                    reason: 'user_requested_forget_or_do_not_store',
                    redactContent: shouldRedactTombstone(packet.latestUserMessageRedacted || '', memory.content),
                    source: packet.sourceHook || 'delete_or_suppress',
                });
            }
            else {
                this.store.softDeleteMemory(memory.id);
            }
        }
        return finalMatches.length;
    }
    isSafeToStore(candidate) {
        const risk = classifySensitiveValue(`${candidate.distilledText} ${candidate.positive ?? ''}`, candidate.type === 'recall_rule' ? 'recall_rule' : undefined);
        if (risk.kind === 'credential_secret')
            return false;
        if (candidate.type === 'recall_rule') {
            return candidate.disclosure === 'on_explicit_user_request_only'
                && candidate.proactiveInjectionAllowed === false
                && candidate.scope.kind !== 'global_user'
                && Boolean(candidate.scope.key);
        }
        if (risk.kind === 'ambiguous_codeword')
            return false;
        return true;
    }
    upsertCandidate(candidate, packet) {
        const existing = this.canonicalExistingMemory(candidate, packet);
        if (existing && sameMemoryContent(existing, candidate)) {
            const mergedTags = [...new Set([...(existing.tags || []), ...(candidate.tags || [])])];
            const updated = this.store.updateMemory(existing.id, {
                tags: mergedTags,
                importance: Math.max(existing.importance, clamp01(candidate.importanceHint)),
                confidence: Math.max(existing.confidence, clamp01(candidate.confidence)),
                freshness: Math.max(existing.freshness, 0.85),
                captureCount: existing.captureCount + 1,
                lastSeenAt: new Date().toISOString(),
                sourceHook: packet.sourceHook,
                sourceTurnId: packet.turnId,
                sourceSessionId: packet.sessionId,
            }) || existing;
            this.store.patchMemoryValidity(updated.id, {
                evidenceConfidence: Math.max(updated.confidence, clamp01(candidate.confidence)),
                currentValidityScore: Math.max(updated.freshness, 0.85),
                behavioralAuthorityScore: Math.max(this.store.getMemoryValidity(updated.id)?.behavioralAuthorityScore ?? 0.85, 0.85),
                stateReason: 'reinforced_same_key_same_value',
            });
            this.store.insertMemoryAuthorityEvent({
                agentId: packet.agentId,
                memoryId: updated.id,
                eventType: 'reinforced',
                source: packet.sourceHook || 'memory_operation',
                turnId: packet.turnId,
                evidenceId: String(packet.metadata.promptHash || ''),
                reason: 'same_key_same_value',
            });
            return updated;
        }
        if (existing) {
            const replacement = this.createMemory(candidate, packet, revisionKey(candidate.normalizedKey, candidate.distilledText));
            this.store.supersedeMemory(existing.id, replacement.id);
            this.store.upsertEdge(packet.agentId, existing.id, replacement.id, 'supersedes');
            this.store.insertMemoryAuthorityEvent({
                agentId: packet.agentId,
                memoryId: replacement.id,
                eventType: 'captured',
                source: packet.sourceHook || 'memory_operation',
                turnId: packet.turnId,
                evidenceId: String(packet.metadata.promptHash || ''),
                oldValue: existing.content,
                newValue: replacement.content,
                reason: 'same_key_changed_value_revision',
            });
            return replacement;
        }
        return this.createMemory(candidate, packet);
    }
    canonicalExistingMemory(candidate, packet) {
        let existing = this.store.findMemoryByNormalizedKey(packet.agentId, candidate.normalizedKey, candidate.scope.kind, candidate.scope.key);
        const seen = new Set();
        while (existing?.supersededBy && !seen.has(existing.id)) {
            seen.add(existing.id);
            const next = this.store.getMemory(existing.supersededBy);
            if (!next || next.deletedAt)
                break;
            existing = next;
        }
        return existing;
    }
    createMemory(candidate, packet, normalizedKey = candidate.normalizedKey) {
        const existingRevision = this.store.findMemoryByNormalizedKey(packet.agentId, normalizedKey, candidate.scope.kind, candidate.scope.key);
        if (existingRevision) {
            return this.store.updateMemory(existingRevision.id, {
                captureCount: existingRevision.captureCount + 1,
                lastSeenAt: new Date().toISOString(),
                importance: Math.max(existingRevision.importance, clamp01(candidate.importanceHint)),
                confidence: Math.max(existingRevision.confidence, clamp01(candidate.confidence)),
            }) || existingRevision;
        }
        return this.store.insertMemory({
            agentId: packet.agentId,
            type: candidate.type,
            content: redactText(candidate.distilledText, 2000),
            positive: candidate.positive ? redactText(candidate.positive, 500) : undefined,
            negative: candidate.negative ? redactText(candidate.negative, 500) : undefined,
            scopeKind: candidate.scope.kind,
            scopeKey: candidate.scope.key,
            normalizedKey,
            tags: candidate.tags,
            importance: clamp01(candidate.importanceHint),
            freshness: 1,
            confidence: clamp01(candidate.confidence),
            useCount: 0,
            usefulCount: 0,
            captureCount: 1,
            distilledByModel: this.config.llm.feedbackModel || this.config.llm.plannerModel || undefined,
            distillerPromptVersion: 'feedback-distiller-v1',
            distillationConfidence: clamp01(candidate.confidence),
            evidenceKind: packet.sourceHook,
            evidenceHash: String(packet.metadata.promptHash || ''),
            sourceHook: packet.sourceHook,
            sourceTurnId: packet.turnId,
            sourceSessionId: packet.sessionId,
        });
    }
    isBlockedByTombstone(candidate, packet) {
        const existing = this.store.findMemoryByNormalizedKeyAny(packet.agentId, candidate.normalizedKey, candidate.scope.kind, candidate.scope.key);
        if (!existing)
            return false;
        const validity = this.store.getMemoryValidity(existing.id);
        return validity?.retentionState === 'tombstoned'
            || validity?.privacyClass === 'do_not_restore'
            || validity?.behavioralAvailability === 'never_use' && validity?.stateReason?.includes('tombstone');
    }
}
function deletionQuery(text) {
    return text
        .replace(/\b(forget|delete|remove|do not remember|don't remember|do not store|don't store|stop using|suppress)\b/ig, ' ')
        .replace(/\b(memory|rule|old|that|this|the)\b/ig, ' ')
        .replace(/\s+/g, ' ')
        .trim() || text;
}
function clamp01(value) {
    if (!Number.isFinite(value))
        return 0;
    return Math.max(0, Math.min(1, value));
}
function sameMemoryContent(existing, candidate) {
    return normalizeMemoryText(existing.content) === normalizeMemoryText(candidate.distilledText)
        && normalizeMemoryText(existing.positive || '') === normalizeMemoryText(candidate.positive || '')
        && normalizeMemoryText(existing.negative || '') === normalizeMemoryText(candidate.negative || '');
}
function normalizeMemoryText(value) {
    return redactText(value || '', 2000).toLowerCase().replace(/\s+/g, ' ').trim();
}
function revisionKey(normalizedKey, content) {
    return `${normalizedKey}:rev:${shortHash(content)}`;
}
function shouldCreateTombstone(text) {
    return /\b(do not store|don't store|never store|do not remember|don't remember|tombstone|hard delete|codeword|passphrase|secret|password|token)\b/i.test(text);
}
function shouldRedactTombstone(requestText, memoryText) {
    return /\b(codeword|passphrase|secret|password|token|api key|private key)\b/i.test(`${requestText} ${memoryText}`);
}
function deletionQueryMatchesMemory(query, memory) {
    const tokens = query.toLowerCase().match(/[a-z][a-z0-9_-]{2,}/g) || [];
    const meaningful = tokens.filter((token) => !new Set(['anymore', 'memory', 'rule', 'old', 'this', 'that', 'the']).has(token));
    if (meaningful.length === 0)
        return false;
    const haystack = `${memory.content} ${memory.normalizedKey} ${(memory.tags || []).join(' ')}`.toLowerCase();
    const hits = meaningful.filter((token) => haystack.includes(token)).length;
    return hits >= Math.min(2, meaningful.length);
}
