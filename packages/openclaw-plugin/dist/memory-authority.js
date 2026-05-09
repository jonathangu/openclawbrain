export class MemoryAuthorityResolver {
    config;
    store;
    constructor(options = {}) {
        this.config = options.config || {};
        this.store = options.store;
    }
    resolve(input) {
        return input.candidates.map((memory) => this.resolveOne(memory, input.packet, input.plan));
    }
    resolveOne(memory, packet, plan) {
        const persisted = this.store?.getMemoryValidity?.(memory.id) ?? undefined;
        const validity = persisted ?? defaultValidityForMemory(memory);
        const lower = packet.latestUserMessageRedacted.toLowerCase();
        const relevanceScore = estimateRelevance(memory, lower, plan);
        const age = ageDays(memory.lastSeenAt || memory.updatedAt || memory.createdAt);
        const temporal = temporalState(memory, validity, age);
        const evidenceConfidence = clamp01(validity.evidenceConfidence || memory.confidence);
        const currentValidity = clamp01(validity.currentValidityScore * temporal.multiplier);
        const behavioralAuthority = clamp01(validity.behavioralAuthorityScore);
        const scopeScore = scopeSpecificity(memory);
        const authorityScore = clamp01((evidenceConfidence * 0.28) + (currentValidity * 0.28) + (behavioralAuthority * 0.24) + (scopeScore * 0.12) + (memory.importance * 0.08));
        const reasons = [...temporal.reasons];
        const risk = riskForMemory(memory, validity);
        const validationStrategy = validationStrategyForMemory(memory, validity);
        const currentOverride = currentInstructionOverrides(memory, lower);
        const explicitRecall = plan.retrievalIntent?.intent === 'recall_value_request';
        const material = isMaterial(memory, lower, relevanceScore, risk);
        if (memory.deletedAt || validity.retentionState === 'soft_deleted') {
            return decision(memory, 'never_use', authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, validationStrategy, ['retention:soft_deleted'], 'none', risk);
        }
        if (validity.retentionState === 'hard_deleted' || validity.retentionState === 'tombstoned') {
            return decision(memory, 'never_use', authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, validationStrategy, [`retention:${validity.retentionState}`], 'none', 'high');
        }
        if (validity.behavioralAvailability === 'never_use') {
            return decision(memory, 'never_use', authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, validationStrategy, ['availability:never_use'], 'none', risk);
        }
        if (memory.supersededBy) {
            return decision(memory, 'audit_only', authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, validationStrategy, [`superseded_by:${memory.supersededBy}`], 'none', risk);
        }
        if (currentOverride) {
            return decision(memory, currentOverride.weak ? 'weak_context' : 'abstain', authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, validationStrategy, [currentOverride.reason], 'none', risk);
        }
        if (memory.type === 'recall_rule' || validity.privacyClass === 'recall_only' || validity.behavioralAvailability === 'explicit_request_only') {
            if (!explicitRecall) {
                return decision(memory, 'never_use', authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, 'never_proactive', ['privacy:explicit_request_only'], 'explicit_recall', 'high');
            }
            if (validity.privacyClass === 'do_not_reveal_proactively') {
                reasons.push('privacy:explicit_recall_request_required');
            }
        }
        if (validity.privacyClass === 'do_not_restore') {
            return decision(memory, 'never_use', authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, validationStrategy, ['privacy:do_not_restore'], 'none', 'high');
        }
        if (temporal.state === 'expired') {
            const nextDecision = material && validationStrategy === 'user_confirm' ? 'confirm_before_use' : 'abstain';
            return decision(memory, nextDecision, authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, validationStrategy, [...reasons, 'temporal:expired'], nextDecision === 'confirm_before_use' ? 'ask_user' : 'none', risk);
        }
        if (validity.behavioralAvailability === 'confirm_before_use') {
            return decision(memory, 'confirm_before_use', authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, validationStrategy, [...reasons, 'availability:confirm_before_use'], 'ask_user', risk);
        }
        if (temporal.state === 'stale' || validity.temporalValidity === 'stale') {
            if (material && validationStrategy === 'environment_check') {
                return decision(memory, 'verify_before_use', authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, validationStrategy, [...reasons, 'temporal:stale', 'verification:environment_available'], 'verify_environment', risk);
            }
            if (material && validationStrategy === 'user_confirm') {
                return decision(memory, 'confirm_before_use', authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, validationStrategy, [...reasons, 'temporal:stale'], 'ask_user', risk);
            }
            return decision(memory, 'weak_context', authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, validationStrategy, [...reasons, 'temporal:stale_low_materiality'], 'none', risk);
        }
        if (authorityScore < 0.32) {
            return decision(memory, 'abstain', authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, validationStrategy, [...reasons, 'authority:low_score'], 'none', risk);
        }
        if (memory.type === 'preference' || memory.type === 'outcome' || memory.type === 'context') {
            return decision(memory, 'weak_context', authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, validationStrategy, [...reasons, 'policy:soft_prior'], 'none', risk);
        }
        return decision(memory, 'inject', authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, validationStrategy, [...reasons, 'authority:current'], 'none', risk);
    }
}
export function defaultValidityForMemory(memory, overrides = {}) {
    const privacyClass = defaultPrivacyClass(memory, overrides.privacyClass);
    const validationStrategy = overrides.validationStrategy ?? defaultValidationStrategy(memory, privacyClass);
    const behavioralAvailability = overrides.behavioralAvailability
        ?? (memory.type === 'recall_rule' || privacyClass === 'recall_only' ? 'explicit_request_only' : 'injectable');
    const retentionState = overrides.retentionState ?? (memory.deletedAt ? 'soft_deleted' : 'stored');
    const ts = new Date().toISOString();
    return {
        memoryId: memory.id,
        retentionState,
        behavioralAvailability,
        temporalValidity: overrides.temporalValidity ?? 'current',
        privacyClass,
        decayPolicy: overrides.decayPolicy ?? defaultDecayPolicy(memory),
        validationStrategy,
        validFrom: overrides.validFrom,
        validUntil: overrides.validUntil,
        expiresAt: overrides.expiresAt,
        lastConfirmedAt: overrides.lastConfirmedAt,
        lastVerifiedAt: overrides.lastVerifiedAt,
        lastSuccessfulUseAt: overrides.lastSuccessfulUseAt,
        lastFailedUseAt: overrides.lastFailedUseAt,
        lastContradictedAt: overrides.lastContradictedAt,
        revalidateAfter: overrides.revalidateAfter,
        halfLifeDays: overrides.halfLifeDays ?? defaultHalfLifeDays(memory),
        evidenceConfidence: clamp01(overrides.evidenceConfidence ?? memory.confidence),
        currentValidityScore: clamp01(overrides.currentValidityScore ?? memory.freshness),
        behavioralAuthorityScore: clamp01(overrides.behavioralAuthorityScore ?? (behavioralAvailability === 'injectable' ? 0.85 : 0.55)),
        stateReason: overrides.stateReason,
        updatedAt: overrides.updatedAt ?? ts,
    };
}
export function authorityEventTypeForDecision(decisionKind) {
    switch (decisionKind) {
        case 'inject':
            return 'used';
        case 'weak_context':
            return 'weak_context_used';
        case 'verify_before_use':
            return 'verification_requested';
        case 'confirm_before_use':
            return 'confirmation_requested';
        case 'audit_only':
            return 'audit_only';
        case 'never_use':
            return 'suppressed';
        default:
            return 'abstained';
    }
}
function decision(memory, decisionKind, authorityScore, relevanceScore, evidenceConfidence, currentValidity, behavioralAuthority, validationStrategy, reasons, requiredAction, risk) {
    return {
        memoryId: memory.id,
        decision: decisionKind,
        authorityScore: round3(authorityScore),
        relevanceScore: round3(relevanceScore),
        evidenceConfidence: round3(evidenceConfidence),
        currentValidity: round3(currentValidity),
        behavioralAuthority: round3(behavioralAuthority),
        validationStrategy,
        reasons: [...new Set(reasons.filter(Boolean))],
        requiredAction,
        risk,
    };
}
function temporalState(memory, validity, age) {
    const reasons = [];
    if (validity.expiresAt && Date.parse(validity.expiresAt) <= Date.now()) {
        return { state: 'expired', multiplier: 0.05, reasons: ['temporal:expires_at_elapsed'] };
    }
    if (validity.validUntil && Date.parse(validity.validUntil) <= Date.now()) {
        return { state: 'expired', multiplier: 0.05, reasons: ['temporal:valid_until_elapsed'] };
    }
    if (validity.revalidateAfter && Date.parse(validity.revalidateAfter) <= Date.now()) {
        return { state: 'stale', multiplier: 0.6, reasons: ['temporal:revalidate_after_elapsed'] };
    }
    const halfLife = Number(validity.halfLifeDays || defaultHalfLifeDays(memory));
    if (Number.isFinite(halfLife) && halfLife > 0 && age > halfLife * 4) {
        return { state: 'expired', multiplier: 0.1, reasons: ['temporal:age_expired'] };
    }
    if (Number.isFinite(halfLife) && halfLife > 0 && age > halfLife) {
        reasons.push('temporal:age_stale');
        return { state: 'stale', multiplier: Math.max(0.35, Math.pow(0.5, age / halfLife)), reasons };
    }
    if (validity.temporalValidity === 'stale')
        return { state: 'stale', multiplier: 0.65, reasons: ['temporal:stored_stale'] };
    if (validity.temporalValidity === 'expired')
        return { state: 'expired', multiplier: 0.1, reasons: ['temporal:stored_expired'] };
    return { state: validity.temporalValidity, multiplier: 1, reasons };
}
function estimateRelevance(memory, lowerMessage, plan) {
    let score = memory.importance * 0.35 + memory.confidence * 0.35 + memory.freshness * 0.2;
    if (plan.retrievalPlan.memoryTypes.includes(memory.type))
        score += 0.15;
    const contentTokens = significantTokens(memory.content);
    const matches = contentTokens.filter((token) => lowerMessage.includes(token)).length;
    score += Math.min(0.25, matches * 0.04);
    return clamp01(score);
}
function isMaterial(memory, lowerMessage, relevanceScore, risk) {
    if (risk === 'high')
        return true;
    if (relevanceScore >= 0.72)
        return true;
    if (['correction', 'workflow', 'tool_convention', 'routing_rule', 'agent_assignment'].includes(memory.type))
        return true;
    if (/\b(build|test|install|deploy|delete|publish|release|token|secret|password)\b/.test(lowerMessage))
        return true;
    return false;
}
function currentInstructionOverrides(memory, lowerMessage) {
    const content = memory.content.toLowerCase();
    if (/\b(ignore|skip|do not use|don't use)\b.*\b(old|prior|previous|memory|preference|rule)\b/.test(lowerMessage)) {
        return { reason: 'current_instruction:ignore_prior_memory' };
    }
    if (memory.type === 'preference') {
        if (/\b(concise|short|brief|terse)\b/.test(content) && /\b(deep|deeply|detailed|thorough|comprehensive|ultimate|long-form|long form)\b/.test(lowerMessage)) {
            return { reason: 'current_instruction:asks_for_depth' };
        }
        if (/\b(deep|detailed|thorough|long)\b/.test(content) && /\b(concise|short|brief|quick|one-liner|one liner)\b/.test(lowerMessage)) {
            return { reason: 'current_instruction:asks_for_concision' };
        }
        if (/\b(default|prefer|usually)\b/.test(content) && /\b(for this|this time|today|right now)\b/.test(lowerMessage)) {
            return { reason: 'current_instruction:local_task_specificity', weak: true };
        }
    }
    return null;
}
function riskForMemory(memory, validity) {
    if (memory.type === 'recall_rule' || validity.privacyClass !== 'normal')
        return 'high';
    if (['routing_rule', 'agent_assignment', 'tool_convention', 'workflow', 'correction'].includes(memory.type))
        return 'medium';
    return 'low';
}
function validationStrategyForMemory(memory, validity) {
    if (validity.validationStrategy && validity.validationStrategy !== 'none')
        return validity.validationStrategy;
    return defaultValidationStrategy(memory, validity.privacyClass);
}
function defaultValidationStrategy(memory, privacyClass) {
    if (memory.type === 'recall_rule' || privacyClass === 'recall_only' || privacyClass === 'do_not_reveal_proactively')
        return 'never_proactive';
    if (['workflow', 'tool_convention', 'project_fact'].includes(memory.type))
        return 'environment_check';
    if (['correction', 'preference', 'routing_rule', 'agent_assignment'].includes(memory.type))
        return 'user_confirm';
    return 'none';
}
function defaultPrivacyClass(memory, override) {
    if (override)
        return override;
    if (memory.type === 'recall_rule')
        return 'recall_only';
    const text = `${memory.content} ${memory.positive ?? ''}`.toLowerCase();
    if (/\b(password|token|secret|api key|private key|codeword|passphrase)\b/.test(text))
        return 'sensitive';
    return 'normal';
}
function defaultDecayPolicy(memory) {
    if (memory.type === 'recall_rule')
        return 'sensitive_no_decay';
    if (['workflow', 'tool_convention', 'project_fact'].includes(memory.type))
        return 'environment_verified';
    if (memory.type === 'preference')
        return 'soft_preference';
    if (memory.type === 'correction')
        return 'correction_lineage';
    return 'standard';
}
function defaultHalfLifeDays(memory) {
    switch (memory.type) {
        case 'workflow':
        case 'tool_convention':
        case 'project_fact':
            return 45;
        case 'routing_rule':
        case 'agent_assignment':
            return 60;
        case 'correction':
            return 120;
        case 'preference':
            return 180;
        case 'recall_rule':
            return 0;
        default:
            return 90;
    }
}
function scopeSpecificity(memory) {
    switch (memory.scopeKind) {
        case 'task':
            return 1;
        case 'session':
            return 0.96;
        case 'repo':
        case 'project':
        case 'app':
            return 0.88;
        case 'tool':
        case 'channel':
            return 0.78;
        case 'agent':
            return 0.7;
        case 'global_user':
            return 0.58;
        default:
            return 0.5;
    }
}
function significantTokens(text) {
    const stop = new Set(['this', 'that', 'with', 'from', 'have', 'like', 'user', 'prefer', 'prefers', 'should', 'would', 'there', 'their', 'about', 'when']);
    return [...new Set(text.toLowerCase().match(/[a-z][a-z0-9_-]{2,}/g) || [])].filter((token) => !stop.has(token)).slice(0, 32);
}
function ageDays(value) {
    const ts = value ? Date.parse(value) : NaN;
    if (!Number.isFinite(ts))
        return 0;
    return Math.max(0, (Date.now() - ts) / 86400000);
}
function clamp01(value) {
    if (!Number.isFinite(value))
        return 0;
    return Math.max(0, Math.min(1, value));
}
function round3(value) {
    return Number(clamp01(value).toFixed(3));
}
