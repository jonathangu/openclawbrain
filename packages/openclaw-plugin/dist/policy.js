import { safeString } from './redact.js';
export const DECISIONS = Object.freeze(['stay_silent', 'proof_only', 'correction_only', 'full_context']);
export const SLICES = Object.freeze(['direct-answer', 'continuation', 'correction-follow-up', 'retrieval-heavy', 'tool-heavy', 'stale-memory-conflict', 'unknown']);
export function decidePolicy(input = {}) {
    const mode = safeString(input.mode || input.runtimeMode || 'conservative');
    const slice = classifyTurn(input);
    if (mode === 'off')
        return decision('stay_silent', slice, 'mode_off');
    if (mode === 'proof-only')
        return decision('proof_only', slice, `proof_only_${slice}`);
    if (slice === 'direct-answer')
        return decision('stay_silent', slice, 'direct_answer_no_help_needed');
    if (slice === 'unknown')
        return decision('stay_silent', slice, 'unknown_or_low_confidence');
    if (slice === 'correction-follow-up')
        return decision('correction_only', slice, 'correction_follow_up');
    if (slice === 'stale-memory-conflict')
        return decision('correction_only', slice, 'stale_memory_conflict');
    if (slice === 'tool-heavy')
        return decision('full_context', slice, 'tool_heavy_verify_before_claiming', true);
    if (slice === 'retrieval-heavy')
        return decision('full_context', slice, 'retrieval_heavy_needs_context');
    if (slice === 'continuation')
        return decision('full_context', slice, 'continuation_needs_bounded_context');
    return decision('stay_silent', 'unknown', 'unknown_or_low_confidence');
}
export function classifyTurn(input = {}) {
    const explicit = normalizeSlice(input.turnType || input.slice || input.event?.turnType || input.event?.turn_type);
    if (explicit)
        return explicit;
    const prompt = safeString(input.redactedPrompt || input.prompt || input.summary || input.redactedTurn?.summary).toLowerCase();
    const tools = Array.isArray(input.tools) ? input.tools : Array.isArray(input.event?.tools) ? input.event.tools : [];
    if (!prompt || prompt.length < 3)
        return 'unknown';
    if (/\b(stale|conflict|wrong memory|old memory|outdated|contradict|incorrect memory)\b/.test(prompt))
        return 'stale-memory-conflict';
    if (/\b(correction|correct this|remember that|actually|you should have|next time|use .* instead|not the .* one)\b/.test(prompt))
        return 'correction-follow-up';
    if (/\b(continue|resume|carry on|pick up|next step|where we left off|keep going)\b/.test(prompt))
        return 'continuation';
    if (/\b(search|retrieve|retrieval|look up|find in|cite|sources|docs|documentation|knowledge base)\b/.test(prompt))
        return 'retrieval-heavy';
    if (tools.length > 0 || /\b(tool|run|execute|test|build|lint|shell|terminal|read file|write file|apply patch|inspect repo|git)\b/.test(prompt))
        return 'tool-heavy';
    if (/\b(what is|who is|define|explain briefly|answer directly|calculate|convert|translate|summarize)\b/.test(prompt))
        return 'direct-answer';
    return 'unknown';
}
export function normalizeSlice(value) {
    const slice = safeString(value).toLowerCase().replaceAll('_', '-');
    if (SLICES.includes(slice))
        return slice;
    if (slice === 'direct')
        return 'direct-answer';
    if (slice === 'correction')
        return 'correction-follow-up';
    if (slice === 'stale-conflict')
        return 'stale-memory-conflict';
    if (slice === 'retrieval')
        return 'retrieval-heavy';
    if (slice === 'tool')
        return 'tool-heavy';
    return '';
}
function decision(kind, slice, reasonCode, verificationHint = false) {
    return { kind, slice, reasonCode, verificationHint };
}
