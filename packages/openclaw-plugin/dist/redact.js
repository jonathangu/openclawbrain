import { createHash, randomUUID } from 'node:crypto';
export function safeString(value) {
    return typeof value === 'string' ? value.trim() : String(value ?? '').trim();
}
export function clipText(value, maxChars = 3000) {
    const text = safeString(value);
    if (text.length <= maxChars)
        return text;
    return `${text.slice(0, Math.max(0, maxChars - 1))}…`;
}
export function hashText(value) {
    return `sha256:${createHash('sha256').update(String(value ?? '')).digest('hex')}`;
}
export function shortHash(value) {
    return createHash('sha256').update(String(value ?? '')).digest('hex').slice(0, 16);
}
export function eventId(prefix = 'evt') {
    if (typeof randomUUID === 'function')
        return `${prefix}_${randomUUID()}`;
    return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}
export function redactText(value, maxChars = 3000) {
    let text = safeString(value);
    text = text.replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi, '[redacted-email]');
    text = text.replace(/\b(?:https?|ftp):\/\/[^\s<>()]+/gi, '[redacted-url]');
    text = text.replace(/\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b/g, '[redacted-phone]');
    text = text.replace(/\b(?:api[_-]?key|token|secret|password|passwd|authorization|bearer)\b\s*[:=]\s*['"]?[^'"\s,;]+/gi, (match) => {
        const key = match.split(/[:=]/)[0]?.trim() || 'secret';
        return `${key}=[redacted-secret]`;
    });
    text = text.replace(/\b(?:sk|pk|ghp|github_pat|xox[baprs])-?[A-Za-z0-9_\-]{16,}\b/g, '[redacted-secret]');
    text = text.replace(/\b[A-Fa-f0-9]{32,}\b/g, '[redacted-blob]');
    text = text.replace(/\b[A-Za-z0-9+/]{40,}={0,2}\b/g, '[redacted-blob]');
    return clipText(text, maxChars);
}
export function latestUserTextFromEvent(event = {}) {
    if (typeof event.userMessage === 'string')
        return event.userMessage;
    if (typeof event.user_message === 'string')
        return event.user_message;
    if (typeof event.userMessageRedacted === 'string')
        return event.userMessageRedacted;
    if (typeof event.user_message_redacted === 'string')
        return event.user_message_redacted;
    if (typeof event.input === 'string')
        return event.input;
    if (Array.isArray(event.messages)) {
        const latest = [...event.messages].reverse().find((message) => message?.role === 'user');
        if (typeof latest?.content === 'string')
            return latest.content;
        if (typeof latest?.redactedText === 'string')
            return latest.redactedText;
    }
    return safeString(event.summary ?? '');
}
export function sanitizeForProof(value) {
    if (Array.isArray(value))
        return value.map(sanitizeForProof);
    if (!value || typeof value !== 'object')
        return value;
    const blocked = new Set(['prompt', 'rawPrompt', 'userText', 'rawUserText', 'transcript', 'messages', 'input', 'output', 'response', 'completion']);
    const result = {};
    for (const [key, entry] of Object.entries(value)) {
        if (blocked.has(key))
            continue;
        result[key] = sanitizeForProof(entry);
    }
    return result;
}
