import { createHash, randomUUID } from 'node:crypto';

export function safeString(value: any) {
  return typeof value === 'string' ? value.trim() : String(value ?? '').trim();
}

export function clipText(value: any, maxChars = 3000) {
  const text = safeString(value);
  if (text.length <= maxChars) return text;
  return `${text.slice(0, Math.max(0, maxChars - 1))}…`;
}

export function hashText(value: any) {
  return `sha256:${createHash('sha256').update(String(value ?? '')).digest('hex')}`;
}

export function shortHash(value: any) {
  return createHash('sha256').update(String(value ?? '')).digest('hex').slice(0, 16);
}

export function eventId(prefix = 'evt') {
  if (typeof randomUUID === 'function') return `${prefix}_${randomUUID()}`;
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

export function redactText(value: any, maxChars = 3000) {
  const redactedValue = redactStructuredValue(value);
  let text = typeof redactedValue === 'string' ? safeString(redactedValue) : safeString(JSON.stringify(redactedValue));
  text = text.replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi, '[redacted-email]');
  text = text.replace(/\b(?:https?|ftp):\/\/[^\s<>()]+/gi, '[redacted-url]');
  text = text.replace(/\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b/g, '[redacted-phone]');
  text = text.replace(/(["']?\b(?:api[_-]?key|token|secret|password|passwd|authorization|bearer|client[_-]?secret|access[_-]?token|refresh[_-]?token|session[_-]?cookie|private[_-]?key|secret[_-]?key)\b["']?\s*:\s*)["'][^"']{1,4096}["']/gi, '$1"[redacted-secret]"');
  text = text.replace(/\b([A-Za-z0-9_.-]*(?:apiKey|accessToken|refreshToken|clientSecret|sessionCookie|privateKey|secretKey|password|passwd|token|secret|authorization|bearer)[A-Za-z0-9_.-]*)\b\s*[:=]\s*['"]?[^'"\s,;}{\]]+/gi, '$1=[redacted-secret]');
  text = text.replace(/\b(?:api[_-]?key|token|secret|password|passwd|authorization|bearer)\b\s*[:=]\s*['"]?[^'"\s,;]+/gi, (match) => {
    const key = match.split(/[:=]/)[0]?.trim() || 'secret';
    return `${key}=[redacted-secret]`;
  });
  text = text.replace(/\b(?:sk|pk|ghp|github_pat|xox[baprs])-?[A-Za-z0-9_\-]{16,}\b/g, '[redacted-secret]');
  text = text.replace(/\b[A-Fa-f0-9]{32,}\b/g, '[redacted-blob]');
  text = text.replace(/\b[A-Za-z0-9+/]{40,}={0,2}\b/g, '[redacted-blob]');
  return clipText(text, maxChars);
}

export function redactJsonValue(value: any): any {
  return redactStructuredValue(value);
}

function redactStructuredValue(value: any): any {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
      try {
        return JSON.stringify(redactStructuredValue(JSON.parse(trimmed)));
      } catch {
        return value;
      }
    }
    return value;
  }
  if (Array.isArray(value)) return value.map(redactStructuredValue);
  if (!value || typeof value !== 'object') return value;
  const result: Record<string, any> = {};
  for (const [key, entry] of Object.entries(value)) {
    result[key] = isSensitiveKey(key) ? '[redacted-secret]' : redactStructuredValue(entry);
  }
  return result;
}

function isSensitiveKey(key: string) {
  return /^(?:api[_-]?key|token|secret|password|passwd|authorization|bearer|client[_-]?secret|access[_-]?token|refresh[_-]?token|session[_-]?cookie|private[_-]?key|secret[_-]?key)$/i.test(key)
    || /(?:api[_-]?key|client[_-]?secret|access[_-]?token|refresh[_-]?token|session[_-]?cookie|private[_-]?key|secret[_-]?key)$/i.test(key);
}

export function latestUserTextFromEvent(event: any = {}) {
  if (typeof event.userMessage === 'string') return event.userMessage;
  if (typeof event.user_message === 'string') return event.user_message;
  if (typeof event.userMessageRedacted === 'string') return event.userMessageRedacted;
  if (typeof event.user_message_redacted === 'string') return event.user_message_redacted;
  if (typeof event.input === 'string') return event.input;
  if (Array.isArray(event.messages)) {
    const latest = [...event.messages].reverse().find((message) => message?.role === 'user');
    const extracted = textFromMessageContent(latest?.content) || safeString(latest?.redactedText ?? '');
    if (extracted) return extracted;
  }
  // OpenClaw's typed `before_prompt_build` hook exposes the active user turn as
  // `prompt` plus prepared session messages. Use it only after explicit
  // user-message fields / messages so tests and future richer hook payloads can
  // override it with already-redacted text.
  if (typeof event.prompt === 'string') return event.prompt;
  return '';
}

function textFromMessageContent(content: any): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .map((part) => {
      if (typeof part === 'string') return part;
      if (!part || typeof part !== 'object') return '';
      if (typeof part.text === 'string') return part.text;
      if (typeof part.content === 'string') return part.content;
      return '';
    })
    .filter(Boolean)
    .join('\n')
    .trim();
}

export function sanitizeForProof(value: any): any {
  if (Array.isArray(value)) return value.map(sanitizeForProof);
  if (!value || typeof value !== 'object') return value;
  const blocked = new Set(['prompt', 'rawPrompt', 'userText', 'rawUserText', 'transcript', 'messages', 'input', 'output', 'response', 'completion']);
  const result: Record<string, any> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (blocked.has(key)) continue;
    result[key] = sanitizeForProof(entry);
  }
  return result;
}
