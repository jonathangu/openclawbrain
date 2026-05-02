import { safeString } from './redact.js';
export function scopeContextFromPacket(packet = {}) {
    const metadata = packet.metadata || {};
    const text = String(packet.latestUserMessageRedacted || '');
    return {
        agentId: safeString(packet.agentId || 'main') || 'main',
        sessionId: safeString(packet.sessionId || ''),
        sessionKey: safeString(packet.sessionKey || ''),
        channelId: safeString(metadata.channelId || ''),
        messageProvider: safeString(metadata.messageProvider || ''),
        repo: safeString(metadata.repo || metadata.repository || inferRepo(text)),
        project: safeString(metadata.project || inferProject(text)),
        app: safeString(metadata.app || inferApp(text)),
        task: safeString(metadata.task || metadata.taskId || ''),
        tool: safeString(metadata.tool || ''),
        person: safeString(metadata.person || ''),
    };
}
function inferRepo(text) {
    if (/\bopenclawbrain|openclaw brain|ocb\b/i.test(text))
        return 'openclawbrain';
    return '';
}
function inferProject(text) {
    if (/\bpelican\b/i.test(text))
        return 'Pelican';
    if (/\bbountiful\b/i.test(text))
        return 'Bountiful Garden';
    return '';
}
function inferApp(text) {
    if (/\bcormorantai\b/i.test(text))
        return 'CormorantAI';
    return '';
}
export function defaultScopeContext(agentId = 'main') {
    return { agentId: safeString(agentId || 'main') || 'main' };
}
export function memoryInScope(memory, context) {
    if (memory.agentId !== context.agentId)
        return false;
    const key = safeString(memory.scopeKey || '');
    switch (memory.scopeKind) {
        case 'global_user':
            return true;
        case 'agent':
            return !key || key === context.agentId;
        case 'session':
            return Boolean(key) && (key === context.sessionId || key === context.sessionKey);
        case 'channel':
            return !key || Boolean(context.channelId || context.messageProvider) && (key === context.channelId || key === context.messageProvider);
        case 'repo':
            return !key || Boolean(context.repo) && (key === context.repo || key === 'current_repo');
        case 'project':
            return !key || Boolean(context.project) && (key === context.project || key === 'current_project');
        case 'app':
            return !key || Boolean(context.app) && key === context.app;
        case 'task':
            return Boolean(key) && key === context.task;
        case 'tool':
            return !key || Boolean(context.tool) && key === context.tool;
        case 'person':
            return Boolean(key) && key === context.person;
        default:
            return false;
    }
}
export function filterMemoriesForScope(memories, context) {
    if (!context)
        return memories;
    return memories.filter((memory) => memoryInScope(memory, context));
}
