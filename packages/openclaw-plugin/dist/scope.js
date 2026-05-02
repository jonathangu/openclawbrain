import { safeString } from './redact.js';
export function scopeContextFromPacket(packet = {}) {
    const metadata = packet.metadata || {};
    return {
        agentId: safeString(packet.agentId || 'main') || 'main',
        sessionId: safeString(packet.sessionId || ''),
        sessionKey: safeString(packet.sessionKey || ''),
        channelId: safeString(metadata.channelId || ''),
        messageProvider: safeString(metadata.messageProvider || ''),
        repo: safeString(metadata.repo || metadata.repository || ''),
        project: safeString(metadata.project || ''),
        app: safeString(metadata.app || ''),
        task: safeString(metadata.task || metadata.taskId || ''),
        tool: safeString(metadata.tool || ''),
        person: safeString(metadata.person || ''),
    };
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
            return !key || key === context.channelId || key === context.messageProvider;
        case 'repo':
            return !context.repo || !key || key === context.repo || (key === 'current_repo' && Boolean(context.repo));
        case 'project':
            return !context.project || !key || key === context.project || (key === 'current_project' && Boolean(context.project));
        case 'app':
            return !context.app || !key || key === context.app;
        case 'task':
            return Boolean(key) && key === context.task;
        case 'tool':
            return !key || key === context.tool;
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
