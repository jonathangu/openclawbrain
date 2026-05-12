import { randomUUID, createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { openDatabase } from './sqlite-driver.js';
import { clipText, redactText, safeString } from './redact.js';
import { graphHelpText, handleGraphBrainCommand } from './graph-maintenance.js';
export const DEFAULT_CODEX_BRIDGE_CONFIG = Object.freeze({
    enabled: true,
    statePaths: Object.freeze(['~/.codex/state_5.sqlite']),
    bridgeStatePath: '~/.openclawbrain/activation/${agentId}/codex-continuity.sqlite',
    preferAppServer: false,
    appServerCommand: 'codex',
    appServerArgs: Object.freeze(['app-server', 'proxy']),
    appServerUrl: '',
    appServerTimeoutMs: 1200,
    staleAfterMs: 10 * 60 * 1000,
    maxThreads: 10,
    watchPollIntervalMs: 60 * 1000,
    messageWatchesEnabled: true,
    directMessageCopyEnabled: true,
    telegramForwardingMode: 'redacted',
    enableTelegramWrites: false,
    enableTelegramSteer: false,
    trustOpenClawAuth: true,
    allowLatestTargetForWrites: false,
    highRiskTelegramWrites: false,
    trustedTelegramSenders: Object.freeze([]),
    repoAllowlist: Object.freeze([]),
    readAllowlist: Object.freeze([]),
    writeAllowlist: Object.freeze([]),
    destructiveWriteAllowlist: Object.freeze([]),
    notifyChannel: 'telegram',
    notifyTarget: '',
});
export function normalizeCodexBridgeConfig(source = {}) {
    const input = source && typeof source === 'object' ? source : {};
    const statePaths = Array.isArray(input.statePaths)
        ? input.statePaths.map((value) => safeString(value)).filter(Boolean)
        : typeof input.statePath === 'string'
            ? [input.statePath]
            : [...DEFAULT_CODEX_BRIDGE_CONFIG.statePaths];
    const appServerArgs = Array.isArray(input.appServerArgs)
        ? input.appServerArgs.map((value) => safeString(value)).filter(Boolean)
        : [...DEFAULT_CODEX_BRIDGE_CONFIG.appServerArgs];
    return {
        enabled: input.enabled !== false,
        statePaths: statePaths.length ? statePaths : [...DEFAULT_CODEX_BRIDGE_CONFIG.statePaths],
        bridgeStatePath: nonEmptyString(input.bridgeStatePath) || DEFAULT_CODEX_BRIDGE_CONFIG.bridgeStatePath,
        preferAppServer: input.preferAppServer === true,
        appServerCommand: nonEmptyString(input.appServerCommand) || DEFAULT_CODEX_BRIDGE_CONFIG.appServerCommand,
        appServerArgs: appServerArgs.length ? appServerArgs : [...DEFAULT_CODEX_BRIDGE_CONFIG.appServerArgs],
        appServerUrl: nonEmptyString(input.appServerUrl) || DEFAULT_CODEX_BRIDGE_CONFIG.appServerUrl,
        appServerTimeoutMs: clampInteger(input.appServerTimeoutMs, DEFAULT_CODEX_BRIDGE_CONFIG.appServerTimeoutMs, 100, 30000),
        staleAfterMs: clampInteger(input.staleAfterMs, DEFAULT_CODEX_BRIDGE_CONFIG.staleAfterMs, 1000, 86400000),
        maxThreads: clampInteger(input.maxThreads, DEFAULT_CODEX_BRIDGE_CONFIG.maxThreads, 1, 100),
        watchPollIntervalMs: clampInteger(input.watchPollIntervalMs, DEFAULT_CODEX_BRIDGE_CONFIG.watchPollIntervalMs, 5000, 86400000),
        messageWatchesEnabled: input.messageWatchesEnabled !== false,
        directMessageCopyEnabled: input.directMessageCopyEnabled !== false,
        telegramForwardingMode: ['redacted', 'raw_trusted', 'metadata_only'].includes(String(input.telegramForwardingMode))
            ? String(input.telegramForwardingMode)
            : DEFAULT_CODEX_BRIDGE_CONFIG.telegramForwardingMode,
        enableTelegramWrites: input.enableTelegramWrites === true,
        enableTelegramSteer: input.enableTelegramSteer === true,
        trustOpenClawAuth: input.trustOpenClawAuth !== false,
        allowLatestTargetForWrites: input.allowLatestTargetForWrites === true,
        highRiskTelegramWrites: input.highRiskTelegramWrites === true,
        trustedTelegramSenders: Array.isArray(input.trustedTelegramSenders)
            ? input.trustedTelegramSenders.map((value) => safeString(value)).filter(Boolean)
            : [],
        repoAllowlist: Array.isArray(input.repoAllowlist)
            ? input.repoAllowlist.map((value) => safeString(value)).filter(Boolean)
            : [],
        readAllowlist: Array.isArray(input.readAllowlist)
            ? input.readAllowlist.map((value) => safeString(value)).filter(Boolean)
            : [],
        writeAllowlist: Array.isArray(input.writeAllowlist)
            ? input.writeAllowlist.map((value) => safeString(value)).filter(Boolean)
            : [],
        destructiveWriteAllowlist: Array.isArray(input.destructiveWriteAllowlist)
            ? input.destructiveWriteAllowlist.map((value) => safeString(value)).filter(Boolean)
            : [],
        notifyChannel: nonEmptyString(input.notifyChannel) || DEFAULT_CODEX_BRIDGE_CONFIG.notifyChannel,
        notifyTarget: nonEmptyString(input.notifyTarget) || DEFAULT_CODEX_BRIDGE_CONFIG.notifyTarget,
    };
}
export async function buildCodexBridgeStatus(config, agentId = 'main', deps = {}) {
    const bridgeConfig = normalizeCodexBridgeConfig(config.codexBridge);
    const generatedAt = new Date(deps.nowMs?.() ?? Date.now()).toISOString();
    const store = new CodexBridgeStore({ config, agentId });
    const watched = store.listWatches(agentId, { activeOnly: true }).length;
    store.close();
    if (!bridgeConfig.enabled) {
        return emptyStatus(generatedAt, watched, 'codex_bridge_disabled');
    }
    const errors = [];
    let appServerAvailable = false;
    if (bridgeConfig.preferAppServer && deps.appServerReader) {
        try {
            const response = await deps.appServerReader.listThreads({ limit: bridgeConfig.maxThreads, timeoutMs: bridgeConfig.appServerTimeoutMs });
            const threads = normalizeAppServerThreads(response, deps.nowMs?.() ?? Date.now());
            if (threads.length > 0) {
                appServerAvailable = true;
                return statusFromThreads({
                    threads,
                    source: 'app_server',
                    stale: false,
                    generatedAt,
                    watched,
                    errors,
                    appServerAvailable,
                    sqliteFallbackAvailable: true,
                    writeEnabled: bridgeConfig.enableTelegramWrites,
                    steerEnabled: bridgeConfig.enableTelegramWrites && bridgeConfig.enableTelegramSteer,
                });
            }
            appServerAvailable = true;
        }
        catch (error) {
            errors.push(`app_server_unavailable:${clipText(String(error?.message || error), 160)}`);
        }
    }
    else if (bridgeConfig.preferAppServer) {
        errors.push('app_server_unavailable:no_host_reader_configured');
    }
    const sqlite = readCodexThreadsFromSqlite(bridgeConfig, { limit: bridgeConfig.maxThreads, nowMs: deps.nowMs?.() ?? Date.now() });
    errors.push(...sqlite.errors);
    if (sqlite.threads.length > 0) {
        const newest = Math.max(...sqlite.threads.map((thread) => thread.updatedAtMs || 0));
        const age = (deps.nowMs?.() ?? Date.now()) - newest;
        return statusFromThreads({
            threads: sqlite.threads,
            source: 'sqlite_fallback',
            stale: true,
            staleReason: age > bridgeConfig.staleAfterMs ? `sqlite_snapshot_age_ms:${age}` : 'sqlite_fallback_read_only',
            generatedAt,
            watched,
            errors,
            appServerAvailable,
            sqliteFallbackAvailable: true,
            writeEnabled: bridgeConfig.enableTelegramWrites,
            steerEnabled: bridgeConfig.enableTelegramWrites && bridgeConfig.enableTelegramSteer,
        });
    }
    return {
        ...emptyStatus(generatedAt, watched, 'no_codex_state_available'),
        errors,
        capabilities: {
            canReadThreads: false,
            canReadGoals: false,
            canReadMessages: false,
            canSubscribe: false,
            canStartTurn: false,
            canSteerTurn: false,
            canWrite: false,
            appServerAvailable,
            sqliteFallbackAvailable: false,
        },
    };
}
export function readCodexThreadsFromSqlite(bridgeConfig, options = {}) {
    const errors = [];
    const limit = Math.min(100, Math.max(1, Number(options.limit || bridgeConfig.maxThreads || 10)));
    for (const rawPath of bridgeConfig.statePaths) {
        const statePath = expandPathTemplate(rawPath, 'main');
        if (!existsSync(statePath)) {
            errors.push(`sqlite_missing:${statePath}`);
            continue;
        }
        let db;
        try {
            db = openReadOnlyBetterSqlite(statePath);
            const search = safeString(options.searchTerm).toLowerCase();
            const threadColumns = tableColumns(db, 'threads');
            const rolloutSelect = threadColumns.has('rollout_path') ? 't.rollout_path' : 'NULL AS rollout_path';
            const firstUserSelect = threadColumns.has('first_user_message') ? 't.first_user_message' : 'NULL AS first_user_message';
            const rows = db.prepare(`
        SELECT
          t.id,
          t.title,
          t.cwd,
          t.git_branch,
          t.git_sha,
          t.updated_at,
          t.updated_at_ms,
          t.archived,
          t.model,
          t.reasoning_effort,
          ${rolloutSelect},
          ${firstUserSelect},
          g.goal_id,
          g.objective,
          g.status AS goal_status,
          g.token_budget,
          g.tokens_used,
          g.time_used_seconds,
          g.updated_at_ms AS goal_updated_at_ms
        FROM threads t
        LEFT JOIN thread_goals g ON g.thread_id = t.id
        WHERE t.archived = 0
        ORDER BY COALESCE(t.updated_at_ms, t.updated_at * 1000) DESC
        LIMIT ?
      `).all(limit * 3);
            const threads = rows
                .map((row) => rowToThread(row, 'sqlite_fallback'))
                .filter((thread) => !search || `${thread.title} ${thread.cwd} ${thread.goal?.objective || ''}`.toLowerCase().includes(search))
                .slice(0, limit);
            return { threads, sourcePath: statePath, errors };
        }
        catch (error) {
            errors.push(`sqlite_read_failed:${statePath}:${clipText(String(error?.message || error), 160)}`);
        }
        finally {
            try {
                db?.close?.();
            }
            catch { /* ignore */ }
        }
    }
    return { threads: [], errors };
}
export function readCodexTranscriptMessages(thread, options = {}) {
    const rolloutPath = thread.rolloutPath ? expandPathTemplate(thread.rolloutPath, 'main') : '';
    if (!rolloutPath) {
        return { ok: false, messages: [], errors: ['thread_has_no_rollout_path'], truncated: false };
    }
    if (!existsSync(rolloutPath)) {
        return { ok: false, messages: [], errors: [`rollout_missing:${rolloutPath}`], truncated: false, rolloutPath };
    }
    let raw = '';
    try {
        raw = readFileSync(rolloutPath, 'utf8');
    }
    catch (error) {
        return { ok: false, messages: [], errors: [`rollout_read_failed:${clipText(String(error?.message || error), 160)}`], truncated: false, rolloutPath };
    }
    const parsed = parseCodexRolloutJsonl(raw, thread.id, options.afterLine || 0);
    const role = options.role || 'all';
    const messages = parsed.messages
        .filter((message) => role === 'all' || message.role === role)
        .slice(-(Math.min(50, Math.max(1, Number(options.limit || 5)))));
    return { ok: true, messages, errors: parsed.errors, truncated: parsed.truncated, rolloutPath };
}
function parseCodexRolloutJsonl(raw, threadId, afterLine = 0) {
    const lines = raw.split(/\r?\n/);
    const primary = [];
    const fallback = [];
    const errors = [];
    let byteOffset = 0;
    lines.forEach((line, index) => {
        const lineNumber = index + 1;
        const currentOffset = byteOffset;
        byteOffset += Buffer.byteLength(line, 'utf8') + 1;
        if (lineNumber <= afterLine || !line.trim())
            return;
        let parsed;
        try {
            parsed = JSON.parse(line);
        }
        catch (error) {
            if (index === lines.length - 1) {
                errors.push('partial_trailing_jsonl_line');
            }
            else {
                errors.push(`malformed_jsonl_line:${lineNumber}`);
            }
            return;
        }
        const timestamp = safeString(parsed.timestamp) || new Date(0).toISOString();
        if (parsed.type === 'response_item' && parsed.payload?.type === 'message') {
            const role = parsed.payload.role === 'assistant' ? 'assistant' : parsed.payload.role === 'user' ? 'user' : '';
            const text = extractMessageContentText(parsed.payload.content);
            if (!role || !text)
                return;
            primary.push(buildTranscriptMessage({
                id: safeString(parsed.payload.id) || `response:${lineNumber}`,
                threadId,
                role,
                text,
                timestamp,
                source: 'rollout_jsonl',
                lineNumber,
                byteOffset: currentOffset,
                messageKind: 'final_message',
            }));
            return;
        }
        if (parsed.type === 'event_msg' && (parsed.payload?.type === 'user_message' || parsed.payload?.type === 'agent_message')) {
            const role = parsed.payload.type === 'agent_message' ? 'assistant' : 'user';
            const text = safeString(parsed.payload.message);
            if (!text)
                return;
            fallback.push(buildTranscriptMessage({
                id: `event:${lineNumber}`,
                threadId,
                role,
                text,
                timestamp,
                source: 'event_msg',
                lineNumber,
                byteOffset: currentOffset,
                messageKind: 'ui_event',
            }));
        }
    });
    const primaryTextHashes = new Set(primary.map((message) => sha256(`${message.role}:${normalizeTranscriptText(message.text)}`)));
    const messages = primary.concat(fallback.filter((message) => !primaryTextHashes.has(sha256(`${message.role}:${normalizeTranscriptText(message.text)}`))));
    messages.sort((left, right) => left.lineNumber - right.lineNumber);
    return { messages, errors, truncated: false };
}
function buildTranscriptMessage(input) {
    return {
        ...input,
        hash: sha256(`${input.threadId}:${input.role}:${input.lineNumber}:${normalizeTranscriptText(input.text)}`),
    };
}
function extractMessageContentText(content) {
    if (!Array.isArray(content))
        return '';
    const parts = content
        .map((item) => {
        if (!item || typeof item !== 'object')
            return '';
        if (item.type === 'input_text' || item.type === 'output_text' || item.type === 'text') {
            return typeof item.text === 'string' ? item.text : '';
        }
        return '';
    })
        .filter(Boolean);
    return parts.join('\n').trim();
}
function normalizeTranscriptText(text) {
    return safeString(text).replace(/\s+/g, ' ').trim();
}
export function formatCodexMessages(thread, result, options = {}) {
    if (result.messages.length === 0) {
        return `No Codex messages found for ${thread.id}${result.errors?.length ? ` (${result.errors.join(', ')})` : ''}.`;
    }
    const lines = [
        options.title || `Codex messages: ${thread.title} [${thread.id}]`,
        '',
        ...result.messages.flatMap((message) => [
            `${message.role} ${message.timestamp} line ${message.lineNumber}:`,
            message.text,
            '',
        ]),
    ];
    return clipText(lines.join('\n').trim(), options.full ? 50000 : 12000);
}
function parseMessageCommand(rest, lastOnly) {
    let limit = lastOnly ? 1 : 5;
    let role = 'all';
    let full = false;
    let target;
    for (let index = 0; index < rest.length; index += 1) {
        const item = rest[index];
        if (item === '--limit') {
            limit = clampInteger(rest[index + 1], limit, 1, 50);
            index += 1;
            continue;
        }
        if (item?.startsWith('--limit=')) {
            limit = clampInteger(item.slice('--limit='.length), limit, 1, 50);
            continue;
        }
        if (item === '--role') {
            role = parseRole(rest[index + 1], role);
            index += 1;
            continue;
        }
        if (item?.startsWith('--role=')) {
            role = parseRole(item.slice('--role='.length), role);
            continue;
        }
        if (item === '--full') {
            full = true;
            continue;
        }
        if (!target)
            target = item;
    }
    return { target, limit, role, full };
}
function parseRole(value, fallback) {
    return value === 'assistant' || value === 'user' || value === 'all' ? value : fallback;
}
function resolveThreadTarget(input) {
    const target = safeString(input.target);
    if (input.mode === 'write' && target === '--latest') {
        return { reason: '--latest is not allowed for Codex writes. Bind a thread or provide an explicit thread id.' };
    }
    if (target && target !== '--latest' && target !== '--bound') {
        const exact = input.status.latestThreads.find((thread) => thread.id === target);
        return exact ? { thread: exact, reason: 'explicit_thread_id' } : { reason: `No visible Codex thread matched ${target}.` };
    }
    if (target === '--bound' || input.mode === 'write') {
        const binding = input.store.getBinding(input.agentId, contextChatKey(input.ctx));
        if (!binding)
            return { reason: 'No Codex thread is bound to this Telegram chat. Use /brain codex bind <thread-id> first.' };
        const thread = input.status.latestThreads.find((item) => item.id === binding.threadId) ?? {
            id: binding.threadId,
            title: binding.title || binding.threadId,
            cwd: binding.cwd || '',
            updatedAtMs: Date.now(),
            archived: false,
            source: input.status.source,
        };
        return { thread, reason: 'bound_thread' };
    }
    const latest = input.status.activeGoals[0] ?? input.status.latestThreads[0];
    return latest ? { thread: latest, reason: 'latest_read_target' } : { reason: 'No Codex thread is visible yet.' };
}
function contextChatKey(ctx) {
    return safeString(ctx.chatId ?? ctx.chat_id ?? ctx.to ?? ctx.from ?? ctx.channelId ?? ctx.message?.chat?.id ?? 'local');
}
function contextSenderKey(ctx) {
    return safeString(ctx.senderId ?? ctx.sender_id ?? ctx.userId ?? ctx.user_id ?? ctx.from ?? ctx.accountId ?? 'local');
}
function formatBinding(binding) {
    return [
        'Codex binding:',
        `- Thread: ${binding.threadId}`,
        `- Title: ${binding.title || 'unknown'}`,
        `- Workspace: ${binding.cwd || 'unknown'}`,
        `- Updated: ${binding.updatedAt}`,
    ].join('\n');
}
function createCodexTerminalWatch(input) {
    const watch = input.store.createWatch({
        agentId: input.agentId,
        scope: 'thread',
        threadId: input.thread.id,
        notifyChannel: input.config.notifyChannel || safeString(input.ctx.channel) || 'telegram',
        notifyTarget: input.config.notifyTarget || safeString(input.ctx.to ?? input.ctx.from ?? ''),
        accountId: safeString(input.ctx.accountId),
        messageThreadId: input.ctx.messageThreadId == null ? undefined : String(input.ctx.messageThreadId),
        allowedClasses: ['completion', 'failure', 'blocker', 'approval_required', 'auth_failure'],
        sensitivity: 'normal',
        verbosity: 'blockers_and_completion',
    });
    input.store.recordEvent({
        agentId: input.agentId,
        eventType: 'watch_created',
        eventClass: 'watch_created',
        source: input.status.source,
        threadId: input.thread.id,
        goalKey: undefined,
        decision: 'recorded',
        notified: false,
        reason: 'explicit_user_terminal_watch_request',
        redactedSummary: `Watch created for Codex thread ${input.thread.id}`,
        dedupeKey: `watch:${watch.id}`,
    });
    return watch;
}
function createCodexMessageWatch(input) {
    const watch = input.store.createWatch({
        agentId: input.agentId,
        scope: 'thread',
        threadId: input.thread.id,
        notifyChannel: input.config.notifyChannel || safeString(input.ctx.channel) || 'telegram',
        notifyTarget: input.config.notifyTarget || safeString(input.ctx.to ?? input.ctx.from ?? ''),
        accountId: safeString(input.ctx.accountId),
        messageThreadId: input.ctx.messageThreadId == null ? undefined : String(input.ctx.messageThreadId),
        allowedClasses: ['assistant_message', 'completion', 'failure', 'blocker', 'approval_required', 'auth_failure'],
        sensitivity: input.config.telegramForwardingMode === 'metadata_only' ? 'no_telegram_details' : 'normal',
        verbosity: 'assistant_messages',
    });
    const transcript = readCodexTranscriptMessages(input.thread, { limit: 1, role: 'all' });
    const latest = transcript.messages.at(-1);
    if (input.thread.rolloutPath) {
        input.store.upsertMessageCursor({
            watchId: watch.id,
            agentId: input.agentId,
            threadId: input.thread.id,
            rolloutPath: expandPathTemplate(input.thread.rolloutPath, 'main'),
            parseCursorLine: latest?.lineNumber || 0,
            parseCursorByteOffset: latest?.byteOffset || 0,
            deliveryCursorLine: latest?.lineNumber || 0,
            deliveryCursorByteOffset: latest?.byteOffset || 0,
            lastMessageId: latest?.id,
            lastMessageHash: latest?.hash,
            fileIdentity: fileIdentity(input.thread.rolloutPath),
        });
    }
    input.store.recordEvent({
        agentId: input.agentId,
        eventType: 'watch_created',
        eventClass: 'watch_created',
        source: input.thread.source,
        threadId: input.thread.id,
        decision: 'recorded',
        notified: false,
        reason: 'explicit_user_message_watch_request',
        redactedSummary: `Message watch created for Codex thread ${input.thread.id}`,
        dedupeKey: `watch:${watch.id}`,
    });
    return watch;
}
export function buildCodexHandoff(status, threadId) {
    const selected = threadId
        ? status.latestThreads.find((thread) => thread.id === threadId)
        : status.activeGoals[0] ?? status.latestThreads[0];
    const generatedAt = new Date().toISOString();
    if (!selected) {
        return {
            ok: false,
            source: status.source,
            observedFacts: ['No local Codex thread was visible through app-server or SQLite fallback.'],
            codexReportedClaims: [],
            evidence: [`Bridge source: ${status.source}`, `Generated: ${generatedAt}`],
            interpretation: ['There is no safe thread-specific handoff yet.'],
            nextActions: ['Open Codex UI or run a Codex turn, then ask for a handoff again.'],
            stale: true,
            generatedAt,
        };
    }
    const goal = selected.goal;
    return {
        ok: true,
        source: status.source,
        threadId: selected.id,
        title: selected.title,
        observedFacts: [
            `Thread: ${selected.id}`,
            `Workspace: ${selected.cwd || 'unknown'}`,
            `Branch: ${selected.branch || 'unknown'}`,
            `Updated: ${new Date(selected.updatedAtMs || Date.now()).toISOString()}`,
            ...(goal ? [`Goal status: ${goal.status}`, `Goal tokens used: ${goal.tokensUsed}`] : ['Goal: none visible']),
        ],
        codexReportedClaims: goal ? [`Goal objective: ${goal.objective}`] : [],
        evidence: [
            `Read source: ${status.source}`,
            status.stale ? `Stale label: ${status.staleReason || 'read-only fallback'}` : 'App-server/current source available',
            'No command output, raw deltas, full transcript, or diff content stored in durable memory.',
        ],
        interpretation: [
            goal?.status === 'active'
                ? 'Codex appears to have an active goal on this thread.'
                : goal?.status === 'complete'
                    ? 'Codex reports this goal complete; verify code/tests from the worktree before acting on claims.'
                    : 'This is a status handoff, not proof of code correctness.',
        ],
        nextActions: [
            'Use Codex UI for high-bandwidth work.',
            'Use /brain codex watch if you want one concise completion/blocker update.',
            'Verify any reported tests or file changes from the repository, not from summary text alone.',
        ],
        stale: status.stale,
        generatedAt,
    };
}
export async function handleBrainCommand(ctx, config, api = {}) {
    const args = splitArgs(ctx.args || '');
    const [namespace = 'help', subcommand = 'status', ...rest] = args;
    if (namespace === 'help' || namespace === '--help')
        return { text: brainHelpText() };
    if (namespace === 'graph')
        return handleGraphBrainCommand(ctx, [subcommand, ...rest], config);
    if (namespace !== 'codex')
        return { text: brainHelpText() };
    const agentId = safeString(ctx.agentId ?? config.scopes?.agents?.[0] ?? 'main') || 'main';
    const bridgeConfig = normalizeCodexBridgeConfig(config.codexBridge);
    const status = await buildCodexBridgeStatus(config, agentId);
    if (subcommand === 'status')
        return { text: formatCodexStatus(status) };
    if (subcommand === 'threads')
        return { text: formatCodexThreads(status, rest.join(' ')) };
    if (subcommand === 'messages' || subcommand === 'last') {
        if (!bridgeConfig.directMessageCopyEnabled)
            return { text: 'Codex direct message copy is disabled.' };
        const parsed = parseMessageCommand(rest, subcommand === 'last');
        const store = new CodexBridgeStore({ config, agentId });
        try {
            const resolved = resolveThreadTarget({ status, store, agentId, ctx, target: parsed.target, mode: 'read' });
            if (!resolved.thread)
                return { text: resolved.reason };
            const result = readCodexTranscriptMessages(resolved.thread, {
                limit: subcommand === 'last' ? 1 : parsed.limit,
                role: subcommand === 'last' ? 'assistant' : parsed.role,
            });
            return { text: formatCodexMessages(resolved.thread, result, { title: subcommand === 'last' ? `Latest Codex assistant message: ${resolved.thread.title} [${resolved.thread.id}]` : undefined, full: parsed.full }) };
        }
        finally {
            store.close();
        }
    }
    if (subcommand === 'handoff') {
        const brief = buildCodexHandoff(status, rest[0]);
        return { text: formatHandoffBrief(brief) };
    }
    if (subcommand === 'bind') {
        const target = rest[0];
        if (!target)
            return { text: 'Usage: /brain codex bind <thread-id>' };
        const store = new CodexBridgeStore({ config, agentId });
        try {
            const resolved = resolveThreadTarget({ status, store, agentId, ctx, target, mode: 'read' });
            if (!resolved.thread || resolved.thread.id !== target)
                return { text: resolved.thread ? 'Bind requires an explicit full thread id.' : resolved.reason };
            const binding = store.bindConversation({ agentId, chatKey: contextChatKey(ctx), senderKey: contextSenderKey(ctx), thread: resolved.thread });
            store.recordEvent({
                agentId,
                eventType: 'binding_created',
                eventClass: 'binding_created',
                source: status.source,
                threadId: resolved.thread.id,
                decision: 'recorded',
                notified: false,
                reason: 'explicit_user_bind_request',
                redactedSummary: `Bound Telegram chat to Codex thread ${resolved.thread.id}`,
                dedupeKey: `binding:${binding.id}:${binding.updatedAt}`,
            });
            return { text: `Bound this Telegram chat to Codex thread ${resolved.thread.id}: ${resolved.thread.title}` };
        }
        finally {
            store.close();
        }
    }
    if (subcommand === 'binding') {
        const store = new CodexBridgeStore({ config, agentId });
        try {
            const binding = store.getBinding(agentId, contextChatKey(ctx));
            return { text: binding ? formatBinding(binding) : 'No Codex thread is bound to this Telegram chat.' };
        }
        finally {
            store.close();
        }
    }
    if (subcommand === 'unbind') {
        const store = new CodexBridgeStore({ config, agentId });
        try {
            const removed = store.unbindConversation(agentId, contextChatKey(ctx));
            if (removed) {
                store.recordEvent({
                    agentId,
                    eventType: 'binding_removed',
                    eventClass: 'binding_removed',
                    source: status.source,
                    decision: 'recorded',
                    notified: false,
                    reason: 'explicit_user_unbind_request',
                    redactedSummary: 'Removed Codex binding for Telegram chat',
                    dedupeKey: `binding_removed:${sha256(contextChatKey(ctx))}:${Date.now()}`,
                });
            }
            return { text: removed ? 'Removed the Codex thread binding for this Telegram chat.' : 'No Codex binding was attached.' };
        }
        finally {
            store.close();
        }
    }
    if (subcommand === 'watches') {
        const store = new CodexBridgeStore({ config, agentId });
        try {
            const watches = store.listWatches(agentId, { activeOnly: false });
            return { text: formatWatches(watches) };
        }
        finally {
            store.close();
        }
    }
    if (subcommand === 'unwatch') {
        const target = rest[0];
        if (!target)
            return { text: 'Usage: /brain codex unwatch <watch-id|thread-id>' };
        const store = new CodexBridgeStore({ config, agentId });
        try {
            const removed = store.pauseWatches(agentId, target);
            return { text: removed > 0 ? `Paused ${removed} Codex watch(es).` : 'No matching active Codex watch found.' };
        }
        finally {
            store.close();
        }
    }
    if (subcommand === 'tail') {
        if (!bridgeConfig.messageWatchesEnabled)
            return { text: 'Codex message watches are disabled.' };
        const parsed = parseMessageCommand(rest, false);
        const store = new CodexBridgeStore({ config, agentId });
        try {
            const resolved = resolveThreadTarget({ status, store, agentId, ctx, target: parsed.target, mode: 'read' });
            if (!resolved.thread)
                return { text: resolved.reason };
            const watch = createCodexMessageWatch({ store, agentId, ctx, config: bridgeConfig, thread: resolved.thread });
            return { text: `Tailing completed assistant messages for Codex thread ${resolved.thread.id}. I will forward new replies and keep tool chatter quiet.` };
        }
        finally {
            store.close();
        }
    }
    if (subcommand === 'watch') {
        const messageMode = rest.includes('--messages') || rest.includes('--all');
        const target = rest.find((item) => !item.startsWith('--'));
        const store = new CodexBridgeStore({ config, agentId });
        try {
            const resolved = resolveThreadTarget({ status, store, agentId, ctx, target, mode: 'read' });
            if (!resolved.thread)
                return { text: resolved.reason };
            const watch = messageMode
                ? createCodexMessageWatch({ store, agentId, ctx, config: bridgeConfig, thread: resolved.thread })
                : createCodexTerminalWatch({ store, agentId, ctx, config: bridgeConfig, status, thread: resolved.thread });
            return { text: messageMode ? `Watching Codex thread ${resolved.thread.id} for new assistant messages.` : `Watching Codex thread ${resolved.thread.id}. I will stay quiet unless it completes, fails, blocks, needs approval, or hits auth trouble.` };
        }
        finally {
            store.close();
        }
    }
    if (subcommand === 'reply' || subcommand === 'send' || subcommand === 'steer') {
        return handleCodexWriteCommand({ subcommand, rest, ctx, config, bridgeConfig, status, agentId, api });
    }
    if (subcommand === 'goal') {
        return { text: '/brain codex goal is still a later phase. Use /brain codex bind plus /brain codex reply or /brain codex steer for existing threads.' };
    }
    return { text: brainHelpText() };
}
export async function processCodexBridgeWatches(config, api = {}, deps = {}) {
    const bridgeConfig = normalizeCodexBridgeConfig(config.codexBridge);
    if (!bridgeConfig.enabled)
        return { ok: true, processed: 0, notified: 0 };
    const agents = Array.isArray(config.scopes?.agents) && config.scopes.agents.length ? config.scopes.agents : ['main'];
    let processed = 0;
    let notified = 0;
    for (const agentId of agents) {
        const store = new CodexBridgeStore({ config, agentId });
        try {
            const status = await buildCodexBridgeStatus(config, agentId, deps);
            for (const watch of store.listWatches(agentId, { activeOnly: true })) {
                processed += 1;
                const thread = status.latestThreads.find((item) => item.id === watch.threadId);
                if (thread && watch.allowedClasses.includes('assistant_message') && bridgeConfig.messageWatchesEnabled) {
                    const messageResult = await processMessageWatch({ store, watch, thread, bridgeConfig, api, agentId });
                    notified += messageResult.notified;
                }
                const eventClass = classifyThreadForWatch(thread);
                if (!thread || eventClass === 'quiet' || !watch.allowedClasses.includes(eventClass))
                    continue;
                const dedupeKey = watchDedupeKey(watch, thread, eventClass);
                if (watch.dedupeKeyLastSeen === dedupeKey)
                    continue;
                const text = formatWatchNotification(thread, eventClass, status.stale);
                const sendResult = await sendBridgeNotification(api, watch, text);
                store.updateWatchEvent(watch.id, { dedupeKeyLastSeen: dedupeKey, lastEventAt: new Date().toISOString(), status: eventClass === 'completion' ? 'completed' : 'active' });
                store.recordEvent({
                    agentId,
                    eventType: 'watch_terminal_or_attention_event',
                    eventClass,
                    source: status.source,
                    threadId: thread.id,
                    goalKey: thread.goal?.goalId,
                    decision: sendResult.ok ? 'notified' : 'recorded',
                    notified: sendResult.ok,
                    reason: sendResult.ok ? 'watch_allowed_class' : sendResult.reason,
                    redactedSummary: text,
                    dedupeKey,
                });
                if (sendResult.ok)
                    notified += 1;
            }
        }
        finally {
            store.close();
        }
    }
    return { ok: true, processed, notified };
}
async function processMessageWatch(input) {
    if (!input.thread.rolloutPath)
        return { notified: 0 };
    const rolloutPath = expandPathTemplate(input.thread.rolloutPath, 'main');
    const cursor = input.store.getMessageCursor(input.watch.id, input.thread.id, rolloutPath);
    const afterLine = Number(cursor?.delivery_cursor_line || 0);
    const result = readCodexTranscriptMessages(input.thread, { limit: 50, role: 'assistant', afterLine });
    const messages = result.messages.filter((message) => message.lineNumber > afterLine).slice(0, 5);
    if (messages.length === 0) {
        if (!cursor && result.ok) {
            const latest = readCodexTranscriptMessages(input.thread, { limit: 1, role: 'all' }).messages.at(-1);
            input.store.upsertMessageCursor({
                watchId: input.watch.id,
                agentId: input.agentId,
                threadId: input.thread.id,
                rolloutPath,
                parseCursorLine: latest?.lineNumber || 0,
                parseCursorByteOffset: latest?.byteOffset || 0,
                deliveryCursorLine: latest?.lineNumber || 0,
                deliveryCursorByteOffset: latest?.byteOffset || 0,
                lastMessageId: latest?.id,
                lastMessageHash: latest?.hash,
                fileIdentity: fileIdentity(rolloutPath),
            });
        }
        return { notified: 0 };
    }
    let notified = 0;
    for (const message of messages) {
        const text = formatForwardedTranscriptMessage(input.thread, message, input.bridgeConfig);
        const sendResult = await sendBridgeNotification(input.api, input.watch, text);
        if (!sendResult.ok) {
            input.store.recordPendingDelivery({ watchId: input.watch.id, threadId: input.thread.id, message, chatKey: input.watch.notifyTarget, status: 'pending', error: sendResult.reason });
            input.store.recordEvent({
                agentId: input.agentId,
                eventType: 'telegram_delivery_failed',
                eventClass: 'delivery_failed',
                source: input.thread.source,
                threadId: input.thread.id,
                decision: 'recorded',
                notified: false,
                reason: sendResult.reason,
                redactedSummary: `Failed to deliver Codex assistant message ${message.id}`,
                dedupeKey: `delivery_failed:${input.watch.id}:${message.hash}:${Date.now()}`,
            });
            break;
        }
        input.store.recordPendingDelivery({ watchId: input.watch.id, threadId: input.thread.id, message, chatKey: input.watch.notifyTarget, status: 'delivered' });
        input.store.upsertMessageCursor({
            watchId: input.watch.id,
            agentId: input.agentId,
            threadId: input.thread.id,
            rolloutPath,
            parseCursorLine: message.lineNumber,
            parseCursorByteOffset: message.byteOffset,
            deliveryCursorLine: message.lineNumber,
            deliveryCursorByteOffset: message.byteOffset,
            lastMessageId: message.id,
            lastMessageHash: message.hash,
            fileIdentity: fileIdentity(rolloutPath),
        });
        input.store.recordEvent({
            agentId: input.agentId,
            eventType: 'assistant_message_forwarded',
            eventClass: 'assistant_message',
            source: input.thread.source,
            threadId: input.thread.id,
            decision: 'notified',
            notified: true,
            reason: 'message_watch_completed_assistant_message',
            redactedSummary: redactText(message.text, 500),
            dedupeKey: `assistant_message:${input.watch.id}:${message.hash}`,
        });
        notified += 1;
    }
    return { notified };
}
async function handleCodexWriteCommand(input) {
    const store = new CodexBridgeStore({ config: input.config, agentId: input.agentId });
    try {
        if (!input.bridgeConfig.enableTelegramWrites) {
            return { text: 'Telegram-to-Codex writes are disabled in package defaults. Jonathan local profiles should set codexBridge.enableTelegramWrites=true.' };
        }
        if (input.subcommand === 'steer' && !input.bridgeConfig.enableTelegramSteer) {
            return { text: 'Codex steering is disabled in package defaults. Jonathan local profiles should set codexBridge.enableTelegramSteer=true.' };
        }
        if (!isTrustedWriteContext(input.ctx, input.bridgeConfig)) {
            return { text: 'Codex write rejected: this Telegram sender/chat is not trusted for writes.' };
        }
        let target = '--bound';
        let message = '';
        if (input.subcommand === 'send') {
            target = safeString(input.rest[0]);
            message = input.rest.slice(1).join(' ').trim();
            if (!target || !message)
                return { text: 'Usage: /brain codex send <thread-id|--bound> <message>' };
            if (target === '--latest' && !input.bridgeConfig.allowLatestTargetForWrites) {
                return { text: '--latest is not allowed for Codex writes. Use /brain codex bind <thread-id> or an explicit thread id.' };
            }
        }
        else if (input.subcommand === 'steer') {
            const first = safeString(input.rest[0]);
            const knownThreadId = input.status.latestThreads.some((thread) => thread.id === first);
            if (first === '--bound' || knownThreadId || looksLikeCodexThreadId(first)) {
                target = first;
                message = input.rest.slice(1).join(' ').trim();
            }
            else {
                message = input.rest.join(' ').trim();
            }
            if (!message)
                return { text: 'Usage: /brain codex steer [thread-id|--bound] <message>' };
            if (target === '--latest') {
                return { text: '--latest is not allowed for Codex steering. Use /brain codex bind <thread-id> or an explicit thread id.' };
            }
        }
        else {
            message = input.rest.join(' ').trim();
            if (!message)
                return { text: 'Usage: /brain codex reply <message>' };
        }
        const resolved = resolveThreadTarget({ status: input.status, store, agentId: input.agentId, ctx: input.ctx, target, mode: 'write' });
        if (!resolved.thread)
            return { text: resolved.reason };
        if (!isRepoAllowedForWrite(resolved.thread.cwd, input.bridgeConfig)) {
            return { text: `Codex write rejected: ${resolved.thread.cwd || 'unknown workspace'} is not in codexBridge.writeAllowlist.` };
        }
        const riskClass = classifyWriteRisk(message);
        if (riskClass === 'high' && !input.bridgeConfig.highRiskTelegramWrites) {
            return { text: 'Codex write rejected: high-risk Telegram writes are disabled. Use Codex UI at the computer for publish/deploy/delete/secrets/full-access requests.' };
        }
        const idempotencyKey = sha256(`${input.agentId}:${contextChatKey(input.ctx)}:${input.subcommand}:${resolved.thread.id}:${message}`);
        store.recordOutbound({
            agentId: input.agentId,
            sourceChannel: safeString(input.ctx.channel) || 'telegram',
            sourceSender: contextSenderKey(input.ctx),
            sourceMessageId: safeString(input.ctx.messageId ?? input.ctx.message_id),
            threadId: resolved.thread.id,
            repoPath: resolved.thread.cwd,
            riskClass,
            confirmationState: 'not_required',
            status: 'validated',
            redactedPreview: message,
            idempotencyKey,
        });
        const writer = input.api.codexAppServerWriter || defaultCodexAppServerWriter(input.bridgeConfig);
        const appServerMethod = input.subcommand === 'steer' ? 'turn/steer' : 'turn/start';
        if (input.subcommand === 'steer' && typeof writer.steerMessage !== 'function') {
            return { text: 'Codex steering failed: the configured app-server writer does not support turn/steer.' };
        }
        const result = input.subcommand === 'steer'
            ? await writer.steerMessage({
                threadId: resolved.thread.id,
                cwd: resolved.thread.cwd,
                model: resolved.thread.model,
                message,
                timeoutMs: Math.max(input.bridgeConfig.appServerTimeoutMs, 10000),
            })
            : await writer.sendMessage({
                threadId: resolved.thread.id,
                cwd: resolved.thread.cwd,
                model: resolved.thread.model,
                message,
                timeoutMs: Math.max(input.bridgeConfig.appServerTimeoutMs, 10000),
            });
        const status = result.ok ? 'accepted' : result.possiblySent ? 'possibly_sent' : 'failed';
        store.recordOutbound({
            agentId: input.agentId,
            sourceChannel: safeString(input.ctx.channel) || 'telegram',
            sourceSender: contextSenderKey(input.ctx),
            sourceMessageId: safeString(input.ctx.messageId ?? input.ctx.message_id),
            threadId: resolved.thread.id,
            repoPath: resolved.thread.cwd,
            riskClass,
            confirmationState: 'not_required',
            appServerMethod,
            appServerTurnId: result.turnId,
            status,
            redactedPreview: message,
            error: result.error,
            idempotencyKey,
        });
        store.recordEvent({
            agentId: input.agentId,
            eventType: 'outbound_write',
            eventClass: 'outbound_write',
            source: resolved.thread.source,
            threadId: resolved.thread.id,
            decision: result.ok ? 'recorded' : 'rejected',
            notified: false,
            reason: result.ok ? (input.subcommand === 'steer' ? 'app_server_turn_steer_accepted' : 'app_server_turn_start_accepted') : (result.error || `${appServerMethod}_failed`),
            redactedSummary: `Telegram ${input.subcommand === 'steer' ? 'steer' : 'message'} sent to Codex thread ${resolved.thread.id}: ${redactText(message, 240)}`,
            dedupeKey: `outbound:${idempotencyKey}:${status}`,
        });
        if (result.ok) {
            return {
                text: input.subcommand === 'steer'
                    ? `Steered Codex thread ${resolved.thread.id}${result.activeTurnId ? ` (active turn ${result.activeTurnId})` : ''}${result.turnId ? `; resulting turn ${result.turnId}` : ''}.`
                    : `Sent to Codex thread ${resolved.thread.id}${result.turnId ? ` (turn ${result.turnId})` : ''}.`,
            };
        }
        if (result.possiblySent)
            return { text: `Codex app-server timed out after the write may have been accepted for ${resolved.thread.id}. Check /brain codex last before retrying.` };
        return { text: `Codex ${input.subcommand === 'steer' ? 'steer' : 'write'} failed for ${resolved.thread.id}: ${result.error || 'unknown error'}` };
    }
    finally {
        store.close();
    }
}
function looksLikeCodexThreadId(value) {
    return /^([0-9a-f]{8,}|thread[-_][\w.-]+|[A-Za-z0-9_-]{12,})$/.test(safeString(value));
}
function isTrustedWriteContext(ctx, config) {
    if (config.trustOpenClawAuth && ctx.isAuthorizedSender === true)
        return true;
    const trusted = config.trustedTelegramSenders;
    if (trusted.includes('*'))
        return true;
    const candidates = [contextSenderKey(ctx), contextChatKey(ctx), safeString(ctx.from), safeString(ctx.to), safeString(ctx.accountId)].filter(Boolean);
    return trusted.some((item) => candidates.includes(item));
}
function isRepoAllowedForWrite(cwd, config) {
    const allowlist = config.writeAllowlist.length ? config.writeAllowlist : config.repoAllowlist;
    if (allowlist.length === 0)
        return false;
    const workspace = path.resolve(safeString(cwd) || '/');
    return allowlist.some((entry) => {
        const allowed = path.resolve(expandPathTemplate(entry, 'main'));
        return workspace === allowed || workspace.startsWith(`${allowed}${path.sep}`);
    });
}
function classifyWriteRisk(message) {
    if (/\b(delete|rm\s+-rf|deploy|publish|release|token|secret|password|credential|full[- ]?access|yolo|trade|order|broker|prod|production)\b/i.test(message)) {
        return 'high';
    }
    if (/\b(edit|patch|fix|write|run|test|commit|push|merge|pr|pull request)\b/i.test(message))
        return 'medium';
    return 'low';
}
export class CodexBridgeStore {
    db;
    constructor(options) {
        const bridgeConfig = normalizeCodexBridgeConfig(options.config.codexBridge);
        const dbPath = expandPathTemplate(bridgeConfig.bridgeStatePath, options.agentId);
        mkdirSync(path.dirname(dbPath), { recursive: true, mode: 0o700 });
        this.db = openDatabase(dbPath).db;
        this.migrate();
    }
    close() {
        this.db.close();
    }
    createWatch(input) {
        const id = input.id || randomUUID();
        const ts = new Date().toISOString();
        this.db.prepare(`
      INSERT INTO codex_bridge_watches (
        id, agent_id, scope, thread_id, goal_key, notify_channel, notify_target, account_id,
        message_thread_id, allowed_classes_json, expires_at, status, dedupe_key_last_seen,
        last_event_at, sensitivity, verbosity, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL, NULL, ?, ?, ?, ?)
    `).run(id, input.agentId, input.scope, input.threadId ?? null, input.goalKey ?? null, input.notifyChannel, input.notifyTarget, input.accountId ?? null, input.messageThreadId ?? null, JSON.stringify(input.allowedClasses), input.expiresAt ?? null, input.sensitivity, input.verbosity, ts, ts);
        return this.getWatch(id);
    }
    getWatch(id) {
        const row = this.db.prepare('SELECT * FROM codex_bridge_watches WHERE id = ?').get(id);
        return row ? rowToWatch(row) : null;
    }
    listWatches(agentId, options = {}) {
        const rows = options.activeOnly
            ? this.db.prepare("SELECT * FROM codex_bridge_watches WHERE agent_id = ? AND status = 'active' ORDER BY created_at DESC").all(agentId)
            : this.db.prepare('SELECT * FROM codex_bridge_watches WHERE agent_id = ? ORDER BY created_at DESC').all(agentId);
        return rows.map(rowToWatch);
    }
    updateWatchEvent(id, patch) {
        const existing = this.getWatch(id);
        if (!existing)
            return null;
        this.db.prepare(`
      UPDATE codex_bridge_watches
      SET dedupe_key_last_seen = ?, last_event_at = ?, status = ?, updated_at = ?
      WHERE id = ?
    `).run(patch.dedupeKeyLastSeen ?? existing.dedupeKeyLastSeen ?? null, patch.lastEventAt ?? existing.lastEventAt ?? null, patch.status ?? existing.status, new Date().toISOString(), id);
        return this.getWatch(id);
    }
    pauseWatches(agentId, target) {
        const ts = new Date().toISOString();
        const result = this.db.prepare(`
      UPDATE codex_bridge_watches
      SET status = 'paused', updated_at = ?
      WHERE agent_id = ? AND status = 'active' AND (id = ? OR thread_id = ?)
    `).run(ts, agentId, target, target);
        return Number(result?.changes || 0);
    }
    recordEvent(input) {
        const id = input.id || randomUUID();
        const createdAt = input.createdAt || new Date().toISOString();
        const redactedSummary = redactText(clipText(input.redactedSummary || '', 800), 800);
        this.db.prepare(`
      INSERT OR IGNORE INTO codex_bridge_events (
        id, agent_id, event_type, event_class, source, thread_id, goal_key, decision,
        notified, reason, redacted_summary, dedupe_key, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(id, input.agentId, input.eventType, input.eventClass, input.source, input.threadId ?? null, input.goalKey ?? null, input.decision, input.notified ? 1 : 0, input.reason, redactedSummary, input.dedupeKey, createdAt);
        return { ...input, id, createdAt, redactedSummary };
    }
    listEvents(agentId, limit = 50) {
        const safeLimit = Math.min(500, Math.max(1, Number(limit || 50)));
        const rows = this.db.prepare('SELECT * FROM codex_bridge_events WHERE agent_id = ? ORDER BY created_at DESC LIMIT ?').all(agentId, safeLimit);
        return rows.map(rowToEvent);
    }
    bindConversation(input) {
        const id = randomUUID();
        const ts = new Date().toISOString();
        const chatKeyHash = sha256(input.chatKey);
        const senderKeyHash = sha256(input.senderKey);
        this.db.prepare(`
      INSERT INTO codex_conversation_bindings (
        id, agent_id, chat_key_hash, sender_key_hash, thread_id, title, cwd, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(agent_id, chat_key_hash) DO UPDATE SET
        sender_key_hash = excluded.sender_key_hash,
        thread_id = excluded.thread_id,
        title = excluded.title,
        cwd = excluded.cwd,
        updated_at = excluded.updated_at
    `).run(id, input.agentId, chatKeyHash, senderKeyHash, input.thread.id, input.thread.title, input.thread.cwd, ts, ts);
        return this.getBinding(input.agentId, input.chatKey);
    }
    getBinding(agentId, chatKey) {
        const row = this.db.prepare('SELECT * FROM codex_conversation_bindings WHERE agent_id = ? AND chat_key_hash = ?').get(agentId, sha256(chatKey));
        return row ? rowToBinding(row) : null;
    }
    unbindConversation(agentId, chatKey) {
        const result = this.db.prepare('DELETE FROM codex_conversation_bindings WHERE agent_id = ? AND chat_key_hash = ?').run(agentId, sha256(chatKey));
        return Number(result?.changes || 0) > 0;
    }
    getMessageCursor(watchId, threadId, rolloutPath) {
        return this.db.prepare('SELECT * FROM codex_message_cursors WHERE watch_id = ? AND thread_id = ? AND rollout_path_hash = ?').get(watchId, threadId, sha256(rolloutPath)) || null;
    }
    upsertMessageCursor(input) {
        const id = sha256(`${input.watchId}:${input.threadId}:${input.rolloutPath}`);
        const ts = new Date().toISOString();
        this.db.prepare(`
      INSERT INTO codex_message_cursors (
        id, agent_id, watch_id, thread_id, rollout_path, rollout_path_hash,
        parse_cursor_line, parse_cursor_byte_offset, delivery_cursor_line, delivery_cursor_byte_offset,
        last_message_id, last_message_hash, file_identity, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        parse_cursor_line = excluded.parse_cursor_line,
        parse_cursor_byte_offset = excluded.parse_cursor_byte_offset,
        delivery_cursor_line = excluded.delivery_cursor_line,
        delivery_cursor_byte_offset = excluded.delivery_cursor_byte_offset,
        last_message_id = excluded.last_message_id,
        last_message_hash = excluded.last_message_hash,
        file_identity = excluded.file_identity,
        updated_at = excluded.updated_at
    `).run(id, input.agentId, input.watchId, input.threadId, input.rolloutPath, sha256(input.rolloutPath), input.parseCursorLine, input.parseCursorByteOffset, input.deliveryCursorLine, input.deliveryCursorByteOffset, input.lastMessageId ?? null, input.lastMessageHash ?? null, input.fileIdentity ?? null, ts, ts);
    }
    recordPendingDelivery(input) {
        const id = sha256(`${input.watchId}:${input.message.hash}:${input.chatKey}`);
        const ts = new Date().toISOString();
        this.db.prepare(`
      INSERT INTO codex_pending_deliveries (
        id, watch_id, thread_id, message_id, message_hash, source_line, source_byte_offset,
        telegram_chat_id_hash, status, attempt_count, last_error, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        status = excluded.status,
        attempt_count = codex_pending_deliveries.attempt_count + 1,
        last_error = excluded.last_error,
        updated_at = excluded.updated_at
    `).run(id, input.watchId, input.threadId, input.message.id, input.message.hash, input.message.lineNumber, input.message.byteOffset, sha256(input.chatKey), input.status, input.error ?? null, ts, ts);
    }
    recordOutbound(input) {
        const id = input.idempotencyKey || randomUUID();
        const ts = new Date().toISOString();
        this.db.prepare(`
      INSERT INTO codex_outbound_messages (
        id, agent_id, source_channel, source_sender, source_message_id, thread_id, repo_path,
        risk_class, confirmation_state, app_server_method, app_server_turn_id, status,
        redacted_preview, error, created_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        status = excluded.status,
        app_server_turn_id = excluded.app_server_turn_id,
        error = excluded.error,
        updated_at = excluded.updated_at
    `).run(id, input.agentId, input.sourceChannel, sha256(input.sourceSender), input.sourceMessageId ?? null, input.threadId, input.repoPath ?? null, input.riskClass, input.confirmationState, input.appServerMethod ?? null, input.appServerTurnId ?? null, input.status, redactText(input.redactedPreview, 500), input.error ?? null, ts, ts);
    }
    migrate() {
        this.db.exec(`
      CREATE TABLE IF NOT EXISTS codex_bridge_watches (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        scope TEXT NOT NULL,
        thread_id TEXT,
        goal_key TEXT,
        notify_channel TEXT NOT NULL,
        notify_target TEXT NOT NULL,
        account_id TEXT,
        message_thread_id TEXT,
        allowed_classes_json TEXT NOT NULL,
        expires_at TEXT,
        status TEXT NOT NULL,
        dedupe_key_last_seen TEXT,
        last_event_at TEXT,
        sensitivity TEXT NOT NULL,
        verbosity TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_codex_bridge_watches_agent ON codex_bridge_watches(agent_id, status, created_at);
      CREATE INDEX IF NOT EXISTS idx_codex_bridge_watches_thread ON codex_bridge_watches(thread_id, status);

      CREATE TABLE IF NOT EXISTS codex_bridge_events (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        event_class TEXT NOT NULL,
        source TEXT NOT NULL,
        thread_id TEXT,
        goal_key TEXT,
        decision TEXT NOT NULL,
        notified INTEGER NOT NULL DEFAULT 0,
        reason TEXT NOT NULL,
        redacted_summary TEXT NOT NULL,
        dedupe_key TEXT NOT NULL,
        created_at TEXT NOT NULL
      );
      CREATE UNIQUE INDEX IF NOT EXISTS idx_codex_bridge_events_dedupe ON codex_bridge_events(agent_id, dedupe_key);
      CREATE INDEX IF NOT EXISTS idx_codex_bridge_events_agent ON codex_bridge_events(agent_id, created_at);

      CREATE TABLE IF NOT EXISTS codex_conversation_bindings (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        chat_key_hash TEXT NOT NULL,
        sender_key_hash TEXT NOT NULL,
        thread_id TEXT NOT NULL,
        title TEXT,
        cwd TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(agent_id, chat_key_hash)
      );
      CREATE INDEX IF NOT EXISTS idx_codex_bindings_agent ON codex_conversation_bindings(agent_id, updated_at);

      CREATE TABLE IF NOT EXISTS codex_message_cursors (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        watch_id TEXT NOT NULL,
        thread_id TEXT NOT NULL,
        rollout_path TEXT NOT NULL,
        rollout_path_hash TEXT NOT NULL,
        parse_cursor_line INTEGER NOT NULL DEFAULT 0,
        parse_cursor_byte_offset INTEGER NOT NULL DEFAULT 0,
        delivery_cursor_line INTEGER NOT NULL DEFAULT 0,
        delivery_cursor_byte_offset INTEGER NOT NULL DEFAULT 0,
        last_message_id TEXT,
        last_message_hash TEXT,
        file_identity TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_codex_message_cursors_watch ON codex_message_cursors(watch_id, thread_id);

      CREATE TABLE IF NOT EXISTS codex_pending_deliveries (
        id TEXT PRIMARY KEY,
        watch_id TEXT NOT NULL,
        thread_id TEXT NOT NULL,
        message_id TEXT,
        message_hash TEXT NOT NULL,
        source_line INTEGER,
        source_byte_offset INTEGER,
        telegram_chat_id_hash TEXT NOT NULL,
        status TEXT NOT NULL,
        attempt_count INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_codex_pending_deliveries_watch ON codex_pending_deliveries(watch_id, status, updated_at);

      CREATE TABLE IF NOT EXISTS codex_outbound_messages (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        source_channel TEXT NOT NULL,
        source_sender TEXT NOT NULL,
        source_message_id TEXT,
        thread_id TEXT NOT NULL,
        repo_path TEXT,
        risk_class TEXT NOT NULL,
        confirmation_state TEXT NOT NULL,
        app_server_method TEXT,
        app_server_turn_id TEXT,
        status TEXT NOT NULL,
        redacted_preview TEXT NOT NULL,
        error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_codex_outbound_agent ON codex_outbound_messages(agent_id, updated_at);
    `);
    }
}
function statusFromThreads(input) {
    const activeGoals = input.threads.filter((thread) => thread.goal?.status === 'active');
    return {
        ok: true,
        bridge: 'openclawbrain-codex-continuity',
        source: input.source,
        stale: input.stale,
        staleReason: input.staleReason,
        generatedAt: input.generatedAt,
        capabilities: {
            canReadThreads: true,
            canReadGoals: true,
            canReadMessages: input.threads.some((thread) => Boolean(thread.rolloutPath)),
            canSubscribe: input.source === 'app_server',
            canStartTurn: input.writeEnabled,
            canSteerTurn: input.steerEnabled,
            canWrite: input.writeEnabled,
            appServerAvailable: input.appServerAvailable,
            sqliteFallbackAvailable: input.sqliteFallbackAvailable,
        },
        counts: { threads: input.threads.length, activeGoals: activeGoals.length, watched: input.watched },
        latestThreads: input.threads,
        activeGoals,
        errors: input.errors,
        writeControl: {
            enabled: input.writeEnabled,
            reason: input.writeEnabled ? (input.steerEnabled ? 'write_and_steer_enabled_for_trusted_contexts' : 'write_enabled_steer_disabled') : 'disabled_by_default',
        },
    };
}
function emptyStatus(generatedAt, watched, reason) {
    return {
        ok: false,
        bridge: 'openclawbrain-codex-continuity',
        source: 'none',
        stale: true,
        staleReason: reason,
        generatedAt,
        capabilities: {
            canReadThreads: false,
            canReadGoals: false,
            canReadMessages: false,
            canSubscribe: false,
            canStartTurn: false,
            canSteerTurn: false,
            canWrite: false,
            appServerAvailable: false,
            sqliteFallbackAvailable: false,
        },
        counts: { threads: 0, activeGoals: 0, watched },
        latestThreads: [],
        activeGoals: [],
        errors: [],
        writeControl: { enabled: false, reason: 'disabled_by_default' },
    };
}
function normalizeAppServerThreads(response, nowMs) {
    const root = response && typeof response === 'object' ? response : {};
    const items = Array.isArray(root.threads) ? root.threads : Array.isArray(root.items) ? root.items : Array.isArray(response) ? response : [];
    return items.map((item) => {
        const goal = item.goal && typeof item.goal === 'object' ? item.goal : undefined;
        return {
            id: safeString(item.id ?? item.threadId),
            title: redactText(clipText(safeString(item.title ?? item.name ?? 'Untitled Codex thread'), 220), 220),
            cwd: safeString(item.cwd ?? item.worktree ?? item.workspace ?? ''),
            branch: safeString(item.gitBranch ?? item.git_branch ?? item.branch),
            sha: safeString(item.gitSha ?? item.git_sha ?? item.sha),
            model: safeString(item.model),
            reasoningEffort: safeString(item.reasoningEffort ?? item.reasoning_effort),
            rolloutPath: safeString(item.rolloutPath ?? item.rollout_path),
            firstUserMessage: redactText(clipText(safeString(item.firstUserMessage ?? item.first_user_message), 500), 500),
            updatedAtMs: normalizeMs(item.updatedAtMs ?? item.updated_at_ms ?? item.updatedAt ?? item.updated_at, nowMs),
            archived: Boolean(item.archived),
            goal: goal ? {
                goalId: safeString(goal.goalId ?? goal.goal_id),
                objective: redactText(clipText(safeString(goal.objective), 500), 500),
                status: safeString(goal.status),
                tokenBudget: optionalNumber(goal.tokenBudget ?? goal.token_budget),
                tokensUsed: Number(goal.tokensUsed ?? goal.tokens_used ?? 0),
                timeUsedSeconds: Number(goal.timeUsedSeconds ?? goal.time_used_seconds ?? 0),
                updatedAtMs: normalizeMs(goal.updatedAtMs ?? goal.updated_at_ms, nowMs),
            } : undefined,
            source: 'app_server',
        };
    }).filter((thread) => thread.id);
}
function rowToThread(row, source) {
    const updatedAtMs = normalizeMs(row.updated_at_ms ?? row.updated_at, Date.now());
    const goal = row.objective ? {
        goalId: safeString(row.goal_id),
        objective: redactText(clipText(safeString(row.objective), 500), 500),
        status: safeString(row.goal_status || 'unknown'),
        tokenBudget: optionalNumber(row.token_budget),
        tokensUsed: Number(row.tokens_used ?? 0),
        timeUsedSeconds: Number(row.time_used_seconds ?? 0),
        updatedAtMs: normalizeMs(row.goal_updated_at_ms, updatedAtMs),
    } : undefined;
    return {
        id: safeString(row.id),
        title: redactText(clipText(safeString(row.title || 'Untitled Codex thread'), 220), 220),
        cwd: safeString(row.cwd),
        branch: safeString(row.git_branch),
        sha: safeString(row.git_sha),
        model: safeString(row.model),
        reasoningEffort: safeString(row.reasoning_effort),
        rolloutPath: safeString(row.rollout_path),
        firstUserMessage: redactText(clipText(safeString(row.first_user_message), 500), 500),
        updatedAtMs,
        archived: Number(row.archived || 0) === 1,
        goal,
        source,
    };
}
function classifyThreadForWatch(thread) {
    const status = thread?.goal?.status;
    if (!thread || !status || status === 'active')
        return 'quiet';
    if (status === 'complete')
        return 'completion';
    if (status === 'budget_limited' || status === 'paused')
        return 'blocker';
    if (/fail|error/i.test(status))
        return 'failure';
    return 'quiet';
}
function watchDedupeKey(watch, thread, eventClass) {
    return sha256(`${watch.id}:${thread.id}:${thread.goal?.status || 'none'}:${thread.goal?.updatedAtMs || thread.updatedAtMs}:${eventClass}`);
}
function formatWatchNotification(thread, eventClass, delayed) {
    const prefix = delayed ? 'Delayed Codex update' : 'Codex update';
    const goal = thread.goal;
    return [
        `${prefix}: ${eventClass.replace(/_/g, ' ')}`,
        `Thread: ${thread.id}`,
        `Title: ${thread.title}`,
        goal ? `Goal: ${goal.status} - ${goal.objective}` : 'Goal: none visible',
        `Workspace: ${thread.cwd || 'unknown'}`,
    ].join('\n');
}
function formatForwardedTranscriptMessage(thread, message, config) {
    if (config.telegramForwardingMode === 'metadata_only') {
        return [
            'Codex message:',
            `Thread: ${thread.id}`,
            `Title: ${thread.title}`,
            `Role: ${message.role}`,
            `Time: ${message.timestamp}`,
            'Content hidden by metadata_only forwarding mode.',
        ].join('\n');
    }
    const body = config.telegramForwardingMode === 'redacted' ? redactText(message.text, 8000) : message.text;
    return [
        `Codex ${message.role} message`,
        `Thread: ${thread.title} [${thread.id}]`,
        '',
        body,
    ].join('\n');
}
async function sendBridgeNotification(api, watch, text) {
    if (!watch.notifyChannel || !watch.notifyTarget)
        return { ok: false, reason: 'missing_notify_route' };
    try {
        const adapter = await api.runtime?.channel?.outbound?.loadAdapter?.(watch.notifyChannel);
        const send = adapter?.sendText;
        if (typeof send !== 'function')
            return { ok: false, reason: 'outbound_adapter_unavailable' };
        const chunks = chunkTelegramText(text);
        for (const chunk of chunks) {
            await send({
                cfg: api.runtime?.config?.current?.(),
                to: watch.notifyTarget,
                text: chunk,
                accountId: watch.accountId,
                threadId: watch.messageThreadId,
            });
        }
        return { ok: true, reason: 'sent' };
    }
    catch (error) {
        return { ok: false, reason: `telegram_send_failed:${clipText(String(error?.message || error), 120)}` };
    }
}
function chunkTelegramText(text, maxChars = 3500) {
    const clean = safeString(text);
    if (clean.length <= maxChars)
        return [clean];
    const chunks = [];
    for (let offset = 0; offset < clean.length; offset += maxChars) {
        chunks.push(clean.slice(offset, offset + maxChars));
    }
    return chunks.map((chunk, index) => `[${index + 1}/${chunks.length}]\n${chunk}`);
}
function defaultCodexAppServerWriter(config) {
    return {
        async sendMessage(input) {
            const client = await createJsonRpcWebSocketClient({
                url: config.appServerUrl,
                timeoutMs: input.timeoutMs || config.appServerTimeoutMs,
            });
            try {
                await client.request('initialize', {
                    clientInfo: { name: 'openclawbrain-codex-telegram-bridge', version: '1' },
                    capabilities: { experimentalApi: true },
                });
                client.notify('initialized');
                await client.request('thread/resume', {
                    threadId: input.threadId,
                    persistExtendedHistory: true,
                    ...(input.model ? { model: input.model } : {}),
                });
                const response = await client.request('turn/start', {
                    threadId: input.threadId,
                    input: [{ type: 'text', text: input.message, text_elements: [] }],
                    ...(input.cwd ? { cwd: input.cwd } : {}),
                    ...(input.model ? { model: input.model } : {}),
                });
                const turn = response && typeof response === 'object' && !Array.isArray(response) ? response.turn : undefined;
                return { ok: true, turnId: safeString(turn?.id ?? response?.turnId), status: safeString(turn?.status) };
            }
            catch (error) {
                const message = clipText(String(error?.message || error), 240);
                return { ok: false, possiblySent: /timeout/i.test(message), error: message };
            }
            finally {
                client.close();
            }
        },
        async steerMessage(input) {
            const client = await createJsonRpcWebSocketClient({
                url: config.appServerUrl,
                timeoutMs: input.timeoutMs || config.appServerTimeoutMs,
            });
            try {
                await client.request('initialize', {
                    clientInfo: { name: 'openclawbrain-codex-telegram-bridge', version: '1' },
                    capabilities: { experimentalApi: true },
                });
                client.notify('initialized');
                const resumed = await client.request('thread/resume', {
                    threadId: input.threadId,
                    persistExtendedHistory: true,
                    ...(input.model ? { model: input.model } : {}),
                });
                let activeTurnId = input.expectedTurnId || findActiveCodexTurnId(resumed);
                if (!activeTurnId) {
                    try {
                        const read = await client.request('thread/read', { threadId: input.threadId, includeTurns: true });
                        activeTurnId = findActiveCodexTurnId(read);
                    }
                    catch {
                        // If thread/read is absent, fall through to the clearer no-active-turn error.
                    }
                }
                if (!activeTurnId) {
                    return { ok: false, error: 'no active in-progress Codex turn found to steer; use /brain codex reply for an idle thread' };
                }
                const response = await client.request('turn/steer', {
                    threadId: input.threadId,
                    expectedTurnId: activeTurnId,
                    input: [{ type: 'text', text: input.message, text_elements: [] }],
                });
                return {
                    ok: true,
                    activeTurnId,
                    turnId: safeString(response?.turnId ?? response?.turn?.id),
                    status: safeString(response?.turn?.status),
                };
            }
            catch (error) {
                const message = clipText(String(error?.message || error), 240);
                return { ok: false, possiblySent: /timeout/i.test(message), error: message };
            }
            finally {
                client.close();
            }
        },
    };
}
function findActiveCodexTurnId(response) {
    const root = response && typeof response === 'object' ? response : {};
    const thread = root.thread && typeof root.thread === 'object' ? root.thread : root;
    const turns = Array.isArray(thread.turns) ? thread.turns : Array.isArray(root.turns) ? root.turns : [];
    for (let index = turns.length - 1; index >= 0; index -= 1) {
        const turn = turns[index];
        if (!turn || typeof turn !== 'object')
            continue;
        const status = safeString(turn.status);
        if (status === 'inProgress')
            return safeString(turn.id ?? turn.turnId);
    }
    return '';
}
async function createJsonRpcWebSocketClient(options) {
    const url = safeString(options.url);
    if (!/^wss?:\/\/(127\.0\.0\.1|localhost)(:\d+)?(\/.*)?$/i.test(url)) {
        throw new Error('codex app-server WebSocket URL is not configured; set codexBridge.appServerUrl to a localhost ws:// endpoint');
    }
    const WebSocketCtor = globalThis.WebSocket;
    if (typeof WebSocketCtor !== 'function') {
        throw new Error('WebSocket is unavailable in this Node runtime');
    }
    const socket = new WebSocketCtor(url);
    let nextId = 1;
    const pending = new Map();
    await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('codex app-server WebSocket connection timeout')), options.timeoutMs);
        socket.onopen = () => {
            clearTimeout(timer);
            resolve();
        };
        socket.onerror = () => {
            clearTimeout(timer);
            reject(new Error('codex app-server WebSocket connection failed'));
        };
    });
    socket.onmessage = (event) => {
        let message;
        try {
            message = JSON.parse(String(event.data));
        }
        catch {
            return;
        }
        const id = Number(message.id);
        const current = pending.get(id);
        if (!current)
            return;
        clearTimeout(current.timer);
        pending.delete(id);
        if (message.error) {
            current.reject(new Error(safeString(message.error.message) || 'codex app-server rpc error'));
        }
        else {
            current.resolve(message.result);
        }
    };
    socket.onerror = () => {
        for (const [id, current] of pending) {
            clearTimeout(current.timer);
            current.reject(new Error('codex app-server WebSocket error'));
            pending.delete(id);
        }
    };
    socket.onclose = () => {
        for (const [id, current] of pending) {
            clearTimeout(current.timer);
            current.reject(new Error('codex app-server WebSocket closed'));
            pending.delete(id);
        }
    };
    return {
        request(method, params) {
            const id = nextId++;
            const payload = JSON.stringify({ id, method, params });
            return new Promise((resolve, reject) => {
                const timer = setTimeout(() => {
                    pending.delete(id);
                    reject(new Error(`codex app-server request timeout for ${method}`));
                }, options.timeoutMs);
                pending.set(id, { resolve, reject, timer });
                try {
                    socket.send(payload);
                }
                catch (error) {
                    clearTimeout(timer);
                    pending.delete(id);
                    reject(error);
                }
            });
        },
        notify(method, params) {
            socket.send(JSON.stringify({ method, params }));
        },
        close() {
            try {
                socket.close();
            }
            catch { /* ignore */ }
            for (const [id, current] of pending) {
                clearTimeout(current.timer);
                current.reject(new Error('codex app-server client closed'));
                pending.delete(id);
            }
        },
    };
}
export function formatCodexStatus(status) {
    if (!status.ok)
        return `Codex continuity: unavailable (${status.staleReason || 'unknown'}).`;
    const top = status.activeGoals[0] ?? status.latestThreads[0];
    return [
        `Codex continuity: ${status.source}${status.stale ? ' (read-only/stale-labeled)' : ''}`,
        `Threads visible: ${status.counts.threads}; active goals: ${status.counts.activeGoals}; watches: ${status.counts.watched}`,
        top ? `Latest: ${top.title} [${top.id}]` : 'Latest: none',
        top?.goal ? `Goal: ${top.goal.status} - ${top.goal.objective}` : undefined,
        `Writes: ${status.capabilities.canStartTurn ? 'on for trusted contexts' : 'disabled'}`,
        `Steer: ${status.capabilities.canSteerTurn ? 'on for active turns' : 'disabled'}`,
    ].filter(Boolean).join('\n');
}
export function formatCodexThreads(status, filter = '') {
    const lower = filter.trim().toLowerCase();
    const threads = status.latestThreads.filter((thread) => !lower || `${thread.title} ${thread.cwd} ${thread.goal?.objective || ''}`.toLowerCase().includes(lower));
    if (threads.length === 0)
        return 'No Codex threads matched.';
    return [
        `Codex threads (${status.source}${status.stale ? ', read-only/stale-labeled' : ''}):`,
        ...threads.slice(0, 10).map((thread) => {
            const goal = thread.goal ? ` goal=${thread.goal.status}` : '';
            const branch = thread.branch ? ` branch=${thread.branch}` : '';
            return `- ${thread.id}${goal}${branch}: ${thread.title}`;
        }),
    ].join('\n');
}
function formatWatches(watches) {
    if (watches.length === 0)
        return 'No Codex watches yet.';
    return [
        'Codex watches:',
        ...watches.slice(0, 20).map((watch) => `- ${watch.id} status=${watch.status} thread=${watch.threadId || 'none'} classes=${watch.allowedClasses.join(',')}`),
    ].join('\n');
}
export function formatHandoffBrief(brief) {
    return [
        brief.ok ? `Codex handoff: ${brief.title || brief.threadId}` : 'Codex handoff unavailable',
        '',
        'Observed facts:',
        ...brief.observedFacts.map((item) => `- ${item}`),
        '',
        'Codex-reported claims:',
        ...(brief.codexReportedClaims.length ? brief.codexReportedClaims : ['None visible.']).map((item) => `- ${item}`),
        '',
        'Evidence:',
        ...brief.evidence.map((item) => `- ${item}`),
        '',
        'Next actions:',
        ...brief.nextActions.map((item) => `- ${item}`),
    ].join('\n');
}
function brainHelpText() {
    return [
        'OpenClawBrain commands:',
        '- /brain codex status',
        '- /brain codex threads [filter]',
        '- /brain codex messages [thread-id|--latest|--bound] [--limit N] [--role assistant|user|all]',
        '- /brain codex last [thread-id|--latest|--bound]',
        '- /brain codex bind <thread-id>',
        '- /brain codex binding',
        '- /brain codex unbind',
        '- /brain codex tail [thread-id|--latest|--bound]',
        '- /brain codex watch [thread-id|--latest|--bound] [--messages]',
        '- /brain codex watches',
        '- /brain codex unwatch <watch-id|thread-id>',
        '- /brain codex reply <message>',
        '- /brain codex send <thread-id|--bound> <message>',
        '- /brain codex steer [thread-id|--bound] <message>',
        '- /brain codex handoff [thread-id]',
        '',
        graphHelpText(),
    ].join('\n');
}
function splitArgs(input) {
    return safeString(input).trim().split(/\s+/).filter(Boolean).map((item) => item.toLowerCase() === '/brain' ? 'help' : item);
}
function rowToWatch(row) {
    return {
        id: safeString(row.id),
        agentId: safeString(row.agent_id),
        scope: safeString(row.scope),
        threadId: safeString(row.thread_id) || undefined,
        goalKey: safeString(row.goal_key) || undefined,
        notifyChannel: safeString(row.notify_channel),
        notifyTarget: safeString(row.notify_target),
        accountId: safeString(row.account_id) || undefined,
        messageThreadId: safeString(row.message_thread_id) || undefined,
        allowedClasses: safeJsonArray(row.allowed_classes_json).filter(isEventClass),
        expiresAt: safeString(row.expires_at) || undefined,
        status: safeString(row.status),
        dedupeKeyLastSeen: safeString(row.dedupe_key_last_seen) || undefined,
        lastEventAt: safeString(row.last_event_at) || undefined,
        sensitivity: safeString(row.sensitivity),
        verbosity: safeString(row.verbosity),
        createdAt: safeString(row.created_at),
        updatedAt: safeString(row.updated_at),
    };
}
function rowToEvent(row) {
    return {
        id: safeString(row.id),
        agentId: safeString(row.agent_id),
        eventType: safeString(row.event_type),
        eventClass: safeString(row.event_class),
        source: safeString(row.source),
        threadId: safeString(row.thread_id) || undefined,
        goalKey: safeString(row.goal_key) || undefined,
        decision: safeString(row.decision),
        notified: Number(row.notified || 0) === 1,
        reason: safeString(row.reason),
        redactedSummary: safeString(row.redacted_summary),
        dedupeKey: safeString(row.dedupe_key),
        createdAt: safeString(row.created_at),
    };
}
function rowToBinding(row) {
    return {
        id: safeString(row.id),
        agentId: safeString(row.agent_id),
        chatKeyHash: safeString(row.chat_key_hash),
        senderKeyHash: safeString(row.sender_key_hash),
        threadId: safeString(row.thread_id),
        title: safeString(row.title) || undefined,
        cwd: safeString(row.cwd) || undefined,
        createdAt: safeString(row.created_at),
        updatedAt: safeString(row.updated_at),
    };
}
function openReadOnlyBetterSqlite(filename) {
    return openDatabase(filename, { readonly: true, fileMustExist: true }).db;
}
function tableColumns(db, table) {
    try {
        const rows = db.prepare(`PRAGMA table_info(${table})`).all();
        return new Set(rows.map((row) => safeString(row.name)).filter(Boolean));
    }
    catch {
        return new Set();
    }
}
function expandPathTemplate(template, agentId) {
    let expanded = safeString(template).replace(/\$\{agentId\}/g, agentId);
    if (expanded.startsWith('~/'))
        expanded = path.join(os.homedir(), expanded.slice(2));
    return path.resolve(expanded);
}
function fileIdentity(filename) {
    const resolved = filename ? expandPathTemplate(filename, 'main') : '';
    if (!resolved)
        return '';
    try {
        const stats = statSync(resolved);
        return `${stats.dev}:${stats.ino}:${stats.size}:${stats.mtimeMs}`;
    }
    catch {
        return '';
    }
}
function normalizeMs(value, fallback) {
    const numeric = Number(value || 0);
    if (!Number.isFinite(numeric) || numeric <= 0)
        return fallback;
    return numeric < 10_000_000_000 ? numeric * 1000 : numeric;
}
function optionalNumber(value) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : undefined;
}
function safeJsonArray(value) {
    try {
        const parsed = JSON.parse(value || '[]');
        return Array.isArray(parsed) ? parsed.map((item) => safeString(item)).filter(Boolean) : [];
    }
    catch {
        return [];
    }
}
function isEventClass(value) {
    return [
        'completion',
        'failure',
        'blocker',
        'approval_required',
        'auth_failure',
        'assistant_message',
        'user_message',
        'turn_started',
        'turn_completed',
        'status_snapshot',
        'watch_created',
        'binding_created',
        'binding_removed',
        'outbound_write',
        'delivery_failed',
        'handoff',
        'quiet',
    ].includes(value);
}
function nonEmptyString(value) {
    const text = safeString(value).trim();
    return text || undefined;
}
function clampInteger(value, fallback, min, max) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric))
        return fallback;
    return Math.min(max, Math.max(min, Math.trunc(numeric)));
}
function sha256(value) {
    return createHash('sha256').update(value).digest('hex');
}
