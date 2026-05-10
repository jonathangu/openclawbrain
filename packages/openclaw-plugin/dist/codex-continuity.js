import { spawn } from 'node:child_process';
import { randomUUID, createHash } from 'node:crypto';
import { existsSync, mkdirSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createInterface } from 'node:readline';
import { createRequire } from 'node:module';
import { openDatabase } from './sqlite-driver.js';
import { clipText, redactText, safeString } from './redact.js';
const require = createRequire(import.meta.url);
export const DEFAULT_CODEX_BRIDGE_CONFIG = Object.freeze({
    enabled: true,
    statePaths: Object.freeze(['~/.codex/state_5.sqlite']),
    bridgeStatePath: '~/.openclawbrain/activation/${agentId}/codex-continuity.sqlite',
    preferAppServer: true,
    appServerCommand: 'codex',
    appServerArgs: Object.freeze(['app-server', 'proxy']),
    appServerTimeoutMs: 1200,
    staleAfterMs: 10 * 60 * 1000,
    maxThreads: 10,
    watchPollIntervalMs: 60 * 1000,
    enableTelegramWrites: false,
    trustedTelegramSenders: Object.freeze([]),
    repoAllowlist: Object.freeze([]),
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
        preferAppServer: input.preferAppServer !== false,
        appServerCommand: nonEmptyString(input.appServerCommand) || DEFAULT_CODEX_BRIDGE_CONFIG.appServerCommand,
        appServerArgs: appServerArgs.length ? appServerArgs : [...DEFAULT_CODEX_BRIDGE_CONFIG.appServerArgs],
        appServerTimeoutMs: clampInteger(input.appServerTimeoutMs, DEFAULT_CODEX_BRIDGE_CONFIG.appServerTimeoutMs, 100, 30000),
        staleAfterMs: clampInteger(input.staleAfterMs, DEFAULT_CODEX_BRIDGE_CONFIG.staleAfterMs, 1000, 86400000),
        maxThreads: clampInteger(input.maxThreads, DEFAULT_CODEX_BRIDGE_CONFIG.maxThreads, 1, 100),
        watchPollIntervalMs: clampInteger(input.watchPollIntervalMs, DEFAULT_CODEX_BRIDGE_CONFIG.watchPollIntervalMs, 5000, 86400000),
        enableTelegramWrites: input.enableTelegramWrites === true,
        trustedTelegramSenders: Array.isArray(input.trustedTelegramSenders)
            ? input.trustedTelegramSenders.map((value) => safeString(value)).filter(Boolean)
            : [],
        repoAllowlist: Array.isArray(input.repoAllowlist)
            ? input.repoAllowlist.map((value) => safeString(value)).filter(Boolean)
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
    if (bridgeConfig.preferAppServer) {
        try {
            const reader = deps.appServerReader ?? new CodexProxyAppServerReader(bridgeConfig);
            const response = await reader.listThreads({ limit: bridgeConfig.maxThreads, timeoutMs: bridgeConfig.appServerTimeoutMs });
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
                });
            }
            appServerAvailable = true;
        }
        catch (error) {
            errors.push(`app_server_unavailable:${clipText(String(error?.message || error), 160)}`);
        }
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
        });
    }
    return {
        ...emptyStatus(generatedAt, watched, 'no_codex_state_available'),
        errors,
        capabilities: {
            canReadThreads: false,
            canReadGoals: false,
            canSubscribe: false,
            canStartTurn: false,
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
    if (namespace !== 'codex')
        return { text: brainHelpText() };
    const agentId = safeString(ctx.agentId ?? config.scopes?.agents?.[0] ?? 'main') || 'main';
    const bridgeConfig = normalizeCodexBridgeConfig(config.codexBridge);
    const status = await buildCodexBridgeStatus(config, agentId);
    if (subcommand === 'status')
        return { text: formatCodexStatus(status) };
    if (subcommand === 'threads')
        return { text: formatCodexThreads(status, rest.join(' ')) };
    if (subcommand === 'handoff') {
        const brief = buildCodexHandoff(status, rest[0]);
        return { text: formatHandoffBrief(brief) };
    }
    if (subcommand === 'watch') {
        const threadId = rest[0] && rest[0] !== '--latest' ? rest[0] : status.activeGoals[0]?.id ?? status.latestThreads[0]?.id;
        if (!threadId)
            return { text: 'No Codex thread is visible to watch yet.' };
        const store = new CodexBridgeStore({ config, agentId });
        const watch = store.createWatch({
            agentId,
            scope: 'thread',
            threadId,
            notifyChannel: bridgeConfig.notifyChannel || safeString(ctx.channel) || 'telegram',
            notifyTarget: bridgeConfig.notifyTarget || safeString(ctx.to ?? ctx.from ?? ''),
            accountId: safeString(ctx.accountId),
            messageThreadId: ctx.messageThreadId == null ? undefined : String(ctx.messageThreadId),
            allowedClasses: ['completion', 'failure', 'blocker', 'approval_required', 'auth_failure'],
            sensitivity: 'normal',
            verbosity: 'blockers_and_completion',
        });
        store.recordEvent({
            agentId,
            eventType: 'watch_created',
            eventClass: 'watch_created',
            source: status.source,
            threadId,
            goalKey: undefined,
            decision: 'recorded',
            notified: false,
            reason: 'explicit_user_watch_request',
            redactedSummary: `Watch created for Codex thread ${threadId}`,
            dedupeKey: `watch:${watch.id}`,
        });
        store.close();
        return { text: `Watching Codex thread ${threadId}. I will stay quiet unless it completes, fails, blocks, needs approval, or hits auth trouble.` };
    }
    if (subcommand === 'goal' || subcommand === 'steer') {
        if (!bridgeConfig.enableTelegramWrites) {
            return {
                text: 'Telegram-to-Codex writes are disabled. Set codexBridge.enableTelegramWrites=true plus trusted senders and repo allowlists before /brain codex goal or steer can run.',
            };
        }
        return { text: 'Telegram-to-Codex write control is feature-flagged but not enabled in this OpenClawBrain build path.' };
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
    recordEvent(input) {
        const id = input.id || randomUUID();
        const createdAt = input.createdAt || new Date().toISOString();
        const redactedSummary = redactText(clipText(input.redactedSummary || '', 800), 800);
        this.db.prepare(`
      INSERT INTO codex_bridge_events (
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
    `);
    }
}
class CodexProxyAppServerReader {
    config;
    constructor(config) {
        this.config = config;
    }
    async listThreads(options) {
        const client = new OneShotJsonRpcProxy({
            command: this.config.appServerCommand,
            args: this.config.appServerArgs,
            timeoutMs: options.timeoutMs,
        });
        try {
            await client.request('initialize', {
                clientInfo: { name: 'openclawbrain', title: 'OpenClawBrain Codex Continuity', version: '0.2' },
                capabilities: { experimentalApi: true },
            });
            return await client.request('thread/list', {
                limit: options.limit,
                ...(options.searchTerm ? { searchTerm: options.searchTerm } : {}),
            });
        }
        finally {
            client.close();
        }
    }
}
class OneShotJsonRpcProxy {
    child;
    timeoutMs;
    nextId = 1;
    constructor(options) {
        this.child = spawn(options.command, options.args, { stdio: ['pipe', 'pipe', 'pipe'] });
        this.timeoutMs = options.timeoutMs;
    }
    async request(method, params) {
        const id = this.nextId++;
        const lineReader = createInterface({ input: this.child.stdout });
        const timeout = setTimeout(() => {
            this.child.kill();
        }, this.timeoutMs);
        timeout.unref?.();
        try {
            this.child.stdin.write(`${JSON.stringify({ id, method, params })}\n`);
            for await (const line of lineReader) {
                if (!line.trim())
                    continue;
                const parsed = JSON.parse(line);
                if (parsed.id !== id)
                    continue;
                if (parsed.error)
                    throw new Error(parsed.error.message || `${method} failed`);
                return parsed.result;
            }
            throw new Error(`${method} returned no response`);
        }
        finally {
            clearTimeout(timeout);
            lineReader.close();
        }
    }
    close() {
        try {
            this.child.kill();
        }
        catch { /* ignore */ }
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
            canSubscribe: input.source === 'app_server',
            canStartTurn: false,
            canWrite: false,
            appServerAvailable: input.appServerAvailable,
            sqliteFallbackAvailable: input.sqliteFallbackAvailable,
        },
        counts: { threads: input.threads.length, activeGoals: activeGoals.length, watched: input.watched },
        latestThreads: input.threads,
        activeGoals,
        errors: input.errors,
        writeControl: {
            enabled: input.writeEnabled,
            reason: input.writeEnabled ? 'feature_flag_enabled_but_write_path_requires_confirmation' : 'disabled_by_default',
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
            canSubscribe: false,
            canStartTurn: false,
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
async function sendBridgeNotification(api, watch, text) {
    if (!watch.notifyChannel || !watch.notifyTarget)
        return { ok: false, reason: 'missing_notify_route' };
    try {
        const adapter = await api.runtime?.channel?.outbound?.loadAdapter?.(watch.notifyChannel);
        const send = adapter?.sendText;
        if (typeof send !== 'function')
            return { ok: false, reason: 'outbound_adapter_unavailable' };
        await send({
            cfg: api.runtime?.config?.current?.(),
            to: watch.notifyTarget,
            text,
            accountId: watch.accountId,
            threadId: watch.messageThreadId,
        });
        return { ok: true, reason: 'sent' };
    }
    catch (error) {
        return { ok: false, reason: `telegram_send_failed:${clipText(String(error?.message || error), 120)}` };
    }
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
        `Writes: ${status.writeControl.enabled ? 'feature-flagged' : 'disabled'}`,
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
        '- /brain codex watch [thread-id|--latest]',
        '- /brain codex handoff [thread-id]',
        '- /brain codex goal ... (disabled by default)',
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
function openReadOnlyBetterSqlite(filename) {
    const mod = require('better-sqlite3');
    const BetterSqlite3 = mod.default || mod;
    return new BetterSqlite3(filename, { readonly: true, fileMustExist: true });
}
function expandPathTemplate(template, agentId) {
    let expanded = safeString(template).replace(/\$\{agentId\}/g, agentId);
    if (expanded.startsWith('~/'))
        expanded = path.join(os.homedir(), expanded.slice(2));
    return path.resolve(expanded);
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
    return ['completion', 'failure', 'blocker', 'approval_required', 'auth_failure', 'status_snapshot', 'watch_created', 'handoff', 'quiet'].includes(value);
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
