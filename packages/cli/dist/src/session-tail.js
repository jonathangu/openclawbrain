import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import path from "node:path";
import { CONTRACT_IDS, buildNormalizedEventExport, createInteractionEvent, sortNormalizedEvents } from "@openclawbrain/contracts";
import { scanSession } from "@openclawbrain/events";
import { extractFeedbackEventsFromInteractionRecords } from "@openclawbrain/event-export";
import { hashOpenClawStableJson, summarizeOpenClawSessionFile, summarizeOpenClawSessionIndex, discoverOpenClawSessionStores, loadOpenClawSessionIndex, readOpenClawSessionFile } from "./session-store.js";
import { inspectOpenClawHome } from "./openclaw-home-layout.js";
const DEFAULT_SCANNER_ID = "openclaw-local-session-tail";
const DEFAULT_SCANNER_LANE = "local_session_tail";
function normalizeIsoTimestamp(value, fallback) {
    if (typeof value === "string" && !Number.isNaN(Date.parse(value))) {
        return new Date(value).toISOString();
    }
    return fallback;
}
function normalizePositiveInteger(value, fieldName, fallback) {
    const resolved = value ?? fallback;
    if (!Number.isInteger(resolved) || resolved <= 0) {
        throw new Error(`${fieldName} must be a positive integer`);
    }
    return resolved;
}
function normalizeNonNegativeDurationMs(value, fieldName, fallback) {
    const resolved = value ?? fallback;
    if (!Number.isInteger(resolved) || resolved < 0) {
        throw new Error(`${fieldName} must be a non-negative integer`);
    }
    return resolved;
}
function normalizeText(value) {
    if (typeof value !== "string") {
        return null;
    }
    const trimmed = value.trim();
    return trimmed.length === 0 ? null : trimmed;
}
function sanitizeToken(value) {
    const normalized = value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
    return normalized.length > 0 ? normalized : "unknown";
}
function deterministicEventId(value) {
    const digest = createHash("sha256").update(JSON.stringify(value)).digest("hex");
    return `evt-${digest.slice(0, 16)}`;
}
function delay(ms) {
    return new Promise((resolve) => {
        setTimeout(resolve, ms);
    });
}
function cursorKey(sourceIndexPath, sessionKey) {
    return `${sourceIndexPath}::${sessionKey}`;
}
function cloneCursor(cursor) {
    return {
        sourceIndexPath: cursor.sourceIndexPath,
        sessionKey: cursor.sessionKey,
        sessionId: cursor.sessionId,
        sessionFile: cursor.sessionFile,
        updatedAt: cursor.updatedAt,
        rawRecordCount: cursor.rawRecordCount,
        bridgedEventCount: cursor.bridgedEventCount
    };
}
function snapshotCursor(cursorBySession) {
    return [...cursorBySession.values()]
        .map((cursor) => cloneCursor(cursor))
        .sort((left, right) => {
        if (left.sourceIndexPath !== right.sourceIndexPath) {
            return left.sourceIndexPath.localeCompare(right.sourceIndexPath);
        }
        return left.sessionKey.localeCompare(right.sessionKey);
    });
}
function validateCursorEntry(cursor, index) {
    const label = `cursor[${index}]`;
    if (typeof cursor.sourceIndexPath !== "string" || cursor.sourceIndexPath.trim().length === 0) {
        throw new Error(`${label}.sourceIndexPath must be a non-empty string`);
    }
    if (typeof cursor.sessionKey !== "string" || cursor.sessionKey.trim().length === 0) {
        throw new Error(`${label}.sessionKey must be a non-empty string`);
    }
    if (typeof cursor.sessionId !== "string" || cursor.sessionId.trim().length === 0) {
        throw new Error(`${label}.sessionId must be a non-empty string`);
    }
    if (cursor.sessionFile !== null && (typeof cursor.sessionFile !== "string" || cursor.sessionFile.trim().length === 0)) {
        throw new Error(`${label}.sessionFile must be null or a non-empty string`);
    }
    if (!Number.isFinite(cursor.updatedAt)) {
        throw new Error(`${label}.updatedAt must be a finite number`);
    }
    if (!Number.isInteger(cursor.rawRecordCount) || cursor.rawRecordCount < 0) {
        throw new Error(`${label}.rawRecordCount must be a non-negative integer`);
    }
    if (!Number.isInteger(cursor.bridgedEventCount) || cursor.bridgedEventCount < 0) {
        throw new Error(`${label}.bridgedEventCount must be a non-negative integer`);
    }
}
const profileIdCache = new Map();
function buildProfileId(profileRoot) {
    const resolvedRoot = path.resolve(profileRoot);
    const cached = profileIdCache.get(resolvedRoot);
    if (cached !== undefined) {
        return cached;
    }
    const inspection = inspectOpenClawHome(resolvedRoot);
    const profileId = inspection.profileId ??
        (inspection.layout === "shared_home_profiles_in_config" || inspection.layout === "single_openclaw_home"
            ? "current_profile"
            : path.basename(resolvedRoot));
    profileIdCache.set(resolvedRoot, profileId);
    return profileId;
}
function deriveSessionChannel(entry) {
    const directValue = typeof entry.chatType === "string" && entry.chatType.trim().length > 0
        ? entry.chatType.trim()
        : typeof entry.deliveryContext?.chatType === "string" && entry.deliveryContext.chatType.trim().length > 0
            ? entry.deliveryContext.chatType.trim()
            : "unknown";
    return directValue;
}
function buildSourceStream(source, entry) {
    const profileId = sanitizeToken(buildProfileId(source.profileRoot));
    const channel = sanitizeToken(deriveSessionChannel(entry));
    return `openclaw/session-store/${profileId}/${source.agentId}/${channel}`;
}
function sessionContentHasToolCalls(content) {
    return content.some((part) => part.type === "toolCall");
}
function buildScannerRawMessage(record) {
    const role = record.message.role === "toolResult" ? "tool" : record.message.role;
    const toolCalls = record.message.content
        .filter((part) => part.type === "toolCall")
        .map((part) => ({
        id: part.id,
        name: part.name,
        arguments: part.arguments
    }));
    return {
        id: record.id,
        messageId: record.id,
        parentId: record.parentId,
        createdAt: record.timestamp,
        role,
        type: role === "tool" ? "tool_result" : "message",
        content: record.message.content,
        ...(sessionContentHasToolCalls(record.message.content) ? { toolCalls } : {}),
        ...(record.message.toolCallId === undefined ? {} : { toolCallId: record.message.toolCallId }),
        ...(record.message.toolName === undefined ? {} : { toolName: record.message.toolName }),
        ...(record.message.details === undefined ? {} : { details: record.message.details }),
        ...(record.message.provider === undefined ? {} : { provider: record.message.provider }),
        ...(record.message.model === undefined ? {} : { model: record.message.model }),
        ...(record.message.api === undefined ? {} : { api: record.message.api }),
        ...(record.message.stopReason === undefined ? {} : { stopReason: record.message.stopReason })
    };
}
function buildScannerRawSession(input) {
    const header = input.records.find((record) => record.type === "session");
    const messageRecords = input.records.filter((record) => record.type === "message");
    return {
        id: input.entry.sessionId,
        sessionId: input.entry.sessionId,
        sessionKey: input.sessionKey,
        profileId: buildProfileId(input.source.profileRoot),
        createdAt: header?.timestamp ?? new Date(input.entry.updatedAt).toISOString(),
        updatedAt: new Date(input.entry.updatedAt).toISOString(),
        channel: deriveSessionChannel(input.entry),
        agentId: input.source.agentId,
        sourceStream: buildSourceStream(input.source, input.entry),
        model: input.entry.model,
        modelProvider: input.entry.modelProvider,
        origin: input.entry.origin,
        deliveryContext: input.entry.deliveryContext,
        systemPromptReport: input.entry.systemPromptReport,
        sessionFile: input.entry.sessionFile,
        messages: messageRecords.map((record) => buildScannerRawMessage(record))
    };
}
function buildInteractionEventId(event, source) {
    return deterministicEventId({
        bridge: "openclaw_local_session_tail",
        source,
        scannedEventId: event.eventId,
        kind: "message_delivered",
        messageId: event.messageId ?? null
    });
}
function buildBridgeableEvents(scannedSession, observedAt) {
    const scannerWarnings = [...scannedSession.warnings];
    const interactionEvents = [];
    const feedbackRecords = [];
    let skippedAssistantTurns = 0;
    for (const event of scannedSession.events) {
        const createdAt = event.createdAt ?? observedAt;
        const content = normalizeText(event.content);
        if ((event.actor === "assistant" || event.actor === "system") && event.kind === "assistant_turn") {
            if (event.contentFormat !== "text" || content === null) {
                skippedAssistantTurns += 1;
                continue;
            }
            const interactionId = buildInteractionEventId(event, scannedSession.session.source);
            interactionEvents.push(createInteractionEvent({
                eventId: interactionId,
                agentId: scannedSession.session.agentId ?? "main",
                sessionId: scannedSession.session.sessionId,
                channel: scannedSession.session.channel,
                sequence: event.sequence,
                kind: "message_delivered",
                createdAt,
                source: scannedSession.session.source,
                ...(event.messageId === null ? {} : { messageId: event.messageId })
            }));
            feedbackRecords.push({
                recordId: event.eventId,
                format: "raw",
                createdAt,
                content,
                sequence: event.sequence,
                interactionId,
                messageId: event.messageId,
                actorRole: event.actor === "system" ? "system" : "assistant"
            });
            continue;
        }
        if (event.actor === "user" && event.kind === "user_turn" && event.contentFormat === "text" && content !== null) {
            feedbackRecords.push({
                recordId: event.eventId,
                format: "raw",
                createdAt,
                content,
                sequence: event.sequence,
                messageId: event.messageId,
                actorRole: "user"
            });
        }
    }
    if (skippedAssistantTurns > 0) {
        scannerWarnings.push(`skipped ${skippedAssistantTurns} non-text assistant turns while bridging local session tail output`);
    }
    const extracted = extractFeedbackEventsFromInteractionRecords({
        agentId: scannedSession.session.agentId ?? "main",
        sessionId: scannedSession.session.sessionId,
        channel: scannedSession.session.channel,
        source: scannedSession.session.source,
        records: feedbackRecords
    });
    const sortedEvents = sortNormalizedEvents([
        ...interactionEvents,
        ...extracted.events.map((entry) => entry.feedbackEvent)
    ]);
    if (sortedEvents.length === 0) {
        scannerWarnings.push("scanned session did not produce bridgeable assistant deliveries or extracted feedback");
    }
    return {
        rawRecordCount: scannedSession.events.length,
        bridgedEventCount: sortedEvents.length,
        sortedEvents,
        scannerWarnings,
        source: scannedSession.session.source
    };
}
function isInteractionEvent(event) {
    return event.contract === CONTRACT_IDS.interactionEvents;
}

function hashSessionSnapshotSeed(seed) {
    return hashOpenClawStableJson(seed);
}

function summarizeCorpusSource(change, sessionIndexDigest, sessionFileDigest) {
    const scannedEventExport = change.scannedEventExport;
    const interactionEvents = scannedEventExport?.interactionEvents ?? [];
    const feedbackEvents = scannedEventExport?.feedbackEvents ?? [];
    const eventDigest = interactionEvents.length + feedbackEvents.length === 0
        ? null
        : hashSessionSnapshotSeed({
            interactionEvents,
            feedbackEvents
        });
    const sourceSeed = {
        sourceIndexPath: change.source.indexPath,
        sessionKey: change.sessionKey,
        sessionId: change.sessionId,
        sessionFile: change.sessionFile,
        sessionIndexDigest,
        sessionFileDigest,
        changeKind: change.changeKind,
        rawRecordCount: change.rawRecordCount,
        bridgedEventCount: change.bridgedEventCount,
        eventDigest,
        scannerDigest: scannedEventExport?.scanner?.sourceManifestDigest ?? null
    };
    return {
        sourceId: `graphify-session-source:${hashSessionSnapshotSeed(sourceSeed).slice(0, 32)}`,
        profileRoot: change.source.profileRoot,
        agentId: change.source.agentId,
        sessionsDir: change.source.sessionsDir,
        sourceIndexPath: change.source.indexPath,
        sessionKey: change.sessionKey,
        sessionId: change.sessionId,
        sessionFile: change.sessionFile,
        sessionIndexDigest,
        sessionFileDigest,
        changeKind: change.changeKind,
        rawRecordCount: change.rawRecordCount,
        bridgedEventCount: change.bridgedEventCount,
        eventDigest,
        sourceManifestDigest: scannedEventExport?.scanner?.sourceManifestDigest ?? null,
        scanner: scannedEventExport?.scanner ?? null,
        warnings: [...(change.warnings ?? [])],
        eventCounts: {
            interaction: interactionEvents.length,
            feedback: feedbackEvents.length,
            total: interactionEvents.length + feedbackEvents.length
        }
    };
}

export function buildOpenClawSessionCorpusSnapshot(input = {}) {
    const observedAt = normalizeIsoTimestamp(input.observedAt, new Date().toISOString());
    const tail = createOpenClawLocalSessionTail({
        ...(input.homeDir === undefined ? {} : { homeDir: input.homeDir }),
        ...(input.profileRoots === undefined ? {} : { profileRoots: input.profileRoots }),
        ...(input.cursor === undefined ? {} : { cursor: input.cursor }),
        emitExistingOnFirstPoll: input.emitExistingOnFirstPoll !== false
    });
    const poll = tail.pollOnce({ observedAt });
    const sourceSummaries = [];
    const interactionEvents = [];
    const feedbackEvents = [];
    for (const change of poll.changes) {
        const index = loadOpenClawSessionIndex(change.source.indexPath);
        const sessionIndexDigest = summarizeOpenClawSessionIndex(index).digest;
        const sessionFileDigest = change.sessionFile === null
            ? null
            : summarizeOpenClawSessionFile(readOpenClawSessionFile(change.sessionFile)).digest;
        sourceSummaries.push(summarizeCorpusSource(change, sessionIndexDigest, sessionFileDigest));
        if (change.scannedEventExport !== null) {
            interactionEvents.push(...change.scannedEventExport.interactionEvents);
            feedbackEvents.push(...change.scannedEventExport.feedbackEvents);
        }
    }
    const sortedEvents = sortNormalizedEvents([...interactionEvents, ...feedbackEvents]);
    const sortedInteractionEvents = sortedEvents.filter(isInteractionEvent);
    const sortedFeedbackEvents = sortedEvents.filter((event) => !isInteractionEvent(event));
    const normalizedEventExport = sortedEvents.length === 0
        ? null
        : buildNormalizedEventExport({
            interactionEvents: sortedInteractionEvents,
            feedbackEvents: sortedFeedbackEvents
        });
    const corpusDigest = hashSessionSnapshotSeed({
        lane: poll.lane,
        observedAt,
        noopReason: poll.noopReason,
        warnings: poll.warnings,
        sourceSummaries,
        normalizedEventExportDigest: normalizedEventExport?.provenance.exportDigest ?? null
    });
    return {
        runtimeOwner: "openclaw",
        lane: poll.lane,
        observedAt,
        poll,
        sourceSummaries,
        interactionEvents: sortedInteractionEvents,
        feedbackEvents: sortedFeedbackEvents,
        normalizedEventExport,
        corpusDigest,
        corpusId: `graphify-source-corpus:${corpusDigest.slice(0, 32)}`
    };
}

function buildScannerManifest(input) {
    const sourceManifestDigest = createHash("sha256")
        .update(JSON.stringify({
        sessionKey: input.sessionKey,
        sessionId: input.entry.sessionId,
        updatedAt: input.entry.updatedAt,
        sessionFile: input.entry.sessionFile ?? null,
        chatType: input.entry.chatType ?? null,
        origin: input.entry.origin ?? null,
        deliveryContext: input.entry.deliveryContext ?? null,
        systemPromptReport: input.entry.systemPromptReport ?? null
    }))
        .digest("hex");
    return {
        scannerId: DEFAULT_SCANNER_ID,
        lane: DEFAULT_SCANNER_LANE,
        status: "complete",
        producedAt: input.producedAt,
        sourceManifestPath: input.sourceIndexPath,
        sourceManifestDigest: `sha256-${sourceManifestDigest}`,
        warnings: [...new Set(input.warnings)],
        failures: []
    };
}
function buildScannedEventExport(events, scanner) {
    if (events.length === 0) {
        return null;
    }
    const interactionEvents = events.filter(isInteractionEvent);
    const feedbackEvents = events.filter((event) => event.contract !== CONTRACT_IDS.interactionEvents);
    return {
        interactionEvents,
        feedbackEvents,
        scanner: {
            scannerId: scanner.scannerId,
            lane: scanner.lane,
            status: scanner.status,
            producedAt: scanner.producedAt,
            sourceManifestPath: scanner.sourceManifestPath,
            sourceManifestDigest: scanner.sourceManifestDigest,
            warnings: [...scanner.warnings],
            failures: [...scanner.failures]
        }
    };
}
function createCursor(sourceIndexPath, sessionKey, sessionId, sessionFile, updatedAt, rawRecordCount, bridgedEventCount) {
    return {
        sourceIndexPath,
        sessionKey,
        sessionId,
        sessionFile,
        updatedAt,
        rawRecordCount,
        bridgedEventCount
    };
}
function createChange(input) {
    const scannedEventExport = input.scannedEventExport ?? null;
    const emittedEventCount = scannedEventExport === null ? 0 : scannedEventExport.interactionEvents.length + scannedEventExport.feedbackEvents.length;
    return {
        source: input.source,
        sessionKey: input.sessionKey,
        sessionId: input.sessionId,
        sessionFile: input.sessionFile,
        changeKind: input.changeKind,
        rawRecordCount: input.rawRecordCount,
        bridgedEventCount: input.bridgedEventCount,
        emittedEventCount,
        lastUserMessageAt: input.lastUserMessageAt ?? null,
        lastUserMessageText: normalizeText(input.lastUserMessageText ?? null),
        warnings: input.warnings === undefined ? [] : [...new Set(input.warnings)],
        scannedEventExport
    };
}
function extractLatestUserMessageFromRecords(records, startIndex) {
    let latest = null;
    for (const record of records.slice(Math.max(0, startIndex))) {
        if (record.type !== "message" || record.message.role !== "user") {
            continue;
        }
        const text = normalizeText(record.message.content
            .filter((part) => part.type === "text")
            .map((part) => part.text)
            .join("\n"));
        if (text === null) {
            continue;
        }
        const candidate = {
            at: record.timestamp,
            text
        };
        if (latest === null || Date.parse(candidate.at) >= Date.parse(latest.at)) {
            latest = candidate;
        }
    }
    return latest;
}
export class OpenClawLocalSessionTail {
    homeDir;
    profileRoots;
    initialized;
    emitExistingOnFirstPoll;
    cursorBySession = new Map();
    constructor(input = {}) {
        this.homeDir = input.homeDir;
        this.profileRoots = input.profileRoots?.map((root) => path.resolve(root));
        for (const [index, cursor] of (input.cursor ?? []).entries()) {
            validateCursorEntry(cursor, index);
            this.cursorBySession.set(cursorKey(cursor.sourceIndexPath, cursor.sessionKey), cloneCursor(cursor));
        }
        this.initialized = this.cursorBySession.size > 0;
        this.emitExistingOnFirstPoll = input.emitExistingOnFirstPoll === true;
    }
    snapshot() {
        return snapshotCursor(this.cursorBySession);
    }
    pollOnce(options = {}) {
        const polledAt = normalizeIsoTimestamp(options.observedAt, new Date().toISOString());
        const firstPoll = this.initialized === false;
        const warnings = [];
        const changes = [];
        const observedKeys = new Set();
        const sources = discoverOpenClawSessionStores({
            ...(this.homeDir === undefined ? {} : { homeDir: this.homeDir }),
            ...(this.profileRoots === undefined ? {} : { profileRoots: this.profileRoots })
        });
        if (sources.length === 0) {
            this.initialized = true;
            return {
                runtimeOwner: "openclaw",
                lane: "local_session_tail",
                polledAt,
                sources: [],
                changes: [],
                noopReason: "no_local_session_stores",
                warnings: [],
                cursor: this.snapshot()
            };
        }
        for (const source of sources) {
            const index = loadOpenClawSessionIndex(source.indexPath);
            const sessionKeys = Object.keys(index).sort((left, right) => left.localeCompare(right));
            for (const sessionKey of sessionKeys) {
                const entry = index[sessionKey];
                if (entry === undefined) {
                    continue;
                }
                const key = cursorKey(source.indexPath, sessionKey);
                observedKeys.add(key);
                const existing = this.cursorBySession.get(key) ?? null;
                const sessionFile = typeof entry.sessionFile === "string" && entry.sessionFile.trim().length > 0 ? path.resolve(entry.sessionFile) : null;
                if (sessionFile === null) {
                    const nextCursor = createCursor(source.indexPath, sessionKey, entry.sessionId, null, entry.updatedAt, 0, 0);
                    this.cursorBySession.set(key, nextCursor);
                    const changed = existing === null ||
                        existing.sessionId !== entry.sessionId ||
                        existing.sessionFile !== null ||
                        existing.updatedAt !== entry.updatedAt ||
                        existing.rawRecordCount !== 0 ||
                        existing.bridgedEventCount !== 0;
                    if (!changed) {
                        continue;
                    }
                    changes.push(createChange({
                        source,
                        sessionKey,
                        sessionId: entry.sessionId,
                        sessionFile: null,
                        changeKind: "missing_session_path",
                        rawRecordCount: 0,
                        bridgedEventCount: 0,
                        warnings: ["session manifest entry did not include a sessionFile path"]
                    }));
                    continue;
                }
                if (!existsSync(sessionFile)) {
                    const nextCursor = createCursor(source.indexPath, sessionKey, entry.sessionId, sessionFile, entry.updatedAt, 0, 0);
                    this.cursorBySession.set(key, nextCursor);
                    const changed = existing === null ||
                        existing.sessionId !== entry.sessionId ||
                        existing.sessionFile !== sessionFile ||
                        existing.updatedAt !== entry.updatedAt ||
                        existing.rawRecordCount !== 0 ||
                        existing.bridgedEventCount !== 0;
                    if (!changed) {
                        continue;
                    }
                    changes.push(createChange({
                        source,
                        sessionKey,
                        sessionId: entry.sessionId,
                        sessionFile,
                        changeKind: "missing_session_file",
                        rawRecordCount: 0,
                        bridgedEventCount: 0,
                        warnings: [`session file does not exist on disk: ${sessionFile}`]
                    }));
                    continue;
                }
                let records;
                let bridge;
                try {
                    records = readOpenClawSessionFile(sessionFile);
                    const scannerRawSession = buildScannerRawSession({
                        source,
                        sessionKey,
                        entry,
                        records
                    });
                    const scannedSession = scanSession(scannerRawSession, {
                        observedAt: polledAt,
                        defaultAgentId: source.agentId,
                        defaultChannel: deriveSessionChannel(entry),
                        defaultSourceStream: buildSourceStream(source, entry)
                    });
                    bridge = buildBridgeableEvents(scannedSession, polledAt);
                }
                catch (error) {
                    const message = error instanceof Error ? error.message : String(error);
                    warnings.push(`session tail skipped ${sessionFile} for ${source.agentId}:${sessionKey}: ${message}`);
                    continue;
                }
                const scanner = buildScannerManifest({
                    producedAt: polledAt,
                    sourceIndexPath: source.indexPath,
                    sessionKey,
                    entry,
                    warnings: bridge.scannerWarnings
                });
                const nextCursor = createCursor(source.indexPath, sessionKey, entry.sessionId, sessionFile, entry.updatedAt, records.length, bridge.bridgedEventCount);
                if (existing === null) {
                    this.cursorBySession.set(key, nextCursor);
                    if (firstPoll && !this.emitExistingOnFirstPoll) {
                        continue;
                    }
                    const latestUserMessage = extractLatestUserMessageFromRecords(records, 0);
                    changes.push(createChange({
                        source,
                        sessionKey,
                        sessionId: entry.sessionId,
                        sessionFile,
                        changeKind: "new_session",
                        rawRecordCount: records.length,
                        bridgedEventCount: bridge.bridgedEventCount,
                        lastUserMessageAt: latestUserMessage?.at ?? null,
                        lastUserMessageText: latestUserMessage?.text ?? null,
                        warnings: bridge.scannerWarnings,
                        scannedEventExport: buildScannedEventExport(bridge.sortedEvents, scanner)
                    }));
                    continue;
                }
                const resetDetected = existing.sessionFile !== sessionFile ||
                    existing.sessionId !== entry.sessionId ||
                    records.length < existing.rawRecordCount ||
                    bridge.bridgedEventCount < existing.bridgedEventCount;
                if (resetDetected) {
                    this.cursorBySession.set(key, nextCursor);
                    changes.push(createChange({
                        source,
                        sessionKey,
                        sessionId: entry.sessionId,
                        sessionFile,
                        changeKind: "session_reset",
                        rawRecordCount: records.length,
                        bridgedEventCount: bridge.bridgedEventCount,
                        warnings: [
                            "session tail cursor reset because the session file, session id, or append-only counts moved backwards",
                            ...bridge.scannerWarnings
                        ]
                    }));
                    continue;
                }
                const changed = existing.updatedAt !== entry.updatedAt ||
                    existing.rawRecordCount !== records.length ||
                    existing.bridgedEventCount !== bridge.bridgedEventCount;
                if (!changed) {
                    continue;
                }
                this.cursorBySession.set(key, nextCursor);
                const deltaEvents = bridge.sortedEvents.slice(existing.bridgedEventCount);
                const deltaExport = buildScannedEventExport(deltaEvents, scanner);
                const latestUserMessage = extractLatestUserMessageFromRecords(records, existing.rawRecordCount);
                changes.push(createChange({
                    source,
                    sessionKey,
                    sessionId: entry.sessionId,
                    sessionFile,
                    changeKind: deltaEvents.length > 0 ? "appended_records" : "metadata_only",
                    rawRecordCount: records.length,
                    bridgedEventCount: bridge.bridgedEventCount,
                    lastUserMessageAt: latestUserMessage?.at ?? null,
                    lastUserMessageText: latestUserMessage?.text ?? null,
                    warnings: bridge.scannerWarnings,
                    scannedEventExport: deltaExport
                }));
            }
        }
        let prunedCursorCount = 0;
        for (const key of [...this.cursorBySession.keys()]) {
            if (observedKeys.has(key)) {
                continue;
            }
            this.cursorBySession.delete(key);
            prunedCursorCount += 1;
        }
        if (prunedCursorCount > 0) {
            warnings.push(`session tail pruned ${prunedCursorCount} stale cursor entr${prunedCursorCount === 1 ? "y" : "ies"}`);
        }
        this.initialized = true;
        const noopReason = firstPoll && !changes.some((change) => change.scannedEventExport !== null)
            ? "seeded_existing_sessions"
            : changes.length === 0
                ? "no_session_changes"
                : null;
        return {
            runtimeOwner: "openclaw",
            lane: "local_session_tail",
            polledAt,
            sources,
            changes,
            noopReason,
            warnings,
            cursor: this.snapshot()
        };
    }
    async runLoop(options = {}) {
        const pollIntervalMs = normalizeNonNegativeDurationMs(options.pollIntervalMs, "pollIntervalMs", 0);
        const maxPasses = normalizePositiveInteger(options.maxPasses, "maxPasses", 1);
        const stopWhenIdle = options.stopWhenIdle !== false;
        let passCount = 0;
        let changedSessionCount = 0;
        let emittedEventCount = 0;
        let lastPoll = null;
        let stoppedReason = "max_passes";
        while (passCount < maxPasses) {
            if (options.signal?.aborted) {
                stoppedReason = "aborted";
                break;
            }
            lastPoll = this.pollOnce();
            passCount += 1;
            changedSessionCount += lastPoll.changes.length;
            emittedEventCount += lastPoll.changes.reduce((total, change) => total + change.emittedEventCount, 0);
            if (options.onPass !== undefined) {
                await options.onPass(lastPoll);
            }
            if (stopWhenIdle && lastPoll.changes.length === 0) {
                stoppedReason = "idle";
                break;
            }
            if (passCount >= maxPasses) {
                stoppedReason = "max_passes";
                break;
            }
            if (pollIntervalMs > 0) {
                await delay(pollIntervalMs);
            }
        }
        return {
            runtimeOwner: "openclaw",
            passCount,
            changedSessionCount,
            emittedEventCount,
            stoppedReason,
            lastPoll,
            cursor: this.snapshot()
        };
    }
}
export function createOpenClawLocalSessionTail(input = {}) {
    return new OpenClawLocalSessionTail(input);
}
//# sourceMappingURL=session-tail.js.map
