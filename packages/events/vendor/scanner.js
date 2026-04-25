import { canonicalJson, checksumJsonPayload } from "@openclawbrain/contracts";
const ROOT_CONTAINER_KEYS = ["turns", "messages", "events", "items", "entries", "conversation", "transcript"];
const NESTED_CONTAINER_KEYS = ["messages", "events", "items", "entries"];
const TOOL_CALL_ARRAY_KEYS = ["toolCalls", "tool_calls", "functionCalls", "function_calls"];
const TOOL_RESULT_ARRAY_KEYS = ["toolResults", "tool_results", "toolOutputs", "tool_outputs"];
const USER_MESSAGE_FIELDS = ["userMessage", "user_message", "prompt", "input"];
const ASSISTANT_MESSAGE_FIELDS = ["assistantMessage", "assistant_message", "assistantResponse", "assistantReply", "response", "output"];
const ROOT_TIMESTAMP_FIELDS = ["recordedAt", "createdAt", "startedAt", "updatedAt"];
const EVENT_TIMESTAMP_FIELDS = ["createdAt", "timestamp", "occurredAt", "sentAt", "startedAt"];
const ASSISTANT_TIMESTAMP_FIELDS = ["deliveredAt", "completedAt", "respondedAt", ...EVENT_TIMESTAMP_FIELDS];
const TOOL_RESULT_TIMESTAMP_FIELDS = ["completedAt", "finishedAt", ...EVENT_TIMESTAMP_FIELDS];
const TOOL_CALL_TYPES = new Set(["tool_call", "function_call", "tool-invocation"]);
const TOOL_RESULT_TYPES = new Set(["tool_result", "function_result", "tool-output"]);
const USER_ROLES = new Set(["user", "human", "customer"]);
const ASSISTANT_ROLES = new Set(["assistant", "model", "agent", "system"]);
const TOOL_ROLES = new Set(["tool", "function"]);
function isRecord(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
function normalizeString(value) {
    if (typeof value !== "string") {
        return null;
    }
    const trimmed = value.trim();
    return trimmed.length === 0 ? null : trimmed;
}
function normalizeIsoTimestamp(value) {
    if (typeof value === "string") {
        const trimmed = value.trim();
        if (trimmed.length === 0) {
            return null;
        }
        const parsed = Date.parse(trimmed);
        return Number.isNaN(parsed) ? null : new Date(parsed).toISOString();
    }
    if (typeof value === "number" && Number.isFinite(value)) {
        return new Date(value).toISOString();
    }
    return null;
}
function getNestedValue(value, path) {
    let current = value;
    for (const segment of path) {
        if (!isRecord(current)) {
            return undefined;
        }
        current = current[segment];
    }
    return current;
}
function getFirstString(value, candidatePaths) {
    for (const path of candidatePaths) {
        const resolved = normalizeString(getNestedValue(value, path));
        if (resolved !== null) {
            return resolved;
        }
    }
    return null;
}
function getFirstTimestamp(value, candidatePaths) {
    for (const path of candidatePaths) {
        const resolved = normalizeIsoTimestamp(getNestedValue(value, path));
        if (resolved !== null) {
            return resolved;
        }
    }
    return null;
}
function sanitizeToken(value) {
    const sanitized = value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
    return sanitized.length > 0 ? sanitized : "scanner";
}
function pathToString(path) {
    if (path.length === 0) {
        return "$";
    }
    return `$${path.map((segment) => (typeof segment === "number" ? `[${segment}]` : `.${segment}`)).join("")}`;
}
function collectContentParts(value) {
    if (value === null || value === undefined) {
        return [];
    }
    if (typeof value === "string") {
        const trimmed = value.trim();
        return trimmed.length === 0 ? [] : [trimmed];
    }
    if (typeof value === "number" || typeof value === "boolean") {
        return [String(value)];
    }
    if (Array.isArray(value)) {
        return value.flatMap((entry) => collectContentParts(entry));
    }
    if (!isRecord(value)) {
        return [];
    }
    const textPart = normalizeString(value.text) ??
        normalizeString(value.content) ??
        normalizeString(value.value) ??
        normalizeString(value.message) ??
        normalizeString(value.input_text) ??
        normalizeString(value.output_text);
    if (textPart !== null) {
        return [textPart];
    }
    if ("content" in value) {
        return collectContentParts(value.content);
    }
    if ("parts" in value) {
        return collectContentParts(value.parts);
    }
    return [];
}
function resolveContent(value) {
    const parts = collectContentParts(value);
    if (parts.length > 0) {
        return {
            content: parts.join("\n"),
            format: "text"
        };
    }
    if (isRecord(value) || Array.isArray(value)) {
        return {
            content: canonicalJson(value),
            format: "json"
        };
    }
    return {
        content: null,
        format: "empty"
    };
}
function resolveTimestamp(record, fieldCandidates, context, sessionCreatedAt, observedAt) {
    for (const field of fieldCandidates) {
        const resolved = isRecord(record) ? normalizeIsoTimestamp(record[field]) : null;
        if (resolved !== null) {
            return { value: resolved, source: "event" };
        }
    }
    if (context.turnCreatedAt !== null) {
        return {
            value: context.turnCreatedAt,
            source: "turn"
        };
    }
    if (sessionCreatedAt !== null) {
        return {
            value: sessionCreatedAt,
            source: "session"
        };
    }
    if (observedAt !== null) {
        return {
            value: observedAt,
            source: "observed"
        };
    }
    return {
        value: null,
        source: "missing"
    };
}
function extractMessageId(record) {
    return getFirstString(record, [["messageId"], ["message_id"], ["id"]]);
}
function extractRecordId(record) {
    return getFirstString(record, [["eventId"], ["event_id"], ["id"]]);
}
function extractRole(record) {
    return getFirstString(record, [["role"], ["author", "role"], ["sender"], ["actor"]]);
}
function extractKind(record) {
    return getFirstString(record, [["kind"], ["type"], ["eventType"]]);
}
function extractToolCallId(record) {
    return getFirstString(record, [["toolCallId"], ["tool_call_id"], ["callId"], ["call_id"], ["id"]]);
}
function extractToolName(record) {
    return getFirstString(record, [["toolName"], ["tool"], ["name"], ["function", "name"]]);
}
function extractProfileId(value) {
    return getFirstString(value, [
        ["profileId"],
        ["metadata", "profileId"],
        ["profile", "id"],
        ["metadata", "profile", "id"],
        ["principalId"],
        ["principal", "id"],
        ["userId"],
        ["user", "id"],
        ["accountId"]
    ]);
}
function extractProfileSelector(value) {
    return getFirstString(value, [["profileSelector"], ["metadata", "profileSelector"], ["profile", "selector"]]);
}
function classifyActor(rawRole) {
    if (rawRole === null) {
        return "unknown";
    }
    const normalized = rawRole.toLowerCase();
    if (USER_ROLES.has(normalized)) {
        return "user";
    }
    if (TOOL_ROLES.has(normalized)) {
        return "tool";
    }
    if (normalized === "system") {
        return "system";
    }
    if (ASSISTANT_ROLES.has(normalized)) {
        return "assistant";
    }
    return "unknown";
}
function roleToKind(actor) {
    if (actor === "user") {
        return "user_turn";
    }
    if (actor === "assistant" || actor === "system") {
        return "assistant_turn";
    }
    if (actor === "tool") {
        return "tool_result";
    }
    return null;
}
function deriveSourceStream(rawSession, channel, options) {
    const rawStream = getFirstString(rawSession, [["sourceStream"], ["stream"], ["source", "stream"]]);
    if (rawStream !== null) {
        return {
            value: rawStream,
            source: "raw"
        };
    }
    if (options.defaultSourceStream !== undefined) {
        return {
            value: options.defaultSourceStream,
            source: "defaulted"
        };
    }
    return {
        value: `openclaw/runtime/${sanitizeToken(channel)}`,
        source: "inferred"
    };
}
function buildSessionMetadata(rawSession, options, warnings) {
    const rawSessionHash = checksumJsonPayload(rawSession);
    const sessionIdFromRaw = getFirstString(rawSession, [["sessionId"], ["session_id"], ["traceId"], ["session", "id"], ["id"]]);
    const channelFromRaw = getFirstString(rawSession, [["channel"], ["transport"], ["medium"], ["source", "channel"], ["session", "channel"]]);
    const agentIdFromRaw = getFirstString(rawSession, [["agentId"], ["assistantId"], ["runtime", "agentId"], ["metadata", "agentId"]]);
    const profileIdFromRaw = extractProfileId(rawSession);
    const profileSelectorFromRaw = extractProfileSelector(rawSession);
    const sessionCreatedAt = getFirstTimestamp(rawSession, ROOT_TIMESTAMP_FIELDS.map((field) => [field]));
    const observedAt = normalizeIsoTimestamp(options.observedAt);
    const sourceStream = deriveSourceStream(rawSession, channelFromRaw ?? options.defaultChannel ?? "unknown", options);
    const sessionId = sessionIdFromRaw ?? `session-${rawSessionHash.slice(0, 12)}`;
    const channel = channelFromRaw ?? options.defaultChannel ?? "unknown";
    const provenance = [
        ...[
            normalizeString(getNestedValue(rawSession, ["contract"])) !== null ? `contract:${normalizeString(getNestedValue(rawSession, ["contract"]))}` : null,
            normalizeString(getNestedValue(rawSession, ["source"])) !== null ? `source:${normalizeString(getNestedValue(rawSession, ["source"]))}` : null,
            normalizeString(getNestedValue(rawSession, ["traceId"])) !== null ? `trace:${normalizeString(getNestedValue(rawSession, ["traceId"]))}` : null
        ].filter((value) => value !== null),
        `raw_hash:${rawSessionHash}`
    ];
    if (sessionCreatedAt === null) {
        warnings.push("session timestamp missing; event timestamps may fall back to turn-level or remain null");
    }
    if (channelFromRaw === null && options.defaultChannel === undefined) {
        warnings.push("session channel missing; scanner defaulted channel to unknown");
    }
    return {
        session: {
            sessionId,
            sessionIdSource: sessionIdFromRaw === null ? "inferred" : "raw",
            channel,
            channelSource: channelFromRaw === null ? (options.defaultChannel === undefined ? "inferred" : "defaulted") : "raw",
            agentId: agentIdFromRaw ?? options.defaultAgentId ?? null,
            agentIdSource: agentIdFromRaw !== null ? "raw" : options.defaultAgentId !== undefined ? "defaulted" : "missing",
            profileId: profileIdFromRaw,
            profileIdSource: profileIdFromRaw === null ? "missing" : "raw",
            profileSelector: profileSelectorFromRaw,
            profileSelectorSource: profileSelectorFromRaw === null ? "missing" : "raw",
            source: {
                runtimeOwner: "openclaw",
                stream: sourceStream.value
            },
            sourceStreamSource: sourceStream.source,
            rawSessionHash,
            provenance
        },
        sessionCreatedAt,
        observedAt
    };
}
function resolveEventProfile(rawSession, rawNode, path, session, kind, actor, options) {
    const localProfileId = extractProfileId(rawNode);
    const localProfileSelector = extractProfileSelector(rawNode);
    if (localProfileId !== null || localProfileSelector !== null) {
        return {
            profileId: localProfileId ?? session.profileId,
            profileSelector: localProfileSelector ?? session.profileSelector,
            profileIdSource: localProfileId === null ? session.profileIdSource : "raw",
            profileSelectorSource: localProfileSelector === null ? session.profileSelectorSource : "raw"
        };
    }
    const resolved = options.profileResolver?.({
        rawSession,
        rawNode,
        path,
        session,
        inferredKind: kind,
        inferredActor: actor
    });
    if (resolved !== null && resolved !== undefined && (resolved.profileId !== undefined || resolved.profileSelector !== undefined)) {
        return {
            profileId: resolved.profileId ?? session.profileId,
            profileSelector: resolved.profileSelector ?? session.profileSelector,
            profileIdSource: resolved.profileId !== undefined ? "hook" : session.profileIdSource,
            profileSelectorSource: resolved.profileSelector !== undefined ? "hook" : session.profileSelectorSource
        };
    }
    return {
        profileId: session.profileId,
        profileSelector: session.profileSelector,
        profileIdSource: session.profileIdSource,
        profileSelectorSource: session.profileSelectorSource
    };
}
function buildTempKey(kind, path, rawHash, discoveryIndex) {
    return `${kind}:${pathToString(path)}:${rawHash.slice(0, 16)}:${discoveryIndex}`;
}
function shouldEmitEvent(input, content) {
    if (input.forceEmit === true) {
        return true;
    }
    if (content.content !== null) {
        return true;
    }
    if (input.messageId !== null && input.messageId !== undefined) {
        return true;
    }
    if (input.toolCallId !== null && input.toolCallId !== undefined) {
        return true;
    }
    if (input.toolName !== null && input.toolName !== undefined) {
        return true;
    }
    return false;
}
function collectEvent(provisionalEvents, rawSession, session, warnings, options, input) {
    const content = input.content ?? resolveContent(input.rawNode);
    if (!shouldEmitEvent(input, content)) {
        warnings.push(`skipped empty ${input.kind} at ${pathToString(input.path)}`);
        return null;
    }
    const rawHash = checksumJsonPayload(input.rawNode);
    const timestamp = input.timestamp ?? {
        value: null,
        source: "missing"
    };
    const rawRecord = isRecord(input.rawNode) ? input.rawNode : null;
    const rawEventId = rawRecord === null ? null : extractRecordId(rawRecord);
    const rawMessageId = rawRecord === null ? null : extractMessageId(rawRecord);
    const rawRole = input.rawRole ?? (rawRecord === null ? null : extractRole(rawRecord));
    const rawKind = input.rawKind ?? (rawRecord === null ? null : extractKind(rawRecord));
    const notes = [...(input.notes ?? [])];
    const extraNotes = options.provenanceResolver?.({
        rawSession,
        rawNode: input.rawNode,
        path: input.path,
        session
    });
    if (extraNotes !== null && extraNotes !== undefined) {
        notes.push(...extraNotes
            .map((entry) => normalizeString(entry))
            .filter((entry) => entry !== null));
    }
    const tempKey = buildTempKey(input.kind, input.path, rawHash, provisionalEvents.length);
    provisionalEvents.push({
        tempKey,
        rawNode: input.rawNode,
        rawHash,
        path: input.path,
        turnId: input.context.turnId,
        rawEventId,
        rawMessageId: input.messageId ?? rawMessageId,
        rawRole,
        rawKind,
        notes,
        kind: input.kind,
        actor: input.actor,
        createdAt: timestamp.value,
        createdAtSource: timestamp.source,
        messageId: input.messageId ?? rawMessageId,
        toolCallId: input.toolCallId ?? null,
        toolName: input.toolName ?? null,
        content: content.content,
        contentFormat: content.format,
        parentTempKey: input.parentTempKey ?? null,
        discoveryIndex: provisionalEvents.length
    });
    return tempKey;
}
function scanToolArrays(node, path, context, parentTempKey, rawSession, session, sessionCreatedAt, observedAt, provisionalEvents, warnings, options) {
    for (const key of TOOL_CALL_ARRAY_KEYS) {
        const entries = node[key];
        if (!Array.isArray(entries)) {
            continue;
        }
        entries.forEach((entry, index) => {
            const record = isRecord(entry) ? entry : null;
            const timestamp = resolveTimestamp(entry, EVENT_TIMESTAMP_FIELDS, context, sessionCreatedAt, observedAt);
            const content = resolveContent(record === null
                ? entry
                : record.arguments ?? record.args ?? record.input ?? record.payload ?? entry);
            collectEvent(provisionalEvents, rawSession, session, warnings, options, {
                rawNode: entry,
                path: [...path, key, index],
                context,
                kind: "tool_call",
                actor: "assistant",
                rawKind: record === null ? key : extractKind(record),
                rawRole: "assistant",
                toolCallId: record === null ? null : extractToolCallId(record),
                toolName: record === null ? null : extractToolName(record),
                content,
                timestamp,
                parentTempKey,
                forceEmit: true,
                notes: [`container:${key}`]
            });
        });
    }
    for (const key of TOOL_RESULT_ARRAY_KEYS) {
        const entries = node[key];
        if (!Array.isArray(entries)) {
            continue;
        }
        entries.forEach((entry, index) => {
            const record = isRecord(entry) ? entry : null;
            const timestamp = resolveTimestamp(entry, TOOL_RESULT_TIMESTAMP_FIELDS, context, sessionCreatedAt, observedAt);
            const content = resolveContent(record === null
                ? entry
                : record.result ?? record.output ?? record.response ?? record.content ?? entry);
            collectEvent(provisionalEvents, rawSession, session, warnings, options, {
                rawNode: entry,
                path: [...path, key, index],
                context,
                kind: "tool_result",
                actor: "tool",
                rawKind: record === null ? key : extractKind(record),
                rawRole: record === null ? "tool" : extractRole(record) ?? "tool",
                messageId: record === null ? null : extractMessageId(record),
                toolCallId: record === null ? null : extractToolCallId(record),
                toolName: record === null ? null : extractToolName(record),
                content,
                timestamp,
                parentTempKey,
                forceEmit: true,
                notes: [`container:${key}`]
            });
        });
    }
}
function isTurnLike(node) {
    return (USER_MESSAGE_FIELDS.some((field) => field in node) ||
        ASSISTANT_MESSAGE_FIELDS.some((field) => field in node) ||
        TOOL_CALL_ARRAY_KEYS.some((field) => field in node) ||
        TOOL_RESULT_ARRAY_KEYS.some((field) => field in node) ||
        "runtimeHints" in node ||
        "feedback" in node);
}
function scanNode(rawNode, path, inheritedContext, rawSession, session, sessionCreatedAt, observedAt, provisionalEvents, warnings, options) {
    if (!isRecord(rawNode)) {
        if (normalizeString(rawNode) !== null) {
            const timestamp = inheritedContext.turnCreatedAt === null
                ? {
                    value: sessionCreatedAt,
                    source: sessionCreatedAt === null ? "missing" : "session"
                }
                : {
                    value: inheritedContext.turnCreatedAt,
                    source: "turn"
                };
            collectEvent(provisionalEvents, rawSession, session, warnings, options, {
                rawNode,
                path,
                context: inheritedContext,
                kind: "assistant_turn",
                actor: "assistant",
                content: resolveContent(rawNode),
                timestamp,
                notes: ["primitive_message_node"]
            });
        }
        else {
            warnings.push(`ignored non-record node at ${pathToString(path)}`);
        }
        return;
    }
    const turnId = normalizeString(rawNode.turnId) ?? inheritedContext.turnId;
    const turnCreatedAt = normalizeIsoTimestamp(rawNode.createdAt) ?? inheritedContext.turnCreatedAt;
    const context = {
        turnId,
        turnCreatedAt,
        path
    };
    let assistantTempKey = null;
    let handledNestedMessages = false;
    let handledNestedEvents = false;
    let handledToolArrays = false;
    let handledRoleBasedEvent = false;
    if (isTurnLike(rawNode)) {
        for (const field of USER_MESSAGE_FIELDS) {
            if (!(field in rawNode)) {
                continue;
            }
            const timestamp = resolveTimestamp(rawNode, EVENT_TIMESTAMP_FIELDS, context, sessionCreatedAt, observedAt);
            collectEvent(provisionalEvents, rawSession, session, warnings, options, {
                rawNode: rawNode[field],
                path: [...path, field],
                context,
                kind: "user_turn",
                actor: "user",
                timestamp,
                messageId: extractMessageId(rawNode),
                forceEmit: true,
                notes: [`field:${field}`]
            });
        }
        for (const field of ASSISTANT_MESSAGE_FIELDS) {
            if (!(field in rawNode)) {
                continue;
            }
            const timestamp = resolveTimestamp(rawNode, handledToolArrays || TOOL_CALL_ARRAY_KEYS.some((key) => key in rawNode) ? EVENT_TIMESTAMP_FIELDS : ASSISTANT_TIMESTAMP_FIELDS, context, sessionCreatedAt, observedAt);
            assistantTempKey = collectEvent(provisionalEvents, rawSession, session, warnings, options, {
                rawNode: rawNode[field],
                path: [...path, field],
                context,
                kind: "assistant_turn",
                actor: "assistant",
                timestamp,
                messageId: extractMessageId(rawNode),
                forceEmit: true,
                notes: [`field:${field}`]
            });
        }
        const turnActor = classifyActor(extractRole(rawNode));
        if (assistantTempKey === null && turnActor === "assistant") {
            assistantTempKey = collectEvent(provisionalEvents, rawSession, session, warnings, options, {
                rawNode,
                path,
                context,
                kind: "assistant_turn",
                actor: "assistant",
                rawKind: extractKind(rawNode),
                rawRole: extractRole(rawNode),
                messageId: extractMessageId(rawNode),
                content: resolveContent(rawNode.content ?? rawNode.message ?? rawNode.text ?? rawNode),
                timestamp: resolveTimestamp(rawNode, EVENT_TIMESTAMP_FIELDS, context, sessionCreatedAt, observedAt),
                forceEmit: true,
                notes: ["assistant_parent_for_tool_edges"]
            });
            handledRoleBasedEvent = true;
        }
        scanToolArrays(rawNode, path, context, assistantTempKey, rawSession, session, sessionCreatedAt, observedAt, provisionalEvents, warnings, options);
        handledToolArrays = true;
        if (Array.isArray(rawNode.messages)) {
            handledNestedMessages = true;
            rawNode.messages.forEach((entry, index) => {
                scanNode(entry, [...path, "messages", index], context, rawSession, session, sessionCreatedAt, observedAt, provisionalEvents, warnings, options);
            });
        }
        if (Array.isArray(rawNode.events)) {
            handledNestedEvents = true;
            rawNode.events.forEach((entry, index) => {
                scanNode(entry, [...path, "events", index], context, rawSession, session, sessionCreatedAt, observedAt, provisionalEvents, warnings, options);
            });
        }
    }
    const rawRole = extractRole(rawNode);
    const rawKind = extractKind(rawNode);
    const actor = classifyActor(rawRole);
    if (rawKind !== null && TOOL_CALL_TYPES.has(rawKind.toLowerCase())) {
        collectEvent(provisionalEvents, rawSession, session, warnings, options, {
            rawNode,
            path,
            context,
            kind: "tool_call",
            actor: "assistant",
            rawKind,
            rawRole: rawRole ?? "assistant",
            toolCallId: extractToolCallId(rawNode),
            toolName: extractToolName(rawNode),
            content: resolveContent(rawNode.arguments ?? rawNode.args ?? rawNode.input ?? rawNode.payload ?? rawNode),
            timestamp: resolveTimestamp(rawNode, EVENT_TIMESTAMP_FIELDS, context, sessionCreatedAt, observedAt),
            forceEmit: true
        });
    }
    else if (rawKind !== null && TOOL_RESULT_TYPES.has(rawKind.toLowerCase())) {
        collectEvent(provisionalEvents, rawSession, session, warnings, options, {
            rawNode,
            path,
            context,
            kind: "tool_result",
            actor: "tool",
            rawKind,
            rawRole: rawRole ?? "tool",
            messageId: extractMessageId(rawNode),
            toolCallId: extractToolCallId(rawNode),
            toolName: extractToolName(rawNode),
            content: resolveContent(rawNode.result ?? rawNode.output ?? rawNode.response ?? rawNode.content ?? rawNode),
            timestamp: resolveTimestamp(rawNode, TOOL_RESULT_TIMESTAMP_FIELDS, context, sessionCreatedAt, observedAt),
            forceEmit: true
        });
    }
    else {
        const kind = roleToKind(actor);
        if (kind !== null && !handledRoleBasedEvent) {
            const messageHasToolCalls = Array.isArray(rawNode.tool_calls) || Array.isArray(rawNode.toolCalls);
            const tempKey = collectEvent(provisionalEvents, rawSession, session, warnings, options, {
                rawNode,
                path,
                context,
                kind,
                actor,
                rawKind,
                rawRole,
                messageId: extractMessageId(rawNode),
                toolCallId: actor === "tool" ? extractToolCallId(rawNode) : null,
                toolName: actor === "tool" ? extractToolName(rawNode) : null,
                content: resolveContent(rawNode.content ?? rawNode.message ?? rawNode.text ?? rawNode),
                timestamp: resolveTimestamp(rawNode, kind === "assistant_turn" ? ASSISTANT_TIMESTAMP_FIELDS : EVENT_TIMESTAMP_FIELDS, context, sessionCreatedAt, observedAt),
                forceEmit: actor === "assistant" && messageHasToolCalls
            });
            if (actor === "assistant" && tempKey !== null) {
                assistantTempKey = tempKey;
            }
            if (actor === "assistant" && messageHasToolCalls && !handledToolArrays) {
                scanToolArrays(rawNode, path, context, tempKey, rawSession, session, sessionCreatedAt, observedAt, provisionalEvents, warnings, options);
            }
        }
    }
    for (const key of NESTED_CONTAINER_KEYS) {
        if ((key === "messages" && handledNestedMessages) || (key === "events" && handledNestedEvents)) {
            continue;
        }
        const entries = rawNode[key];
        if (!Array.isArray(entries)) {
            continue;
        }
        entries.forEach((entry, index) => {
            scanNode(entry, [...path, key, index], context, rawSession, session, sessionCreatedAt, observedAt, provisionalEvents, warnings, options);
        });
    }
}
function buildEventId(sessionId, provisional) {
    // Include the path and raw hash so sibling lanes can safely join scanner output back to raw-session fragments.
    const digest = checksumJsonPayload({
        sessionId,
        path: pathToString(provisional.path),
        kind: provisional.kind,
        rawHash: provisional.rawHash
    }).slice(0, 16);
    return `${sessionId}:${digest}`;
}
export function scanSession(rawSession, options = {}) {
    const warnings = [];
    const { session, sessionCreatedAt, observedAt } = buildSessionMetadata(rawSession, options, warnings);
    const provisionalEvents = [];
    const rootNodes = [];
    if (isRecord(rawSession)) {
        for (const key of ROOT_CONTAINER_KEYS) {
            const entries = rawSession[key];
            if (!Array.isArray(entries)) {
                continue;
            }
            entries.forEach((entry, index) => {
                rootNodes.push({
                    node: entry,
                    path: [key, index]
                });
            });
        }
    }
    if (rootNodes.length === 0) {
        rootNodes.push({
            node: rawSession,
            path: []
        });
    }
    for (const rootNode of rootNodes) {
        scanNode(rootNode.node, rootNode.path, {
            turnId: null,
            turnCreatedAt: null,
            path: rootNode.path
        }, rawSession, session, sessionCreatedAt, observedAt, provisionalEvents, warnings, options);
    }
    provisionalEvents.sort((left, right) => {
        if (left.createdAt !== null && right.createdAt !== null) {
            const byTime = left.createdAt.localeCompare(right.createdAt);
            if (byTime !== 0) {
                return byTime;
            }
        }
        else if (left.createdAt !== null || right.createdAt !== null) {
            return left.createdAt === null ? 1 : -1;
        }
        const byDiscovery = left.discoveryIndex - right.discoveryIndex;
        if (byDiscovery !== 0) {
            return byDiscovery;
        }
        const byPath = pathToString(left.path).localeCompare(pathToString(right.path));
        if (byPath !== 0) {
            return byPath;
        }
        return 0;
    });
    const eventIdsByTempKey = new Map();
    const toolCallEventIdsByCallId = new Map();
    const events = provisionalEvents.map((event, index) => {
        const eventId = buildEventId(session.sessionId, event);
        eventIdsByTempKey.set(event.tempKey, eventId);
        if (event.toolCallId !== null && event.kind === "tool_call") {
            toolCallEventIdsByCallId.set(event.toolCallId, eventId);
        }
        const resolvedProfile = resolveEventProfile(rawSession, event.rawNode, event.path, session, event.kind, event.actor, options);
        return {
            eventId,
            sessionId: session.sessionId,
            channel: session.channel,
            agentId: session.agentId,
            profileId: resolvedProfile.profileId,
            profileSelector: resolvedProfile.profileSelector,
            sequence: index,
            kind: event.kind,
            actor: event.actor,
            createdAt: event.createdAt,
            createdAtSource: event.createdAtSource,
            source: session.source,
            messageId: event.messageId,
            toolCallId: event.toolCallId,
            toolName: event.toolName,
            parentEventId: null,
            content: event.content,
            contentFormat: event.contentFormat,
            rawHash: event.rawHash,
            provenance: {
                path: pathToString(event.path),
                turnId: event.turnId,
                rawEventId: event.rawEventId,
                rawMessageId: event.rawMessageId,
                rawRole: event.rawRole,
                rawKind: event.rawKind,
                notes: [...event.notes]
            }
        };
    });
    events.forEach((event, index) => {
        const provisional = provisionalEvents[index];
        if (provisional === undefined) {
            return;
        }
        if (provisional.toolCallId !== null && provisional.kind === "tool_result") {
            event.parentEventId = toolCallEventIdsByCallId.get(provisional.toolCallId) ?? null;
            return;
        }
        if (provisional.parentTempKey !== null) {
            event.parentEventId = eventIdsByTempKey.get(provisional.parentTempKey) ?? null;
            return;
        }
    });
    if (events.length === 0) {
        warnings.push("scanner produced no interaction events from the raw session");
    }
    return {
        session,
        events,
        warnings: [...new Set(warnings)]
    };
}
