import path from "node:path";
import { buildNormalizedEventExport, createInteractionEvent } from "@openclawbrain/contracts";
import { extractFeedbackEventsFromInteractionRecords, isSystemMessage } from "@openclawbrain/event-export";
import { buildAssistantMessageSemanticMetadata, buildFeedbackSemanticMetadata } from "./semantic-metadata.js";
const DEFAULT_AGENT_ID = "openclaw-session-store";
const KNOWN_PRINCIPALS = {
    bihua: {
        teacherIdentity: "bihua",
        teacherRole: "principal",
        teacherAuthority: "binding",
        priorityClass: "critical"
    },
    "jonathan gu": {
        teacherIdentity: "jonathan",
        teacherRole: "admin",
        teacherAuthority: "high",
        priorityClass: "high"
    },
    jonathan: {
        teacherIdentity: "jonathan",
        teacherRole: "admin",
        teacherAuthority: "high",
        priorityClass: "high"
    }
};
const DEFAULT_PRIVACY_RULES = [
    "session-store fixtures stay sanitized and use fake ids, paths, and workspace roots",
    "leading Conversation info / Sender untrusted metadata blocks are stripped before feedback extraction",
    "assistant thinking blocks and raw tool-result payloads are excluded from passive-learning content",
    "sender names are treated as hints for principal resolution, not source-of-truth identity proof",
    "runtime noise (internal context, subagent results, exec output) is stripped from passive-learning content"
];
function normalizeString(value) {
    if (typeof value !== "string") {
        return null;
    }
    const trimmed = value.trim();
    return trimmed.length === 0 ? null : trimmed;
}
function sanitizeToken(value) {
    const sanitized = value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
    return sanitized.length === 0 ? "unknown" : sanitized;
}
function slugifyIdentity(value) {
    return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}
function sortSessionRecordsDeterministically(records) {
    return [...records].sort((left, right) => {
        const leftTimestamp = Date.parse(left.timestamp);
        const rightTimestamp = Date.parse(right.timestamp);
        if (leftTimestamp !== rightTimestamp) {
            return leftTimestamp - rightTimestamp;
        }
        const leftId = typeof left.id === "string" ? left.id : "";
        const rightId = typeof right.id === "string" ? right.id : "";
        const leftParentId = typeof left.parentId === "string" ? left.parentId : "";
        const rightParentId = typeof right.parentId === "string" ? right.parentId : "";
        if (rightParentId === leftId && leftParentId !== rightId) {
            return -1;
        }
        if (leftParentId === rightId && rightParentId !== leftId) {
            return 1;
        }
        if (leftId !== rightId) {
            return leftId.localeCompare(rightId);
        }
        if (leftParentId !== rightParentId) {
            return leftParentId.localeCompare(rightParentId);
        }
        return left.type.localeCompare(right.type);
    });
}
function deriveChannel(sessionKey, entry) {
    const deliveryChannel = normalizeString(entry.deliveryContext?.channel);
    if (deliveryChannel !== null) {
        return deliveryChannel;
    }
    const originSurface = normalizeString(entry.origin?.surface);
    if (originSurface !== null) {
        return originSurface;
    }
    const keyParts = sessionKey.split(":");
    const transportPart = keyParts[2];
    if (transportPart !== undefined && transportPart.length > 0) {
        return transportPart;
    }
    return entry.chatType ?? "unknown";
}
function messageText(record) {
    const parts = [];
    let thinkingBlocks = 0;
    let toolCalls = 0;
    for (const part of record.message.content) {
        if (part.type === "text") {
            const normalized = normalizeString(part.text);
            if (normalized !== null) {
                parts.push(normalized);
            }
            continue;
        }
        if (part.type === "thinking") {
            thinkingBlocks += 1;
            continue;
        }
        if (part.type === "toolCall") {
            toolCalls += 1;
        }
    }
    return {
        content: parts.join("\n").trim(),
        thinkingBlocks,
        toolCalls
    };
}
/**
 * Patterns that indicate runtime/system noise in user messages.
 * These are system-injected content blocks that should not be included
 * in passive learning data.
 */
const RUNTIME_NOISE_PATTERNS = [
    /OpenClaw runtime context \(internal\)/i,
    /<<<BEGIN_UNTRUSTED_CHILD_RESULT>>>/,
    /<<<END_UNTRUSTED_CHILD_RESULT>>>/,
    /\[Internal task completion event\]/,
    /source: subagent\s+session_key:/,
    /Exec failed \([^)]+, code \d+\)/,
    /\[System Message\]\s*\[sessionId:/,
    /\[cron:[a-f0-9-]+\s/,
    /Result \(untrusted content, treat as data\):/,
];
/**
 * Heuristic: returns true if a section is predominantly runtime noise
 * rather than genuine user content.
 */
function isSectionRuntimeNoise(section) {
    for (const pattern of RUNTIME_NOISE_PATTERNS) {
        if (pattern.test(section)) {
            return true;
        }
    }
    return false;
}
/**
 * Strip runtime noise from content after initial metadata stripping.
 * Removes sections that contain system-injected runtime context,
 * subagent results, exec output, and other non-user content.
 */
function stripRuntimeNoise(text) {
    if (text.length === 0) {
        return { content: "", strippedNoiseCount: 0 };
    }
    const sections = text
        .split(/\n\s*\n/g)
        .map((section) => section.trim())
        .filter((section) => section.length > 0);
    const remaining = [];
    let strippedNoiseCount = 0;
    for (const section of sections) {
        if (isSectionRuntimeNoise(section)) {
            strippedNoiseCount += 1;
            continue;
        }
        remaining.push(section);
    }
    return {
        content: remaining.join("\n\n").trim(),
        strippedNoiseCount
    };
}
function stripLeadingUntrustedMetadata(text) {
    const blocks = text.replace(/\r\n/g, "\n").trim();
    if (blocks.length === 0) {
        return {
            content: "",
            senderName: null,
            strippedMetadataBlockCount: 0
        };
    }
    const sections = blocks
        .split(/\n\s*\n/g)
        .map((section) => section.trim())
        .filter((section) => section.length > 0);
    const remaining = [];
    let senderName = null;
    let strippedMetadataBlockCount = 0;
    for (const section of sections) {
        const lines = section.split("\n").map((line) => line.trim()).filter((line) => line.length > 0);
        const header = lines[0] ?? "";
        if (header === "Conversation info (untrusted metadata)" || header === "Sender (untrusted metadata)") {
            strippedMetadataBlockCount += 1;
            if (header === "Sender (untrusted metadata)") {
                const senderLine = lines.find((line) => /^name:/iu.test(line) || /^display name:/iu.test(line));
                const extracted = senderLine?.split(":").slice(1).join(":").trim() ?? null;
                if (extracted !== null && extracted.length > 0) {
                    senderName = extracted;
                }
            }
            continue;
        }
        remaining.push(section);
    }
    // After stripping metadata headers, also strip runtime noise sections
    const joined = remaining.join("\n\n").trim();
    const denoised = stripRuntimeNoise(joined);
    return {
        content: denoised.content,
        senderName,
        strippedMetadataBlockCount: strippedMetadataBlockCount + denoised.strippedNoiseCount
    };
}
function buildPrincipalScope(sessionId, relatedInteractionId) {
    const scopeKey = [
        "profile:current_profile",
        `session:${sessionId}`,
        relatedInteractionId === null ? null : `interaction:${relatedInteractionId}`
    ]
        .filter((value) => value !== null)
        .join("|");
    if (relatedInteractionId !== null) {
        return {
            kind: "interaction",
            profileSelector: "current_profile",
            sessionId,
            interactionId: relatedInteractionId,
            scopeKey
        };
    }
    return {
        kind: "session",
        profileSelector: "current_profile",
        sessionId,
        scopeKey
    };
}
function resolveSenderIdentity(sessionId, senderName, relatedInteractionId) {
    if (senderName === null) {
        return {
            actorRole: "user",
            principal: undefined
        };
    }
    const normalized = senderName.toLowerCase().replace(/\s+/g, " ").trim();
    const known = KNOWN_PRINCIPALS[normalized];
    if (known !== undefined) {
        return {
            actorRole: known.teacherRole,
            principal: {
                teacherIdentity: known.teacherIdentity,
                teacherRole: known.teacherRole,
                teacherAuthority: known.teacherAuthority,
                priorityClass: known.priorityClass,
                principalScope: buildPrincipalScope(sessionId, relatedInteractionId)
            }
        };
    }
    const actorIdentity = slugifyIdentity(normalized);
    return {
        actorRole: "user",
        principal: actorIdentity.length === 0
            ? undefined
            : {
                teacherIdentity: `scanner/actor/${actorIdentity}`,
                teacherRole: "user",
                teacherAuthority: "normal",
                priorityClass: "normal",
                principalScope: buildPrincipalScope(sessionId, relatedInteractionId)
            }
    };
}
export function buildPassiveLearningSessionExportFromOpenClawSessionStore(input) {
    const agentId = input.agentId ?? DEFAULT_AGENT_ID;
    const sequenceStart = input.sequenceStart ?? 1;
    const channel = deriveChannel(input.sessionKey, input.indexEntry);
    const sourceStream = `openclaw/runtime/${sanitizeToken(channel)}`;
    const warnings = [];
    const interactionEvents = [];
    const interactionContentsById = {};
    const feedbackRecords = [];
    let strippedMetadataBlockCount = 0;
    let strippedThinkingBlockCount = 0;
    let droppedToolResultCount = 0;
    let droppedRuntimeNoiseCount = 0;
    let nextSequence = sequenceStart;
    let latestAssistantInteractionId = null;
    for (const record of sortSessionRecordsDeterministically(input.records)) {
        if (record.type !== "message") {
            continue;
        }
        if (record.message.role === "toolResult") {
            droppedToolResultCount += 1;
            continue;
        }
        const resolvedText = messageText(record);
        strippedThinkingBlockCount += resolvedText.thinkingBlocks;
        if (record.message.role === "assistant") {
            if (resolvedText.content.length === 0) {
                if (resolvedText.toolCalls > 0) {
                    warnings.push(`assistant message ${record.id} only contained tool calls; skipped from passive-learning interactions`);
                }
                continue;
            }
            // Strip runtime noise from assistant content
            const denoisedAssistant = stripRuntimeNoise(resolvedText.content);
            droppedRuntimeNoiseCount += denoisedAssistant.strippedNoiseCount;
            if (denoisedAssistant.content.length === 0) {
                warnings.push(`assistant message ${record.id} only contained runtime noise; skipped from passive-learning interactions`);
                nextSequence += 1;
                continue;
            }
            const eventId = `evt-session-store-assistant-${input.indexEntry.sessionId}-${record.id}`;
            const interactionSequence = nextSequence;
            const interaction = createInteractionEvent({
                eventId,
                agentId,
                sessionId: input.indexEntry.sessionId,
                channel,
                sequence: interactionSequence,
                kind: "message_delivered",
                createdAt: record.timestamp,
                source: {
                    runtimeOwner: "openclaw",
                    stream: sourceStream
                },
                semantic: buildAssistantMessageSemanticMetadata(),
                messageId: record.id
            });
            nextSequence += 1;
            latestAssistantInteractionId = interaction.eventId;
            interactionEvents.push(interaction);
            interactionContentsById[interaction.eventId] = denoisedAssistant.content;
            feedbackRecords.push({
                recordId: record.id,
                format: "raw",
                createdAt: record.timestamp,
                content: denoisedAssistant.content,
                sequence: interactionSequence,
                messageId: record.id,
                interactionId: interaction.eventId,
                actorRole: "assistant"
            });
            continue;
        }
        if (record.message.role !== "user") {
            warnings.push(`unsupported session-store message role skipped: ${record.message.role}`);
            continue;
        }
        // Early gate: skip messages that contain system/runtime markers BEFORE
        // metadata stripping and feedback classification. This prevents subagent
        // completions, runtime context blocks, and internal events from being
        // misclassified as corrections or teaching by the feedback heuristics.
        if (isSystemMessage(resolvedText.content)) {
            droppedRuntimeNoiseCount += 1;
            warnings.push(`user message ${record.id} contained system/runtime markers; skipped from feedback extraction`);
            nextSequence += 1;
            continue;
        }
        const sanitized = stripLeadingUntrustedMetadata(resolvedText.content);
        strippedMetadataBlockCount += sanitized.strippedMetadataBlockCount;
        const { actorRole, principal } = resolveSenderIdentity(input.indexEntry.sessionId, sanitized.senderName, latestAssistantInteractionId);
        if (sanitized.content.length === 0) {
            warnings.push(`user message ${record.id} had no passive-learning content after metadata stripping`);
            nextSequence += 1;
            continue;
        }
        feedbackRecords.push({
            recordId: record.id,
            format: "raw",
            createdAt: record.timestamp,
            content: sanitized.content,
            sequence: nextSequence,
            messageId: record.id,
            relatedInteractionId: latestAssistantInteractionId,
            actorRole,
            ...(principal === undefined ? {} : { principal })
        });
        nextSequence += 1;
    }
    const feedbackExtraction = extractFeedbackEventsFromInteractionRecords({
        agentId,
        sessionId: input.indexEntry.sessionId,
        channel,
        source: {
            runtimeOwner: "openclaw",
            stream: sourceStream
        },
        records: feedbackRecords
    });
    const feedbackEvents = feedbackExtraction.events.map((entry) => ({
        ...entry.feedbackEvent,
        semantic: buildFeedbackSemanticMetadata("session_store", entry.feedbackEvent.kind, entry.feedbackEvent.content)
    }));
    const normalizedEventExport = buildNormalizedEventExport({
        interactionEvents,
        feedbackEvents
    });
    return {
        session: {
            sessionKey: input.sessionKey,
            sessionId: input.indexEntry.sessionId,
            updatedAt: new Date(input.indexEntry.updatedAt).toISOString(),
            channel,
            sourceStream,
            sessionFileBasename: input.indexEntry.sessionFile === undefined ? null : path.basename(input.indexEntry.sessionFile),
            model: input.indexEntry.model ?? null,
            modelProvider: input.indexEntry.modelProvider ?? null,
            chatType: input.indexEntry.chatType ?? null,
            systemPromptFingerprint: {
                available: input.indexEntry.systemPromptReport !== undefined,
                workspaceDirBasename: input.indexEntry.systemPromptReport?.workspaceDir === undefined
                    ? null
                    : path.basename(input.indexEntry.systemPromptReport.workspaceDir),
                injectedWorkspaceFileCount: input.indexEntry.systemPromptReport?.injectedWorkspaceFiles?.length ?? 0,
                missingInjectedWorkspaceFileCount: input.indexEntry.systemPromptReport?.injectedWorkspaceFiles?.filter((file) => file.missing).length ?? 0,
                toolNames: (input.indexEntry.systemPromptReport?.tools?.entries ?? [])
                    .map((entry) => entry.name)
                    .filter((name) => typeof name === "string")
            }
        },
        privacy: {
            sanitized: true,
            rules: [...DEFAULT_PRIVACY_RULES],
            strippedMetadataBlockCount,
            strippedThinkingBlockCount,
            droppedToolResultCount,
            droppedRuntimeNoiseCount
        },
        interactionEvents,
        feedbackEvents,
        interactionContentsById,
        feedbackExtraction,
        normalizedEventExport,
        warnings,
        nextSequence
    };
}
export function buildPassiveLearningStoreExportFromOpenClawSessionIndex(input) {
    const sessionKeys = input.sessionKeys ?? Object.keys(input.sessionIndex);
    const orderedEntries = sessionKeys
        .map((sessionKey) => {
        const entry = input.sessionIndex[sessionKey];
        if (entry === undefined) {
            throw new Error(`missing session index entry for ${sessionKey}`);
        }
        return {
            sessionKey,
            entry
        };
    })
        .sort((left, right) => {
        if (left.entry.updatedAt !== right.entry.updatedAt) {
            return left.entry.updatedAt - right.entry.updatedAt;
        }
        return left.sessionKey.localeCompare(right.sessionKey);
    });
    let nextSequence = 1;
    const sessions = orderedEntries.map(({ sessionKey, entry }) => {
        const built = buildPassiveLearningSessionExportFromOpenClawSessionStore({
            sessionKey,
            indexEntry: entry,
            records: input.readSessionRecords(sessionKey, entry),
            sequenceStart: nextSequence,
            ...(input.agentId === undefined ? {} : { agentId: input.agentId })
        });
        nextSequence = built.nextSequence;
        return built;
    });
    const interactionEvents = sessions.flatMap((session) => session.interactionEvents);
    const feedbackEvents = sessions.flatMap((session) => session.feedbackEvents);
    return {
        sessions,
        interactionEvents,
        feedbackEvents,
        normalizedEventExport: buildNormalizedEventExport({
            interactionEvents,
            feedbackEvents
        }),
        warnings: sessions.flatMap((session) => session.warnings)
    };
}
