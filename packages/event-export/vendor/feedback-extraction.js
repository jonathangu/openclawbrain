import { checksumJsonPayload, createFeedbackEvent } from "@openclawbrain/contracts";
const HUMAN_FEEDBACK_ROLES = new Set(["principal", "admin", "operator", "user", "unknown"]);
const INTERACTION_TARGET_ROLES = new Set(["assistant", "system"]);
const STRONG_SUPPRESSION_PATTERNS = [
    /\b(?:stop|don't|do not|never|avoid|skip|block|disable|suppress|mute|silence|remove|drop|prune)\b.{0,48}\b(?:send|sending|show|surface|emit|open|create|reply|message|messages|case|ticket|path|route|context|alert|alerts)\b/u,
    /\b(?:prune|remove|drop|disable)\b.{0,32}\b(?:stale|duplicate|dead|broken)\b/u
];
const EXPLICIT_CORRECTION_PATTERNS = [
    /^no(?:[\s,!.]|$)/u,
    /\b(?:wrong|incorrect|not right|that's not right|that is not right|correction|fix this|fix that|not quite)\b/u,
    /\b(?:why did(?:n't| not) you|why would you)\b/u,
    /\b(?:do|say|use|ask|tell)\b.{0,32}\binstead\b/u,
    /\binstead of\b/u,
    /\bnot\b.{0,24}\bbut\b/u,
    // Implicit corrections: rephrasing, clarifying intent
    /\bactually\b.{0,48}\b(?:i\s+(?:want|need|meant|asked)|it\s+should|that\s+should|please)\b/u,
    /\bthat'?s?\s+not\s+what\s+i\b/u,
    /\bi\s+meant\b/u,
    /\blet\s+me\s+rephrase\b/u,
    // Retry / redo requests
    /\b(?:try\s+again|redo\s+(?:this|that|it)|start\s+over)\b/u
];
const NEGATIVE_CORRECTION_PATTERNS = [
    /\b(?:avoid|don't|do not|never)\b.{0,48}\b(?:claim|promise|tell|say|assume|mention|reply|draft)\b/u
];
const EXPLICIT_TEACHING_PATTERNS = [
    /\b(?:remember to|make sure(?: to)?|be sure(?: to)?|next time)\b/u,
    /\b(?:always|whenever)\b/u,
    /\b(?:ask for|include|tell (?:the )?(?:customer|user)|note that|explain that|prefer|use|keep)\b/u,
    /\b(?:if|when)\b.{0,64}\b(?:then|before|after|ask|tell|include|note|explain|keep|use|avoid)\b/u,
    /\b(?:send|ask|route|give|walk)\b.{0,40}\b(?:these|the following|this)\b.{0,24}\b(?:questions?|questionnaire|question list|checklist)\b/u,
    /\b(?:ask|send)\b.{0,16}\beagle\b.{0,32}\b(?:questions?|questionnaire)\b/u,
    /\bone question at a time\b/u,
    // Style / tone preferences
    /\btoo\s+(?:formal|verbose|long|short|brief|wordy|casual|technical|vague|detailed)\b/u,
    /\bbe\s+(?:more|less)\s+\w+/u,
    // Explicit preference statements
    /\bi'?d\s+(?:rather|prefer)\b/u
];
const POSITIVE_SIGNAL_PATTERNS = [
    /\b(?:approved|approve(?:d)?|looks good|ship it|exactly right|that works|good catch|yes[, ]+(?:that's|that is)\s+right|correct)\b/u
];
const QUESTION_PREFIX_PATTERN = /^(?:what|when|why|how|can|could|should|would|is|are|do|did)\b/u;
const IMPLICIT_POSITIVE_SOURCE_SUFFIX = "/implicit-positive";
const IMPLICIT_POSITIVE_MIN_WORD_COUNT = 3;
const IMPLICIT_POSITIVE_MIN_LENGTH = 12;
const ACKNOWLEDGEMENT_OR_FILLER_PATTERNS = [
    /^(?:ok(?:ay)?|k|kk|sure|yep|yeah|yes|got it|gotcha|understood|makes sense|cool|nice|perfect|awesome|fine|alright|all right)[.! ]*$/u,
    /^(?:sounds good|sounds right|that works|works for me|sgtm)[.! ]*$/u,
    /^(?:thanks|thank you|thx|ty|much appreciated)(?:\s+(?:again|for that|for the help))?[.! ]*$/u,
    /^(?:lol|haha|ha|hmm|uh|um|huh)[.! ]*$/u
];
const PRODUCTIVE_CONTINUATION_PATTERNS = [
    /^(?:now|next|then|also|please|go ahead(?: and)?|continue|proceed|let'?s|lets)\b/u,
    /^(?:add|adjust|apply|build|call|change|check|compare|compile|create|debug|draft|drop|file|fix|handle|implement|keep|message|move|open|prepare|pull|push|refactor|reply|rerun|review|run|send|ship|start|test|update|use|verify|wire|write)\b/u,
    /\b(?:please|need to|let'?s|lets|next|now|then|also|go ahead)\b/u
];
const ASSISTANT_ADJACENT_DIRECTIVE_PATTERNS = [
    /^(?:please\s+)?(?:ask|confirm|verify|check|include|keep|use|mention|note|explain|wait|hold|avoid)\b/u,
    /^(?:also|and|then)\s+(?:ask|confirm|verify|check|include|keep|use|mention|note|explain)\b/u,
    /^(?:you\s+)?still\s+need\s+to\b/u
];
function normalizeContent(value) {
    return value.replace(/\s+/gu, " ").trim();
}
function hasQuestionShape(value) {
    return value.endsWith("?") || QUESTION_PREFIX_PATTERN.test(value);
}
function explicitPrincipalFeedback(record) {
    return (record.actorRole === "principal" ||
        record.principal?.teacherRole === "principal" ||
        record.principal?.teacherAuthority === "binding");
}
function recordActorRole(record) {
    return record.actorRole ?? record.principal?.teacherRole ?? "unknown";
}
function compareRecordEntries(left, right) {
    const leftSequence = left.record.sequence;
    const rightSequence = right.record.sequence;
    if (leftSequence !== undefined && leftSequence !== null && rightSequence !== undefined && rightSequence !== null && leftSequence !== rightSequence) {
        return leftSequence - rightSequence;
    }
    if (left.record.createdAt !== right.record.createdAt) {
        return left.record.createdAt.localeCompare(right.record.createdAt);
    }
    return left.index - right.index;
}
function latestRelatedInteractionId(entries, currentIndex) {
    for (let index = currentIndex - 1; index >= 0; index -= 1) {
        const candidate = entries[index]?.record;
        if (candidate === undefined) {
            continue;
        }
        if (!INTERACTION_TARGET_ROLES.has(recordActorRole(candidate))) {
            continue;
        }
        const interactionId = candidate.interactionId?.trim();
        if (interactionId !== undefined && interactionId.length > 0) {
            return interactionId;
        }
    }
    return null;
}
function deterministicExtractedFeedbackEventId(input) {
    const digest = checksumJsonPayload({
        agentId: input.agentId,
        sessionId: input.sessionId,
        channel: input.channel,
        source: input.source,
        recordId: input.record.recordId,
        format: input.record.format,
        createdAt: input.record.createdAt,
        sequence: input.record.sequence ?? null,
        kind: input.kind,
        content: normalizeContent(input.record.content),
        messageId: input.record.messageId ?? null,
        interactionId: input.record.interactionId ?? null,
        relatedInteractionId: input.relatedInteractionId
    })
        .replace(/^sha256-/u, "")
        .slice(0, 16);
    return `evt-feedback-extract-${digest}`;
}
function cueMatches(value, patterns) {
    return patterns.some((pattern) => pattern.test(value));
}
function wordCount(value) {
    return value.split(/\s+/u).filter((token) => token.length > 0).length;
}
function looksLikeAcknowledgementOrFiller(value) {
    return cueMatches(value, ACKNOWLEDGEMENT_OR_FILLER_PATTERNS);
}
function looksLikeProductiveContinuation(value) {
    return cueMatches(value, PRODUCTIVE_CONTINUATION_PATTERNS);
}
/**
 * Runtime/system message markers that should never be classified as feedback.
 * These are distinctive strings injected by the OpenClaw runtime, subagent
 * completion events, or internal task machinery — not genuine user content.
 */
const SYSTEM_MESSAGE_MARKERS = [
    /openclaw runtime context/i,
    /<<<BEGIN_UNTRUSTED_CHILD_RESULT>>>/,
    /<<<END_UNTRUSTED_CHILD_RESULT>>>/,
    /\[internal task completion event\]/i,
    /source:\s*subagent\s+session_key:/,
    /exec completed\b/i,
    /exec failed \([^)]+, code \d+\)/i,
    /\[system message\]\s*\[sessionid:/i,
    /\[cron:[a-f0-9-]+\s/,
    /result \(untrusted content, treat as data\):/i,
    /^done:\s+\w+\s+/i,
];
/**
 * Returns true if the content looks like a system/runtime message
 * rather than genuine human feedback. Used as an early gate to
 * prevent misclassification of subagent completions, runtime context
 * blocks, and internal events as corrections or teaching.
 */
export function isSystemMessage(content) {
    for (const marker of SYSTEM_MESSAGE_MARKERS) {
        if (marker.test(content)) {
            return true;
        }
    }
    return false;
}
export function classifyFeedbackSignalContent(content, context = {}) {
    const normalized = normalizeContent(content).toLowerCase();
    if (normalized.length === 0) {
        return null;
    }
    if (isSystemMessage(content) || isSystemMessage(normalized)) {
        return null;
    }
    const suppression = cueMatches(normalized, STRONG_SUPPRESSION_PATTERNS);
    const explicitCorrection = cueMatches(normalized, EXPLICIT_CORRECTION_PATTERNS);
    const negativeCorrection = cueMatches(normalized, NEGATIVE_CORRECTION_PATTERNS);
    const teaching = cueMatches(normalized, EXPLICIT_TEACHING_PATTERNS);
    const approval = cueMatches(normalized, POSITIVE_SIGNAL_PATTERNS);
    const isQuestion = hasQuestionShape(normalized);
    if (isQuestion && !suppression && !explicitCorrection && !negativeCorrection && !approval && !teaching) {
        return null;
    }
    const principalFeedback = explicitPrincipalFeedback(context);
    const cues = [];
    const notes = [];
    let kind = null;
    let confidence = 0;
    if (suppression) {
        kind = "suppression";
        confidence = 0.88;
        cues.push("suppression_cue");
        notes.push("matched clear suppression verb tied to emitted behavior");
    }
    else if (explicitCorrection) {
        kind = "correction";
        confidence = 0.9;
        cues.push("explicit_correction_cue");
        notes.push("matched explicit correction language");
    }
    else if (negativeCorrection) {
        kind = "correction";
        confidence = 0.8;
        cues.push("negative_corrective_signal");
        notes.push("matched negative corrective language");
    }
    else if (teaching) {
        kind = "teaching";
        confidence = 0.76;
        cues.push("explicit_teaching_cue");
        notes.push("matched explicit teaching or process guidance");
    }
    else if (approval) {
        kind = "approval";
        confidence = 0.64;
        cues.push("positive_signal");
        notes.push("matched explicit approval language");
    }
    else {
        return null;
    }
    if (principalFeedback) {
        cues.unshift("explicit_principal_feedback");
        if (kind === "correction") {
            cues.push("explicit_principal_correction");
        }
        if (kind === "teaching") {
            cues.push("explicit_principal_teaching");
        }
        confidence = Math.min(0.99, confidence + 0.08);
        notes.push("principal-authored feedback received priority lift");
    }
    if ((explicitCorrection || negativeCorrection) && teaching) {
        notes.push("correction cues outranked broader teaching cues");
    }
    else if (teaching && approval) {
        notes.push("teaching cues outranked weaker approval cues");
    }
    const priority = principalFeedback && kind !== "approval"
        ? "highest"
        : kind === "correction" || kind === "suppression"
            ? "high"
            : kind === "teaching"
                ? "normal"
                : "low";
    return {
        kind,
        confidence,
        priority,
        cues,
        notes
    };
}
function classifyImplicitPositiveContinuation(entries, currentIndex) {
    const current = entries[currentIndex]?.record;
    const previous = entries[currentIndex - 1]?.record;
    if (current === undefined || previous === undefined) {
        return null;
    }
    if (recordActorRole(previous) !== "assistant") {
        return null;
    }
    const previousInteractionId = previous.interactionId?.trim();
    const currentRelatedInteractionId = current.relatedInteractionId?.trim();
    if (previousInteractionId !== undefined &&
        previousInteractionId.length > 0 &&
        currentRelatedInteractionId !== undefined &&
        currentRelatedInteractionId.length > 0 &&
        previousInteractionId !== currentRelatedInteractionId) {
        return null;
    }
    const normalized = normalizeContent(current.content).toLowerCase();
    if (normalized.length === 0) {
        return null;
    }
    if (isSystemMessage(current.content) || isSystemMessage(normalized)) {
        return null;
    }
    if (classifyFeedbackSignalContent(current.content, current) !== null) {
        return null;
    }
    if (hasQuestionShape(normalized)) {
        return null;
    }
    if (looksLikeAcknowledgementOrFiller(normalized)) {
        return null;
    }
    if (normalized.length < IMPLICIT_POSITIVE_MIN_LENGTH || wordCount(normalized) < IMPLICIT_POSITIVE_MIN_WORD_COUNT) {
        return null;
    }
    if (!looksLikeProductiveContinuation(normalized)) {
        return null;
    }
    return {
        kind: "approval",
        confidence: 0.34,
        priority: "low",
        cues: ["implicit_positive_continuation"],
        notes: ["immediately followed an assistant turn with a non-feedback productive continuation"]
    };
}
function classifyAssistantAdjacentDirective(entries, currentIndex) {
    const current = entries[currentIndex]?.record;
    const previous = entries[currentIndex - 1]?.record;
    if (current === undefined || previous === undefined) {
        return null;
    }
    if (recordActorRole(previous) !== "assistant") {
        return null;
    }
    const previousInteractionId = previous.interactionId?.trim();
    const currentRelatedInteractionId = current.relatedInteractionId?.trim();
    if (previousInteractionId !== undefined &&
        previousInteractionId.length > 0 &&
        currentRelatedInteractionId !== undefined &&
        currentRelatedInteractionId.length > 0 &&
        previousInteractionId !== currentRelatedInteractionId) {
        return null;
    }
    const normalized = normalizeContent(current.content).toLowerCase();
    if (normalized.length === 0) {
        return null;
    }
    if (isSystemMessage(current.content) || isSystemMessage(normalized)) {
        return null;
    }
    if (classifyFeedbackSignalContent(current.content, current) !== null) {
        return null;
    }
    if (hasQuestionShape(normalized) || looksLikeAcknowledgementOrFiller(normalized)) {
        return null;
    }
    if (!cueMatches(normalized, ASSISTANT_ADJACENT_DIRECTIVE_PATTERNS)) {
        return null;
    }
    if (/^(?:you\s+)?still\s+need\s+to\b/u.test(normalized)) {
        return {
            kind: "correction",
            confidence: 0.72,
            priority: "high",
            cues: ["assistant_adjacent_directive"],
            notes: ["assistant-adjacent directive called out a missed requirement in the prior reply"]
        };
    }
    return {
        kind: "teaching",
        confidence: 0.62,
        priority: "normal",
        cues: ["assistant_adjacent_directive"],
        notes: ["assistant-adjacent directive supplied reusable instruction for the prior reply"]
    };
}
function feedbackEventSourceForClassification(source, classification) {
    if (!classification.cues.includes("implicit_positive_continuation")) {
        return {
            runtimeOwner: source.runtimeOwner,
            stream: source.stream
        };
    }
    return {
        runtimeOwner: source.runtimeOwner,
        stream: source.stream.endsWith(IMPLICIT_POSITIVE_SOURCE_SUFFIX) ? source.stream : `${source.stream}${IMPLICIT_POSITIVE_SOURCE_SUFFIX}`
    };
}
export function extractFeedbackEventsFromInteractionRecords(input) {
    const entries = input.records.map((record, index) => ({
        index,
        record: {
            ...record,
            content: normalizeContent(record.content)
        }
    })).sort(compareRecordEntries);
    const events = [];
    const skippedRecordIds = [];
    for (const [index, entry] of entries.entries()) {
        const actorRole = recordActorRole(entry.record);
        if (!HUMAN_FEEDBACK_ROLES.has(actorRole) && entry.record.principal === undefined) {
            skippedRecordIds.push(entry.record.recordId);
            continue;
        }
        const classification = classifyFeedbackSignalContent(entry.record.content, entry.record) ??
            classifyAssistantAdjacentDirective(entries, index) ??
            classifyImplicitPositiveContinuation(entries, index);
        if (classification === null) {
            skippedRecordIds.push(entry.record.recordId);
            continue;
        }
        const feedbackSource = feedbackEventSourceForClassification(input.source, classification);
        const relatedInteractionId = entry.record.relatedInteractionId?.trim() ??
            latestRelatedInteractionId(entries, index) ??
            input.defaultRelatedInteractionId?.trim() ??
            null;
        const feedbackEvent = createFeedbackEvent({
            eventId: deterministicExtractedFeedbackEventId({
                agentId: input.agentId,
                sessionId: input.sessionId,
                channel: input.channel,
                source: feedbackSource,
                record: entry.record,
                kind: classification.kind,
                relatedInteractionId
            }),
            agentId: input.agentId,
            sessionId: input.sessionId,
            channel: input.channel,
            sequence: entry.record.sequence ?? index,
            kind: classification.kind,
            createdAt: entry.record.createdAt,
            source: {
                runtimeOwner: feedbackSource.runtimeOwner,
                stream: feedbackSource.stream
            },
            content: entry.record.content,
            ...(entry.record.messageId !== undefined && entry.record.messageId !== null ? { messageId: entry.record.messageId } : {}),
            ...(relatedInteractionId !== null ? { relatedInteractionId } : {}),
            ...(entry.record.principal !== undefined ? { principal: entry.record.principal } : {})
        });
        events.push({
            feedbackEvent,
            extraction: {
                ...classification,
                sourceRecordId: entry.record.recordId,
                sourceFormat: entry.record.format
            }
        });
    }
    return {
        runtimeOwner: input.source.runtimeOwner,
        inputRecordCount: input.records.length,
        extractedCount: events.length,
        skippedRecordIds,
        events
    };
}
function uniqueValue(values) {
    const unique = [...new Set(values)];
    if (unique.length !== 1) {
        throw new Error("normalized interaction extraction requires one agentId/sessionId/channel/source stream");
    }
    return unique[0];
}
export function buildFeedbackExtractionRecordsFromNormalizedInteractions(input) {
    return input.interactionEvents.flatMap((event) => {
        const content = normalizeContent(input.contentsByInteractionId[event.eventId] ?? "");
        if (content.length === 0) {
            return [];
        }
        const principal = input.principalsByInteractionId?.[event.eventId] ?? event.principal;
        return [{
                recordId: event.eventId,
                format: "normalized",
                createdAt: event.createdAt,
                content,
                sequence: event.sequence,
                messageId: event.messageId ?? null,
                interactionId: event.eventId,
                relatedInteractionId: input.relatedInteractionIdsByInteractionId?.[event.eventId] ?? null,
                actorRole: input.actorRolesByInteractionId?.[event.eventId] ?? "unknown",
                ...(principal === undefined ? {} : { principal })
            }];
    });
}
export function extractFeedbackEventsFromNormalizedInteractions(input) {
    if (input.interactionEvents.length === 0) {
        return {
            runtimeOwner: "openclaw",
            inputRecordCount: 0,
            extractedCount: 0,
            skippedRecordIds: [],
            events: []
        };
    }
    const records = buildFeedbackExtractionRecordsFromNormalizedInteractions(input);
    return extractFeedbackEventsFromInteractionRecords({
        agentId: uniqueValue(input.interactionEvents.map((event) => event.agentId)),
        sessionId: uniqueValue(input.interactionEvents.map((event) => event.sessionId)),
        channel: uniqueValue(input.interactionEvents.map((event) => event.channel)),
        source: {
            runtimeOwner: uniqueValue(input.interactionEvents.map((event) => event.source.runtimeOwner)),
            stream: uniqueValue(input.interactionEvents.map((event) => event.source.stream))
        },
        records
    });
}
