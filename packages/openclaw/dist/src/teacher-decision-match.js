const DEFAULT_MATCH_WINDOW_MS = 60_000;
const OPERATIONAL_DECISION_PATTERNS = [
    /^NO_REPLY$/i,
    /^HEARTBEAT_OK$/i,
    /^read heartbeat\.md if it exists\b/i,
    /^a new session was started via \/new or \/reset\./i,
    /\[cron:[a-f0-9-]+\s/i,
    /\[system message\]\s*\[sessionid:/i,
];

function normalizeOptionalString(value) {
    if (typeof value !== "string") {
        return undefined;
    }
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : undefined;
}

function toTimestamp(value) {
    const normalized = normalizeOptionalString(value);
    if (normalized === undefined) {
        return null;
    }
    const parsed = Date.parse(normalized);
    return Number.isFinite(parsed) ? parsed : null;
}

function buildSessionChannelKey(sessionId, channel) {
    const normalizedSessionId = normalizeOptionalString(sessionId);
    const normalizedChannel = normalizeOptionalString(channel);
    if (normalizedSessionId === undefined || normalizedChannel === undefined) {
        return null;
    }
    return `${normalizedSessionId}|${normalizedChannel}`;
}

function buildCandidateKey(sessionId, channel, createdAt) {
    const sessionChannelKey = buildSessionChannelKey(sessionId, channel);
    const normalizedCreatedAt = normalizeOptionalString(createdAt);
    if (sessionChannelKey === null || normalizedCreatedAt === undefined) {
        return null;
    }
    return `${sessionChannelKey}|${normalizedCreatedAt}`;
}

function buildSelectionDigestKey(selectionDigest, activePackGraphChecksum) {
    const normalizedSelectionDigest = normalizeOptionalString(selectionDigest);
    const normalizedGraphChecksum = normalizeOptionalString(activePackGraphChecksum);
    if (normalizedSelectionDigest === undefined || normalizedGraphChecksum === undefined) {
        return null;
    }
    return `${normalizedGraphChecksum}|${normalizedSelectionDigest}`;
}

function toRecord(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : null;
}

function readInteractionExactDecisionRecordId(interaction) {
    return normalizeOptionalString(interaction?.serveDecisionRecordId)
        ?? normalizeOptionalString(toRecord(interaction?.routeMetadata)?.serveDecisionRecordId)
        ?? normalizeOptionalString(toRecord(interaction?.decisionProvenance)?.serveDecisionRecordId)
        ?? normalizeOptionalString(toRecord(interaction?.metadata)?.serveDecisionRecordId)
        ?? undefined;
}

function readInteractionSelectionDigest(interaction) {
    return normalizeOptionalString(interaction?.selectionDigest)
        ?? normalizeOptionalString(toRecord(interaction?.routeMetadata)?.selectionDigest)
        ?? normalizeOptionalString(toRecord(interaction?.decisionProvenance)?.selectionDigest)
        ?? normalizeOptionalString(toRecord(interaction?.metadata)?.selectionDigest)
        ?? undefined;
}

function readInteractionActivePackGraphChecksum(interaction) {
    return normalizeOptionalString(interaction?.activePackGraphChecksum)
        ?? normalizeOptionalString(toRecord(interaction?.routeMetadata)?.activePackGraphChecksum)
        ?? normalizeOptionalString(toRecord(interaction?.decisionProvenance)?.activePackGraphChecksum)
        ?? normalizeOptionalString(toRecord(interaction?.metadata)?.activePackGraphChecksum)
        ?? undefined;
}

function readInteractionExplicitTurnCompileEventId(interaction) {
    return normalizeOptionalString(interaction?.turnCompileEventId)
        ?? normalizeOptionalString(toRecord(interaction?.routeMetadata)?.turnCompileEventId)
        ?? normalizeOptionalString(toRecord(interaction?.decisionProvenance)?.turnCompileEventId)
        ?? normalizeOptionalString(toRecord(interaction?.metadata)?.turnCompileEventId)
        ?? undefined;
}

function buildDecisionTimestamps(decision) {
    const timestamps = [];
    const turnCreatedAt = toTimestamp(decision.turnCreatedAt);
    const recordedAt = toTimestamp(decision.recordedAt);
    if (turnCreatedAt !== null) {
        timestamps.push(turnCreatedAt);
    }
    if (recordedAt !== null && !timestamps.includes(recordedAt)) {
        timestamps.push(recordedAt);
    }
    return timestamps;
}

function isOperationalDecision(decision) {
    const userMessage = normalizeOptionalString(decision.userMessage);
    if (userMessage === undefined) {
        return true;
    }
    return OPERATIONAL_DECISION_PATTERNS.some((pattern) => pattern.test(userMessage));
}

function selectNearestDecision(entries, interactionAt, maxTimeDeltaMs) {
    const candidates = entries
        .map((entry) => {
        const deltas = entry.timestamps.map((timestamp) => Math.abs(timestamp - interactionAt));
        const bestDelta = deltas.length === 0 ? null : Math.min(...deltas);
        return bestDelta === null || bestDelta > maxTimeDeltaMs
            ? null
            : {
                decision: entry.decision,
                deltaMs: bestDelta,
                recordedAt: toTimestamp(entry.decision.recordedAt) ?? 0,
            };
    })
        .filter((entry) => entry !== null)
        .sort((left, right) => {
        if (left.deltaMs !== right.deltaMs) {
            return left.deltaMs - right.deltaMs;
        }
        return right.recordedAt - left.recordedAt;
    });
    const best = candidates[0] ?? null;
    const runnerUp = candidates[1] ?? null;
    if (best === null) {
        return null;
    }
    if (runnerUp !== null && runnerUp.deltaMs === best.deltaMs && runnerUp.decision !== best.decision) {
        return null;
    }
    return best.decision;
}

export function createServeTimeDecisionMatcher(decisions, options = {}) {
    const maxTimeDeltaMs = Number.isInteger(options.maxTimeDeltaMs) && options.maxTimeDeltaMs >= 0
        ? options.maxTimeDeltaMs
        : DEFAULT_MATCH_WINDOW_MS;
    const decisionsByRecordId = new Map();
    const decisionsBySelectionDigest = new Map();
    const ambiguousSelectionDigests = new Set();
    const decisionsByTurnCompileEventId = new Map();
    const ambiguousTurnCompileEventIds = new Set();
    const fallbackDecisions = new Map();
    const ambiguousFallbackDecisionKeys = new Set();
    const decisionsBySessionChannel = new Map();
    const globalFallbackDecisions = [];

    for (const decision of [...decisions].sort((left, right) => Date.parse(right.recordedAt) - Date.parse(left.recordedAt))) {
        const userMessage = normalizeOptionalString(decision.userMessage);
        if (userMessage === undefined) {
            continue;
        }
        const decisionRecordId = normalizeOptionalString(decision.recordId);
        if (decisionRecordId !== undefined && !decisionsByRecordId.has(decisionRecordId)) {
            decisionsByRecordId.set(decisionRecordId, decision);
        }
        const selectionDigestKey = buildSelectionDigestKey(decision.selectionDigest, decision.activePackGraphChecksum);
        if (selectionDigestKey !== null) {
            if (decisionsBySelectionDigest.has(selectionDigestKey)) {
                decisionsBySelectionDigest.delete(selectionDigestKey);
                ambiguousSelectionDigests.add(selectionDigestKey);
            }
            else if (!ambiguousSelectionDigests.has(selectionDigestKey)) {
                decisionsBySelectionDigest.set(selectionDigestKey, decision);
            }
        }
        const turnCompileEventId = normalizeOptionalString(decision.turnCompileEventId);
        if (turnCompileEventId !== undefined) {
            if (decisionsByTurnCompileEventId.has(turnCompileEventId)) {
                decisionsByTurnCompileEventId.delete(turnCompileEventId);
                ambiguousTurnCompileEventIds.add(turnCompileEventId);
            }
            else if (!ambiguousTurnCompileEventIds.has(turnCompileEventId)) {
                decisionsByTurnCompileEventId.set(turnCompileEventId, decision);
            }
        }
        for (const candidateKey of [
            buildCandidateKey(decision.sessionId, decision.channel, decision.turnCreatedAt),
            buildCandidateKey(decision.sessionId, decision.channel, decision.recordedAt),
        ]) {
            if (candidateKey !== null) {
                if (fallbackDecisions.has(candidateKey)) {
                    fallbackDecisions.delete(candidateKey);
                    ambiguousFallbackDecisionKeys.add(candidateKey);
                }
                else if (!ambiguousFallbackDecisionKeys.has(candidateKey)) {
                    fallbackDecisions.set(candidateKey, decision);
                }
            }
        }
        const sessionChannelKey = buildSessionChannelKey(decision.sessionId, decision.channel);
        if (sessionChannelKey === null) {
            if (!isOperationalDecision(decision)) {
                globalFallbackDecisions.push({
                    decision,
                    timestamps: buildDecisionTimestamps(decision),
                });
            }
            continue;
        }
        const indexedEntry = {
            decision,
            timestamps: buildDecisionTimestamps(decision),
            operational: isOperationalDecision(decision),
        };
        const indexed = decisionsBySessionChannel.get(sessionChannelKey) ?? [];
        indexed.push(indexedEntry);
        decisionsBySessionChannel.set(sessionChannelKey, indexed);
        if (!indexedEntry.operational) {
            globalFallbackDecisions.push(indexedEntry);
        }
    }

    return (interaction) => {
        const decisionRecordId = readInteractionExactDecisionRecordId(interaction);
        if (decisionRecordId !== undefined) {
            return decisionsByRecordId.get(decisionRecordId) ?? null;
        }
        const interactionSelectionDigest = readInteractionSelectionDigest(interaction);
        const interactionGraphChecksum = readInteractionActivePackGraphChecksum(interaction);
        const selectionDigestKey = buildSelectionDigestKey(interactionSelectionDigest, interactionGraphChecksum);
        if (selectionDigestKey !== null) {
            if (ambiguousSelectionDigests.has(selectionDigestKey)) {
                return null;
            }
            return decisionsBySelectionDigest.get(selectionDigestKey) ?? null;
        }
        if (interactionSelectionDigest !== undefined || interactionGraphChecksum !== undefined) {
            return null;
        }
        const explicitTurnCompileEventId = readInteractionExplicitTurnCompileEventId(interaction);
        if (explicitTurnCompileEventId !== undefined) {
            if (ambiguousTurnCompileEventIds.has(explicitTurnCompileEventId)) {
                return null;
            }
            return decisionsByTurnCompileEventId.get(explicitTurnCompileEventId) ?? null;
        }
        const softTurnCompileEventId = normalizeOptionalString(interaction.eventId);
        const exact = softTurnCompileEventId === undefined || ambiguousTurnCompileEventIds.has(softTurnCompileEventId)
            ? undefined
            : decisionsByTurnCompileEventId.get(softTurnCompileEventId);
        if (exact !== undefined) {
            return exact;
        }
        const exactFallbackKey = buildCandidateKey(interaction.sessionId, interaction.channel, interaction.createdAt);
        if (exactFallbackKey !== null) {
            if (ambiguousFallbackDecisionKeys.has(exactFallbackKey)) {
                return null;
            }
            const fallback = fallbackDecisions.get(exactFallbackKey);
            if (fallback !== undefined) {
                return fallback;
            }
        }
        const interactionAt = toTimestamp(interaction.createdAt);
        const sessionChannelKey = buildSessionChannelKey(interaction.sessionId, interaction.channel);
        if (interactionAt === null) {
            return null;
        }
        if (sessionChannelKey !== null) {
            const sessionMatch = selectNearestDecision((decisionsBySessionChannel.get(sessionChannelKey) ?? []).filter((entry) => entry.operational !== true), interactionAt, maxTimeDeltaMs);
            if (sessionMatch !== null) {
                return sessionMatch;
            }
        }
        return selectNearestDecision(globalFallbackDecisions, interactionAt, maxTimeDeltaMs);
    };
}
