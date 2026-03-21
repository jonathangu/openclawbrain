const DEFAULT_MATCH_WINDOW_MS = 30_000;

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

export function createServeTimeDecisionMatcher(decisions, options = {}) {
    const maxTimeDeltaMs = Number.isInteger(options.maxTimeDeltaMs) && options.maxTimeDeltaMs >= 0
        ? options.maxTimeDeltaMs
        : DEFAULT_MATCH_WINDOW_MS;
    const exactDecisions = new Map();
    const fallbackDecisions = new Map();
    const decisionsBySessionChannel = new Map();

    for (const decision of [...decisions].sort((left, right) => Date.parse(right.recordedAt) - Date.parse(left.recordedAt))) {
        const userMessage = normalizeOptionalString(decision.userMessage);
        if (userMessage === undefined) {
            continue;
        }
        const turnCompileEventId = normalizeOptionalString(decision.turnCompileEventId);
        if (turnCompileEventId !== undefined && !exactDecisions.has(turnCompileEventId)) {
            exactDecisions.set(turnCompileEventId, decision);
        }
        for (const candidateKey of [
            buildCandidateKey(decision.sessionId, decision.channel, decision.turnCreatedAt),
            buildCandidateKey(decision.sessionId, decision.channel, decision.recordedAt),
        ]) {
            if (candidateKey !== null && !fallbackDecisions.has(candidateKey)) {
                fallbackDecisions.set(candidateKey, decision);
            }
        }
        const sessionChannelKey = buildSessionChannelKey(decision.sessionId, decision.channel);
        if (sessionChannelKey === null) {
            continue;
        }
        const indexed = decisionsBySessionChannel.get(sessionChannelKey) ?? [];
        indexed.push({
            decision,
            timestamps: buildDecisionTimestamps(decision),
        });
        decisionsBySessionChannel.set(sessionChannelKey, indexed);
    }

    return (interaction) => {
        const exact = exactDecisions.get(interaction.eventId);
        if (exact !== undefined) {
            return exact;
        }
        const exactFallbackKey = buildCandidateKey(interaction.sessionId, interaction.channel, interaction.createdAt);
        if (exactFallbackKey !== null) {
            const fallback = fallbackDecisions.get(exactFallbackKey);
            if (fallback !== undefined) {
                return fallback;
            }
        }
        const interactionAt = toTimestamp(interaction.createdAt);
        const sessionChannelKey = buildSessionChannelKey(interaction.sessionId, interaction.channel);
        if (interactionAt === null || sessionChannelKey === null) {
            return null;
        }
        const candidates = (decisionsBySessionChannel.get(sessionChannelKey) ?? [])
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
    };
}
