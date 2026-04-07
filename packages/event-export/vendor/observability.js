import { CONTRACT_IDS, sortNormalizedEvents } from "@openclawbrain/contracts";
function compareEventsByFreshness(left, right) {
    if (left.sequence !== right.sequence) {
        return right.sequence - left.sequence;
    }
    return Date.parse(right.createdAt) - Date.parse(left.createdAt);
}
function isHumanSupervisionEvent(event) {
    return event.contract === CONTRACT_IDS.feedbackEvents || event.kind === "operator_override";
}
function isSelfSupervisionEvent(event) {
    return event.contract === CONTRACT_IDS.interactionEvents && event.kind === "memory_compiled";
}
function uniqueStringsInOrder(values) {
    const seen = new Set();
    const unique = [];
    for (const value of values) {
        if (typeof value !== "string" || value.length === 0 || seen.has(value)) {
            continue;
        }
        seen.add(value);
        unique.push(value);
    }
    return unique;
}
function toSupervisionFreshnessBySource(accumulator) {
    return {
        sourceStream: accumulator.sourceStream,
        eventCount: accumulator.eventCount,
        interactionCount: accumulator.interactionCount,
        feedbackCount: accumulator.feedbackCount,
        humanLabelCount: accumulator.humanLabelCount,
        selfLabelCount: accumulator.selfLabelCount,
        freshestEventId: accumulator.freshestEvent.eventId,
        freshestSequence: accumulator.freshestEvent.sequence,
        freshestCreatedAt: accumulator.freshestEvent.createdAt,
        freshestKind: accumulator.freshestEvent.kind
    };
}
export function describeNormalizedEventExportObservability(normalizedEventExport) {
    const sorted = sortNormalizedEvents([
        ...normalizedEventExport.interactionEvents,
        ...normalizedEventExport.feedbackEvents
    ]);
    const attributedEvents = sorted.filter((event) => event.attribution !== undefined);
    const bySource = new Map();
    let freshestTeacherEvent = null;
    const teacherSources = new Set();
    let humanLabelCount = 0;
    for (const event of sorted) {
        const sourceStream = event.source.stream;
        const existing = bySource.get(sourceStream);
        const freshestEvent = existing === undefined || compareEventsByFreshness(event, existing.freshestEvent) < 0 ? event : existing.freshestEvent;
        bySource.set(sourceStream, {
            sourceStream,
            eventCount: (existing?.eventCount ?? 0) + 1,
            interactionCount: (existing?.interactionCount ?? 0) + (event.contract === CONTRACT_IDS.interactionEvents ? 1 : 0),
            feedbackCount: (existing?.feedbackCount ?? 0) + (event.contract === CONTRACT_IDS.feedbackEvents ? 1 : 0),
            humanLabelCount: (existing?.humanLabelCount ?? 0) + (isHumanSupervisionEvent(event) ? 1 : 0),
            selfLabelCount: (existing?.selfLabelCount ?? 0) + (isSelfSupervisionEvent(event) ? 1 : 0),
            freshestEvent
        });
        if (isHumanSupervisionEvent(event)) {
            humanLabelCount += 1;
            teacherSources.add(sourceStream);
            if (freshestTeacherEvent === null || compareEventsByFreshness(event, freshestTeacherEvent) < 0) {
                freshestTeacherEvent = event;
            }
        }
    }
    return {
        exportDigest: normalizedEventExport.provenance.exportDigest,
        range: { ...normalizedEventExport.range },
        sourceStreams: [...normalizedEventExport.provenance.sourceStreams],
        learningSurface: {
            scanPolicy: normalizedEventExport.provenance.learningSurface.scanPolicy,
            scanSurfaces: [...normalizedEventExport.provenance.learningSurface.scanSurfaces],
            humanLabelCount: normalizedEventExport.provenance.learningSurface.labelHarvest.humanLabels,
            selfLabelCount: normalizedEventExport.provenance.learningSurface.labelHarvest.selfLabels
        },
        supervisionFreshnessBySource: [...bySource.values()]
            .map((accumulator) => toSupervisionFreshnessBySource(accumulator))
            .sort((left, right) => {
            if (left.freshestSequence !== right.freshestSequence) {
                return right.freshestSequence - left.freshestSequence;
            }
            return Date.parse(right.freshestCreatedAt) - Date.parse(left.freshestCreatedAt);
        }),
        teacherFreshness: {
            freshestEventId: freshestTeacherEvent?.eventId ?? null,
            freshestSequence: freshestTeacherEvent?.sequence ?? null,
            freshestCreatedAt: freshestTeacherEvent?.createdAt ?? null,
            freshestKind: freshestTeacherEvent?.kind ?? null,
            sourceStream: freshestTeacherEvent?.source.stream ?? null,
            humanLabelCount,
            sources: [...teacherSources]
        },
        attributionCoverage: {
            totalEventCount: sorted.length,
            attributedEventCount: attributedEvents.length,
            attributedInteractionCount: attributedEvents.filter((event) => event.contract === CONTRACT_IDS.interactionEvents).length,
            attributedFeedbackCount: attributedEvents.filter((event) => event.contract === CONTRACT_IDS.feedbackEvents).length,
            selectionDigestCount: attributedEvents.filter((event) => event.attribution?.selectionDigest !== null).length,
            profileSelectors: uniqueStringsInOrder(attributedEvents.map((event) => event.attribution?.profileSelector)),
            profileIds: uniqueStringsInOrder(attributedEvents.map((event) => event.attribution?.profileId)),
            brainStatuses: uniqueStringsInOrder(attributedEvents.map((event) => event.attribution?.brainStatus)),
            activePackIds: uniqueStringsInOrder(attributedEvents.map((event) => event.attribution?.activePackId)),
            routerIdentities: uniqueStringsInOrder(attributedEvents.map((event) => event.attribution?.routerIdentity))
        }
    };
}
//# sourceMappingURL=observability.js.map