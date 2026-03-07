import {
  CONTRACT_IDS,
  sortNormalizedEvents,
  type FeedbackEventKind,
  type InteractionEventKind,
  type NormalizedEventExportV1,
  type NormalizedEventRangeV1,
  type NormalizedEventV1
} from "@openclawbrain/contracts";

export interface SupervisionFreshnessBySource {
  sourceStream: string;
  eventCount: number;
  interactionCount: number;
  feedbackCount: number;
  humanLabelCount: number;
  selfLabelCount: number;
  freshestEventId: string;
  freshestSequence: number;
  freshestCreatedAt: string;
  freshestKind: FeedbackEventKind | InteractionEventKind;
}

export interface TeacherFreshness {
  freshestEventId: string | null;
  freshestSequence: number | null;
  freshestCreatedAt: string | null;
  freshestKind: FeedbackEventKind | InteractionEventKind | null;
  sourceStream: string | null;
  humanLabelCount: number;
  sources: string[];
}

export interface NormalizedEventExportObservabilityReport {
  exportDigest: string;
  range: NormalizedEventRangeV1;
  sourceStreams: string[];
  supervisionFreshnessBySource: SupervisionFreshnessBySource[];
  teacherFreshness: TeacherFreshness;
}

interface SourceAccumulator {
  sourceStream: string;
  eventCount: number;
  interactionCount: number;
  feedbackCount: number;
  humanLabelCount: number;
  selfLabelCount: number;
  freshestEvent: NormalizedEventV1;
}

function compareEventsByFreshness(left: NormalizedEventV1, right: NormalizedEventV1): number {
  if (left.sequence !== right.sequence) {
    return right.sequence - left.sequence;
  }

  return Date.parse(right.createdAt) - Date.parse(left.createdAt);
}

function isHumanSupervisionEvent(event: NormalizedEventV1): boolean {
  return event.contract === CONTRACT_IDS.feedbackEvents || event.kind === "operator_override";
}

function isSelfSupervisionEvent(event: NormalizedEventV1): boolean {
  return event.contract === CONTRACT_IDS.interactionEvents && event.kind === "memory_compiled";
}

function toSupervisionFreshnessBySource(accumulator: SourceAccumulator): SupervisionFreshnessBySource {
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

export function describeNormalizedEventExportObservability(
  normalizedEventExport: NormalizedEventExportV1
): NormalizedEventExportObservabilityReport {
  const sorted = sortNormalizedEvents([
    ...normalizedEventExport.interactionEvents,
    ...normalizedEventExport.feedbackEvents
  ]);
  const bySource = new Map<string, SourceAccumulator>();
  let freshestTeacherEvent: NormalizedEventV1 | null = null;
  const teacherSources = new Set<string>();
  let humanLabelCount = 0;

  for (const event of sorted) {
    const sourceStream = event.source.stream;
    const existing = bySource.get(sourceStream);
    const freshestEvent =
      existing === undefined || compareEventsByFreshness(event, existing.freshestEvent) < 0 ? event : existing.freshestEvent;

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
    }
  };
}
