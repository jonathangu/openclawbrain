import {
  CONTRACT_IDS,
  buildNormalizedEventExport,
  checksumJsonPayload,
  sortNormalizedEvents,
  validateFeedbackEvent,
  validateInteractionEvent,
  validateNormalizedEventExport,
  type EventContractId,
  type FeedbackEventV1,
  type InteractionEventV1,
  type NormalizedEventExportV1,
  type NormalizedEventSourceV1,
  type NormalizedEventV1
} from "@openclawbrain/contracts";

export const DEFAULT_EVENT_EXPORT_LIVE_SLICE_SIZE = 32;
export const DEFAULT_EVENT_EXPORT_BACKFILL_SLICE_SIZE = 32;

export type EventExportLaneV1 = "live" | "backfill";

export interface EventExportWatermarkV1 {
  runtimeOwner: NormalizedEventSourceV1["runtimeOwner"];
  contract: EventContractId;
  eventId: string;
  sequence: number;
  createdAt: string;
  dedupId: string;
}

export interface EventExportCursorV1 {
  runtimeOwner: NormalizedEventSourceV1["runtimeOwner"];
  live: {
    after: EventExportWatermarkV1 | null;
    exhausted: boolean;
  };
  backfill: {
    before: EventExportWatermarkV1 | null;
    exhausted: boolean;
  };
}

export interface EventExportSliceProvenanceV1 {
  runtimeOwner: NormalizedEventSourceV1["runtimeOwner"];
  lane: EventExportLaneV1;
  sliceDigest: string;
  bridgeDigest: string;
  sourceStreams: string[];
  contracts: EventContractId[];
  dedupedEventCount: number;
  duplicateIdentityCount: number;
}

export interface NormalizedEventExportSliceV1 {
  lane: EventExportLaneV1;
  sliceId: string;
  export: NormalizedEventExportV1;
  eventIdentities: string[];
  dedupedEventCount: number;
  duplicateIdentityCount: number;
  watermark: {
    first: EventExportWatermarkV1 | null;
    last: EventExportWatermarkV1 | null;
  };
  nextCursor: EventExportCursorV1;
  provenance: EventExportSliceProvenanceV1;
}

export interface NormalizedEventExportBridgeV1 {
  runtimeOwner: NormalizedEventSourceV1["runtimeOwner"];
  slices: NormalizedEventExportSliceV1[];
  cursor: EventExportCursorV1;
  dedupedInputCount: number;
  duplicateIdentityCount: number;
  bridgeDigest: string;
}

export interface BuildNormalizedEventExportBridgeInput {
  interactionEvents: readonly InteractionEventV1[];
  feedbackEvents: readonly FeedbackEventV1[];
  cursor?: EventExportCursorV1;
  liveSliceSize?: number;
  backfillSliceSize?: number;
}

interface RawSlicePlan {
  lane: EventExportLaneV1;
  events: NormalizedEventV1[];
  nextCursor: EventExportCursorV1;
}

interface DedupedNormalizedEvents {
  events: NormalizedEventV1[];
  duplicateIdentityCount: number;
}

function isIsoDate(value: string): boolean {
  return !Number.isNaN(Date.parse(value));
}

function uniqueInOrder<T>(values: readonly T[]): T[] {
  const seen = new Set<T>();
  const result: T[] = [];

  for (const value of values) {
    if (seen.has(value)) {
      continue;
    }
    seen.add(value);
    result.push(value);
  }

  return result;
}

function sortEvents<T extends NormalizedEventV1>(events: readonly T[]): T[] {
  return [...events].sort((left, right) => compareEventKeys(left, right));
}

function isChecksum(value: string): boolean {
  return value.startsWith("sha256-") && value.length > "sha256-".length;
}

function toSortedInteractionEvents(events: readonly NormalizedEventV1[]): InteractionEventV1[] {
  return sortEvents(events.filter((event): event is InteractionEventV1 => event.contract === CONTRACT_IDS.interactionEvents));
}

function toSortedFeedbackEvents(events: readonly NormalizedEventV1[]): FeedbackEventV1[] {
  return sortEvents(events.filter((event): event is FeedbackEventV1 => event.contract === CONTRACT_IDS.feedbackEvents));
}

function eventComparisonKey(value: Pick<NormalizedEventV1, "sequence" | "createdAt" | "contract" | "eventId">): readonly [number, string, string, string] {
  return [value.sequence, value.createdAt, value.contract, value.eventId];
}

function compareEventKeys(
  left: Pick<NormalizedEventV1, "sequence" | "createdAt" | "contract" | "eventId">,
  right: Pick<NormalizedEventV1, "sequence" | "createdAt" | "contract" | "eventId">
): number {
  const leftKey = eventComparisonKey(left);
  const rightKey = eventComparisonKey(right);

  if (leftKey[0] !== rightKey[0]) {
    return leftKey[0] - rightKey[0];
  }
  if (leftKey[1] !== rightKey[1]) {
    return leftKey[1].localeCompare(rightKey[1]);
  }
  if (leftKey[2] !== rightKey[2]) {
    return leftKey[2].localeCompare(rightKey[2]);
  }
  return leftKey[3].localeCompare(rightKey[3]);
}

function cloneCursor(value: EventExportCursorV1): EventExportCursorV1 {
  return {
    runtimeOwner: value.runtimeOwner,
    live: {
      after: value.live.after === null ? null : { ...value.live.after },
      exhausted: value.live.exhausted
    },
    backfill: {
      before: value.backfill.before === null ? null : { ...value.backfill.before },
      exhausted: value.backfill.exhausted
    }
  };
}

function chunkEvents(events: readonly NormalizedEventV1[], size: number): NormalizedEventV1[][] {
  const chunks: NormalizedEventV1[][] = [];

  for (let index = 0; index < events.length; index += size) {
    chunks.push(events.slice(index, index + size));
  }

  return chunks;
}

function lastOf<T>(values: readonly T[]): T | null {
  return values.length === 0 ? null : (values[values.length - 1] ?? null);
}

function firstOf<T>(values: readonly T[]): T | null {
  return values.length === 0 ? null : (values[0] ?? null);
}

export function buildNormalizedEventDedupId(event: NormalizedEventV1): string {
  return checksumJsonPayload({
    contract: event.contract,
    eventId: event.eventId,
    agentId: event.agentId,
    sessionId: event.sessionId,
    channel: event.channel,
    sequence: event.sequence,
    kind: event.kind,
    createdAt: event.createdAt,
    source: event.source,
    packId: "packId" in event ? event.packId ?? null : null,
    content: "content" in event ? event.content : null,
    messageId: event.messageId ?? null,
    relatedInteractionId: "relatedInteractionId" in event ? event.relatedInteractionId ?? null : null
  });
}

export function buildEventExportWatermark(event: NormalizedEventV1): EventExportWatermarkV1 {
  return {
    runtimeOwner: event.source.runtimeOwner,
    contract: event.contract,
    eventId: event.eventId,
    sequence: event.sequence,
    createdAt: event.createdAt,
    dedupId: buildNormalizedEventDedupId(event)
  };
}

export function createEventExportCursor(): EventExportCursorV1 {
  return {
    runtimeOwner: "openclaw",
    live: {
      after: null,
      exhausted: false
    },
    backfill: {
      before: null,
      exhausted: false
    }
  };
}

function materializeCursorState(events: readonly NormalizedEventV1[], cursor: EventExportCursorV1): EventExportCursorV1 {
  const nextCursor = cloneCursor(cursor);
  if (nextCursor.backfill.before === null && nextCursor.live.after !== null) {
    nextCursor.backfill.before = { ...nextCursor.live.after };
  }

  nextCursor.live.exhausted =
    nextCursor.live.after === null
      ? events.length === 0
      : events.every((event) => compareEventKeys(event, nextCursor.live.after as EventExportWatermarkV1) <= 0);
  nextCursor.backfill.exhausted =
    nextCursor.backfill.before === null
      ? events.length === 0
      : events.every((event) => compareEventKeys(event, nextCursor.backfill.before as EventExportWatermarkV1) >= 0);

  return nextCursor;
}

function dedupeNormalizedEvents(events: readonly NormalizedEventV1[]): DedupedNormalizedEvents {
  const deduped: NormalizedEventV1[] = [];
  const identities = new Set<string>();
  let duplicateIdentityCount = 0;

  for (const event of sortNormalizedEvents(events)) {
    const identity = buildNormalizedEventDedupId(event);
    if (identities.has(identity)) {
      duplicateIdentityCount += 1;
      continue;
    }
    identities.add(identity);
    deduped.push(event);
  }

  return {
    events: deduped,
    duplicateIdentityCount
  };
}

function validateInputEvents(input: BuildNormalizedEventExportBridgeInput): void {
  const errors = [
    ...input.interactionEvents.flatMap((event, index) => validateInteractionEvent(event).map((message) => `interactionEvents[${index}] ${message}`)),
    ...input.feedbackEvents.flatMap((event, index) => validateFeedbackEvent(event).map((message) => `feedbackEvents[${index}] ${message}`))
  ];

  if (errors.length > 0) {
    throw new Error(`Invalid event export bridge input: ${errors.join("; ")}`);
  }
}

function validateSliceSize(label: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
}

function buildRawSlicePlans(
  events: readonly NormalizedEventV1[],
  cursor: EventExportCursorV1,
  liveSliceSize: number,
  backfillSliceSize: number
): RawSlicePlan[] {
  const plans: RawSlicePlan[] = [];
  const workingCursor = cloneCursor(cursor);

  if (events.length === 0) {
    return plans;
  }

  if (workingCursor.live.after === null) {
    const liveEvents = events.slice(Math.max(events.length - liveSliceSize, 0));
    const firstLiveEvent = firstOf(liveEvents);
    const lastLiveEvent = lastOf(liveEvents);

    if (firstLiveEvent !== null && lastLiveEvent !== null) {
      workingCursor.live.after = buildEventExportWatermark(lastLiveEvent);
      if (workingCursor.backfill.before === null) {
        workingCursor.backfill.before = buildEventExportWatermark(firstLiveEvent);
      }
      plans.push({
        lane: "live",
        events: [...liveEvents],
        nextCursor: materializeCursorState(events, workingCursor)
      });
    }
  } else {
    const liveEligible = events.filter((event) => compareEventKeys(event, workingCursor.live.after as EventExportWatermarkV1) > 0);
    for (const liveChunk of chunkEvents(liveEligible, liveSliceSize)) {
      const lastLiveEvent = lastOf(liveChunk);
      if (lastLiveEvent === null) {
        continue;
      }
      workingCursor.live.after = buildEventExportWatermark(lastLiveEvent);
      plans.push({
        lane: "live",
        events: [...liveChunk],
        nextCursor: materializeCursorState(events, workingCursor)
      });
    }
  }

  if (workingCursor.backfill.before === null && workingCursor.live.after !== null) {
    workingCursor.backfill.before = { ...workingCursor.live.after };
  }

  if (workingCursor.backfill.before !== null) {
    const backfillEligible = events.filter((event) => compareEventKeys(event, workingCursor.backfill.before as EventExportWatermarkV1) < 0);
    const backfillEvents = backfillEligible.slice(Math.max(backfillEligible.length - backfillSliceSize, 0));
    const firstBackfillEvent = firstOf(backfillEvents);
    if (firstBackfillEvent !== null) {
      workingCursor.backfill.before = buildEventExportWatermark(firstBackfillEvent);
      plans.push({
        lane: "backfill",
        events: [...backfillEvents],
        nextCursor: materializeCursorState(events, workingCursor)
      });
    }
  }

  return plans;
}

function buildSliceFromPlan(
  plan: RawSlicePlan,
  bridgeDigest: string,
  duplicateIdentityCount: number,
  dedupedInputCount: number
): NormalizedEventExportSliceV1 {
  const eventExport = buildNormalizedEventExport({
    interactionEvents: toSortedInteractionEvents(plan.events),
    feedbackEvents: toSortedFeedbackEvents(plan.events)
  });
  const eventIdentities = plan.events.map((event) => buildNormalizedEventDedupId(event));
  const firstEvent = firstOf(plan.events);
  const lastEvent = lastOf(plan.events);
  const sourceStreams = uniqueInOrder(plan.events.map((event) => event.source.stream));
  const contracts = uniqueInOrder(plan.events.map((event) => event.contract));
  const sliceId = checksumJsonPayload({
    lane: plan.lane,
    eventIdentities,
    nextCursor: plan.nextCursor,
    bridgeDigest
  });

  return {
    lane: plan.lane,
    sliceId,
    export: eventExport,
    eventIdentities,
    dedupedEventCount: eventIdentities.length,
    duplicateIdentityCount,
    watermark: {
      first: firstEvent === null ? null : buildEventExportWatermark(firstEvent),
      last: lastEvent === null ? null : buildEventExportWatermark(lastEvent)
    },
    nextCursor: cloneCursor(plan.nextCursor),
    provenance: {
      runtimeOwner: "openclaw",
      lane: plan.lane,
      sliceDigest: sliceId,
      bridgeDigest,
      sourceStreams,
      contracts,
      dedupedEventCount: eventIdentities.length,
      duplicateIdentityCount
    }
  };
}

export function buildNormalizedEventExportBridge(input: BuildNormalizedEventExportBridgeInput): NormalizedEventExportBridgeV1 {
  validateInputEvents(input);

  const liveSliceSize = input.liveSliceSize ?? DEFAULT_EVENT_EXPORT_LIVE_SLICE_SIZE;
  const backfillSliceSize = input.backfillSliceSize ?? DEFAULT_EVENT_EXPORT_BACKFILL_SLICE_SIZE;
  validateSliceSize("liveSliceSize", liveSliceSize);
  validateSliceSize("backfillSliceSize", backfillSliceSize);

  const cursor = cloneCursor(input.cursor ?? createEventExportCursor());
  const cursorErrors = validateEventExportCursor(cursor);
  if (cursorErrors.length > 0) {
    throw new Error(`Invalid event export bridge cursor: ${cursorErrors.join("; ")}`);
  }

  const deduped = dedupeNormalizedEvents([...input.interactionEvents, ...input.feedbackEvents]);
  const rawSlicePlans = buildRawSlicePlans(deduped.events, cursor, liveSliceSize, backfillSliceSize);
  const finalCursor = rawSlicePlans.length === 0 ? materializeCursorState(deduped.events, cursor) : cloneCursor(rawSlicePlans[rawSlicePlans.length - 1]?.nextCursor ?? cursor);
  const bridgeDigest = checksumJsonPayload({
    runtimeOwner: "openclaw",
    cursor: finalCursor,
    slicePlan: rawSlicePlans.map((plan) => ({
      lane: plan.lane,
      eventIdentities: plan.events.map((event) => buildNormalizedEventDedupId(event)),
      nextCursor: plan.nextCursor
    })),
    dedupedInputCount: deduped.events.length,
    duplicateIdentityCount: deduped.duplicateIdentityCount
  });
  const slices = rawSlicePlans.map((plan) => buildSliceFromPlan(plan, bridgeDigest, deduped.duplicateIdentityCount, deduped.events.length));

  const bridge: NormalizedEventExportBridgeV1 = {
    runtimeOwner: "openclaw",
    slices,
    cursor: finalCursor,
    dedupedInputCount: deduped.events.length,
    duplicateIdentityCount: deduped.duplicateIdentityCount,
    bridgeDigest
  };

  const errors = validateNormalizedEventExportBridge(bridge);
  if (errors.length > 0) {
    throw new Error(`Invalid normalized event export bridge: ${errors.join("; ")}`);
  }

  return bridge;
}

export function validateEventExportWatermark(value: EventExportWatermarkV1): string[] {
  const errors: string[] = [];

  if (value.runtimeOwner !== "openclaw") {
    errors.push("event export watermark runtimeOwner must be openclaw");
  }
  if (value.contract !== CONTRACT_IDS.interactionEvents && value.contract !== CONTRACT_IDS.feedbackEvents) {
    errors.push("event export watermark contract must reference a normalized event contract");
  }
  if (value.eventId.length === 0) {
    errors.push("event export watermark eventId is required");
  }
  if (value.sequence < 0) {
    errors.push("event export watermark sequence must be non-negative");
  }
  if (!isIsoDate(value.createdAt)) {
    errors.push("event export watermark createdAt must be an ISO timestamp");
  }
  if (!isChecksum(value.dedupId)) {
    errors.push("event export watermark dedupId must be a sha256 digest");
  }

  return errors;
}

export function validateEventExportCursor(value: EventExportCursorV1): string[] {
  const errors: string[] = [];

  if (value.runtimeOwner !== "openclaw") {
    errors.push("event export cursor runtimeOwner must be openclaw");
  }
  if (value.live.after !== null) {
    errors.push(...validateEventExportWatermark(value.live.after));
  }
  if (value.backfill.before !== null) {
    errors.push(...validateEventExportWatermark(value.backfill.before));
  }

  return errors;
}

export function validateNormalizedEventExportSlice(value: NormalizedEventExportSliceV1): string[] {
  const errors: string[] = [];

  if (value.lane !== "live" && value.lane !== "backfill") {
    errors.push("normalized event export slice lane must be live or backfill");
  }
  if (!isChecksum(value.sliceId)) {
    errors.push("normalized event export sliceId must be a sha256 digest");
  }
  if (!isChecksum(value.provenance.sliceDigest)) {
    errors.push("normalized event export slice provenance sliceDigest must be a sha256 digest");
  }
  if (!isChecksum(value.provenance.bridgeDigest)) {
    errors.push("normalized event export slice provenance bridgeDigest must be a sha256 digest");
  }
  if (value.sliceId !== value.provenance.sliceDigest) {
    errors.push("normalized event export sliceId must match provenance.sliceDigest");
  }
  if (value.provenance.runtimeOwner !== "openclaw") {
    errors.push("normalized event export slice provenance runtimeOwner must be openclaw");
  }
  if (value.provenance.lane !== value.lane) {
    errors.push("normalized event export slice provenance lane must match slice lane");
  }
  if (value.eventIdentities.length !== value.export.range.count) {
    errors.push("normalized event export slice eventIdentities must match export range count");
  }
  if (new Set(value.eventIdentities).size !== value.eventIdentities.length) {
    errors.push("normalized event export slice eventIdentities must be unique");
  }
  if (value.dedupedEventCount !== value.eventIdentities.length) {
    errors.push("normalized event export slice dedupedEventCount must match eventIdentities length");
  }

  errors.push(...validateNormalizedEventExport(value.export));
  errors.push(...validateEventExportCursor(value.nextCursor));
  if (value.watermark.first !== null) {
    errors.push(...validateEventExportWatermark(value.watermark.first));
  }
  if (value.watermark.last !== null) {
    errors.push(...validateEventExportWatermark(value.watermark.last));
  }

  if (value.watermark.first?.eventId !== value.export.range.firstEventId) {
    errors.push("normalized event export slice first watermark must match export firstEventId");
  }
  if (value.watermark.last?.eventId !== value.export.range.lastEventId) {
    errors.push("normalized event export slice last watermark must match export lastEventId");
  }
  if (value.lane === "live" && value.watermark.last?.eventId !== value.nextCursor.live.after?.eventId) {
    errors.push("live slice nextCursor.live.after must match the slice last watermark");
  }
  if (value.lane === "backfill" && value.watermark.first?.eventId !== value.nextCursor.backfill.before?.eventId) {
    errors.push("backfill slice nextCursor.backfill.before must match the slice first watermark");
  }

  return errors;
}

export function validateNormalizedEventExportBridge(value: NormalizedEventExportBridgeV1): string[] {
  const errors: string[] = [];
  const seenEventIdentities = new Set<string>();
  let backfillSeen = false;

  if (value.runtimeOwner !== "openclaw") {
    errors.push("normalized event export bridge runtimeOwner must be openclaw");
  }
  if (!isChecksum(value.bridgeDigest)) {
    errors.push("normalized event export bridgeDigest must be a sha256 digest");
  }
  if (value.dedupedInputCount < 0) {
    errors.push("normalized event export bridge dedupedInputCount must be non-negative");
  }
  if (value.duplicateIdentityCount < 0) {
    errors.push("normalized event export bridge duplicateIdentityCount must be non-negative");
  }

  errors.push(...validateEventExportCursor(value.cursor));

  for (const [index, slice] of value.slices.entries()) {
    errors.push(...validateNormalizedEventExportSlice(slice).map((message) => `slices[${index}] ${message}`));

    if (slice.provenance.bridgeDigest !== value.bridgeDigest) {
      errors.push(`slices[${index}] provenance bridgeDigest must match bridge bridgeDigest`);
    }
    if (slice.lane === "backfill") {
      backfillSeen = true;
    }
    if (backfillSeen && slice.lane === "live") {
      errors.push(`slices[${index}] live slices must precede backfill slices`);
    }

    for (const identity of slice.eventIdentities) {
      if (seenEventIdentities.has(identity)) {
        errors.push(`slices[${index}] duplicate event identity across slices: ${identity}`);
        continue;
      }
      seenEventIdentities.add(identity);
    }
  }

  if (value.slices.length > 0) {
    const lastSlice = value.slices[value.slices.length - 1] as NormalizedEventExportSliceV1;
    if (checksumJsonPayload(value.cursor) !== checksumJsonPayload(lastSlice.nextCursor)) {
      errors.push("normalized event export bridge cursor must match the final slice nextCursor");
    }
  }

  return errors;
}
