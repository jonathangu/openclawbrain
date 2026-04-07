import { createHash } from "node:crypto";

function canonicalize(value) {
  if (Array.isArray(value)) {
    return value.map(canonicalize);
  }
  if (value === null || typeof value !== "object") {
    return value;
  }
  const result = {};
  for (const key of Object.keys(value).sort((left, right) => left.localeCompare(right))) {
    const child = canonicalize(value[key]);
    if (child !== undefined) {
      result[key] = child;
    }
  }
  return result;
}

export function canonicalJson(value) {
  return JSON.stringify(canonicalize(value));
}

export function checksumJsonPayload(value) {
  return `sha256:${createHash("sha256").update(canonicalJson(value)).digest("hex")}`;
}

function uniqueStrings(values) {
  return [...new Set(values.filter((value) => typeof value === "string" && value.trim().length > 0))];
}

function normalizeEventTimestamp(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const asDate = Date.parse(value);
    if (!Number.isNaN(asDate)) {
      return asDate;
    }
    const asNumber = Number(value);
    if (Number.isFinite(asNumber)) {
      return asNumber;
    }
  }
  return null;
}

function sortEventTimes(events) {
  return [...events]
    .map((event) => ({ event, ts: normalizeEventTimestamp(event?.createdAt) }))
    .sort((left, right) => {
      if (left.ts !== null && right.ts !== null && left.ts !== right.ts) {
        return left.ts - right.ts;
      }
      if (left.ts !== null && right.ts === null) {
        return -1;
      }
      if (left.ts === null && right.ts !== null) {
        return 1;
      }
      const leftSequence = Number(left.event?.sequence ?? 0);
      const rightSequence = Number(right.event?.sequence ?? 0);
      if (leftSequence !== rightSequence) {
        return leftSequence - rightSequence;
      }
      return String(left.event?.eventId ?? "").localeCompare(String(right.event?.eventId ?? ""));
    })
    .map(({ event }) => event);
}

function resolveSourceStream(event) {
  if (typeof event?.source === "string") {
    return event.source;
  }
  if (event?.source && typeof event.source === "object") {
    return event.source.stream ?? event.source.sourceStream ?? event.source.id ?? event.source.name ?? null;
  }
  if (typeof event?.sourceStream === "string") {
    return event.sourceStream;
  }
  return null;
}

function resolveRuntimeOwner(events) {
  for (const event of events) {
    if (event?.source && typeof event.source === "object" && typeof event.source.runtimeOwner === "string") {
      return event.source.runtimeOwner;
    }
    if (typeof event?.runtimeOwner === "string") {
      return event.runtimeOwner;
    }
  }
  return "openclaw";
}

function resolveSingleValue(events, key, fallback = null) {
  const values = uniqueStrings(events.map((event) => event?.[key]));
  if (values.length === 1) {
    return values[0];
  }
  return fallback;
}

export const CONTRACT_IDS = {
  interactionEvents: "openclawbrain_interaction_events.v1",
  feedbackEvents: "openclawbrain_feedback_events.v1",
  normalizedEventExport: "openclawbrain_normalized_event_export.v1",
  kernelSurface: "openclawbrain_kernel_surface.v1",
  normalizedKernelEventExport: "openclawbrain_normalized_kernel_event_export.v1",
  runtimeCompile: "openclawbrain_runtime_compile.v1",
  runtimeCompileRequest: "openclawbrain_runtime_compile_request.v1",
  teacherSupervisionArtifact: "openclawbrain_teacher_supervision_artifact.v1",
  routerArtifact: "openclawbrain_router_artifact.v1",
};

export function createInteractionEvent(input) {
  return {
    contract: CONTRACT_IDS.interactionEvents,
    ...input,
  };
}

export function createFeedbackEvent(input) {
  return {
    contract: CONTRACT_IDS.feedbackEvents,
    ...input,
  };
}

export function sortNormalizedEvents(events) {
  return sortEventTimes(events);
}

export function buildNormalizedEventExport({ interactionEvents = [], feedbackEvents = [] } = {}) {
  const normalizedInteractions = sortEventTimes(interactionEvents);
  const normalizedFeedback = sortEventTimes(feedbackEvents);
  const allEvents = sortEventTimes([...normalizedInteractions, ...normalizedFeedback]);
  const exportDigest = checksumJsonPayload({ interactionEvents, feedbackEvents });
  const sourceStreams = uniqueStrings(allEvents.map((event) => resolveSourceStream(event)));
  const eventContracts = uniqueStrings(allEvents.map((event) => event?.contract));
  const firstEvent = allEvents[0] ?? null;
  const lastEvent = allEvents[allEvents.length - 1] ?? null;
  const rangeStart = allEvents.length === 0
    ? 0
    : Number.isInteger(firstEvent?.sequence)
      ? firstEvent.sequence
      : 0;
  const rangeEnd = allEvents.length === 0
    ? 0
    : Number.isInteger(lastEvent?.sequence)
      ? lastEvent.sequence
      : rangeStart + allEvents.length - 1;
  const labelHarvest = {
    humanLabels: normalizedFeedback.length,
    selfLabels: normalizedInteractions.filter((event) => event?.kind === "memory_compiled").length,
    approvals: normalizedFeedback.filter((event) => event?.kind === "approval").length,
    corrections: normalizedFeedback.filter((event) => event?.kind === "correction").length,
    teachings: normalizedFeedback.filter((event) => event?.kind === "teaching").length,
    suppressions: normalizedFeedback.filter((event) => event?.kind === "suppression").length,
  };
  const semanticSurface = buildEventSemanticSurface(allEvents);
  return {
    contract: CONTRACT_IDS.normalizedEventExport,
    interactionEvents: normalizedInteractions,
    feedbackEvents: normalizedFeedback,
    range: {
      start: rangeStart,
      end: rangeEnd,
      count: allEvents.length,
      firstCreatedAt: firstEvent?.createdAt ?? null,
      lastCreatedAt: lastEvent?.createdAt ?? null,
    },
    provenance: {
      exportDigest,
      runtimeOwner: resolveRuntimeOwner(allEvents),
      sessionId: resolveSingleValue(allEvents, "sessionId"),
      channel: resolveSingleValue(allEvents, "channel"),
      sourceStreams,
      interactionCount: interactionEvents.length,
      feedbackCount: feedbackEvents.length,
      contracts: uniqueStrings([
        ...eventContracts,
        CONTRACT_IDS.normalizedEventExport,
        semanticSurface.contract,
      ]),
      semanticSurface,
      learningSurface: {
        source: allEvents.length === 0 ? "empty" : "normalized_event_export",
        labelHarvest,
        interactionKinds: uniqueStrings(normalizedInteractions.map((event) => event?.kind)),
        feedbackKinds: uniqueStrings(normalizedFeedback.map((event) => event?.kind)),
      },
    },
  };
}

export function buildEventSemanticSurface(value) {
  const events = Array.isArray(value) ? value : [];
  return {
    contract: CONTRACT_IDS.kernelSurface,
    eventCount: events.length,
    sourceStreams: uniqueStrings(events.map((event) => resolveSourceStream(event))),
    interactionKinds: uniqueStrings(events.filter((event) => event?.contract === CONTRACT_IDS.interactionEvents).map((event) => event?.kind)),
    feedbackKinds: uniqueStrings(events.filter((event) => event?.contract === CONTRACT_IDS.feedbackEvents).map((event) => event?.kind)),
  };
}

export function validateKernelSurface() {
  return [];
}

export function validateNormalizedEventExport() {
  return [];
}

export function validateRuntimeCompileRequest() {
  return [];
}

export function buildRouteArtifactReference(value) {
  return { contract: "openclawbrain_route_artifact_reference.v1", ...value };
}

export const PACK_GRAPH_SCHEMAS = {};
export const ROUTER_PG_PROFILE_V1 = "router_pg_profile.v1";
export const ROUTER_PG_PROFILE_V2 = "router_pg_profile.v2";

export function validateTeacherSupervisionArtifact() {
  return [];
}

export function computeRouterCollectedLabelCounts() { return {}; }
export function computeRouterFreshnessChecksum() { return "sha256:0"; }
export function computeRouterObjectiveChecksum() { return "sha256:0"; }
export function computeRouterQueryChecksum() { return "sha256:0"; }
export function computeRouterWeightsChecksum() { return "sha256:0"; }
