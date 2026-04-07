import { createHash } from "node:crypto";

import {
  CONTRACT_IDS,
  buildEventSemanticSurface,
  buildNormalizedEventExport as buildNormalizedEventExportFromContracts,
  checksumJsonPayload,
  validateNormalizedEventExport as validateNormalizedEventExportFromContracts,
} from "../contracts/index.js";

function unique(values) {
  return [...new Set(values.filter((value) => value !== null && value !== undefined))];
}

function normalizeText(value) {
  return typeof value === "string" ? value.trim() : "";
}

function detectFeedbackKind(record) {
  const explicitKind = normalizeText(record?.kind);
  if (explicitKind.length > 0) {
    return explicitKind;
  }
  const content = normalizeText(record?.content).toLowerCase();
  if (content.includes("correct") || content.includes("wrong")) {
    return "correction";
  }
  if (content.includes("approve") || content.includes("right answer")) {
    return "approval";
  }
  if (content.includes("suppress") || content.includes("avoid")) {
    return "suppression";
  }
  return "teaching";
}

export function buildNormalizedEventDedupId(event = {}) {
  return `dedup-${createHash("sha256").update(JSON.stringify({
    eventId: event?.eventId ?? null,
    sessionId: event?.sessionId ?? null,
    channel: event?.channel ?? null,
    kind: event?.kind ?? null,
    createdAt: event?.createdAt ?? null,
    sequence: event?.sequence ?? null,
  })).digest("hex").slice(0, 16)}`;
}

export function createExplicitEventRange(events = []) {
  const normalized = [...events];
  const count = normalized.length;
  const sequences = normalized.map((event) => Number(event?.sequence ?? 0)).filter((value) => Number.isFinite(value));
  const createdAts = normalized.map((event) => event?.createdAt).filter((value) => typeof value === "string");
  return {
    start: count === 0 ? 0 : Math.min(...sequences),
    end: count === 0 ? 0 : Math.max(...sequences),
    count,
    firstCreatedAt: createdAts[0] ?? null,
    lastCreatedAt: createdAts.at(-1) ?? null,
  };
}

export function createDefaultLearningSurface() {
  return {
    source: "normalized_event_export",
    labelHarvest: {
      humanLabels: 0,
      selfLabels: 0,
      approvals: 0,
      corrections: 0,
      teachings: 0,
      suppressions: 0,
    },
    interactionKinds: [],
    feedbackKinds: [],
  };
}

export function createEventExportCursor(input = {}) {
  return {
    watermark: Number.isInteger(input?.watermark) ? input.watermark : 0,
    exportDigest: input?.exportDigest ?? null,
  };
}

export function buildNormalizedEventExport(input = {}) {
  return buildNormalizedEventExportFromContracts(input);
}

export function buildNormalizedEventExportBridge(input = {}) {
  return {
    contract: "normalized_event_export_bridge.v1",
    cursor: createEventExportCursor(input.cursor),
    slices: Array.isArray(input.slices) ? input.slices : [],
  };
}

export function validateNormalizedEventExport(value) {
  return validateNormalizedEventExportFromContracts(value);
}

export function validateNormalizedEventExportSlice() {
  return [];
}

export function validateNormalizedEventExportBridge() {
  return [];
}

export function describeNormalizedEventExportObservability(normalizedEventExport) {
  return {
    exportDigest: normalizedEventExport?.provenance?.exportDigest ?? null,
    interactionCount: normalizedEventExport?.provenance?.interactionCount ?? 0,
    feedbackCount: normalizedEventExport?.provenance?.feedbackCount ?? 0,
    sourceStreams: normalizedEventExport?.provenance?.sourceStreams ?? [],
    learningSurface: normalizedEventExport?.provenance?.learningSurface ?? createDefaultLearningSurface(),
  };
}

export function classifyFeedbackSignalContent(content) {
  const text = normalizeText(content).toLowerCase();
  return {
    kind: text.includes("correct") || text.includes("wrong")
      ? "correction"
      : text.includes("approve") || text.includes("right answer")
        ? "approval"
        : text.includes("avoid") || text.includes("suppress")
          ? "suppression"
          : "teaching",
    normalizedText: text,
  };
}

export function extractFeedbackEventsFromInteractionRecords({ agentId = "main", sessionId = "unknown-session", channel = "telegram", source = "unknown", records = [] } = {}) {
  const events = records.map((record, index) => {
    const kind = detectFeedbackKind(record);
    const feedbackEvent = {
      contract: CONTRACT_IDS.feedbackEvents,
      eventId: record.recordId ?? record.eventId ?? `feedback-${index}`,
      agentId,
      sessionId,
      channel,
      source,
      createdAt: record.createdAt ?? null,
      sequence: record.sequence ?? index,
      kind,
      content: record.content ?? null,
      actorRole: record.actorRole ?? null,
      interactionId: record.interactionId ?? null,
      relatedInteractionId: record.relatedInteractionId ?? record.interactionId ?? null,
      messageId: record.messageId ?? null,
      dedupId: buildNormalizedEventDedupId(record),
      semantic: buildEventSemanticSurface([{ contract: CONTRACT_IDS.feedbackEvents, kind, source }]),
      recordDigest: checksumJsonPayload(record),
    };
    return { feedbackEvent };
  });
  return { events, warnings: [] };
}

export function isSystemMessage(record) {
  return record?.actorRole === "system" || record?.role === "system";
}
