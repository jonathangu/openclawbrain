import { createHash } from "node:crypto";

export const CONTRACT_IDS = {
  activationPointers: "activation_pointers.v1",
  artifactManifest: "artifact_manifest.v1",
  feedbackEvents: "feedback_events.v1",
  interactionEvents: "interaction_events.v1",
  runtimeCompile: "runtime_compile.v1"
} as const;

export type ContractId = (typeof CONTRACT_IDS)[keyof typeof CONTRACT_IDS];
export type EventContractId = typeof CONTRACT_IDS.interactionEvents | typeof CONTRACT_IDS.feedbackEvents;
export type RouteMode = "heuristic" | "learned";
export type RoutePolicy = "heuristic_allowed" | "requires_learned_routing";
export type RouterAssetKind = "none" | "stub" | "artifact";
export type ActivationPointerSlot = "active" | "candidate" | "previous";
export type InteractionEventKind = "memory_compiled" | "message_delivered" | "operator_override";
export type FeedbackEventKind = "correction" | "teaching" | "approval" | "suppression";

export interface RuntimeCompileRequestV1 {
  contract: typeof CONTRACT_IDS.runtimeCompile;
  agentId: string;
  userMessage: string;
  maxContextBlocks: number;
  modeRequested: RouteMode;
  activePackId?: string;
  runtimeHints?: string[];
}

export interface RuntimeContextBlockV1 {
  id: string;
  source: string;
  text: string;
}

export interface PackContextBlockRecordV1 extends RuntimeContextBlockV1 {
  keywords: string[];
  priority: number;
}

export interface RuntimeCompileDiagnosticsV1 {
  modeRequested: RouteMode;
  modeEffective: RouteMode;
  usedLearnedRouteFn: boolean;
  routerIdentity: string | null;
  notes: string[];
}

export interface RuntimeCompileResponseV1 {
  contract: typeof CONTRACT_IDS.runtimeCompile;
  packId: string;
  selectedContext: RuntimeContextBlockV1[];
  diagnostics: RuntimeCompileDiagnosticsV1;
}

export interface NormalizedEventSourceV1 {
  runtimeOwner: "openclaw";
  stream: string;
}

export interface InteractionEventV1 {
  contract: typeof CONTRACT_IDS.interactionEvents;
  eventId: string;
  agentId: string;
  sessionId: string;
  channel: string;
  sequence: number;
  kind: InteractionEventKind;
  createdAt: string;
  source: NormalizedEventSourceV1;
  packId?: string;
  messageId?: string;
}

export interface FeedbackEventV1 {
  contract: typeof CONTRACT_IDS.feedbackEvents;
  eventId: string;
  agentId: string;
  sessionId: string;
  channel: string;
  sequence: number;
  kind: FeedbackEventKind;
  createdAt: string;
  source: NormalizedEventSourceV1;
  content: string;
  messageId?: string;
  relatedInteractionId?: string;
}

export type NormalizedEventV1 = InteractionEventV1 | FeedbackEventV1;

export interface NormalizedEventRangeV1 {
  start: number;
  end: number;
  count: number;
  firstEventId: string | null;
  lastEventId: string | null;
  firstCreatedAt: string | null;
  lastCreatedAt: string | null;
}

export interface EventExportProvenanceV1 {
  runtimeOwner: "openclaw";
  sessionId: string | null;
  channel: string | null;
  interactionCount: number;
  feedbackCount: number;
  sourceStreams: string[];
  contracts: EventContractId[];
  exportDigest: string;
}

export interface NormalizedEventExportV1 {
  interactionEvents: InteractionEventV1[];
  feedbackEvents: FeedbackEventV1[];
  range: NormalizedEventRangeV1;
  provenance: EventExportProvenanceV1;
}

export interface ArtifactManifestV1 {
  contract: typeof CONTRACT_IDS.artifactManifest;
  packId: string;
  immutable: true;
  routePolicy: RoutePolicy;
  runtimeAssets: {
    graphPath: string;
    vectorPath: string;
    router: {
      kind: RouterAssetKind;
      identity: string | null;
      artifactPath: string | null;
    };
  };
  payloadChecksums: {
    graph: string;
    vector: string;
    router: string | null;
  };
  modelFingerprints: string[];
  provenance: {
    workspaceSnapshot: string;
    eventRange: NormalizedEventRangeV1;
    eventExports: EventExportProvenanceV1 | null;
    builtAt: string;
    offlineArtifacts: string[];
  };
  graphDynamics: {
    hebbian: {
      enabled: boolean;
      learningRate: number;
    };
    decay: {
      enabled: boolean;
      halfLifeDays: number;
    };
    structuralOps: {
      split: number;
      merge: number;
      prune: number;
      connect: number;
    };
  };
}

export interface ActivationPointerRecordV1 {
  slot: ActivationPointerSlot;
  packId: string;
  packRootDir: string;
  manifestPath: string;
  routePolicy: RoutePolicy;
  routerIdentity: string | null;
  eventRange: Pick<NormalizedEventRangeV1, "start" | "end" | "count">;
  eventExportDigest: string | null;
  builtAt: string;
  updatedAt: string;
}

export interface ActivationPointersV1 {
  contract: typeof CONTRACT_IDS.activationPointers;
  active: ActivationPointerRecordV1 | null;
  candidate: ActivationPointerRecordV1 | null;
  previous: ActivationPointerRecordV1 | null;
}

export interface PackVectorEntryV1 {
  blockId: string;
  keywords: string[];
  boost: number;
  weights?: Record<string, number>;
}

export interface PackVectorsPayloadV1 {
  packId: string;
  entries: PackVectorEntryV1[];
}

export interface PackGraphPayloadV1 {
  packId: string;
  blocks: PackContextBlockRecordV1[];
}

export interface RouterArtifactV1 {
  routerIdentity: string;
  strategy: "keyword_overlap_v1";
  trainedAt: string;
  requiresLearnedRouting: boolean;
}

function isIsoDate(value: string): boolean {
  return !Number.isNaN(Date.parse(value));
}

function pushWhenMissing(errors: string[], condition: boolean, message: string): void {
  if (!condition) {
    errors.push(message);
  }
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

function eventSequenceErrors(events: readonly NormalizedEventV1[]): string[] {
  const errors: string[] = [];
  let previous: NormalizedEventV1 | null = null;

  for (const event of sortNormalizedEvents(events)) {
    if (previous !== null) {
      if (event.sequence === previous.sequence && event.eventId === previous.eventId) {
        errors.push(`duplicate normalized event identity: ${event.contract}:${event.eventId}`);
      }
      if (event.sequence < previous.sequence) {
        errors.push(`normalized events must be sorted by sequence: ${event.eventId}`);
      }
      if (event.sequence === previous.sequence && event.createdAt < previous.createdAt) {
        errors.push(`normalized events with sequence ${event.sequence} must be ordered by createdAt`);
      }
    }
    previous = event;
  }

  return errors;
}

export function canonicalJson(value: unknown): string {
  return `${JSON.stringify(value, null, 2)}\n`;
}

export function checksumJsonPayload(value: unknown): string {
  return `sha256-${createHash("sha256").update(canonicalJson(value)).digest("hex")}`;
}

export function createInteractionEvent(
  value: Omit<InteractionEventV1, "contract">
): InteractionEventV1 {
  return {
    contract: CONTRACT_IDS.interactionEvents,
    ...value
  };
}

export function createFeedbackEvent(
  value: Omit<FeedbackEventV1, "contract">
): FeedbackEventV1 {
  return {
    contract: CONTRACT_IDS.feedbackEvents,
    ...value
  };
}

export function sortNormalizedEvents(events: readonly NormalizedEventV1[]): NormalizedEventV1[] {
  return [...events].sort((left, right) => {
    if (left.sequence !== right.sequence) {
      return left.sequence - right.sequence;
    }
    if (left.createdAt !== right.createdAt) {
      return left.createdAt.localeCompare(right.createdAt);
    }
    if (left.contract !== right.contract) {
      return left.contract.localeCompare(right.contract);
    }
    return left.eventId.localeCompare(right.eventId);
  });
}

export function createExplicitEventRange(value: {
  start: number;
  end: number;
}): NormalizedEventRangeV1 {
  return {
    start: value.start,
    end: value.end,
    count: value.end >= value.start ? value.end - value.start + 1 : 0,
    firstEventId: null,
    lastEventId: null,
    firstCreatedAt: null,
    lastCreatedAt: null
  };
}

export function buildNormalizedEventRange(events: readonly NormalizedEventV1[]): NormalizedEventRangeV1 {
  const sorted = sortNormalizedEvents(events);
  if (sorted.length === 0) {
    return createExplicitEventRange({
      start: 0,
      end: -1
    });
  }

  const first = sorted[0] as NormalizedEventV1;
  const last = sorted[sorted.length - 1] as NormalizedEventV1;

  return {
    start: first.sequence,
    end: last.sequence,
    count: sorted.length,
    firstEventId: first.eventId,
    lastEventId: last.eventId,
    firstCreatedAt: first.createdAt,
    lastCreatedAt: last.createdAt
  };
}

export function buildNormalizedEventExport(value: {
  interactionEvents: readonly InteractionEventV1[];
  feedbackEvents: readonly FeedbackEventV1[];
}): NormalizedEventExportV1 {
  const interactionEvents = [...value.interactionEvents];
  const feedbackEvents = [...value.feedbackEvents];
  const events = sortNormalizedEvents([...interactionEvents, ...feedbackEvents]);
  const contracts = uniqueInOrder(events.map((event) => event.contract));
  const sessionIds = uniqueInOrder(events.map((event) => event.sessionId));
  const channels = uniqueInOrder(events.map((event) => event.channel));
  const sourceStreams = uniqueInOrder(events.map((event) => event.source.stream));

  return {
    interactionEvents,
    feedbackEvents,
    range: buildNormalizedEventRange(events),
    provenance: {
      runtimeOwner: "openclaw",
      sessionId: sessionIds.length === 1 ? (sessionIds[0] ?? null) : null,
      channel: channels.length === 1 ? (channels[0] ?? null) : null,
      interactionCount: interactionEvents.length,
      feedbackCount: feedbackEvents.length,
      sourceStreams,
      contracts,
      exportDigest: checksumJsonPayload({
        interactionEvents: sortNormalizedEvents(interactionEvents),
        feedbackEvents: sortNormalizedEvents(feedbackEvents)
      })
    }
  };
}

export function validateRuntimeCompileRequest(value: RuntimeCompileRequestV1): string[] {
  const errors: string[] = [];
  pushWhenMissing(errors, value.contract === CONTRACT_IDS.runtimeCompile, "runtime_compile.v1 contract is required");
  pushWhenMissing(errors, value.agentId.length > 0, "agentId is required");
  pushWhenMissing(errors, value.userMessage.length > 0, "userMessage is required");
  pushWhenMissing(errors, value.maxContextBlocks >= 0, "maxContextBlocks must be non-negative");
  pushWhenMissing(errors, value.modeRequested === "heuristic" || value.modeRequested === "learned", "modeRequested must be heuristic or learned");
  return errors;
}

export function validateRuntimeCompileResponse(value: RuntimeCompileResponseV1): string[] {
  const errors: string[] = [];
  pushWhenMissing(errors, value.contract === CONTRACT_IDS.runtimeCompile, "runtime_compile.v1 contract is required");
  pushWhenMissing(errors, value.packId.length > 0, "packId is required");
  pushWhenMissing(errors, value.diagnostics.modeRequested === "heuristic" || value.diagnostics.modeRequested === "learned", "modeRequested must be explicit");
  pushWhenMissing(errors, value.diagnostics.modeEffective === "heuristic" || value.diagnostics.modeEffective === "learned", "modeEffective must be explicit");

  if (value.diagnostics.modeEffective === "learned" && !value.diagnostics.usedLearnedRouteFn) {
    errors.push("learned mode requires usedLearnedRouteFn=true");
  }

  if (value.diagnostics.usedLearnedRouteFn && value.diagnostics.modeEffective !== "learned") {
    errors.push("usedLearnedRouteFn=true requires modeEffective=learned");
  }

  if (value.diagnostics.usedLearnedRouteFn && value.diagnostics.routerIdentity === null) {
    errors.push("learned routing requires routerIdentity");
  }

  return errors;
}

export function validateNormalizedEventSource(value: NormalizedEventSourceV1): string[] {
  const errors: string[] = [];
  pushWhenMissing(errors, value.runtimeOwner === "openclaw", "normalized events must declare runtimeOwner=openclaw");
  pushWhenMissing(errors, value.stream.length > 0, "normalized events require a source stream");
  return errors;
}

export function validateInteractionEvent(value: InteractionEventV1): string[] {
  const errors: string[] = [];
  pushWhenMissing(errors, value.contract === CONTRACT_IDS.interactionEvents, "interaction_events.v1 contract is required");
  pushWhenMissing(errors, value.eventId.length > 0, "eventId is required");
  pushWhenMissing(errors, value.agentId.length > 0, "agentId is required");
  pushWhenMissing(errors, value.sessionId.length > 0, "sessionId is required");
  pushWhenMissing(errors, value.channel.length > 0, "channel is required");
  pushWhenMissing(errors, value.sequence >= 0, "sequence must be non-negative");
  pushWhenMissing(
    errors,
    value.kind === "memory_compiled" || value.kind === "message_delivered" || value.kind === "operator_override",
    "interaction kind must be explicit"
  );
  pushWhenMissing(errors, isIsoDate(value.createdAt), "createdAt must be an ISO timestamp");
  errors.push(...validateNormalizedEventSource(value.source));
  return errors;
}

export function validateFeedbackEvent(value: FeedbackEventV1): string[] {
  const errors: string[] = [];
  pushWhenMissing(errors, value.contract === CONTRACT_IDS.feedbackEvents, "feedback_events.v1 contract is required");
  pushWhenMissing(errors, value.eventId.length > 0, "eventId is required");
  pushWhenMissing(errors, value.agentId.length > 0, "agentId is required");
  pushWhenMissing(errors, value.sessionId.length > 0, "sessionId is required");
  pushWhenMissing(errors, value.channel.length > 0, "channel is required");
  pushWhenMissing(errors, value.sequence >= 0, "sequence must be non-negative");
  pushWhenMissing(
    errors,
    value.kind === "correction" || value.kind === "teaching" || value.kind === "approval" || value.kind === "suppression",
    "feedback kind must be explicit"
  );
  pushWhenMissing(errors, value.content.length > 0, "content is required");
  pushWhenMissing(errors, isIsoDate(value.createdAt), "createdAt must be an ISO timestamp");
  errors.push(...validateNormalizedEventSource(value.source));
  return errors;
}

export function validateNormalizedEventRange(value: NormalizedEventRangeV1): string[] {
  const errors: string[] = [];
  pushWhenMissing(errors, value.start >= 0 || value.count === 0, "eventRange.start must be non-negative when events exist");
  pushWhenMissing(errors, value.count >= 0, "eventRange.count must be non-negative");

  if (value.count === 0) {
    if (value.end !== value.start - 1) {
      errors.push("empty event ranges must use end=start-1");
    }
    if (value.firstEventId !== null || value.lastEventId !== null) {
      errors.push("empty event ranges must not set event ids");
    }
    if (value.firstCreatedAt !== null || value.lastCreatedAt !== null) {
      errors.push("empty event ranges must not set timestamps");
    }
    return errors;
  }

  pushWhenMissing(errors, value.end >= value.start, "eventRange.end must be >= start");
  pushWhenMissing(errors, value.count <= value.end - value.start + 1, "eventRange.count cannot exceed the numeric span");
  const hasExplicitBoundaryMetadata =
    value.firstEventId !== null || value.lastEventId !== null || value.firstCreatedAt !== null || value.lastCreatedAt !== null;

  if (hasExplicitBoundaryMetadata) {
    pushWhenMissing(errors, value.firstEventId !== null, "eventRange.firstEventId is required when boundary metadata is present");
    pushWhenMissing(errors, value.lastEventId !== null, "eventRange.lastEventId is required when boundary metadata is present");
    pushWhenMissing(errors, value.firstCreatedAt !== null, "eventRange.firstCreatedAt is required when boundary metadata is present");
    pushWhenMissing(errors, value.lastCreatedAt !== null, "eventRange.lastCreatedAt is required when boundary metadata is present");
  }

  if (value.firstCreatedAt !== null && !isIsoDate(value.firstCreatedAt)) {
    errors.push("eventRange.firstCreatedAt must be an ISO timestamp");
  }
  if (value.lastCreatedAt !== null && !isIsoDate(value.lastCreatedAt)) {
    errors.push("eventRange.lastCreatedAt must be an ISO timestamp");
  }
  if (value.firstCreatedAt !== null && value.lastCreatedAt !== null && value.lastCreatedAt < value.firstCreatedAt) {
    errors.push("eventRange timestamps must be monotonic");
  }

  return errors;
}

export function validateEventExportProvenance(
  value: EventExportProvenanceV1,
  eventRange?: NormalizedEventRangeV1
): string[] {
  const errors: string[] = [];
  pushWhenMissing(errors, value.runtimeOwner === "openclaw", "event export provenance requires runtimeOwner=openclaw");
  pushWhenMissing(errors, value.interactionCount >= 0, "interactionCount must be non-negative");
  pushWhenMissing(errors, value.feedbackCount >= 0, "feedbackCount must be non-negative");
  pushWhenMissing(errors, value.exportDigest.length > 0, "event export provenance requires exportDigest");
  pushWhenMissing(errors, value.sourceStreams.length > 0, "event export provenance requires at least one source stream");

  for (const stream of value.sourceStreams) {
    pushWhenMissing(errors, stream.length > 0, "event export provenance source streams must be non-empty");
  }
  for (const contract of value.contracts) {
    pushWhenMissing(
      errors,
      contract === CONTRACT_IDS.interactionEvents || contract === CONTRACT_IDS.feedbackEvents,
      "event export provenance contracts must be event contracts"
    );
  }
  if (value.sessionId !== null) {
    pushWhenMissing(errors, value.sessionId.length > 0, "event export provenance sessionId must be non-empty when set");
  }
  if (value.channel !== null) {
    pushWhenMissing(errors, value.channel.length > 0, "event export provenance channel must be non-empty when set");
  }
  if (eventRange !== undefined && value.interactionCount + value.feedbackCount !== eventRange.count) {
    errors.push("event export provenance counts must match eventRange.count");
  }

  return errors;
}

export function validateNormalizedEventExport(value: NormalizedEventExportV1): string[] {
  const errors: string[] = [
    ...value.interactionEvents.flatMap((event) => validateInteractionEvent(event)),
    ...value.feedbackEvents.flatMap((event) => validateFeedbackEvent(event)),
    ...validateNormalizedEventRange(value.range),
    ...validateEventExportProvenance(value.provenance, value.range),
    ...eventSequenceErrors([...value.interactionEvents, ...value.feedbackEvents])
  ];

  const rebuilt = buildNormalizedEventExport(value);
  if (canonicalJson(rebuilt.range) !== canonicalJson(value.range)) {
    errors.push("normalized event export range does not match the supplied events");
  }
  if (canonicalJson(rebuilt.provenance) !== canonicalJson(value.provenance)) {
    errors.push("normalized event export provenance does not match the supplied events");
  }

  return errors;
}

export function validateArtifactManifest(value: ArtifactManifestV1): string[] {
  const errors: string[] = [];
  pushWhenMissing(errors, value.contract === CONTRACT_IDS.artifactManifest, "artifact_manifest.v1 contract is required");
  pushWhenMissing(errors, value.packId.length > 0, "packId is required");
  pushWhenMissing(errors, value.immutable === true, "pack manifests must be immutable");
  pushWhenMissing(errors, value.runtimeAssets.graphPath.length > 0, "graphPath is required");
  pushWhenMissing(errors, value.runtimeAssets.vectorPath.length > 0, "vectorPath is required");
  pushWhenMissing(errors, value.payloadChecksums.graph.length > 0, "graph checksum is required");
  pushWhenMissing(errors, value.payloadChecksums.vector.length > 0, "vector checksum is required");
  pushWhenMissing(errors, isIsoDate(value.provenance.builtAt), "builtAt must be an ISO timestamp");
  pushWhenMissing(errors, value.provenance.workspaceSnapshot.length > 0, "workspaceSnapshot is required");
  pushWhenMissing(errors, value.graphDynamics.hebbian.learningRate >= 0, "hebbian learningRate must be non-negative");
  pushWhenMissing(errors, value.graphDynamics.decay.halfLifeDays >= 0, "decay halfLifeDays must be non-negative");
  errors.push(...validateNormalizedEventRange(value.provenance.eventRange));

  if (value.provenance.eventExports !== null) {
    errors.push(...validateEventExportProvenance(value.provenance.eventExports, value.provenance.eventRange));
  }

  if (value.routePolicy === "requires_learned_routing") {
    pushWhenMissing(errors, value.runtimeAssets.router.kind !== "none", "learned-routing packs require a router asset");
    pushWhenMissing(errors, value.runtimeAssets.router.identity !== null, "learned-routing packs require router identity");
    pushWhenMissing(errors, value.payloadChecksums.router !== null, "learned-routing packs require a router checksum");
  }

  if (value.runtimeAssets.router.kind === "none" && value.payloadChecksums.router !== null) {
    errors.push("router checksum must be null when no router asset exists");
  }

  return errors;
}

export function validateActivationPointerRecord(
  value: ActivationPointerRecordV1,
  expectedSlot?: ActivationPointerSlot
): string[] {
  const errors: string[] = [];
  const allowedSlots: ActivationPointerSlot[] = ["active", "candidate", "previous"];

  pushWhenMissing(errors, allowedSlots.includes(value.slot), "activation pointer slot must be active, candidate, or previous");
  pushWhenMissing(errors, value.packId.length > 0, "activation pointer packId is required");
  pushWhenMissing(errors, value.packRootDir.length > 0, "activation pointer packRootDir is required");
  pushWhenMissing(errors, value.manifestPath.length > 0, "activation pointer manifestPath is required");
  pushWhenMissing(errors, value.manifestPath.endsWith(".json"), "activation pointer manifestPath must target json");
  pushWhenMissing(
    errors,
    value.routePolicy === "heuristic_allowed" || value.routePolicy === "requires_learned_routing",
    "activation pointer routePolicy must be explicit"
  );
  pushWhenMissing(errors, isIsoDate(value.builtAt), "activation pointer builtAt must be an ISO timestamp");
  pushWhenMissing(errors, isIsoDate(value.updatedAt), "activation pointer updatedAt must be an ISO timestamp");
  pushWhenMissing(errors, value.eventRange.count > 0, "activation pointers require a non-empty eventRange");
  pushWhenMissing(errors, value.eventRange.start >= 0, "activation pointer eventRange.start must be non-negative");
  pushWhenMissing(errors, value.eventRange.end >= value.eventRange.start, "activation pointer eventRange.end must be >= start");

  if (expectedSlot !== undefined && value.slot !== expectedSlot) {
    errors.push(`activation pointer slot ${value.slot} does not match field ${expectedSlot}`);
  }

  if (value.routePolicy === "requires_learned_routing" && value.routerIdentity === null) {
    errors.push("learned-routing activation pointers require routerIdentity");
  }

  return errors;
}

export function validateActivationPointers(value: ActivationPointersV1): string[] {
  const errors: string[] = [];
  pushWhenMissing(errors, value.contract === CONTRACT_IDS.activationPointers, "activation_pointers.v1 contract is required");

  const seenPackIds = new Set<string>();
  for (const slot of ["active", "candidate", "previous"] as const) {
    const record = value[slot];
    if (record === null) {
      continue;
    }

    errors.push(...validateActivationPointerRecord(record, slot));
    if (seenPackIds.has(record.packId)) {
      errors.push(`activation pointers must not reuse packId across slots: ${record.packId}`);
    }
    seenPackIds.add(record.packId);
  }

  return errors;
}

export function validatePackGraphPayload(value: PackGraphPayloadV1, expectedPackId?: string): string[] {
  const errors: string[] = [];
  pushWhenMissing(errors, value.packId.length > 0, "graph packId is required");
  pushWhenMissing(errors, value.blocks.length > 0, "graph must contain at least one context block");
  if (expectedPackId !== undefined && value.packId !== expectedPackId) {
    errors.push(`graph packId ${value.packId} does not match manifest packId ${expectedPackId}`);
  }

  const seen = new Set<string>();
  for (const block of value.blocks) {
    pushWhenMissing(errors, block.id.length > 0, "graph blocks require id");
    pushWhenMissing(errors, block.source.length > 0, `graph block ${block.id || "<unknown>"} requires source`);
    pushWhenMissing(errors, block.text.length > 0, `graph block ${block.id || "<unknown>"} requires text`);
    pushWhenMissing(errors, block.priority >= 0, `graph block ${block.id || "<unknown>"} priority must be non-negative`);
    pushWhenMissing(errors, block.keywords.length > 0, `graph block ${block.id || "<unknown>"} requires keywords`);
    if (seen.has(block.id)) {
      errors.push(`graph block ids must be unique: ${block.id}`);
    }
    seen.add(block.id);
  }

  return errors;
}

export function validatePackVectorsPayload(value: PackVectorsPayloadV1, graph?: PackGraphPayloadV1): string[] {
  const errors: string[] = [];
  pushWhenMissing(errors, value.packId.length > 0, "vector packId is required");
  const seen = new Set<string>();
  const knownBlockIds = graph === undefined ? null : new Set(graph.blocks.map((block) => block.id));

  if (graph !== undefined && value.packId !== graph.packId) {
    errors.push(`vector packId ${value.packId} does not match graph packId ${graph.packId}`);
  }

  for (const entry of value.entries) {
    pushWhenMissing(errors, entry.blockId.length > 0, "vector entries require blockId");
    pushWhenMissing(errors, entry.keywords.length > 0, `vector entry ${entry.blockId || "<unknown>"} requires keywords`);
    pushWhenMissing(errors, entry.boost >= 0, `vector entry ${entry.blockId || "<unknown>"} boost must be non-negative`);
    if (seen.has(entry.blockId)) {
      errors.push(`vector entries must be unique per blockId: ${entry.blockId}`);
    }
    seen.add(entry.blockId);
    if (knownBlockIds !== null && !knownBlockIds.has(entry.blockId)) {
      errors.push(`vector entry references unknown blockId ${entry.blockId}`);
    }
  }

  return errors;
}

export function validateRouterArtifact(value: RouterArtifactV1, manifest?: ArtifactManifestV1): string[] {
  const errors: string[] = [];
  pushWhenMissing(errors, value.routerIdentity.length > 0, "routerIdentity is required");
  pushWhenMissing(errors, value.strategy === "keyword_overlap_v1", "router strategy must be keyword_overlap_v1");
  pushWhenMissing(errors, isIsoDate(value.trainedAt), "router trainedAt must be an ISO timestamp");

  if (manifest !== undefined) {
    if (manifest.routePolicy === "requires_learned_routing") {
      pushWhenMissing(errors, value.requiresLearnedRouting === true, "learned-routing manifests require router requiresLearnedRouting=true");
    }
    if (manifest.runtimeAssets.router.identity !== null && value.routerIdentity !== manifest.runtimeAssets.router.identity) {
      errors.push(`router identity ${value.routerIdentity} does not match manifest router identity ${manifest.runtimeAssets.router.identity}`);
    }
  }

  return errors;
}

export const FIXTURE_PACK_GRAPH: PackGraphPayloadV1 = {
  packId: "pack-fixture",
  blocks: [
    {
      id: "ctx-feedback-scanner",
      source: "memory/2026-03-05-openclawbrain-vnext-roadmap.md",
      text: "Unified feedback scanner runs against local session logs with Ollama qwen3.5:9b-q4_K_M and checkpointed replay.",
      keywords: ["feedback", "scanner", "session", "logs", "ollama", "qwen", "checkpoint"],
      priority: 5
    },
    {
      id: "ctx-runtime-compile",
      source: "docs/openclawbrain-openclaw-rearchitecture-execution-plan.md",
      text: "runtime_compile.v1 is the narrow OpenClaw to OpenClawBrain compile contract for promoted packs and manifest-gated routing.",
      keywords: ["runtime", "compile", "contract", "pack", "manifest", "routing", "openclaw"],
      priority: 4
    },
    {
      id: "ctx-structural-ops",
      source: "docs/openclawbrain-openclaw-rearchitecture-plan.md",
      text: "Structural graph operations like split, merge, prune, and connect remain first-class pack provenance.",
      keywords: ["structural", "split", "merge", "prune", "connect", "graph", "memory"],
      priority: 3
    }
  ]
};

export const FIXTURE_PACK_VECTORS: PackVectorsPayloadV1 = {
  packId: FIXTURE_PACK_GRAPH.packId,
  entries: [
    {
      blockId: "ctx-feedback-scanner",
      keywords: ["feedback", "scanner", "ollama", "qwen", "checkpoint", "sessions"],
      boost: 2,
      weights: {
        feedback: 5,
        scanner: 5,
        ollama: 4,
        qwen: 6,
        checkpoint: 3
      }
    },
    {
      blockId: "ctx-runtime-compile",
      keywords: ["runtime", "compile", "contract", "pack", "manifest", "routing", "openclaw"],
      boost: 2,
      weights: {
        runtime: 5,
        compile: 5,
        contract: 4,
        manifest: 4,
        pack: 3,
        routing: 3
      }
    },
    {
      blockId: "ctx-structural-ops",
      keywords: ["structural", "split", "merge", "prune", "connect", "graph", "memory"],
      boost: 1,
      weights: {
        structural: 5,
        split: 4,
        merge: 4,
        prune: 3,
        connect: 3,
        memory: 2
      }
    }
  ]
};

export const FIXTURE_ROUTER_ARTIFACT: RouterArtifactV1 = {
  routerIdentity: "pack-fixture:route_fn",
  strategy: "keyword_overlap_v1",
  trainedAt: "2026-03-06T00:00:00.000Z",
  requiresLearnedRouting: true
};

export const FIXTURE_INTERACTION_EVENTS: InteractionEventV1[] = [
  createInteractionEvent({
    eventId: "evt-interaction-fixture-1",
    agentId: "agent-fixture",
    sessionId: "session-fixture",
    channel: "whatsapp",
    sequence: 101,
    kind: "memory_compiled",
    createdAt: "2026-03-06T00:00:00.000Z",
    source: {
      runtimeOwner: "openclaw",
      stream: "openclaw/runtime/whatsapp"
    },
    packId: FIXTURE_PACK_GRAPH.packId
  }),
  createInteractionEvent({
    eventId: "evt-interaction-fixture-2",
    agentId: "agent-fixture",
    sessionId: "session-fixture",
    channel: "whatsapp",
    sequence: 103,
    kind: "message_delivered",
    createdAt: "2026-03-06T00:02:00.000Z",
    source: {
      runtimeOwner: "openclaw",
      stream: "openclaw/runtime/whatsapp"
    },
    packId: FIXTURE_PACK_GRAPH.packId,
    messageId: "msg-fixture-1"
  })
];

export const FIXTURE_FEEDBACK_EVENTS: FeedbackEventV1[] = [
  createFeedbackEvent({
    eventId: "evt-feedback-fixture-1",
    agentId: "agent-fixture",
    sessionId: "session-fixture",
    channel: "whatsapp",
    sequence: 102,
    kind: "teaching",
    createdAt: "2026-03-06T00:01:00.000Z",
    source: {
      runtimeOwner: "openclaw",
      stream: "openclaw/runtime/whatsapp"
    },
    content: "Use the unified feedback scanner before enabling default loop scans.",
    messageId: "msg-fixture-1",
    relatedInteractionId: (FIXTURE_INTERACTION_EVENTS[0] as InteractionEventV1).eventId
  }),
  createFeedbackEvent({
    eventId: "evt-feedback-fixture-2",
    agentId: "agent-fixture",
    sessionId: "session-fixture",
    channel: "whatsapp",
    sequence: 104,
    kind: "approval",
    createdAt: "2026-03-06T00:03:00.000Z",
    source: {
      runtimeOwner: "openclaw",
      stream: "openclaw/runtime/whatsapp"
    },
    content: "Learned routing promotion is approved after compile diagnostics stay stable.",
    relatedInteractionId: (FIXTURE_INTERACTION_EVENTS[1] as InteractionEventV1).eventId
  })
];

export const FIXTURE_NORMALIZED_EVENT_EXPORT: NormalizedEventExportV1 = buildNormalizedEventExport({
  interactionEvents: FIXTURE_INTERACTION_EVENTS,
  feedbackEvents: FIXTURE_FEEDBACK_EVENTS
});

export const FIXTURE_ARTIFACT_MANIFEST: ArtifactManifestV1 = {
  contract: CONTRACT_IDS.artifactManifest,
  packId: FIXTURE_PACK_GRAPH.packId,
  immutable: true,
  routePolicy: "requires_learned_routing",
  runtimeAssets: {
    graphPath: "graph.json",
    vectorPath: "vectors.json",
    router: {
      kind: "artifact",
      identity: FIXTURE_ROUTER_ARTIFACT.routerIdentity,
      artifactPath: "router/model.json"
    }
  },
  payloadChecksums: {
    graph: checksumJsonPayload(FIXTURE_PACK_GRAPH),
    vector: checksumJsonPayload(FIXTURE_PACK_VECTORS),
    router: checksumJsonPayload(FIXTURE_ROUTER_ARTIFACT)
  },
  modelFingerprints: ["BAAI/bge-large-en-v1.5", "ollama:qwen3.5:9b-q4_K_M", FIXTURE_ROUTER_ARTIFACT.routerIdentity],
  provenance: {
    workspaceSnapshot: "workspace-fixture",
    eventRange: FIXTURE_NORMALIZED_EVENT_EXPORT.range,
    eventExports: FIXTURE_NORMALIZED_EVENT_EXPORT.provenance,
    builtAt: "2026-03-06T00:00:00.000Z",
    offlineArtifacts: ["feedback_events.v1", "runtime_compile.v1"]
  },
  graphDynamics: {
    hebbian: {
      enabled: true,
      learningRate: 0.2
    },
    decay: {
      enabled: true,
      halfLifeDays: 30
    },
    structuralOps: {
      split: 1,
      merge: 0,
      prune: 2,
      connect: 3
    }
  }
};

export const FIXTURE_ACTIVATION_POINTERS: ActivationPointersV1 = {
  contract: CONTRACT_IDS.activationPointers,
  active: {
    slot: "active",
    packId: "pack-active",
    packRootDir: "/packs/pack-active",
    manifestPath: "/packs/pack-active/manifest.json",
    routePolicy: "heuristic_allowed",
    routerIdentity: null,
    eventRange: {
      start: 1,
      end: 25,
      count: 25
    },
    eventExportDigest: null,
    builtAt: "2026-03-06T00:00:00.000Z",
    updatedAt: "2026-03-06T00:00:00.000Z"
  },
  candidate: {
    slot: "candidate",
    packId: "pack-candidate",
    packRootDir: "/packs/pack-candidate",
    manifestPath: "/packs/pack-candidate/manifest.json",
    routePolicy: "requires_learned_routing",
    routerIdentity: "pack-candidate:route_fn",
    eventRange: {
      start: 26,
      end: 40,
      count: 15
    },
    eventExportDigest: "sha256-candidate-events",
    builtAt: "2026-03-06T00:05:00.000Z",
    updatedAt: "2026-03-06T00:05:00.000Z"
  },
  previous: {
    slot: "previous",
    packId: "pack-previous",
    packRootDir: "/packs/pack-previous",
    manifestPath: "/packs/pack-previous/manifest.json",
    routePolicy: "heuristic_allowed",
    routerIdentity: null,
    eventRange: {
      start: 0,
      end: 0,
      count: 1
    },
    eventExportDigest: null,
    builtAt: "2026-03-05T23:55:00.000Z",
    updatedAt: "2026-03-06T00:10:00.000Z"
  }
};

export const FIXTURE_RUNTIME_COMPILE_REQUEST: RuntimeCompileRequestV1 = {
  contract: CONTRACT_IDS.runtimeCompile,
  agentId: "agent-fixture",
  userMessage: "Compile scanner and manifest context for this turn.",
  maxContextBlocks: 2,
  modeRequested: "heuristic",
  runtimeHints: ["feedback scanner", "manifest"]
};

export const FIXTURE_RUNTIME_COMPILE_RESPONSE: RuntimeCompileResponseV1 = {
  contract: CONTRACT_IDS.runtimeCompile,
  packId: FIXTURE_ARTIFACT_MANIFEST.packId,
  selectedContext: FIXTURE_PACK_GRAPH.blocks.slice(0, 2).map((block) => ({
    id: block.id,
    source: block.source,
    text: block.text
  })),
  diagnostics: {
    modeRequested: "heuristic",
    modeEffective: "learned",
    usedLearnedRouteFn: true,
    routerIdentity: FIXTURE_ROUTER_ARTIFACT.routerIdentity,
    notes: ["selected_context_ids=ctx-feedback-scanner,ctx-runtime-compile", "router_strategy=keyword_overlap_v1"]
  }
};

export const FIXTURE_INTERACTION_EVENT: InteractionEventV1 = FIXTURE_INTERACTION_EVENTS[0] as InteractionEventV1;
export const FIXTURE_FEEDBACK_EVENT: FeedbackEventV1 = FIXTURE_FEEDBACK_EVENTS[0] as FeedbackEventV1;
