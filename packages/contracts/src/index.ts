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
export type RouterRefreshStatus = "updated" | "no_supervision";
export type RouterAssetKind = "none" | "stub" | "artifact";
export type ActivationPointerSlot = "active" | "candidate" | "previous";
export type InteractionEventKind = "memory_compiled" | "message_delivered" | "operator_override";
export type FeedbackEventKind = "correction" | "teaching" | "approval" | "suppression";
export type RouterSupervisionKind = "route_trace" | "human_feedback" | "operator_override" | "self_memory";
export type LearningBootProfile = "fast_boot_defaults";
export type LearningCadence = "passive_background";
export type LearningScanPolicy = "always_on";
export type LearningBlockRole =
  | "boot_default"
  | "background_expectation"
  | "label_surface"
  | "workspace"
  | "structural"
  | "interaction"
  | "feedback";

export interface LearningLabelSourcesV1 {
  human: string[];
  self: string[];
}

export interface LearningLabelHarvestV1 {
  humanLabels: number;
  selfLabels: number;
  corrections: number;
  teachings: number;
  approvals: number;
  suppressions: number;
  operatorOverrideLabels: number;
  memoryCompileLabels: number;
}

export interface LearningSurfaceV1 {
  bootProfile: LearningBootProfile;
  learningCadence: LearningCadence;
  scanPolicy: LearningScanPolicy;
  scanSurfaces: string[];
  labelSources: LearningLabelSourcesV1;
  labelHarvest: LearningLabelHarvestV1;
}

export interface PackBlockLearningSignalsV1 {
  role: LearningBlockRole;
  humanLabels: number;
  selfLabels: number;
  decayHalfLifeDays: number | null;
  hebbianPulse: number;
}
export type ContextCompactionMode = "none" | "native";

export interface RuntimeCompileRequestV1 {
  contract: typeof CONTRACT_IDS.runtimeCompile;
  agentId: string;
  userMessage: string;
  maxContextBlocks: number;
  maxContextChars?: number;
  modeRequested: RouteMode;
  activePackId?: string;
  runtimeHints?: string[];
  compactionMode?: ContextCompactionMode;
}

export interface RuntimeContextBlockV1 {
  id: string;
  source: string;
  text: string;
  tokenCount?: number;
  compactedFrom?: string[];
}

export interface PackContextBlockRecordV1 extends RuntimeContextBlockV1 {
  keywords: string[];
  priority: number;
  learning: PackBlockLearningSignalsV1;
}

export interface RuntimeCompileDiagnosticsV1 {
  modeRequested: RouteMode;
  modeEffective: RouteMode;
  usedLearnedRouteFn: boolean;
  routerIdentity: string | null;
  candidateCount: number;
  selectedCount: number;
  selectedCharCount: number;
  selectedTokenCount: number;
  selectionStrategy: "pack_route_fn_selection_v1";
  selectionDigest: string;
  compactionMode: ContextCompactionMode;
  compactionApplied: boolean;
  notes: string[];
}

export interface RuntimeCompileResponseV1 {
  contract: typeof CONTRACT_IDS.runtimeCompile;
  packId: string;
  selectedContext: RuntimeContextBlockV1[];
  diagnostics: RuntimeCompileDiagnosticsV1;
}

export interface RuntimeCompileTargetV1 {
  packId: string;
  routePolicy: RoutePolicy;
  routerIdentity: string | null;
  workspaceSnapshot: string;
  workspaceRevision: string | null;
  eventRange: Pick<NormalizedEventRangeV1, "start" | "end" | "count">;
  eventExportDigest: string | null;
  builtAt: string;
}

export interface RuntimeCompileExpectationV1 {
  packId?: string;
  routePolicy?: RoutePolicy;
  routerIdentity?: string | null;
  workspaceSnapshot?: string;
  workspaceRevision?: string | null;
  eventRange?: Pick<NormalizedEventRangeV1, "start" | "end" | "count">;
  eventExportDigest?: string | null;
  builtAt?: string;
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
  learningSurface: LearningSurfaceV1;
}

export interface NormalizedEventExportV1 {
  interactionEvents: InteractionEventV1[];
  feedbackEvents: FeedbackEventV1[];
  range: NormalizedEventRangeV1;
  provenance: EventExportProvenanceV1;
}

export interface WorkspaceMetadataV1 {
  workspaceId: string;
  snapshotId: string;
  capturedAt: string;
  rootDir: string;
  branch: string | null;
  revision: string | null;
  dirty: boolean;
  manifestDigest: string | null;
  labels: string[];
  files: string[];
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
    workspace: WorkspaceMetadataV1;
    workspaceSnapshot: string;
    eventRange: NormalizedEventRangeV1;
    eventExports: EventExportProvenanceV1 | null;
    learningSurface: LearningSurfaceV1;
    builtAt: string;
    offlineArtifacts: string[];
  };
  graphDynamics: {
    bootstrapping: {
      fastBootDefaults: boolean;
      passiveBackgroundLearning: boolean;
    };
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

export type ArtifactProvenanceV1 = ArtifactManifestV1["provenance"];

export interface ActivationPointerRecordV1 {
  slot: ActivationPointerSlot;
  packId: string;
  packRootDir: string;
  manifestPath: string;
  manifestDigest: string;
  routePolicy: RoutePolicy;
  routerIdentity: string | null;
  workspaceSnapshot: string;
  workspaceRevision: string | null;
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

export interface RouterPolicyUpdateV1 {
  blockId: string;
  delta: number;
  evidenceCount: number;
  rewardSum: number;
  tokenWeights: Record<string, number>;
  traceIds: string[];
}

export interface RouterTraceV1 {
  traceId: string;
  sourceEventId: string;
  sourceContract: EventContractId;
  sourceKind: InteractionEventKind | FeedbackEventKind;
  supervisionKind: RouterSupervisionKind;
  targetBlockIds: string[];
  reward: number;
  queryTokens: string[];
  queryVector: Record<string, number>;
}

export interface RouterRefreshDiagnosticsV1 {
  status: RouterRefreshStatus;
  eventExportDigest: string | null;
  routeTraceCount: number;
  supervisionCount: number;
  updateCount: number;
  queryChecksum: string;
  weightsChecksum: string;
  freshnessChecksum: string;
  noOpReason: string | null;
}

export interface RouterArtifactV1 {
  routerIdentity: string;
  strategy: "learned_route_fn_v1";
  trainedAt: string;
  requiresLearnedRouting: boolean;
  training: RouterRefreshDiagnosticsV1;
  traces: RouterTraceV1[];
  policyUpdates: RouterPolicyUpdateV1[];
}

function isIsoDate(value: string): boolean {
  return !Number.isNaN(Date.parse(value));
}

function hasOwn(value: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
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

export function createDefaultLearningSurface(scanSurfaces: readonly string[] = ["workspace_snapshot"]): LearningSurfaceV1 {
  const surfaces = uniqueInOrder(scanSurfaces.map((surface) => surface.trim()).filter((surface) => surface.length > 0));

  return {
    bootProfile: "fast_boot_defaults",
    learningCadence: "passive_background",
    scanPolicy: "always_on",
    scanSurfaces: surfaces.length === 0 ? ["workspace_snapshot"] : surfaces,
    labelSources: {
      human: [CONTRACT_IDS.feedbackEvents, `${CONTRACT_IDS.interactionEvents}:operator_override`],
      self: [`${CONTRACT_IDS.interactionEvents}:memory_compiled`]
    },
    labelHarvest: {
      humanLabels: 0,
      selfLabels: 0,
      corrections: 0,
      teachings: 0,
      approvals: 0,
      suppressions: 0,
      operatorOverrideLabels: 0,
      memoryCompileLabels: 0
    }
  };
}

export function buildLearningSurface(events: readonly NormalizedEventV1[]): LearningSurfaceV1 {
  if (events.length === 0) {
    return createDefaultLearningSurface(["event_export:empty"]);
  }

  const sorted = sortNormalizedEvents(events);
  const scanSurfaces = uniqueInOrder(sorted.map((event) => `${event.source.stream}:${event.kind}`));
  const humanSources: string[] = [];
  const selfSources: string[] = [];
  let corrections = 0;
  let teachings = 0;
  let approvals = 0;
  let suppressions = 0;
  let operatorOverrideLabels = 0;
  let memoryCompileLabels = 0;

  for (const event of sorted) {
    const surface = `${event.source.stream}:${event.kind}`;
    if (event.contract === CONTRACT_IDS.feedbackEvents) {
      humanSources.push(surface);
      switch (event.kind) {
        case "correction":
          corrections += 1;
          break;
        case "teaching":
          teachings += 1;
          break;
        case "approval":
          approvals += 1;
          break;
        case "suppression":
          suppressions += 1;
          break;
      }
      continue;
    }

    if (event.kind === "operator_override") {
      humanSources.push(surface);
      operatorOverrideLabels += 1;
      continue;
    }

    if (event.kind === "memory_compiled") {
      selfSources.push(surface);
      memoryCompileLabels += 1;
    }
  }

  return {
    bootProfile: "fast_boot_defaults",
    learningCadence: "passive_background",
    scanPolicy: "always_on",
    scanSurfaces,
    labelSources: {
      human: uniqueInOrder(humanSources),
      self: uniqueInOrder(selfSources)
    },
    labelHarvest: {
      humanLabels: corrections + teachings + approvals + suppressions + operatorOverrideLabels,
      selfLabels: memoryCompileLabels,
      corrections,
      teachings,
      approvals,
      suppressions,
      operatorOverrideLabels,
      memoryCompileLabels
    }
  };
}

export function canonicalJson(value: unknown): string {
  return `${JSON.stringify(value, null, 2)}\n`;
}

export function checksumJsonPayload(value: unknown): string {
  return `sha256-${createHash("sha256").update(canonicalJson(value)).digest("hex")}`;
}

export function computeRouterQueryChecksum(traces: readonly RouterTraceV1[]): string {
  return checksumJsonPayload(
    traces.map((trace) => ({
      traceId: trace.traceId,
      sourceEventId: trace.sourceEventId,
      sourceContract: trace.sourceContract,
      sourceKind: trace.sourceKind,
      supervisionKind: trace.supervisionKind,
      targetBlockIds: [...trace.targetBlockIds],
      reward: trace.reward,
      queryTokens: [...trace.queryTokens],
      queryVector: trace.queryVector
    }))
  );
}

export function computeRouterWeightsChecksum(policyUpdates: readonly RouterPolicyUpdateV1[]): string {
  return checksumJsonPayload(
    policyUpdates.map((update) => ({
      blockId: update.blockId,
      delta: update.delta,
      evidenceCount: update.evidenceCount,
      rewardSum: update.rewardSum,
      tokenWeights: update.tokenWeights,
      traceIds: [...update.traceIds]
    }))
  );
}

export function computeRouterFreshnessChecksum(input: {
  trainedAt: string;
  status: RouterRefreshStatus;
  eventExportDigest: string | null;
  routeTraceCount: number;
  supervisionCount: number;
  updateCount: number;
}): string {
  return checksumJsonPayload({
    trainedAt: input.trainedAt,
    status: input.status,
    eventExportDigest: input.eventExportDigest,
    routeTraceCount: input.routeTraceCount,
    supervisionCount: input.supervisionCount,
    updateCount: input.updateCount
  });
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
      }),
      learningSurface: buildLearningSurface(events)
    }
  };
}

function validateRuntimeContextBlock(value: RuntimeContextBlockV1, label: string): string[] {
  const errors: string[] = [];
  pushWhenMissing(errors, value.id.length > 0, `${label} id is required`);
  pushWhenMissing(errors, value.source.length > 0, `${label} source is required`);
  pushWhenMissing(errors, value.text.length > 0, `${label} text is required`);

  if (value.tokenCount !== undefined) {
    pushWhenMissing(errors, value.tokenCount >= 0, `${label} tokenCount must be non-negative`);
  }

  if (value.compactedFrom !== undefined) {
    pushWhenMissing(errors, value.compactedFrom.length > 0, `${label} compactedFrom must not be empty`);
    const uniqueIds = new Set(value.compactedFrom.filter((id) => id.length > 0));
    if (uniqueIds.size !== value.compactedFrom.length) {
      errors.push(`${label} compactedFrom must contain unique non-empty ids`);
    }
  }

  return errors;
}

function flattenRuntimeContextCoverageIds(value: Pick<RuntimeContextBlockV1, "id" | "compactedFrom">): string[] {
  return value.compactedFrom ?? [value.id];
}

export function validateRuntimeCompileRequest(value: RuntimeCompileRequestV1): string[] {
  const errors: string[] = [];
  pushWhenMissing(errors, value.contract === CONTRACT_IDS.runtimeCompile, "runtime_compile.v1 contract is required");
  pushWhenMissing(errors, value.agentId.length > 0, "agentId is required");
  pushWhenMissing(errors, value.userMessage.length > 0, "userMessage is required");
  pushWhenMissing(errors, value.maxContextBlocks >= 0, "maxContextBlocks must be non-negative");
  pushWhenMissing(errors, value.modeRequested === "heuristic" || value.modeRequested === "learned", "modeRequested must be heuristic or learned");

  if (value.activePackId !== undefined) {
    pushWhenMissing(errors, value.activePackId.length > 0, "activePackId must be non-empty when set");
  }

  if (value.maxContextChars !== undefined) {
    pushWhenMissing(errors, value.maxContextChars >= 0, "maxContextChars must be non-negative");
  }

  if (value.compactionMode !== undefined) {
    pushWhenMissing(errors, value.compactionMode === "none" || value.compactionMode === "native", "compactionMode must be none or native");
  }

  return errors;
}

export function validateRuntimeCompileExpectation(value: RuntimeCompileExpectationV1): string[] {
  const errors: string[] = [];

  if (value.packId !== undefined) {
    pushWhenMissing(errors, value.packId.length > 0, "runtime compile expectation packId must be non-empty when set");
  }
  if (value.routePolicy !== undefined) {
    pushWhenMissing(
      errors,
      value.routePolicy === "heuristic_allowed" || value.routePolicy === "requires_learned_routing",
      "runtime compile expectation routePolicy must be explicit when set"
    );
  }
  if (value.routerIdentity !== undefined && value.routerIdentity !== null) {
    pushWhenMissing(errors, value.routerIdentity.length > 0, "runtime compile expectation routerIdentity must be non-empty when set");
  }
  if (value.workspaceSnapshot !== undefined) {
    pushWhenMissing(errors, value.workspaceSnapshot.length > 0, "runtime compile expectation workspaceSnapshot must be non-empty when set");
  }
  if (value.workspaceRevision !== undefined && value.workspaceRevision !== null) {
    pushWhenMissing(errors, value.workspaceRevision.length > 0, "runtime compile expectation workspaceRevision must be non-empty when set");
  }
  if (value.eventRange !== undefined) {
    pushWhenMissing(errors, value.eventRange.count >= 0, "runtime compile expectation eventRange.count must be non-negative");
    pushWhenMissing(errors, value.eventRange.start >= 0, "runtime compile expectation eventRange.start must be non-negative");
    pushWhenMissing(
      errors,
      value.eventRange.end >= value.eventRange.start,
      "runtime compile expectation eventRange.end must be >= start"
    );
  }
  if (value.eventExportDigest !== undefined && value.eventExportDigest !== null) {
    pushWhenMissing(
      errors,
      value.eventExportDigest.length > 0,
      "runtime compile expectation eventExportDigest must be non-empty when set"
    );
  }
  if (value.builtAt !== undefined) {
    pushWhenMissing(errors, isIsoDate(value.builtAt), "runtime compile expectation builtAt must be an ISO timestamp when set");
  }

  return errors;
}

export function validateRuntimeCompileTargetExpectation(
  target: RuntimeCompileTargetV1,
  expectation: RuntimeCompileExpectationV1
): string[] {
  const errors = validateRuntimeCompileExpectation(expectation);

  if (expectation.packId !== undefined && target.packId !== expectation.packId) {
    errors.push(`runtime compile target packId ${target.packId} does not match expected ${expectation.packId}`);
  }
  if (expectation.routePolicy !== undefined && target.routePolicy !== expectation.routePolicy) {
    errors.push(`runtime compile target routePolicy ${target.routePolicy} does not match expected ${expectation.routePolicy}`);
  }
  if (expectation.routerIdentity !== undefined && target.routerIdentity !== expectation.routerIdentity) {
    errors.push(
      `runtime compile target routerIdentity ${target.routerIdentity ?? "null"} does not match expected ${expectation.routerIdentity ?? "null"}`
    );
  }
  if (expectation.workspaceSnapshot !== undefined && target.workspaceSnapshot !== expectation.workspaceSnapshot) {
    errors.push(
      `runtime compile target workspaceSnapshot ${target.workspaceSnapshot} does not match expected ${expectation.workspaceSnapshot}`
    );
  }
  if (expectation.workspaceRevision !== undefined && target.workspaceRevision !== expectation.workspaceRevision) {
    errors.push(
      `runtime compile target workspaceRevision ${target.workspaceRevision ?? "null"} does not match expected ${expectation.workspaceRevision ?? "null"}`
    );
  }
  if (expectation.eventRange !== undefined) {
    if (target.eventRange.start !== expectation.eventRange.start) {
      errors.push(
        `runtime compile target eventRange.start ${target.eventRange.start} does not match expected ${expectation.eventRange.start}`
      );
    }
    if (target.eventRange.end !== expectation.eventRange.end) {
      errors.push(`runtime compile target eventRange.end ${target.eventRange.end} does not match expected ${expectation.eventRange.end}`);
    }
    if (target.eventRange.count !== expectation.eventRange.count) {
      errors.push(
        `runtime compile target eventRange.count ${target.eventRange.count} does not match expected ${expectation.eventRange.count}`
      );
    }
  }
  if (expectation.eventExportDigest !== undefined && target.eventExportDigest !== expectation.eventExportDigest) {
    errors.push(
      `runtime compile target eventExportDigest ${target.eventExportDigest ?? "null"} does not match expected ${expectation.eventExportDigest ?? "null"}`
    );
  }
  if (expectation.builtAt !== undefined && target.builtAt !== expectation.builtAt) {
    errors.push(`runtime compile target builtAt ${target.builtAt} does not match expected ${expectation.builtAt}`);
  }

  return errors;
}

export function validateRuntimeCompileResponse(value: RuntimeCompileResponseV1): string[] {
  const errors: string[] = [];
  const selectedCoverage = new Map<string, string>();
  pushWhenMissing(errors, value.contract === CONTRACT_IDS.runtimeCompile, "runtime_compile.v1 contract is required");
  pushWhenMissing(errors, value.packId.length > 0, "packId is required");
  pushWhenMissing(errors, value.diagnostics.modeRequested === "heuristic" || value.diagnostics.modeRequested === "learned", "modeRequested must be explicit");
  pushWhenMissing(errors, value.diagnostics.modeEffective === "heuristic" || value.diagnostics.modeEffective === "learned", "modeEffective must be explicit");
  pushWhenMissing(errors, value.diagnostics.candidateCount >= 0, "candidateCount must be non-negative");
  pushWhenMissing(errors, value.diagnostics.selectedCount >= 0, "selectedCount must be non-negative");
  pushWhenMissing(errors, value.diagnostics.selectedCount === value.selectedContext.length, "selectedCount must match selectedContext length");
  pushWhenMissing(errors, value.diagnostics.candidateCount >= value.diagnostics.selectedCount, "candidateCount must be >= selectedCount");
  pushWhenMissing(errors, value.diagnostics.selectedCharCount >= 0, "selectedCharCount must be non-negative");
  pushWhenMissing(errors, value.diagnostics.selectedTokenCount >= 0, "selectedTokenCount must be non-negative");
  pushWhenMissing(errors, value.diagnostics.selectionStrategy === "pack_route_fn_selection_v1", "selectionStrategy must be pack_route_fn_selection_v1");
  pushWhenMissing(errors, value.diagnostics.selectionDigest.length > 0, "selectionDigest is required");
  pushWhenMissing(
    errors,
    value.diagnostics.compactionMode === "none" || value.diagnostics.compactionMode === "native",
    "compactionMode must be none or native"
  );

  if (value.diagnostics.modeEffective === "learned" && !value.diagnostics.usedLearnedRouteFn) {
    errors.push("learned mode requires usedLearnedRouteFn=true");
  }

  if (value.diagnostics.usedLearnedRouteFn && value.diagnostics.modeEffective !== "learned") {
    errors.push("usedLearnedRouteFn=true requires modeEffective=learned");
  }

  if (value.diagnostics.usedLearnedRouteFn && value.diagnostics.routerIdentity === null) {
    errors.push("learned routing requires routerIdentity");
  }

  value.selectedContext.forEach((block, index) => {
    errors.push(...validateRuntimeContextBlock(block, `selectedContext[${index}]`));

    for (const coverageId of flattenRuntimeContextCoverageIds(block)) {
      const existingBlockId = selectedCoverage.get(coverageId);
      if (existingBlockId !== undefined) {
        errors.push(`selectedContext[${index}] overlaps block ${existingBlockId} via ${coverageId}`);
        continue;
      }
      selectedCoverage.set(coverageId, block.id);
    }
  });

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

export function validateWorkspaceMetadata(value: WorkspaceMetadataV1): string[] {
  const errors: string[] = [];

  pushWhenMissing(errors, value.workspaceId.length > 0, "workspaceId is required");
  pushWhenMissing(errors, value.snapshotId.length > 0, "snapshotId is required");
  pushWhenMissing(errors, isIsoDate(value.capturedAt), "workspace capturedAt must be an ISO timestamp");
  pushWhenMissing(errors, value.rootDir.length > 0, "workspace rootDir is required");

  if (value.branch !== null && value.branch.length === 0) {
    errors.push("workspace branch must be null or non-empty");
  }
  if (value.revision !== null && value.revision.length === 0) {
    errors.push("workspace revision must be null or non-empty");
  }
  if (value.manifestDigest !== null && value.manifestDigest.length === 0) {
    errors.push("workspace manifestDigest must be null or non-empty");
  }
  if (value.labels.some((label) => label.length === 0)) {
    errors.push("workspace labels must be non-empty");
  }
  if (value.files.some((file) => file.length === 0)) {
    errors.push("workspace files must be non-empty");
  }

  return errors;
}

export function validateLearningSurface(value: LearningSurfaceV1): string[] {
  const errors: string[] = [];
  pushWhenMissing(errors, value.bootProfile === "fast_boot_defaults", "learning surface bootProfile must be fast_boot_defaults");
  pushWhenMissing(errors, value.learningCadence === "passive_background", "learning surface learningCadence must be passive_background");
  pushWhenMissing(errors, value.scanPolicy === "always_on", "learning surface scanPolicy must be always_on");
  pushWhenMissing(errors, value.scanSurfaces.length > 0, "learning surface requires at least one scan surface");

  for (const surface of value.scanSurfaces) {
    pushWhenMissing(errors, surface.length > 0, "learning surface scan surfaces must be non-empty");
  }
  for (const source of value.labelSources.human) {
    pushWhenMissing(errors, source.length > 0, "learning surface human label sources must be non-empty");
  }
  for (const source of value.labelSources.self) {
    pushWhenMissing(errors, source.length > 0, "learning surface self label sources must be non-empty");
  }

  pushWhenMissing(errors, value.labelHarvest.humanLabels >= 0, "learning surface humanLabels must be non-negative");
  pushWhenMissing(errors, value.labelHarvest.selfLabels >= 0, "learning surface selfLabels must be non-negative");
  pushWhenMissing(errors, value.labelHarvest.corrections >= 0, "learning surface corrections must be non-negative");
  pushWhenMissing(errors, value.labelHarvest.teachings >= 0, "learning surface teachings must be non-negative");
  pushWhenMissing(errors, value.labelHarvest.approvals >= 0, "learning surface approvals must be non-negative");
  pushWhenMissing(errors, value.labelHarvest.suppressions >= 0, "learning surface suppressions must be non-negative");
  pushWhenMissing(errors, value.labelHarvest.operatorOverrideLabels >= 0, "learning surface operatorOverrideLabels must be non-negative");
  pushWhenMissing(errors, value.labelHarvest.memoryCompileLabels >= 0, "learning surface memoryCompileLabels must be non-negative");

  const humanLabels =
    value.labelHarvest.corrections +
    value.labelHarvest.teachings +
    value.labelHarvest.approvals +
    value.labelHarvest.suppressions +
    value.labelHarvest.operatorOverrideLabels;
  const selfLabels = value.labelHarvest.memoryCompileLabels;

  if (value.labelHarvest.humanLabels !== humanLabels) {
    errors.push("learning surface humanLabels must equal feedback and operatorOverride labels");
  }
  if (value.labelHarvest.selfLabels !== selfLabels) {
    errors.push("learning surface selfLabels must equal memoryCompileLabels");
  }

  return errors;
}

export function validatePackBlockLearningSignals(value: PackBlockLearningSignalsV1, blockId?: string): string[] {
  const errors: string[] = [];
  const prefix = blockId === undefined ? "pack block learning" : `pack block ${blockId}`;
  const allowedRoles: LearningBlockRole[] = [
    "boot_default",
    "background_expectation",
    "label_surface",
    "workspace",
    "structural",
    "interaction",
    "feedback"
  ];

  pushWhenMissing(errors, allowedRoles.includes(value.role), `${prefix} role must be explicit`);
  pushWhenMissing(errors, value.humanLabels >= 0, `${prefix} humanLabels must be non-negative`);
  pushWhenMissing(errors, value.selfLabels >= 0, `${prefix} selfLabels must be non-negative`);
  pushWhenMissing(errors, value.hebbianPulse >= 0, `${prefix} hebbianPulse must be non-negative`);
  if (value.decayHalfLifeDays !== null) {
    pushWhenMissing(errors, value.decayHalfLifeDays >= 0, `${prefix} decayHalfLifeDays must be non-negative when set`);
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
  errors.push(...validateLearningSurface(value.learningSurface));

  for (const stream of value.sourceStreams) {
    pushWhenMissing(errors, stream.length > 0, "event export provenance source streams must be non-empty");
    if (!value.learningSurface.scanSurfaces.some((surface) => surface.startsWith(`${stream}:`))) {
      errors.push(`event export provenance learningSurface must include a scan surface for ${stream}`);
    }
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
  if (eventRange !== undefined) {
    if (value.interactionCount + value.feedbackCount !== eventRange.count) {
      errors.push("event export provenance counts must match eventRange.count");
    }
    if (value.learningSurface.labelHarvest.humanLabels + value.learningSurface.labelHarvest.selfLabels > eventRange.count) {
      errors.push("learning surface labels cannot exceed eventRange.count");
    }
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
  errors.push(...validateWorkspaceMetadata(value.provenance.workspace));
  errors.push(...validateLearningSurface(value.provenance.learningSurface));
  pushWhenMissing(errors, value.graphDynamics.bootstrapping.fastBootDefaults === true, "graph bootstrapping fastBootDefaults must stay enabled");
  pushWhenMissing(
    errors,
    value.graphDynamics.bootstrapping.passiveBackgroundLearning === true,
    "graph bootstrapping passiveBackgroundLearning must stay enabled"
  );
  pushWhenMissing(errors, value.graphDynamics.hebbian.learningRate >= 0, "hebbian learningRate must be non-negative");
  pushWhenMissing(errors, value.graphDynamics.decay.halfLifeDays >= 0, "decay halfLifeDays must be non-negative");
  errors.push(...validateNormalizedEventRange(value.provenance.eventRange));

  if (value.provenance.workspace.snapshotId !== value.provenance.workspaceSnapshot) {
    errors.push("workspaceSnapshot must match provenance.workspace.snapshotId");
  }

  if (value.provenance.eventExports !== null) {
    errors.push(...validateEventExportProvenance(value.provenance.eventExports, value.provenance.eventRange));
    if (canonicalJson(value.provenance.eventExports.learningSurface) !== canonicalJson(value.provenance.learningSurface)) {
      errors.push("artifact provenance learningSurface must match event export learningSurface");
    }
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
  pushWhenMissing(errors, value.manifestDigest.startsWith("sha256-"), "activation pointer manifestDigest must be a sha256 digest");
  pushWhenMissing(
    errors,
    value.routePolicy === "heuristic_allowed" || value.routePolicy === "requires_learned_routing",
    "activation pointer routePolicy must be explicit"
  );
  pushWhenMissing(errors, value.workspaceSnapshot.length > 0, "activation pointer workspaceSnapshot is required");
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
    errors.push(...validatePackBlockLearningSignals(block.learning, block.id));
    if (block.tokenCount !== undefined) {
      pushWhenMissing(errors, block.tokenCount >= 0, `graph block ${block.id || "<unknown>"} tokenCount must be non-negative`);
    }
    if (block.compactedFrom !== undefined) {
      pushWhenMissing(errors, block.compactedFrom.length > 0, `graph block ${block.id || "<unknown>"} compactedFrom must not be empty`);
      const compactedFrom = new Set(block.compactedFrom.filter((id) => id.length > 0));
      if (compactedFrom.size !== block.compactedFrom.length) {
        errors.push(`graph block ${block.id || "<unknown>"} compactedFrom must contain unique non-empty ids`);
      }
    }
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
  pushWhenMissing(errors, value.strategy === "learned_route_fn_v1", "router strategy must be learned_route_fn_v1");
  pushWhenMissing(errors, isIsoDate(value.trainedAt), "router trainedAt must be an ISO timestamp");
  pushWhenMissing(
    errors,
    value.training.status === "updated" || value.training.status === "no_supervision",
    "router training status must be updated or no_supervision"
  );
  pushWhenMissing(errors, value.training.routeTraceCount === value.traces.length, "router routeTraceCount must match trace count");
  pushWhenMissing(errors, value.training.updateCount === value.policyUpdates.length, "router updateCount must match policyUpdates length");

  const supervisionCount = value.traces.filter((trace) => trace.supervisionKind !== "route_trace" && trace.reward !== 0).length;
  pushWhenMissing(errors, value.training.supervisionCount === supervisionCount, "router supervisionCount must match supervised traces");
  pushWhenMissing(errors, value.training.queryChecksum === computeRouterQueryChecksum(value.traces), "router queryChecksum does not match traces");
  pushWhenMissing(
    errors,
    value.training.weightsChecksum === computeRouterWeightsChecksum(value.policyUpdates),
    "router weightsChecksum does not match policyUpdates"
  );
  pushWhenMissing(
    errors,
    value.training.freshnessChecksum ===
      computeRouterFreshnessChecksum({
        trainedAt: value.trainedAt,
        status: value.training.status,
        eventExportDigest: value.training.eventExportDigest,
        routeTraceCount: value.training.routeTraceCount,
        supervisionCount: value.training.supervisionCount,
        updateCount: value.training.updateCount
      }),
    "router freshnessChecksum does not match router freshness metadata"
  );

  if (value.training.status === "updated") {
    pushWhenMissing(errors, value.training.updateCount > 0, "updated routers must record at least one policy update");
    pushWhenMissing(errors, value.training.noOpReason === null, "updated routers must not set noOpReason");
  }
  if (value.training.status === "no_supervision") {
    pushWhenMissing(errors, value.training.updateCount === 0, "no-supervision routers must not record policy updates");
    pushWhenMissing(
      errors,
      typeof value.training.noOpReason === "string" && value.training.noOpReason.length > 0,
      "no-supervision routers must expose a non-empty noOpReason"
    );
  }

  const seenTraceIds = new Set<string>();
  for (const trace of value.traces) {
    pushWhenMissing(errors, trace.traceId.length > 0, "router traces require traceId");
    pushWhenMissing(errors, trace.sourceEventId.length > 0, "router traces require sourceEventId");
    pushWhenMissing(errors, trace.sourceKind.length > 0, "router traces require sourceKind");
    pushWhenMissing(
      errors,
      trace.supervisionKind === "route_trace" ||
        trace.supervisionKind === "human_feedback" ||
        trace.supervisionKind === "operator_override" ||
        trace.supervisionKind === "self_memory",
      "router traces require a supported supervisionKind"
    );
    pushWhenMissing(errors, Number.isFinite(trace.reward), "router traces require finite reward values");
    if (seenTraceIds.has(trace.traceId)) {
      errors.push(`duplicate router traceId ${trace.traceId}`);
    }
    seenTraceIds.add(trace.traceId);
    for (const weight of Object.values(trace.queryVector)) {
      pushWhenMissing(errors, Number.isFinite(weight), "router trace queryVector values must be finite");
    }
  }

  const seenBlockIds = new Set<string>();
  for (const update of value.policyUpdates) {
    pushWhenMissing(errors, update.blockId.length > 0, "router policyUpdates require blockId");
    pushWhenMissing(errors, Number.isFinite(update.delta), "router policyUpdates require finite delta values");
    pushWhenMissing(errors, Number.isFinite(update.rewardSum), "router policyUpdates require finite rewardSum values");
    pushWhenMissing(errors, update.evidenceCount > 0, "router policyUpdates require evidenceCount > 0");
    if (seenBlockIds.has(update.blockId)) {
      errors.push(`duplicate router policy update blockId ${update.blockId}`);
    }
    seenBlockIds.add(update.blockId);
    for (const weight of Object.values(update.tokenWeights)) {
      pushWhenMissing(errors, Number.isFinite(weight), "router policy update tokenWeights must be finite");
    }
  }

  if (manifest !== undefined) {
    if (manifest.routePolicy === "requires_learned_routing") {
      pushWhenMissing(errors, value.requiresLearnedRouting === true, "learned-routing manifests require router requiresLearnedRouting=true");
    }
    if (manifest.runtimeAssets.router.identity !== null && value.routerIdentity !== manifest.runtimeAssets.router.identity) {
      errors.push(`router identity ${value.routerIdentity} does not match manifest router identity ${manifest.runtimeAssets.router.identity}`);
    }
    if (
      manifest.provenance.eventExports?.exportDigest !== undefined &&
      manifest.provenance.eventExports !== null &&
      value.training.eventExportDigest !== manifest.provenance.eventExports.exportDigest
    ) {
      errors.push(
        `router eventExportDigest ${value.training.eventExportDigest ?? "null"} does not match manifest event export digest ${manifest.provenance.eventExports.exportDigest}`
      );
    }
  }

  return errors;
}

export const FIXTURE_PACK_GRAPH: PackGraphPayloadV1 = {
  packId: "pack-fixture",
  blocks: [
    {
      id: "ctx-feedback-scanner",
      source: "docs/openclaw-attach-quickstart.md",
      text: "Always-on feedback scanner harvests human labels from local session logs with Ollama qwen3.5:9b-q4_K_M and checkpointed replay.",
      keywords: ["feedback", "scanner", "always-on", "session", "logs", "ollama", "qwen", "checkpoint"],
      priority: 5,
      tokenCount: 16,
      learning: {
        role: "label_surface",
        humanLabels: 2,
        selfLabels: 0,
        decayHalfLifeDays: 30,
        hebbianPulse: 5
      }
    },
    {
      id: "ctx-runtime-compile",
      source: "docs/contracts-v1.md",
      text: "runtime_compile.v1 keeps fast boot defaults available while passive background learning hydrates promoted packs, explicit budgets, and manifest-gated routing.",
      keywords: ["runtime", "compile", "fast", "boot", "passive", "background", "pack", "manifest", "routing", "openclaw", "budget"],
      priority: 4,
      tokenCount: 19,
      learning: {
        role: "boot_default",
        humanLabels: 0,
        selfLabels: 0,
        decayHalfLifeDays: null,
        hebbianPulse: 2
      }
    },
    {
      id: "ctx-structural-ops",
      source: "docs/learning-first-convergence.md",
      text: "Structural graph operations like split, merge, prune, and connect stay first-class beside Hebbian reinforcement and decay.",
      keywords: ["structural", "split", "merge", "prune", "connect", "graph", "memory", "hebbian", "decay"],
      priority: 3,
      tokenCount: 15,
      learning: {
        role: "structural",
        humanLabels: 0,
        selfLabels: 0,
        decayHalfLifeDays: 30,
        hebbianPulse: 4
      }
    },
    {
      id: "ctx-context-compact",
      source: "pack/pack-fixture:structural-compaction",
      text: "Compacted pack context keeps fast boot defaults and passive background learning deterministic across human label, self label, and structural graph sources.",
      keywords: ["pack", "structural", "compaction", "context", "deterministic", "fast", "boot", "background", "labels"],
      priority: 4,
      tokenCount: 18,
      compactedFrom: ["ctx-feedback-scanner", "ctx-runtime-compile", "ctx-structural-ops"],
      learning: {
        role: "background_expectation",
        humanLabels: 2,
        selfLabels: 1,
        decayHalfLifeDays: 30,
        hebbianPulse: 4
      }
    }
  ]
};

export const FIXTURE_PACK_VECTORS: PackVectorsPayloadV1 = {
  packId: FIXTURE_PACK_GRAPH.packId,
  entries: [
    {
      blockId: "ctx-feedback-scanner",
      keywords: ["feedback", "scanner", "human_label", "always_on", "ollama", "qwen", "checkpoint", "sessions"],
      boost: 4,
      weights: {
        feedback: 5,
        scanner: 5,
        human_label: 6,
        always_on: 4,
        ollama: 4,
        qwen: 6,
        checkpoint: 3
      }
    },
    {
      blockId: "ctx-runtime-compile",
      keywords: ["runtime", "compile", "fast_boot", "passive_background", "pack", "manifest", "routing", "openclaw", "budget"],
      boost: 3,
      weights: {
        runtime: 5,
        compile: 5,
        fast_boot: 5,
        passive_background: 4,
        manifest: 4,
        pack: 3,
        routing: 3,
        budget: 3
      }
    },
    {
      blockId: "ctx-structural-ops",
      keywords: ["structural", "split", "merge", "prune", "connect", "graph", "memory", "hebbian", "decay"],
      boost: 2,
      weights: {
        structural: 5,
        split: 4,
        merge: 4,
        prune: 3,
        connect: 3,
        hebbian: 4,
        decay: 3,
        memory: 2
      }
    },
    {
      blockId: "ctx-context-compact",
      keywords: ["pack", "structural", "compaction", "context", "deterministic", "fast_boot", "passive_background", "human_label", "self_label"],
      boost: 4,
      weights: {
        structural: 5,
        compaction: 5,
        context: 4,
        deterministic: 4,
        fast_boot: 4,
        passive_background: 4,
        human_label: 3,
        self_label: 3
      }
    }
  ]
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

const FIXTURE_ROUTER_TRACES: RouterTraceV1[] = [
  {
    traceId: "trace-fixture-memory-compiled",
    sourceEventId: "evt-interaction-fixture-1",
    sourceContract: CONTRACT_IDS.interactionEvents,
    sourceKind: "memory_compiled",
    supervisionKind: "self_memory",
    targetBlockIds: ["ctx-runtime-compile", "ctx-context-compact"],
    reward: 2,
    queryTokens: ["memory", "compiled", "runtime", "pack"],
    queryVector: {
      memory: 2,
      compiled: 1,
      runtime: 1,
      pack: 1
    }
  },
  {
    traceId: "trace-fixture-feedback-scanner",
    sourceEventId: "evt-feedback-fixture-1",
    sourceContract: CONTRACT_IDS.feedbackEvents,
    sourceKind: "teaching",
    supervisionKind: "human_feedback",
    targetBlockIds: ["ctx-feedback-scanner", "ctx-runtime-compile"],
    reward: 4,
    queryTokens: ["feedback", "scanner", "default", "loop", "scans"],
    queryVector: {
      feedback: 3,
      scanner: 3,
      default: 1,
      loop: 1,
      scans: 1
    }
  },
  {
    traceId: "trace-fixture-approval-route",
    sourceEventId: "evt-feedback-fixture-2",
    sourceContract: CONTRACT_IDS.feedbackEvents,
    sourceKind: "approval",
    supervisionKind: "human_feedback",
    targetBlockIds: ["ctx-runtime-compile", "ctx-context-compact"],
    reward: 2,
    queryTokens: ["learned", "routing", "promotion", "compile", "diagnostics"],
    queryVector: {
      learned: 1,
      routing: 2,
      promotion: 1,
      compile: 2,
      diagnostics: 1
    }
  },
  {
    traceId: "trace-fixture-message-route",
    sourceEventId: "evt-interaction-fixture-2",
    sourceContract: CONTRACT_IDS.interactionEvents,
    sourceKind: "message_delivered",
    supervisionKind: "route_trace",
    targetBlockIds: ["ctx-context-compact"],
    reward: 0,
    queryTokens: ["message", "delivered", "route"],
    queryVector: {
      message: 1,
      delivered: 1,
      route: 1
    }
  }
];

const FIXTURE_ROUTER_POLICY_UPDATES: RouterPolicyUpdateV1[] = [
  {
    blockId: "ctx-feedback-scanner",
    delta: 11,
    evidenceCount: 2,
    rewardSum: 6,
    tokenWeights: {
      feedback: 5,
      scanner: 5,
      default: 1
    },
    traceIds: ["trace-fixture-feedback-scanner", "trace-fixture-approval-route"]
  },
  {
    blockId: "ctx-runtime-compile",
    delta: 8,
    evidenceCount: 3,
    rewardSum: 8,
    tokenWeights: {
      runtime: 2,
      compile: 3,
      routing: 2,
      promotion: 1
    },
    traceIds: ["trace-fixture-memory-compiled", "trace-fixture-feedback-scanner", "trace-fixture-approval-route"]
  },
  {
    blockId: "ctx-context-compact",
    delta: 5,
    evidenceCount: 2,
    rewardSum: 4,
    tokenWeights: {
      pack: 1,
      compile: 1,
      diagnostics: 1,
      route: 2
    },
    traceIds: ["trace-fixture-memory-compiled", "trace-fixture-approval-route"]
  }
];

export const FIXTURE_ROUTER_ARTIFACT: RouterArtifactV1 = {
  routerIdentity: "pack-fixture:route_fn",
  strategy: "learned_route_fn_v1",
  trainedAt: "2026-03-06T00:00:00.000Z",
  requiresLearnedRouting: true,
  training: {
    status: "updated",
    eventExportDigest: FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.exportDigest,
    routeTraceCount: FIXTURE_ROUTER_TRACES.length,
    supervisionCount: FIXTURE_ROUTER_TRACES.filter((trace) => trace.supervisionKind !== "route_trace" && trace.reward !== 0).length,
    updateCount: FIXTURE_ROUTER_POLICY_UPDATES.length,
    queryChecksum: computeRouterQueryChecksum(FIXTURE_ROUTER_TRACES),
    weightsChecksum: computeRouterWeightsChecksum(FIXTURE_ROUTER_POLICY_UPDATES),
    freshnessChecksum: computeRouterFreshnessChecksum({
      trainedAt: "2026-03-06T00:00:00.000Z",
      status: "updated",
      eventExportDigest: FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.exportDigest,
      routeTraceCount: FIXTURE_ROUTER_TRACES.length,
      supervisionCount: FIXTURE_ROUTER_TRACES.filter((trace) => trace.supervisionKind !== "route_trace" && trace.reward !== 0).length,
      updateCount: FIXTURE_ROUTER_POLICY_UPDATES.length
    }),
    noOpReason: null
  },
  traces: FIXTURE_ROUTER_TRACES,
  policyUpdates: FIXTURE_ROUTER_POLICY_UPDATES
};

export const FIXTURE_WORKSPACE_METADATA: WorkspaceMetadataV1 = {
  workspaceId: "workspace-fixture",
  snapshotId: "workspace-fixture@snapshot-2026-03-06",
  capturedAt: "2026-03-06T00:00:00.000Z",
  rootDir: "/workspace/openclawbrain",
  branch: "codex/20260306/ts-public-converge",
  revision: "fixture-rev-20260306",
  dirty: false,
  manifestDigest: "sha256-workspace-fixture",
  labels: ["learning-first", "public-surface"],
  files: ["README.md", "packages/contracts/src/index.ts", "packages/learner/src/index.ts"]
};

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
    workspace: FIXTURE_WORKSPACE_METADATA,
    workspaceSnapshot: FIXTURE_WORKSPACE_METADATA.snapshotId,
    eventRange: FIXTURE_NORMALIZED_EVENT_EXPORT.range,
    eventExports: FIXTURE_NORMALIZED_EVENT_EXPORT.provenance,
    learningSurface: FIXTURE_NORMALIZED_EVENT_EXPORT.provenance.learningSurface,
    builtAt: "2026-03-06T00:00:00.000Z",
    offlineArtifacts: ["feedback_events.v1", "runtime_compile.v1"]
  },
  graphDynamics: {
    bootstrapping: {
      fastBootDefaults: true,
      passiveBackgroundLearning: true
    },
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
    manifestDigest: "sha256-pack-active-manifest",
    routePolicy: "heuristic_allowed",
    routerIdentity: null,
    workspaceSnapshot: "workspace-active@snapshot-2026-03-06",
    workspaceRevision: "workspace-active-rev",
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
    manifestDigest: "sha256-pack-candidate-manifest",
    routePolicy: "requires_learned_routing",
    routerIdentity: "pack-candidate:route_fn",
    workspaceSnapshot: FIXTURE_WORKSPACE_METADATA.snapshotId,
    workspaceRevision: FIXTURE_WORKSPACE_METADATA.revision,
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
    manifestDigest: "sha256-pack-previous-manifest",
    routePolicy: "heuristic_allowed",
    routerIdentity: null,
    workspaceSnapshot: "workspace-previous@snapshot-2026-03-05",
    workspaceRevision: "workspace-previous-rev",
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
  maxContextBlocks: 3,
  maxContextChars: 240,
  modeRequested: "heuristic",
  runtimeHints: ["feedback scanner", "manifest", "structural compaction"],
  compactionMode: "native"
};

const FIXTURE_RUNTIME_COMPILE_SELECTED_CONTEXT: RuntimeContextBlockV1[] = [
  {
    id: "ctx-context-compact",
    source: "pack/pack-fixture:structural-compaction",
    text: "Pack-backed structural compaction keeps larger context windows deterministic across feedback, runtime compile, and structural graph sources.",
    tokenCount: 16,
    compactedFrom: ["ctx-feedback-scanner", "ctx-runtime-compile", "ctx-structural-ops"]
  }
];

export const FIXTURE_RUNTIME_COMPILE_RESPONSE: RuntimeCompileResponseV1 = {
  contract: CONTRACT_IDS.runtimeCompile,
  packId: FIXTURE_ARTIFACT_MANIFEST.packId,
  selectedContext: FIXTURE_RUNTIME_COMPILE_SELECTED_CONTEXT,
  diagnostics: {
    modeRequested: "heuristic",
    modeEffective: "learned",
    usedLearnedRouteFn: true,
    routerIdentity: FIXTURE_ROUTER_ARTIFACT.routerIdentity,
    candidateCount: FIXTURE_PACK_GRAPH.blocks.length,
    selectedCount: FIXTURE_RUNTIME_COMPILE_SELECTED_CONTEXT.length,
    selectedCharCount: FIXTURE_RUNTIME_COMPILE_SELECTED_CONTEXT.reduce((sum, block) => sum + block.text.length, 0),
    selectedTokenCount: FIXTURE_RUNTIME_COMPILE_SELECTED_CONTEXT.reduce((sum, block) => sum + (block.tokenCount ?? 0), 0),
    selectionStrategy: "pack_route_fn_selection_v1",
    selectionDigest: checksumJsonPayload({
      packId: FIXTURE_ARTIFACT_MANIFEST.packId,
      selectedContext: FIXTURE_RUNTIME_COMPILE_SELECTED_CONTEXT
    }),
    compactionMode: "native",
    compactionApplied: false,
    notes: [
      "selected_context_ids=ctx-context-compact",
      "selection_mode=token_match(feedback,scanner,manifest,structural,compaction)",
      "selection_tiers=token_match_only",
      "selection_strategy=pack_route_fn_selection_v1",
      "selection_compaction_deduped=3",
      "router_strategy=learned_route_fn_v1"
    ]
  }
};

export const FIXTURE_INTERACTION_EVENT: InteractionEventV1 = FIXTURE_INTERACTION_EVENTS[0] as InteractionEventV1;
export const FIXTURE_FEEDBACK_EVENT: FeedbackEventV1 = FIXTURE_FEEDBACK_EVENTS[0] as FeedbackEventV1;
