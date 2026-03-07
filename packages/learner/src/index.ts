import { mkdirSync, rmSync } from "node:fs";
import path from "node:path";

import {
  CONTRACT_IDS,
  checksumJsonPayload,
  sortNormalizedEvents,
  type ArtifactManifestV1,
  type FeedbackEventV1,
  type InteractionEventV1,
  type LearningSurfaceV1,
  type NormalizedEventExportV1,
  type NormalizedEventV1,
  type PackBlockLearningSignalsV1,
  type PackContextBlockRecordV1,
  type PackGraphPayloadV1,
  type PackVectorEntryV1,
  type PackVectorsPayloadV1,
  type RouterArtifactV1
} from "@openclawbrain/contracts";
import {
  buildNormalizedEventDedupId,
  buildNormalizedEventExport,
  buildNormalizedEventExportBridge,
  createDefaultLearningSurface,
  createEventExportCursor,
  createExplicitEventRange,
  validateNormalizedEventExport,
  validateNormalizedEventExportBridge,
  validateNormalizedEventExportSlice,
  type EventExportCursorV1,
  type EventExportLaneV1,
  type NormalizedEventExportBridgeV1,
  type NormalizedEventExportSliceV1
} from "@openclawbrain/event-export";
import { computePayloadChecksum, loadPack, PACK_LAYOUT, type PackDescriptor, writePackFile } from "@openclawbrain/pack-format";
import { buildArtifactProvenance } from "@openclawbrain/provenance";
import { createWorkspaceMetadata, type WorkspaceMetadataInput } from "@openclawbrain/workspace-metadata";

export interface CandidatePackEventExports {
  interactionEvents: InteractionEventV1[];
  feedbackEvents: FeedbackEventV1[];
}

export interface CandidatePackBuildInput {
  packLabel: string;
  workspace: WorkspaceMetadataInput;
  eventRange: {
    start: number;
    end: number;
  };
  eventExports?: CandidatePackEventExports;
  learnedRouting: boolean;
  builtAt?: string;
  offlineArtifacts?: string[];
  structuralOps?: Partial<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
}

export interface CandidatePackFromNormalizedEventExportInput {
  packLabel: string;
  workspace: WorkspaceMetadataInput;
  normalizedEventExport: NormalizedEventExportV1;
  learnedRouting: boolean;
  builtAt?: string;
  offlineArtifacts?: string[];
  structuralOps?: Partial<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
}

interface CandidatePackBridgeInputBase {
  packLabel: string;
  workspace: WorkspaceMetadataInput;
  learnedRouting: boolean;
  builtAt?: string;
  offlineArtifacts?: string[];
  structuralOps?: Partial<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
}

export interface CandidatePackFromNormalizedEventExportSliceInput extends CandidatePackBridgeInputBase {
  normalizedEventExportSlice: NormalizedEventExportSliceV1;
}

export interface CandidatePackBundleFromNormalizedEventExportBridgeInput extends CandidatePackBridgeInputBase {
  normalizedEventExportBridge: NormalizedEventExportBridgeV1;
}

export interface CandidatePackPayloads {
  graph: PackGraphPayloadV1;
  vectors: PackVectorsPayloadV1;
  router: RouterArtifactV1 | null;
}

export interface CandidatePackBuildResult {
  manifest: ArtifactManifestV1;
  payloads: CandidatePackPayloads;
  summary: {
    packId: string;
    immutable: true;
    routePolicy: ArtifactManifestV1["routePolicy"];
    workspaceSnapshot: string;
    eventRange: ArtifactManifestV1["provenance"]["eventRange"];
    eventExportDigest: string | null;
    learningSurface: ArtifactManifestV1["provenance"]["learningSurface"];
    bootstrapping: ArtifactManifestV1["graphDynamics"]["bootstrapping"];
  };
}

export interface CandidatePackBundleEntry {
  lane: EventExportLaneV1;
  sliceId: string;
  packLabel: string;
  normalizedEventExport: NormalizedEventExportV1;
  nextCursor: EventExportCursorV1;
  watermark: NormalizedEventExportSliceV1["watermark"];
  build: CandidatePackBuildResult;
}

export interface CandidatePackBundleBuildResult {
  runtimeOwner: "openclaw";
  bridgeDigest: string;
  bundleDigest: string;
  cursor: EventExportCursorV1;
  dedupedInputCount: number;
  duplicateIdentityCount: number;
  entries: CandidatePackBundleEntry[];
}

export interface MaterializedCandidatePackBundleEntry extends CandidatePackBundleEntry {
  rootDir: string;
  descriptor: PackDescriptor;
}

export interface CandidatePackBundleMaterializationResult {
  runtimeOwner: "openclaw";
  bridgeDigest: string;
  bundleDigest: string;
  cursor: EventExportCursorV1;
  dedupedInputCount: number;
  duplicateIdentityCount: number;
  entries: MaterializedCandidatePackBundleEntry[];
}

export const DEFAULT_ALWAYS_ON_LEARNING_LIVE_SLICES_PER_CYCLE = 1;
export const DEFAULT_ALWAYS_ON_LEARNING_BACKFILL_SLICES_PER_CYCLE = 1;

export interface AlwaysOnLearningCadenceV1 {
  liveSlicesPerCycle: number;
  backfillSlicesPerCycle: number;
}

export interface AlwaysOnLearningPendingSlicesV1 {
  live: NormalizedEventExportSliceV1[];
  backfill: NormalizedEventExportSliceV1[];
}

export interface AlwaysOnLearningRuntimeStateV1 {
  runtimeOwner: "openclaw";
  hotPathLearning: false;
  attachBlocksOnFullReplay: false;
  cursor: EventExportCursorV1;
  pending: AlwaysOnLearningPendingSlicesV1;
  learnedEventExport: NormalizedEventExportV1 | null;
  lastMaterializedAt: string | null;
  materializationCount: number;
}

export interface AdvanceAlwaysOnLearningRuntimeInput {
  packLabel: string;
  workspace: WorkspaceMetadataInput;
  interactionEvents: readonly InteractionEventV1[];
  feedbackEvents: readonly FeedbackEventV1[];
  learnedRouting: boolean;
  state?: AlwaysOnLearningRuntimeStateV1;
  builtAt?: string;
  offlineArtifacts?: string[];
  structuralOps?: Partial<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
  liveSliceSize?: number;
  backfillSliceSize?: number;
  cadence?: Partial<AlwaysOnLearningCadenceV1>;
}

export interface AlwaysOnLearningMaterializationJobV1 {
  jobId: string;
  lane: EventExportLaneV1;
  priority: "immediate" | "background";
  reason: "attach_bootstrap" | "fresh_live_events" | "passive_history_catchup";
  selectedSliceIds: string[];
  selectedEventRange: NormalizedEventExportV1["range"];
  normalizedEventExport: NormalizedEventExportV1;
  candidateInput: CandidatePackFromNormalizedEventExportInput;
  candidate: CandidatePackBuildResult;
}

export interface AdvanceAlwaysOnLearningRuntimeResultV1 {
  runtimeOwner: "openclaw";
  hotPathLearning: false;
  attachBlocksOnFullReplay: false;
  bridge: NormalizedEventExportBridgeV1;
  selectedSlices: NormalizedEventExportSliceV1[];
  deferred: {
    live: number;
    backfill: number;
  };
  materialization: AlwaysOnLearningMaterializationJobV1 | null;
  state: AlwaysOnLearningRuntimeStateV1;
}

export interface DrainAlwaysOnLearningRuntimeInput extends AdvanceAlwaysOnLearningRuntimeInput {
  maxCycles?: number;
}

export interface AlwaysOnLearningRuntimeCycleV1 extends AdvanceAlwaysOnLearningRuntimeResultV1 {
  cycle: number;
}

export interface DrainAlwaysOnLearningRuntimeResultV1 {
  runtimeOwner: "openclaw";
  drained: boolean;
  stopReason: "idle" | "max_cycles" | "no_progress";
  cycles: AlwaysOnLearningRuntimeCycleV1[];
  materializations: AlwaysOnLearningMaterializationJobV1[];
  state: AlwaysOnLearningRuntimeStateV1;
}

function stableHash(value: string): string {
  let hash = 0;
  for (const char of value) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
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

function cloneSliceWatermark(value: NormalizedEventExportSliceV1["watermark"]): NormalizedEventExportSliceV1["watermark"] {
  return {
    first: value.first === null ? null : { ...value.first },
    last: value.last === null ? null : { ...value.last }
  };
}

function cloneNormalizedEventExportSlice(value: NormalizedEventExportSliceV1): NormalizedEventExportSliceV1 {
  return structuredClone(value);
}

function cloneAlwaysOnLearningRuntimeState(value: AlwaysOnLearningRuntimeStateV1): AlwaysOnLearningRuntimeStateV1 {
  return structuredClone(value);
}

function cloneAlwaysOnLearningMaterializationJob(
  value: AlwaysOnLearningMaterializationJobV1
): AlwaysOnLearningMaterializationJobV1 {
  return structuredClone(value);
}

function compareSliceRecency(left: NormalizedEventExportSliceV1, right: NormalizedEventExportSliceV1): number {
  if (left.export.range.end !== right.export.range.end) {
    return right.export.range.end - left.export.range.end;
  }
  if (left.export.range.start !== right.export.range.start) {
    return right.export.range.start - left.export.range.start;
  }

  return left.sliceId.localeCompare(right.sliceId);
}

function sortPendingSlicesByRecency(slices: readonly NormalizedEventExportSliceV1[]): NormalizedEventExportSliceV1[] {
  return [...slices].sort(compareSliceRecency);
}

function materializeCandidatePackResult(rootDir: string, result: CandidatePackBuildResult): PackDescriptor {
  rmSync(rootDir, { recursive: true, force: true });
  mkdirSync(rootDir, { recursive: true });

  writePackFile(rootDir, PACK_LAYOUT.graph, result.payloads.graph);
  writePackFile(rootDir, PACK_LAYOUT.vectors, result.payloads.vectors);
  if (result.payloads.router !== null) {
    writePackFile(rootDir, PACK_LAYOUT.router, result.payloads.router);
  }
  writePackFile(rootDir, PACK_LAYOUT.manifest, result.manifest);

  return loadPack(path.resolve(rootDir));
}

function buildBridgePackLabel(basePackLabel: string, slice: NormalizedEventExportSliceV1, index: number): string {
  return `${basePackLabel}-${String(index + 1).padStart(2, "0")}-${slice.lane}-${slice.export.range.start}-${slice.export.range.end}`;
}

function buildBundleEntryRootDir(rootDir: string, entry: CandidatePackBundleEntry, index: number): string {
  const digestSuffix = entry.sliceId.replace(/^sha256-/u, "").slice(0, 12);

  return path.join(
    rootDir,
    `${String(index + 1).padStart(2, "0")}-${entry.lane}-${entry.normalizedEventExport.range.start}-${entry.normalizedEventExport.range.end}-${digestSuffix}`
  );
}

function assertPositiveInteger(label: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
}

function assertNonNegativeInteger(label: string, value: number): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${label} must be a non-negative integer`);
  }
}

function normalizeAlwaysOnLearningCadence(
  input: Partial<AlwaysOnLearningCadenceV1> | undefined
): AlwaysOnLearningCadenceV1 {
  const cadence: AlwaysOnLearningCadenceV1 = {
    liveSlicesPerCycle: input?.liveSlicesPerCycle ?? DEFAULT_ALWAYS_ON_LEARNING_LIVE_SLICES_PER_CYCLE,
    backfillSlicesPerCycle: input?.backfillSlicesPerCycle ?? DEFAULT_ALWAYS_ON_LEARNING_BACKFILL_SLICES_PER_CYCLE
  };

  assertPositiveInteger("cadence.liveSlicesPerCycle", cadence.liveSlicesPerCycle);
  assertNonNegativeInteger("cadence.backfillSlicesPerCycle", cadence.backfillSlicesPerCycle);

  return cadence;
}

function mergePendingSlices(
  pending: AlwaysOnLearningPendingSlicesV1,
  discovered: readonly NormalizedEventExportSliceV1[]
): AlwaysOnLearningPendingSlicesV1 {
  const live = pending.live.map(cloneNormalizedEventExportSlice);
  const backfill = pending.backfill.map(cloneNormalizedEventExportSlice);
  const seenSliceIds = new Set([...live, ...backfill].map((slice) => slice.sliceId));

  for (const slice of discovered) {
    if (seenSliceIds.has(slice.sliceId)) {
      continue;
    }

    if (slice.lane === "live") {
      live.push(cloneNormalizedEventExportSlice(slice));
    } else {
      backfill.push(cloneNormalizedEventExportSlice(slice));
    }
    seenSliceIds.add(slice.sliceId);
  }

  return {
    live: sortPendingSlicesByRecency(live),
    backfill: sortPendingSlicesByRecency(backfill)
  };
}

function mergeNormalizedEventExports(
  current: NormalizedEventExportV1 | null,
  additions: readonly NormalizedEventExportSliceV1[]
): NormalizedEventExportV1 | null {
  if (current === null && additions.length === 0) {
    return null;
  }

  const mergedEvents = sortNormalizedEvents([
    ...(current?.interactionEvents ?? []),
    ...(current?.feedbackEvents ?? []),
    ...additions.flatMap((slice) => [...slice.export.interactionEvents, ...slice.export.feedbackEvents])
  ]);
  const deduped: NormalizedEventV1[] = [];
  const seenDedupIds = new Set<string>();

  for (const event of mergedEvents) {
    const dedupId = buildNormalizedEventDedupId(event);
    if (seenDedupIds.has(dedupId)) {
      continue;
    }

    seenDedupIds.add(dedupId);
    deduped.push(event);
  }

  return buildNormalizedEventExport({
    interactionEvents: deduped.filter((event): event is InteractionEventV1 => event.contract === CONTRACT_IDS.interactionEvents),
    feedbackEvents: deduped.filter((event): event is FeedbackEventV1 => event.contract === CONTRACT_IDS.feedbackEvents)
  });
}

function createAlwaysOnLearningPendingSlices(): AlwaysOnLearningPendingSlicesV1 {
  return {
    live: [],
    backfill: []
  };
}

export function createAlwaysOnLearningRuntimeState(): AlwaysOnLearningRuntimeStateV1 {
  return {
    runtimeOwner: "openclaw",
    hotPathLearning: false,
    attachBlocksOnFullReplay: false,
    cursor: createEventExportCursor(),
    pending: createAlwaysOnLearningPendingSlices(),
    learnedEventExport: null,
    lastMaterializedAt: null,
    materializationCount: 0
  };
}

function hasPendingSlices(pending: AlwaysOnLearningPendingSlicesV1): boolean {
  return pending.live.length > 0 || pending.backfill.length > 0;
}

function buildAlwaysOnLearningMaterializationJob(
  input: AdvanceAlwaysOnLearningRuntimeInput,
  current: AlwaysOnLearningRuntimeStateV1,
  selectedSlices: readonly NormalizedEventExportSliceV1[],
  normalizedEventExport: NormalizedEventExportV1
): AlwaysOnLearningMaterializationJobV1 {
  const lane: EventExportLaneV1 = selectedSlices.some((slice) => slice.lane === "live") ? "live" : "backfill";
  const reason: AlwaysOnLearningMaterializationJobV1["reason"] =
    lane === "live"
      ? current.learnedEventExport === null
        ? "attach_bootstrap"
        : "fresh_live_events"
      : "passive_history_catchup";
  const candidateInput: CandidatePackFromNormalizedEventExportInput = {
    packLabel: input.packLabel,
    workspace: input.workspace,
    normalizedEventExport,
    learnedRouting: input.learnedRouting,
    ...(input.builtAt !== undefined ? { builtAt: input.builtAt } : {}),
    ...(input.offlineArtifacts !== undefined ? { offlineArtifacts: input.offlineArtifacts } : {}),
    ...(input.structuralOps !== undefined ? { structuralOps: input.structuralOps } : {})
  };
  const candidate = buildCandidatePackFromNormalizedEventExport(candidateInput);
  const selectedSliceIds = selectedSlices.map((slice) => slice.sliceId);

  return {
    jobId: `learning-${stableHash(checksumJsonPayload({ lane, reason, exportDigest: normalizedEventExport.provenance.exportDigest, selectedSliceIds }))}`,
    lane,
    priority: lane === "live" ? "immediate" : "background",
    reason,
    selectedSliceIds,
    selectedEventRange: normalizedEventExport.range,
    normalizedEventExport,
    candidateInput,
    candidate
  };
}

export function advanceAlwaysOnLearningRuntime(
  input: AdvanceAlwaysOnLearningRuntimeInput
): AdvanceAlwaysOnLearningRuntimeResultV1 {
  const cadence = normalizeAlwaysOnLearningCadence(input.cadence);
  const current = cloneAlwaysOnLearningRuntimeState(input.state ?? createAlwaysOnLearningRuntimeState());
  const bridge = buildNormalizedEventExportBridge({
    interactionEvents: [...input.interactionEvents],
    feedbackEvents: [...input.feedbackEvents],
    cursor: current.cursor,
    ...(input.liveSliceSize !== undefined ? { liveSliceSize: input.liveSliceSize } : {}),
    ...(input.backfillSliceSize !== undefined ? { backfillSliceSize: input.backfillSliceSize } : {})
  });
  const pending = mergePendingSlices(current.pending, bridge.slices);
  const selectedLive = pending.live.slice(0, cadence.liveSlicesPerCycle).map(cloneNormalizedEventExportSlice);
  const bootstrapLiveFirst = current.learnedEventExport === null && selectedLive.length > 0;
  const selectedBackfill = bootstrapLiveFirst
    ? []
    : pending.backfill.slice(0, cadence.backfillSlicesPerCycle).map(cloneNormalizedEventExportSlice);
  const selectedSlices = [...selectedLive, ...selectedBackfill];
  const learnedEventExport = mergeNormalizedEventExports(current.learnedEventExport, selectedSlices);
  const materialization =
    learnedEventExport === null || selectedSlices.length === 0
      ? null
      : buildAlwaysOnLearningMaterializationJob(input, current, selectedSlices, learnedEventExport);
  const nextState: AlwaysOnLearningRuntimeStateV1 = {
    runtimeOwner: "openclaw",
    hotPathLearning: false,
    attachBlocksOnFullReplay: false,
    cursor: bridge.cursor,
    pending: {
      live: pending.live.slice(selectedLive.length).map(cloneNormalizedEventExportSlice),
      backfill: pending.backfill.slice(selectedBackfill.length).map(cloneNormalizedEventExportSlice)
    },
    learnedEventExport,
    lastMaterializedAt: materialization?.candidate.manifest.provenance.builtAt ?? current.lastMaterializedAt,
    materializationCount: current.materializationCount + (materialization === null ? 0 : 1)
  };

  return {
    runtimeOwner: "openclaw",
    hotPathLearning: false,
    attachBlocksOnFullReplay: false,
    bridge: structuredClone(bridge),
    selectedSlices: selectedSlices.map(cloneNormalizedEventExportSlice),
    deferred: {
      live: nextState.pending.live.length,
      backfill: nextState.pending.backfill.length
    },
    materialization: materialization === null ? null : cloneAlwaysOnLearningMaterializationJob(materialization),
    state: cloneAlwaysOnLearningRuntimeState(nextState)
  };
}

export function drainAlwaysOnLearningRuntime(
  input: DrainAlwaysOnLearningRuntimeInput
): DrainAlwaysOnLearningRuntimeResultV1 {
  const maxCycles = input.maxCycles ?? 64;

  assertPositiveInteger("maxCycles", maxCycles);

  const cycles: AlwaysOnLearningRuntimeCycleV1[] = [];
  const materializations: AlwaysOnLearningMaterializationJobV1[] = [];
  let state = cloneAlwaysOnLearningRuntimeState(input.state ?? createAlwaysOnLearningRuntimeState());
  let stopReason: DrainAlwaysOnLearningRuntimeResultV1["stopReason"] = "max_cycles";

  for (let cycle = 1; cycle <= maxCycles; cycle += 1) {
    const result = advanceAlwaysOnLearningRuntime({
      ...input,
      state
    });

    cycles.push({
      cycle,
      ...structuredClone(result)
    });

    if (result.materialization !== null) {
      materializations.push(cloneAlwaysOnLearningMaterializationJob(result.materialization));
    }

    state = cloneAlwaysOnLearningRuntimeState(result.state);

    const idle = result.selectedSlices.length === 0 && result.bridge.slices.length === 0 && !hasPendingSlices(state.pending);
    if (idle) {
      stopReason = "idle";
      break;
    }

    if (result.selectedSlices.length === 0) {
      stopReason = "no_progress";
      break;
    }
  }

  return {
    runtimeOwner: "openclaw",
    drained: stopReason === "idle",
    stopReason,
    cycles,
    materializations,
    state
  };
}

export function materializeAlwaysOnLearningCandidatePack(
  rootDir: string,
  job: AlwaysOnLearningMaterializationJobV1
): PackDescriptor {
  return materializeCandidatePackFromNormalizedEventExport(rootDir, job.candidateInput);
}

export function buildCandidatePackFromNormalizedEventExportSlice(
  input: CandidatePackFromNormalizedEventExportSliceInput
): CandidatePackBuildResult {
  const validationErrors = validateNormalizedEventExportSlice(input.normalizedEventExportSlice);
  if (validationErrors.length > 0) {
    throw new Error(`normalized event export slice is invalid: ${validationErrors.join("; ")}`);
  }

  return buildCandidatePackFromNormalizedEventExport({
    packLabel: input.packLabel,
    workspace: input.workspace,
    normalizedEventExport: input.normalizedEventExportSlice.export,
    learnedRouting: input.learnedRouting,
    ...(input.builtAt !== undefined ? { builtAt: input.builtAt } : {}),
    ...(input.offlineArtifacts !== undefined ? { offlineArtifacts: input.offlineArtifacts } : {}),
    ...(input.structuralOps !== undefined ? { structuralOps: input.structuralOps } : {})
  });
}

export function buildCandidatePackBundleFromNormalizedEventExportBridge(
  input: CandidatePackBundleFromNormalizedEventExportBridgeInput
): CandidatePackBundleBuildResult {
  const validationErrors = validateNormalizedEventExportBridge(input.normalizedEventExportBridge);
  if (validationErrors.length > 0) {
    throw new Error(`normalized event export bridge is invalid: ${validationErrors.join("; ")}`);
  }

  const entries = input.normalizedEventExportBridge.slices.map((slice, index): CandidatePackBundleEntry => {
    const packLabel = buildBridgePackLabel(input.packLabel, slice, index);

    return {
      lane: slice.lane,
      sliceId: slice.sliceId,
      packLabel,
      normalizedEventExport: slice.export,
      nextCursor: cloneCursor(slice.nextCursor),
      watermark: cloneSliceWatermark(slice.watermark),
      build: buildCandidatePackFromNormalizedEventExportSlice({
        packLabel,
        workspace: input.workspace,
        normalizedEventExportSlice: slice,
        learnedRouting: input.learnedRouting,
        ...(input.builtAt !== undefined ? { builtAt: input.builtAt } : {}),
        ...(input.offlineArtifacts !== undefined ? { offlineArtifacts: input.offlineArtifacts } : {}),
        ...(input.structuralOps !== undefined ? { structuralOps: input.structuralOps } : {})
      })
    };
  });

  const bundleDigest = checksumJsonPayload({
    runtimeOwner: input.normalizedEventExportBridge.runtimeOwner,
    bridgeDigest: input.normalizedEventExportBridge.bridgeDigest,
    cursor: input.normalizedEventExportBridge.cursor,
    entries: entries.map((entry) => ({
      lane: entry.lane,
      sliceId: entry.sliceId,
      packLabel: entry.packLabel,
      packId: entry.build.summary.packId,
      nextCursor: entry.nextCursor
    })),
    dedupedInputCount: input.normalizedEventExportBridge.dedupedInputCount,
    duplicateIdentityCount: input.normalizedEventExportBridge.duplicateIdentityCount
  });

  return {
    runtimeOwner: input.normalizedEventExportBridge.runtimeOwner,
    bridgeDigest: input.normalizedEventExportBridge.bridgeDigest,
    bundleDigest,
    cursor: cloneCursor(input.normalizedEventExportBridge.cursor),
    dedupedInputCount: input.normalizedEventExportBridge.dedupedInputCount,
    duplicateIdentityCount: input.normalizedEventExportBridge.duplicateIdentityCount,
    entries
  };
}

function structuralOpsSummary(input: CandidatePackBuildInput): Required<ArtifactManifestV1["graphDynamics"]["structuralOps"]> {
  return {
    split: input.structuralOps?.split ?? 0,
    merge: input.structuralOps?.merge ?? 0,
    prune: input.structuralOps?.prune ?? 0,
    connect: input.structuralOps?.connect ?? 0
  };
}

function keywordTokens(value: string): string[] {
  return [...new Set(value.toLowerCase().split(/[^a-z0-9]+/u).filter((token) => token.length >= 3 && /[a-z]/u.test(token)))].slice(0, 16);
}

function eventPriority(event: NormalizedEventV1): number {
  if (event.contract === CONTRACT_IDS.feedbackEvents) {
    switch (event.kind) {
      case "correction":
      case "teaching":
        return 5;
      case "approval":
      case "suppression":
        return 4;
      default:
        return 4;
    }
  }

  switch (event.kind) {
    case "operator_override":
      return 5;
    case "memory_compiled":
      return 4;
    case "message_delivered":
      return 3;
    default:
      return 3;
  }
}

function sentenceCase(value: string): string {
  return value.length === 0 ? value : `${value[0]?.toUpperCase() ?? ""}${value.slice(1)}`;
}

function learningSignals(input: {
  role: PackBlockLearningSignalsV1["role"];
  humanLabels?: number;
  selfLabels?: number;
  decayHalfLifeDays?: number | null;
  hebbianPulse?: number;
}): PackBlockLearningSignalsV1 {
  return {
    role: input.role,
    humanLabels: input.humanLabels ?? 0,
    selfLabels: input.selfLabels ?? 0,
    decayHalfLifeDays: input.decayHalfLifeDays ?? null,
    hebbianPulse: input.hebbianPulse ?? 0
  };
}

function summarizeEvent(event: NormalizedEventV1): string {
  if (event.contract === CONTRACT_IDS.feedbackEvents) {
    const relation = event.relatedInteractionId === undefined ? "" : ` Related interaction: ${event.relatedInteractionId}.`;
    return `${sentenceCase(event.kind)} feedback on ${event.channel} session ${event.sessionId}: ${event.content}.${relation}`;
  }

  const messagePart = event.messageId === undefined ? "" : ` Message: ${event.messageId}.`;
  const packPart = event.packId === undefined ? "" : ` Pack: ${event.packId}.`;
  return `Interaction ${event.kind} on ${event.channel} session ${event.sessionId}.${packPart}${messagePart}`;
}

function eventLearningSignals(event: NormalizedEventV1): PackBlockLearningSignalsV1 {
  if (event.contract === CONTRACT_IDS.feedbackEvents) {
    return learningSignals({
      role: "feedback",
      humanLabels: 1,
      decayHalfLifeDays: 30,
      hebbianPulse: eventPriority(event)
    });
  }

  if (event.kind === "operator_override") {
    return learningSignals({
      role: "interaction",
      humanLabels: 1,
      decayHalfLifeDays: 30,
      hebbianPulse: eventPriority(event)
    });
  }

  if (event.kind === "memory_compiled") {
    return learningSignals({
      role: "interaction",
      selfLabels: 1,
      decayHalfLifeDays: 30,
      hebbianPulse: eventPriority(event)
    });
  }

  return learningSignals({
    role: "interaction",
    decayHalfLifeDays: 14,
    hebbianPulse: 1
  });
}

function eventBlock(packId: string, event: NormalizedEventV1): PackContextBlockRecordV1 {
  const text = summarizeEvent(event);
  return {
    id: `${packId}:event:${event.eventId}`,
    source: `${event.source.stream}:${event.kind}`,
    text,
    keywords: keywordTokens(`${event.kind} ${event.channel} ${event.source.stream} ${text}`),
    priority: eventPriority(event),
    learning: eventLearningSignals(event)
  };
}

function staticLifecycleBlocks(
  packId: string,
  input: CandidatePackBuildInput,
  workspace: ReturnType<typeof createWorkspaceMetadata>,
  learningSurface: LearningSurfaceV1
): PackContextBlockRecordV1[] {
  const structuralOps = structuralOpsSummary(input);
  const humanSources = learningSurface.labelSources.human.join(", ") || "feedback_events.v1";
  const selfSources = learningSurface.labelSources.self.join(", ") || "interaction_events.v1:memory_compiled";

  return [
    {
      id: `${packId}:feedback-scanner`,
      source: "docs/openclaw-attach-quickstart.md",
      text: "Always-on feedback scanner harvests human labels from local session logs with Ollama qwen3.5:9b-q4_K_M, checkpointed resumes, and deduplicated background scans.",
      keywords: ["feedback", "scanner", "always", "background", "labels", "ollama", "qwen", "checkpoint", "dedup"],
      priority: 5,
      learning: learningSignals({
        role: "label_surface",
        humanLabels: learningSurface.labelHarvest.humanLabels,
        decayHalfLifeDays: 30,
        hebbianPulse: 5
      })
    },
    {
      id: `${packId}:fast-boot-defaults`,
      source: "product/always-on-learning",
      text: "Fast boot defaults stay live at startup so OpenClaw can answer immediately while passive background learning hydrates richer graph state.",
      keywords: ["fast", "boot", "defaults", "startup", "background", "learning", "openclaw"],
      priority: 5,
      learning: learningSignals({
        role: "boot_default",
        decayHalfLifeDays: null,
        hebbianPulse: 2
      })
    },
    {
      id: `${packId}:passive-background-learning`,
      source: "product/always-on-learning",
      text: `Learning cadence is ${learningSurface.learningCadence} with ${learningSurface.scanPolicy} scans across ${learningSurface.scanSurfaces.join(", ")}.`,
      keywords: ["passive", "background", "always", "scan", "learning", ...keywordTokens(learningSurface.scanSurfaces.join(" "))],
      priority: 5,
      learning: learningSignals({
        role: "background_expectation",
        decayHalfLifeDays: 30,
        hebbianPulse: 3
      })
    },
    {
      id: `${packId}:human-label-harvest`,
      source: humanSources,
      text: `Human label harvest is first-class with ${learningSurface.labelHarvest.humanLabels} labels sourced from ${humanSources}.`,
      keywords: ["human", "labels", "harvest", "feedback", "approval", "teaching", "correction", "suppression"],
      priority: 4,
      learning: learningSignals({
        role: "label_surface",
        humanLabels: learningSurface.labelHarvest.humanLabels,
        decayHalfLifeDays: 30,
        hebbianPulse: Math.max(1, learningSurface.labelHarvest.humanLabels)
      })
    },
    {
      id: `${packId}:self-label-harvest`,
      source: selfSources,
      text: `Self label harvest stays visible with ${learningSurface.labelHarvest.selfLabels} memory-side labels sourced from ${selfSources}.`,
      keywords: ["self", "labels", "harvest", "memory", "compiled", "graph"],
      priority: 4,
      learning: learningSignals({
        role: "label_surface",
        selfLabels: learningSurface.labelHarvest.selfLabels,
        decayHalfLifeDays: 30,
        hebbianPulse: Math.max(1, learningSurface.labelHarvest.selfLabels)
      })
    },
    {
      id: `${packId}:workspace`,
      source: workspace.rootDir,
      text: `Workspace snapshot ${workspace.snapshotId} for ${workspace.workspaceId} captured at ${workspace.capturedAt} with revision ${workspace.revision ?? "unversioned"}.`,
      keywords: keywordTokens(
        `workspace ${workspace.workspaceId} ${workspace.snapshotId} ${workspace.branch ?? ""} ${workspace.revision ?? ""} ${workspace.labels.join(" ")}`
      ),
      priority: 4,
      learning: learningSignals({
        role: "workspace",
        decayHalfLifeDays: null,
        hebbianPulse: 1
      })
    },
    {
      id: `${packId}:structural-ops`,
      source: "docs/learning-first-convergence.md",
      text: `Structural graph learning stays first-class with Hebbian reinforcement, decay half-life 30 days, and ops split=${structuralOps.split}, merge=${structuralOps.merge}, prune=${structuralOps.prune}, connect=${structuralOps.connect}.`,
      keywords: ["structural", "hebbian", "decay", "split", "merge", "prune", "connect", "graph", "memory"],
      priority: 4,
      learning: learningSignals({
        role: "structural",
        decayHalfLifeDays: 30,
        hebbianPulse: 4
      })
    }
  ];
}

function eventExportBlocks(packId: string, eventExport: NormalizedEventExportV1): PackContextBlockRecordV1[] {
  const allEvents = [...eventExport.interactionEvents, ...eventExport.feedbackEvents];
  const learningSurface = eventExport.provenance.learningSurface;
  const summaryKeywords = keywordTokens(
    `normalized event export ${eventExport.provenance.sourceStreams.join(" ")} interaction ${eventExport.provenance.interactionCount} feedback ${eventExport.provenance.feedbackCount} human ${learningSurface.labelHarvest.humanLabels} self ${learningSurface.labelHarvest.selfLabels}`
  );

  return [
    {
      id: `${packId}:event-export`,
      source: `contracts/${CONTRACT_IDS.interactionEvents}+${CONTRACT_IDS.feedbackEvents}`,
      text: `Normalized event export covers ${eventExport.provenance.interactionCount} interaction events and ${eventExport.provenance.feedbackCount} feedback events across sequences ${eventExport.range.start}-${eventExport.range.end}; harvested labels human=${learningSurface.labelHarvest.humanLabels}, self=${learningSurface.labelHarvest.selfLabels}.`,
      keywords: summaryKeywords,
      priority: 5,
      learning: learningSignals({
        role: "label_surface",
        humanLabels: learningSurface.labelHarvest.humanLabels,
        selfLabels: learningSurface.labelHarvest.selfLabels,
        decayHalfLifeDays: 30,
        hebbianPulse: 5
      })
    },
    ...allEvents.map((event) => eventBlock(packId, event))
  ];
}

function createGraphPayload(
  packId: string,
  input: CandidatePackBuildInput,
  workspace: ReturnType<typeof createWorkspaceMetadata>,
  eventExport: NormalizedEventExportV1 | null,
  learningSurface: LearningSurfaceV1
): PackGraphPayloadV1 {
  return {
    packId,
    blocks: [...staticLifecycleBlocks(packId, input, workspace, learningSurface), ...(eventExport === null ? [] : eventExportBlocks(packId, eventExport))]
  };
}

function learningVectorKeywords(block: PackContextBlockRecordV1): string[] {
  const keywords: string[] = [block.learning.role];

  if (block.learning.role === "boot_default") {
    keywords.push("fast_boot");
  }
  if (block.learning.role === "background_expectation") {
    keywords.push("passive_background", "always_on");
  }
  if (block.learning.humanLabels > 0) {
    keywords.push("human_label");
  }
  if (block.learning.selfLabels > 0) {
    keywords.push("self_label");
  }
  if (block.learning.hebbianPulse > 0) {
    keywords.push("hebbian");
  }
  if (block.learning.decayHalfLifeDays !== null) {
    keywords.push("decay");
  }

  return keywords;
}

function vectorEntryFromBlock(block: PackContextBlockRecordV1): PackVectorEntryV1 {
  const keywords = [...new Set([...block.keywords, ...learningVectorKeywords(block)])];
  const weights = Object.fromEntries(
    keywords.map((keyword, index) => [keyword, Math.max(1, block.priority - Math.min(index, Math.max(0, block.priority - 1)))])
  );

  if (block.learning.humanLabels > 0) {
    weights.human_label = Math.max(weights.human_label ?? 0, block.priority + block.learning.humanLabels);
  }
  if (block.learning.selfLabels > 0) {
    weights.self_label = Math.max(weights.self_label ?? 0, block.priority + block.learning.selfLabels);
  }
  if (block.learning.hebbianPulse > 0) {
    weights.hebbian = Math.max(weights.hebbian ?? 0, block.learning.hebbianPulse + 1);
  }
  if (block.learning.role === "boot_default") {
    weights.fast_boot = Math.max(weights.fast_boot ?? 0, block.priority + 1);
  }
  if (block.learning.role === "background_expectation") {
    weights.passive_background = Math.max(weights.passive_background ?? 0, block.priority + 1);
  }

  return {
    blockId: block.id,
    keywords,
    boost:
      Math.max(1, Math.ceil(block.priority / 2)) +
      Math.min(3, block.learning.humanLabels + block.learning.selfLabels) +
      Math.min(2, Math.ceil(block.learning.hebbianPulse / 3)),
    weights
  };
}

function createVectorsPayload(graph: PackGraphPayloadV1): PackVectorsPayloadV1 {
  return {
    packId: graph.packId,
    entries: graph.blocks.map((block) => vectorEntryFromBlock(block))
  };
}

function createRouterArtifact(packId: string, builtAt: string): RouterArtifactV1 {
  return {
    routerIdentity: `${packId}:route_fn`,
    strategy: "learned_route_fn_v1",
    trainedAt: builtAt,
    requiresLearnedRouting: true
  };
}

function resolveEventExport(input: CandidatePackBuildInput): NormalizedEventExportV1 | null {
  if (input.eventExports === undefined) {
    return null;
  }

  const eventExport = buildNormalizedEventExport(input.eventExports);
  const validationErrors = validateNormalizedEventExport(eventExport);
  if (validationErrors.length > 0) {
    throw new Error(`normalized event export is invalid: ${validationErrors.join("; ")}`);
  }

  if (eventExport.range.start !== input.eventRange.start || eventExport.range.end !== input.eventRange.end) {
    throw new Error(
      `event export range ${eventExport.range.start}-${eventExport.range.end} does not match requested range ${input.eventRange.start}-${input.eventRange.end}`
    );
  }

  return eventExport;
}

function defaultLearningSurface(workspace: ReturnType<typeof createWorkspaceMetadata>, offlineArtifacts: readonly string[]): LearningSurfaceV1 {
  return createDefaultLearningSurface([
    `workspace:${workspace.workspaceId}`,
    ...offlineArtifacts.map((artifact) => `offline:${artifact}`)
  ]);
}

export function buildCandidatePack(input: CandidatePackBuildInput): CandidatePackBuildResult {
  const builtAt = input.builtAt ?? "2026-03-06T00:00:00.000Z";
  const routePolicy = input.learnedRouting ? "requires_learned_routing" : "heuristic_allowed";
  const eventExport = resolveEventExport(input);
  const eventRange = eventExport?.range ?? createExplicitEventRange(input.eventRange);
  const workspace = createWorkspaceMetadata(input.workspace);
  const offlineArtifacts = input.offlineArtifacts ?? [];
  const learningSurface = eventExport?.provenance.learningSurface ?? defaultLearningSurface(workspace, offlineArtifacts);
  const seed = JSON.stringify({
    packLabel: input.packLabel,
    workspace,
    eventRange,
    learnedRouting: input.learnedRouting,
    offlineArtifacts,
    eventExportDigest: eventExport?.provenance.exportDigest ?? null,
    learningSurface
  });
  const packId = `pack-${stableHash(seed)}`;

  const graph = createGraphPayload(packId, input, workspace, eventExport, learningSurface);
  const vectors = createVectorsPayload(graph);
  const router = input.learnedRouting ? createRouterArtifact(packId, builtAt) : null;

  const payloads: CandidatePackPayloads = {
    graph,
    vectors,
    router
  };

  const manifest: ArtifactManifestV1 = {
    contract: CONTRACT_IDS.artifactManifest,
    packId,
    immutable: true,
    routePolicy,
    runtimeAssets: {
      graphPath: PACK_LAYOUT.graph,
      vectorPath: PACK_LAYOUT.vectors,
      router: input.learnedRouting
        ? {
            kind: "artifact",
            identity: payloads.router?.routerIdentity ?? null,
            artifactPath: PACK_LAYOUT.router
          }
        : {
            kind: "none",
            identity: null,
            artifactPath: null
          }
    },
    payloadChecksums: {
      graph: computePayloadChecksum(payloads.graph),
      vector: computePayloadChecksum(payloads.vectors),
      router: payloads.router === null ? null : computePayloadChecksum(payloads.router)
    },
    modelFingerprints: input.learnedRouting
      ? ["BAAI/bge-large-en-v1.5", "ollama:qwen3.5:9b-q4_K_M", payloads.router?.routerIdentity ?? "router:missing"]
      : ["BAAI/bge-large-en-v1.5"],
    provenance: buildArtifactProvenance({
      workspace,
      eventRange,
      eventExports: eventExport?.provenance ?? null,
      learningSurface,
      builtAt,
      offlineArtifacts
    }),
    graphDynamics: {
      bootstrapping: {
        fastBootDefaults: true,
        passiveBackgroundLearning: true
      },
      hebbian: {
        enabled: true,
        learningRate: 0.1
      },
      decay: {
        enabled: true,
        halfLifeDays: 30
      },
      structuralOps: structuralOpsSummary(input)
    }
  };

  return {
    manifest,
    payloads,
    summary: {
      packId,
      immutable: true,
      routePolicy,
      workspaceSnapshot: workspace.snapshotId,
      eventRange,
      eventExportDigest: eventExport?.provenance.exportDigest ?? null,
      learningSurface: manifest.provenance.learningSurface,
      bootstrapping: manifest.graphDynamics.bootstrapping
    }
  };
}

export function buildCandidatePackFromNormalizedEventExport(
  input: CandidatePackFromNormalizedEventExportInput
): CandidatePackBuildResult {
  const validationErrors = validateNormalizedEventExport(input.normalizedEventExport);
  if (validationErrors.length > 0) {
    throw new Error(`normalized event export is invalid: ${validationErrors.join("; ")}`);
  }

  const candidateInput: CandidatePackBuildInput = {
    packLabel: input.packLabel,
    workspace: input.workspace,
    eventRange: {
      start: input.normalizedEventExport.range.start,
      end: input.normalizedEventExport.range.end
    },
    eventExports: {
      interactionEvents: [...input.normalizedEventExport.interactionEvents],
      feedbackEvents: [...input.normalizedEventExport.feedbackEvents]
    },
    learnedRouting: input.learnedRouting,
    ...(input.builtAt !== undefined ? { builtAt: input.builtAt } : {}),
    ...(input.offlineArtifacts !== undefined ? { offlineArtifacts: input.offlineArtifacts } : {}),
    ...(input.structuralOps !== undefined ? { structuralOps: input.structuralOps } : {})
  };

  return buildCandidatePack(candidateInput);
}

export function materializeCandidatePack(rootDir: string, input: CandidatePackBuildInput): PackDescriptor {
  const result = buildCandidatePack(input);
  return materializeCandidatePackResult(rootDir, result);
}

export function materializeCandidatePackFromNormalizedEventExport(
  rootDir: string,
  input: CandidatePackFromNormalizedEventExportInput
): PackDescriptor {
  const result = buildCandidatePackFromNormalizedEventExport(input);
  return materializeCandidatePackResult(rootDir, result);
}

export function materializeCandidatePackFromNormalizedEventExportSlice(
  rootDir: string,
  input: CandidatePackFromNormalizedEventExportSliceInput
): PackDescriptor {
  const result = buildCandidatePackFromNormalizedEventExportSlice(input);
  return materializeCandidatePackResult(rootDir, result);
}

export function materializeCandidatePackBundleFromNormalizedEventExportBridge(
  rootDir: string,
  input: CandidatePackBundleFromNormalizedEventExportBridgeInput
): CandidatePackBundleMaterializationResult {
  const bundle = buildCandidatePackBundleFromNormalizedEventExportBridge(input);

  rmSync(rootDir, { recursive: true, force: true });
  mkdirSync(rootDir, { recursive: true });

  const entries = bundle.entries.map((entry, index): MaterializedCandidatePackBundleEntry => {
    const entryRootDir = buildBundleEntryRootDir(rootDir, entry, index);

    return {
      ...entry,
      rootDir: path.resolve(entryRootDir),
      descriptor: materializeCandidatePackResult(entryRootDir, entry.build)
    };
  });

  return {
    runtimeOwner: bundle.runtimeOwner,
    bridgeDigest: bundle.bridgeDigest,
    bundleDigest: bundle.bundleDigest,
    cursor: cloneCursor(bundle.cursor),
    dedupedInputCount: bundle.dedupedInputCount,
    duplicateIdentityCount: bundle.duplicateIdentityCount,
    entries
  };
}
