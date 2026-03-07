import { mkdirSync, rmSync } from "node:fs";
import path from "node:path";

import {
  CONTRACT_IDS,
  checksumJsonPayload,
  computeRouterFreshnessChecksum,
  computeRouterQueryChecksum,
  computeRouterWeightsChecksum,
  sortNormalizedEvents,
  type ArtifactManifestV1,
  type FeedbackEventV1,
  type InteractionEventV1,
  type LearningSurfaceV1,
  type PackBlockStateV1,
  type NormalizedEventExportV1,
  type NormalizedEventV1,
  type PackBlockLearningSignalsV1,
  type PackContextBlockRecordV1,
  type PackGraphEdgeV1,
  type PackGraphPayloadV1,
  type RouterPolicyUpdateV1,
  type RouterSupervisionKind,
  type RouterTraceV1,
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
    learnedRouter: {
      routerIdentity: string | null;
      refreshStatus: RouterArtifactV1["training"]["status"] | null;
      updateCount: number;
      supervisionCount: number;
      weightsChecksum: string | null;
      visibleDelta: string[];
      noOpReason: string | null;
    };
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
  learnedGraph: PackGraphPayloadV1 | null;
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
    learnedGraph: null,
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
    learnedGraph: materialization?.candidate.payloads.graph ?? current.learnedGraph,
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

interface LearnerGraphBlockMeta {
  createdAt: string;
  sourceStream: string;
  sessionId?: string;
  channel?: string;
  relatedInteractionId?: string;
  syntheticRole: "base" | "split" | "merge";
  splitDepth: number;
}

interface LearnerGraphEvolutionResult {
  blocks: PackContextBlockRecordV1[];
  evolution: NonNullable<PackGraphPayloadV1["evolution"]>;
}

interface ConnectPairCandidate {
  leftId: string;
  rightId: string;
  score: number;
}

function roundMetric(value: number): number {
  return Math.round(value * 1_000) / 1_000;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function estimateTokenCount(text: string): number {
  return Math.max(1, keywordTokens(text).length);
}

function uniqueKeywords(values: readonly string[]): string[] {
  return [...new Set(values)].slice(0, 16);
}

function genericKeyword(token: string): boolean {
  return ["feedback", "interaction", "session", "openclaw", "runtime", "message", "memory", "pack"].includes(token);
}

function daysBetween(fromIso: string, toIso: string): number {
  return Math.max(0, Date.parse(toIso) - Date.parse(fromIso)) / 86_400_000;
}

function compareIsoDates(left: string, right: string): number {
  return Date.parse(left) - Date.parse(right);
}

function decayFreshness(createdAt: string, builtAt: string, halfLifeDays: number | null): number {
  if (halfLifeDays === null) {
    return 1;
  }

  return roundMetric(clamp(Math.pow(0.5, daysBetween(createdAt, builtAt) / halfLifeDays), 0.05, 1));
}

function keywordOverlap(left: readonly string[], right: readonly string[]): number {
  const rightSet = new Set(right);
  return left.reduce((count, keyword) => count + (rightSet.has(keyword) ? 1 : 0), 0);
}

function cloneGraphBlock(block: PackContextBlockRecordV1): PackContextBlockRecordV1 {
  return structuredClone(block);
}

function addEdge(edgesById: Map<string, PackGraphEdgeV1[]>, fromId: string, edge: PackGraphEdgeV1): boolean {
  const edges = edgesById.get(fromId);
  if (edges === undefined) {
    edgesById.set(fromId, [structuredClone(edge)]);
    return true;
  }

  const existing = edges.find((candidate) => candidate.kind === edge.kind && candidate.targetBlockId === edge.targetBlockId);
  if (existing !== undefined) {
    existing.weight = Math.max(existing.weight, edge.weight);
    return false;
  }

  edges.push(structuredClone(edge));
  return true;
}

function topFocusKeywords(block: PackContextBlockRecordV1): string[] {
  const focused = block.keywords.filter((keyword) => !genericKeyword(keyword));
  return (focused.length > 0 ? focused : block.keywords).slice(0, 4);
}

function splitBlockText(parent: PackContextBlockRecordV1): string {
  const focusKeywords = topFocusKeywords(parent);
  const focus = focusKeywords.length === 0 ? "learned memory" : focusKeywords.join(", ");
  const content = parent.text.split(/(?<=[.!?])/u).map((part) => part.trim()).find((part) => part.length >= 24) ?? parent.text;
  return `Focused memory on ${focus}: ${content}`;
}

function mergeBlockText(left: PackContextBlockRecordV1, right: PackContextBlockRecordV1): string {
  return `Merged memory path: ${left.text} ${right.text}`;
}

function buildBlockMetadata(
  packId: string,
  workspace: ReturnType<typeof createWorkspaceMetadata>,
  eventExport: NormalizedEventExportV1 | null
): Map<string, LearnerGraphBlockMeta> {
  const metadata = new Map<string, LearnerGraphBlockMeta>();
  const baseCreatedAt = workspace.capturedAt;

  for (const [blockId, sourceStream] of [
    [`${packId}:feedback-scanner`, "docs/openclaw-attach-quickstart.md"],
    [`${packId}:fast-boot-defaults`, "product/always-on-learning"],
    [`${packId}:passive-background-learning`, "product/always-on-learning"],
    [`${packId}:human-label-harvest`, "feedback_events.v1"],
    [`${packId}:self-label-harvest`, "interaction_events.v1:memory_compiled"],
    [`${packId}:workspace`, workspace.rootDir],
    [`${packId}:structural-ops`, "docs/learning-first-convergence.md"]
  ] as const) {
    metadata.set(blockId, {
      createdAt: baseCreatedAt,
      sourceStream,
      syntheticRole: "base",
      splitDepth: 0
    });
  }

  if (eventExport === null) {
    return metadata;
  }

  for (const event of [...eventExport.interactionEvents, ...eventExport.feedbackEvents]) {
    metadata.set(`${packId}:event:${event.eventId}`, {
      createdAt: event.createdAt,
      sourceStream: event.source.stream,
      sessionId: event.sessionId,
      channel: event.channel,
      ...(event.contract === CONTRACT_IDS.feedbackEvents && event.relatedInteractionId !== undefined
        ? { relatedInteractionId: event.relatedInteractionId }
        : {}),
      syntheticRole: "base",
      splitDepth: 0
    });
  }

  return metadata;
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

function addFeedbackEdges(
  edgesById: Map<string, PackGraphEdgeV1[]>,
  packId: string,
  eventExport: NormalizedEventExportV1 | null
): void {
  if (eventExport === null) {
    return;
  }

  for (const event of eventExport.feedbackEvents) {
    if (event.relatedInteractionId === undefined) {
      continue;
    }

    const feedbackBlockId = `${packId}:event:${event.eventId}`;
    const interactionBlockId = `${packId}:event:${event.relatedInteractionId}`;
    addEdge(edgesById, feedbackBlockId, {
      targetBlockId: interactionBlockId,
      kind: "feedback",
      weight: Math.max(2, eventPriority(event))
    });
    addEdge(edgesById, interactionBlockId, {
      targetBlockId: feedbackBlockId,
      kind: "feedback",
      weight: Math.max(2, eventPriority(event) - 1)
    });
  }
}

function connectPairs(
  blocks: readonly PackContextBlockRecordV1[],
  metadataById: ReadonlyMap<string, LearnerGraphBlockMeta>,
  edgesById: Map<string, PackGraphEdgeV1[]>,
  connectLimit: number
): number {
  const candidates: ConnectPairCandidate[] = [];

  for (let index = 0; index < blocks.length; index += 1) {
    const left = blocks[index];
    if (left === undefined) {
      continue;
    }

    for (let peerIndex = index + 1; peerIndex < blocks.length; peerIndex += 1) {
      const right = blocks[peerIndex];
      if (right === undefined) {
        continue;
      }

      const leftMeta = metadataById.get(left.id);
      const rightMeta = metadataById.get(right.id);
      const overlap = keywordOverlap(left.keywords, right.keywords);
      const sameStream = leftMeta?.sourceStream === rightMeta?.sourceStream ? 2 : 0;
      const sameSession = leftMeta?.sessionId !== undefined && leftMeta.sessionId === rightMeta?.sessionId ? 1 : 0;
      const score = overlap + sameStream + sameSession;
      if (score < 3) {
        continue;
      }

      candidates.push({
        leftId: left.id,
        rightId: right.id,
        score
      });
    }
  }

  candidates.sort((left, right) => {
    if (right.score !== left.score) {
      return right.score - left.score;
    }
    if (left.leftId !== right.leftId) {
      return left.leftId.localeCompare(right.leftId);
    }
    return left.rightId.localeCompare(right.rightId);
  });

  let applied = 0;
  for (const candidate of candidates) {
    if (applied >= connectLimit) {
      break;
    }

    const weight = Math.max(2, candidate.score);
    const createdLeft = addEdge(edgesById, candidate.leftId, {
      targetBlockId: candidate.rightId,
      kind: "connect",
      weight
    });
    const createdRight = addEdge(edgesById, candidate.rightId, {
      targetBlockId: candidate.leftId,
      kind: "connect",
      weight
    });

    if (createdLeft || createdRight) {
      applied += 1;
    }
  }

  return applied;
}

function splitBlocks(
  packId: string,
  blocks: PackContextBlockRecordV1[],
  metadataById: Map<string, LearnerGraphBlockMeta>,
  edgesById: Map<string, PackGraphEdgeV1[]>,
  splitLimit: number
): number {
  const candidates = blocks
    .filter(
      (block) =>
        block.id.includes(":event:") &&
        (block.learning.humanLabels > 0 ||
          block.learning.hebbianPulse >= 4 ||
          block.source.includes(":teaching") ||
          block.source.includes(":correction"))
    )
    .sort((left, right) => {
      const leftScore = left.learning.hebbianPulse + left.learning.humanLabels + left.priority;
      const rightScore = right.learning.hebbianPulse + right.learning.humanLabels + right.priority;
      if (rightScore !== leftScore) {
        return rightScore - leftScore;
      }
      return left.id.localeCompare(right.id);
    });

  let applied = 0;
  for (const parent of candidates) {
    if (applied >= splitLimit) {
      break;
    }

    const meta = metadataById.get(parent.id);
    if (meta === undefined) {
      continue;
    }

    const blockId = `${parent.id}:split:${applied + 1}`;
    const text = splitBlockText(parent);
    const splitBlock: PackContextBlockRecordV1 = {
      id: blockId,
      source: `split:${parent.source}`,
      text,
      tokenCount: estimateTokenCount(text),
      compactedFrom: [parent.id],
      keywords: uniqueKeywords([...topFocusKeywords(parent), ...keywordTokens(text), "split", "focused"]),
      priority: parent.priority + 1,
      learning: learningSignals({
        role: parent.learning.role,
        humanLabels: parent.learning.humanLabels,
        selfLabels: parent.learning.selfLabels,
        decayHalfLifeDays: parent.learning.decayHalfLifeDays,
        hebbianPulse: parent.learning.hebbianPulse + 2
      })
    };

    blocks.push(splitBlock);
    metadataById.set(blockId, {
      createdAt: meta.createdAt,
      sourceStream: meta.sourceStream,
      ...(meta.sessionId !== undefined ? { sessionId: meta.sessionId } : {}),
      ...(meta.channel !== undefined ? { channel: meta.channel } : {}),
      ...(meta.relatedInteractionId !== undefined ? { relatedInteractionId: meta.relatedInteractionId } : {}),
      syntheticRole: "split",
      splitDepth: meta.splitDepth + 1
    });
    edgesById.set(blockId, []);
    addEdge(edgesById, parent.id, { targetBlockId: blockId, kind: "split", weight: parent.learning.hebbianPulse + 1 });
    addEdge(edgesById, blockId, { targetBlockId: parent.id, kind: "merge", weight: Math.max(1, parent.priority - 1) });
    applied += 1;
  }

  return applied;
}

function mergeBlocks(
  packId: string,
  blocks: PackContextBlockRecordV1[],
  metadataById: Map<string, LearnerGraphBlockMeta>,
  edgesById: Map<string, PackGraphEdgeV1[]>,
  mergeLimit: number
): number {
  const candidates: ConnectPairCandidate[] = [];

  for (let index = 0; index < blocks.length; index += 1) {
    const left = blocks[index];
    if (left === undefined) {
      continue;
    }
    if (left.compactedFrom !== undefined) {
      continue;
    }

    for (let peerIndex = index + 1; peerIndex < blocks.length; peerIndex += 1) {
      const right = blocks[peerIndex];
      if (right === undefined || right.compactedFrom !== undefined) {
        continue;
      }

      const leftMeta = metadataById.get(left.id);
      const rightMeta = metadataById.get(right.id);
      const overlap = keywordOverlap(left.keywords, right.keywords);
      const related = leftMeta?.relatedInteractionId === right.id.replace(`${packId}:event:`, "") || rightMeta?.relatedInteractionId === left.id.replace(`${packId}:event:`, "") ? 3 : 0;
      const score = overlap + related + (leftMeta?.sourceStream === rightMeta?.sourceStream ? 1 : 0);

      if (score < 3) {
        continue;
      }

      candidates.push({ leftId: left.id, rightId: right.id, score });
    }
  }

  candidates.sort((left, right) => {
    if (right.score !== left.score) {
      return right.score - left.score;
    }
    if (left.leftId !== right.leftId) {
      return left.leftId.localeCompare(right.leftId);
    }
    return left.rightId.localeCompare(right.rightId);
  });

  const used = new Set<string>();
  let applied = 0;

  for (const candidate of candidates) {
    if (applied >= mergeLimit) {
      break;
    }
    if (used.has(candidate.leftId) || used.has(candidate.rightId)) {
      continue;
    }

    const left = blocks.find((block) => block.id === candidate.leftId);
    const right = blocks.find((block) => block.id === candidate.rightId);
    if (left === undefined || right === undefined) {
      continue;
    }

    const leftMeta = metadataById.get(left.id);
    const rightMeta = metadataById.get(right.id);
    if (leftMeta === undefined || rightMeta === undefined) {
      continue;
    }

    const blockId = `${packId}:merge:${applied + 1}`;
    const text = mergeBlockText(left, right);
    const mergedBlock: PackContextBlockRecordV1 = {
      id: blockId,
      source: `merge:${left.source}+${right.source}`,
      text,
      tokenCount: estimateTokenCount(text),
      compactedFrom: [left.id, right.id],
      keywords: uniqueKeywords([...left.keywords, ...right.keywords, "merge", "path", "connected"]),
      priority: Math.max(left.priority, right.priority) + 1,
      learning: learningSignals({
        role: left.learning.role === right.learning.role ? left.learning.role : "structural",
        humanLabels: left.learning.humanLabels + right.learning.humanLabels,
        selfLabels: left.learning.selfLabels + right.learning.selfLabels,
        decayHalfLifeDays: left.learning.decayHalfLifeDays ?? right.learning.decayHalfLifeDays,
        hebbianPulse: left.learning.hebbianPulse + right.learning.hebbianPulse
      })
    };

    blocks.push(mergedBlock);
    metadataById.set(blockId, {
      createdAt: compareIsoDates(leftMeta.createdAt, rightMeta.createdAt) >= 0 ? leftMeta.createdAt : rightMeta.createdAt,
      sourceStream: `${leftMeta.sourceStream}|${rightMeta.sourceStream}`,
      ...(leftMeta.sessionId !== undefined ? { sessionId: leftMeta.sessionId } : rightMeta.sessionId !== undefined ? { sessionId: rightMeta.sessionId } : {}),
      ...(leftMeta.channel !== undefined ? { channel: leftMeta.channel } : rightMeta.channel !== undefined ? { channel: rightMeta.channel } : {}),
      syntheticRole: "merge",
      splitDepth: 0
    });
    edgesById.set(blockId, []);
    addEdge(edgesById, blockId, { targetBlockId: left.id, kind: "merge", weight: Math.max(2, candidate.score) });
    addEdge(edgesById, blockId, { targetBlockId: right.id, kind: "merge", weight: Math.max(2, candidate.score) });
    addEdge(edgesById, left.id, { targetBlockId: blockId, kind: "connect", weight: Math.max(2, candidate.score - 1) });
    addEdge(edgesById, right.id, { targetBlockId: blockId, kind: "connect", weight: Math.max(2, candidate.score - 1) });

    used.add(left.id);
    used.add(right.id);
    applied += 1;
  }

  return applied;
}

function assignGraphState(
  blocks: readonly PackContextBlockRecordV1[],
  metadataById: ReadonlyMap<string, LearnerGraphBlockMeta>,
  edgesById: ReadonlyMap<string, PackGraphEdgeV1[]>,
  builtAt: string
): void {
  for (const block of blocks) {
    const metadata = metadataById.get(block.id);
    const freshness = decayFreshness(metadata?.createdAt ?? builtAt, builtAt, block.learning.decayHalfLifeDays);
    const edgeCount = edgesById.get(block.id)?.length ?? 0;
    const mergedFromCount = block.compactedFrom?.length ?? 0;
    const splitDepth = metadata?.splitDepth ?? 0;
    const hebbianGain = 1 + block.learning.humanLabels * 0.35 + block.learning.selfLabels * 0.2 + block.learning.hebbianPulse * 0.12;
    const structuralGain = 1 + mergedFromCount * 0.1 + splitDepth * 0.12 + edgeCount * 0.06;
    const evidenceCount = Math.max(1, block.learning.humanLabels + block.learning.selfLabels + mergedFromCount + (splitDepth > 0 ? 1 : 0));

    const state: PackBlockStateV1 = {
      strength: roundMetric(Math.max(0.25, block.priority * freshness * hebbianGain * structuralGain)),
      freshness,
      traversalBias: roundMetric(Math.max(0, freshness * 2 + edgeCount * 1.15 + mergedFromCount * 0.75 + splitDepth * 0.5)),
      evidenceCount,
      splitDepth,
      mergedFromCount,
      pruned: false
    };

    block.state = state;
  }
}

function pruneBlocks(
  blocks: readonly PackContextBlockRecordV1[],
  metadataById: ReadonlyMap<string, LearnerGraphBlockMeta>,
  eventExport: NormalizedEventExportV1 | null,
  pruneLimit: number
): string[] {
  if (pruneLimit === 0) {
    return [];
  }

  const suppressedInteractionIds = new Set(
    eventExport?.feedbackEvents
      .filter((event) => event.kind === "suppression" && event.relatedInteractionId !== undefined)
      .map((event) => `${event.relatedInteractionId}`) ?? []
  );

  const candidates = blocks
    .filter((block) => {
      const meta = metadataById.get(block.id);
      const eventId = block.id.includes(":event:") ? block.id.split(":event:")[1] : undefined;
      const suppressed = eventId !== undefined && suppressedInteractionIds.has(eventId);
      const lowStrength = (block.state?.strength ?? block.priority) <= 3.5;
      const labelFree = block.learning.humanLabels === 0 && block.learning.selfLabels === 0;
      const eventLike = meta?.syntheticRole === "base" && block.learning.role === "interaction";
      return suppressed || (eventLike && labelFree && lowStrength);
    })
    .sort((left, right) => {
      const leftEventId = left.id.includes(":event:") ? left.id.split(":event:")[1] : undefined;
      const rightEventId = right.id.includes(":event:") ? right.id.split(":event:")[1] : undefined;
      const leftSuppressed = leftEventId !== undefined && suppressedInteractionIds.has(leftEventId);
      const rightSuppressed = rightEventId !== undefined && suppressedInteractionIds.has(rightEventId);
      if (leftSuppressed !== rightSuppressed) {
        return leftSuppressed ? -1 : 1;
      }
      const leftStrength = left.state?.strength ?? left.priority;
      const rightStrength = right.state?.strength ?? right.priority;
      if (leftStrength !== rightStrength) {
        return leftStrength - rightStrength;
      }
      return left.id.localeCompare(right.id);
    });

  return candidates.slice(0, pruneLimit).map((block) => block.id);
}

function applyGraphEvolution(
  packId: string,
  builtAt: string,
  blocks: readonly PackContextBlockRecordV1[],
  metadataById: Map<string, LearnerGraphBlockMeta>,
  structuralOps: Required<ArtifactManifestV1["graphDynamics"]["structuralOps"]>,
  eventExport: NormalizedEventExportV1 | null
): LearnerGraphEvolutionResult {
  const workingBlocks = blocks.map((block) => cloneGraphBlock(block));
  const edgesById = new Map<string, PackGraphEdgeV1[]>(workingBlocks.map((block) => [block.id, []] as const));

  addFeedbackEdges(edgesById, packId, eventExport);
  const appliedSplit = splitBlocks(packId, workingBlocks, metadataById, edgesById, structuralOps.split);
  const appliedMerge = mergeBlocks(packId, workingBlocks, metadataById, edgesById, structuralOps.merge);
  const appliedConnect = connectPairs(workingBlocks, metadataById, edgesById, structuralOps.connect);

  assignGraphState(workingBlocks, metadataById, edgesById, builtAt);
  const prunedBlockIds = pruneBlocks(workingBlocks, metadataById, eventExport, structuralOps.prune);
  const pruned = new Set(prunedBlockIds);
  const survivors = workingBlocks.filter((block) => !pruned.has(block.id));
  const survivorIds = new Set(survivors.map((block) => block.id));

  for (const block of survivors) {
    const edges = (edgesById.get(block.id) ?? [])
      .filter((edge) => survivorIds.has(edge.targetBlockId))
      .sort((left, right) => {
        if (right.weight !== left.weight) {
          return right.weight - left.weight;
        }
        if (left.kind !== right.kind) {
          return left.kind.localeCompare(right.kind);
        }
        return left.targetBlockId.localeCompare(right.targetBlockId);
      });

    if (edges.length > 0) {
      block.edges = edges;
    }
  }

  assignGraphState(survivors, metadataById, new Map(survivors.map((block) => [block.id, block.edges ?? []] as const)), builtAt);

  survivors.sort((left, right) => {
    const leftStrength = left.state?.strength ?? left.priority;
    const rightStrength = right.state?.strength ?? right.priority;
    if (rightStrength !== leftStrength) {
      return rightStrength - leftStrength;
    }
    if (right.priority !== left.priority) {
      return right.priority - left.priority;
    }
    return left.id.localeCompare(right.id);
  });

  const strongestBlockId = survivors[0]?.id ?? null;

  return {
    blocks: survivors,
    evolution: {
      builtAt,
      hebbianApplied: true,
      decayApplied: true,
      structuralOps: {
        split: appliedSplit,
        merge: appliedMerge,
        prune: prunedBlockIds.length,
        connect: appliedConnect
      },
      prunedBlockIds,
      strongestBlockId
    }
  };
}

function createGraphPayload(
  packId: string,
  input: CandidatePackBuildInput,
  workspace: ReturnType<typeof createWorkspaceMetadata>,
  eventExport: NormalizedEventExportV1 | null,
  learningSurface: LearningSurfaceV1,
  builtAt: string
): PackGraphPayloadV1 {
  const baseBlocks = [...staticLifecycleBlocks(packId, input, workspace, learningSurface), ...(eventExport === null ? [] : eventExportBlocks(packId, eventExport))];
  const metadataById = buildBlockMetadata(packId, workspace, eventExport);
  const evolved = applyGraphEvolution(packId, builtAt, baseBlocks, metadataById, structuralOpsSummary(input), eventExport);

  return {
    packId,
    blocks: evolved.blocks,
    evolution: evolved.evolution
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
  if ((block.compactedFrom?.length ?? 0) > 1) {
    keywords.push("merged");
  }
  if ((block.state?.splitDepth ?? 0) > 0) {
    keywords.push("split");
  }
  if ((block.edges?.length ?? 0) > 0) {
    keywords.push("connected");
  }
  if ((block.state?.strength ?? 0) >= 6) {
    keywords.push("reinforced");
  }
  if ((block.state?.freshness ?? 1) < 0.5) {
    keywords.push("decayed");
  }
  for (const edge of block.edges ?? []) {
    keywords.push(edge.kind);
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
  if ((block.state?.strength ?? 0) > 0) {
    weights.reinforced = Math.max(weights.reinforced ?? 0, Math.ceil(block.state?.strength ?? 0));
  }
  if ((block.state?.freshness ?? 1) < 1) {
    weights.decayed = Math.max(weights.decayed ?? 0, Math.ceil((1 - (block.state?.freshness ?? 1)) * 4));
  }
  if ((block.edges?.length ?? 0) > 0) {
    weights.connected = Math.max(weights.connected ?? 0, block.edges?.length ?? 0);
    for (const edge of block.edges ?? []) {
      weights[edge.kind] = Math.max(weights[edge.kind] ?? 0, Math.ceil(edge.weight));
    }
  }

  return {
    blockId: block.id,
    keywords,
    boost:
      Math.max(1, Math.ceil(block.priority / 2)) +
      Math.min(3, block.learning.humanLabels + block.learning.selfLabels) +
      Math.min(2, Math.ceil(block.learning.hebbianPulse / 3)) +
      Math.min(3, Math.ceil((block.state?.traversalBias ?? 0) / 3)) +
      Math.min(2, block.edges?.length ?? 0) -
      ((block.state?.freshness ?? 1) < 0.35 ? 1 : 0),
    weights
  };
}

function createVectorsPayload(graph: PackGraphPayloadV1): PackVectorsPayloadV1 {
  return {
    packId: graph.packId,
    entries: graph.blocks.map((block) => vectorEntryFromBlock(block))
  };
}

function countKeywordWeights(value: string): Record<string, number> {
  const counts = new Map<string, number>();

  for (const token of value.toLowerCase().split(/[^a-z0-9]+/u)) {
    if (token.length < 3 || !/[a-z]/u.test(token)) {
      continue;
    }
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }

  return Object.fromEntries(counts.entries());
}

function lifecycleBlockIds(packId: string): {
  feedbackScanner: string;
  fastBootDefaults: string;
  passiveBackgroundLearning: string;
  humanLabelHarvest: string;
  selfLabelHarvest: string;
  workspace: string;
  structuralOps: string;
} {
  return {
    feedbackScanner: `${packId}:feedback-scanner`,
    fastBootDefaults: `${packId}:fast-boot-defaults`,
    passiveBackgroundLearning: `${packId}:passive-background-learning`,
    humanLabelHarvest: `${packId}:human-label-harvest`,
    selfLabelHarvest: `${packId}:self-label-harvest`,
    workspace: `${packId}:workspace`,
    structuralOps: `${packId}:structural-ops`
  };
}

function eventQueryTokens(event: NormalizedEventV1): string[] {
  const base =
    event.contract === CONTRACT_IDS.feedbackEvents
      ? `${event.content} ${summarizeEvent(event)} ${event.relatedInteractionId ?? ""} ${event.messageId ?? ""}`
      : `${summarizeEvent(event)} ${event.packId ?? ""} ${event.messageId ?? ""}`;
  return keywordTokens(`${event.kind} ${event.channel} ${event.source.stream} ${base}`);
}

function supervisionKindForEvent(event: NormalizedEventV1): RouterSupervisionKind {
  if (event.contract === CONTRACT_IDS.feedbackEvents) {
    return "human_feedback";
  }
  if (event.kind === "operator_override") {
    return "operator_override";
  }
  if (event.kind === "memory_compiled") {
    return "self_memory";
  }
  return "route_trace";
}

function rewardForEvent(event: NormalizedEventV1): number {
  if (event.contract === CONTRACT_IDS.feedbackEvents) {
    switch (event.kind) {
      case "correction":
        return 4;
      case "teaching":
        return 3;
      case "approval":
        return 2;
      case "suppression":
        return -3;
    }
  }

  switch (event.kind) {
    case "operator_override":
      return 4;
    case "memory_compiled":
      return 2;
    default:
      return 0;
  }
}

function targetBlockIdsForEvent(packId: string, event: NormalizedEventV1): string[] {
  const lifecycleIds = lifecycleBlockIds(packId);
  const targetBlockIds = new Set<string>([`${packId}:event:${event.eventId}`]);

  if (event.contract === CONTRACT_IDS.feedbackEvents) {
    targetBlockIds.add(lifecycleIds.feedbackScanner);
    targetBlockIds.add(lifecycleIds.humanLabelHarvest);
    targetBlockIds.add(lifecycleIds.fastBootDefaults);
    if (event.kind === "suppression") {
      targetBlockIds.add(lifecycleIds.passiveBackgroundLearning);
    } else {
      targetBlockIds.add(lifecycleIds.workspace);
    }
    if (event.relatedInteractionId !== undefined) {
      targetBlockIds.add(`${packId}:event:${event.relatedInteractionId}`);
    }
    return [...targetBlockIds];
  }

  if (event.kind === "operator_override") {
    targetBlockIds.add(lifecycleIds.humanLabelHarvest);
    targetBlockIds.add(lifecycleIds.workspace);
    return [...targetBlockIds];
  }

  if (event.kind === "memory_compiled") {
    targetBlockIds.add(lifecycleIds.selfLabelHarvest);
    targetBlockIds.add(lifecycleIds.structuralOps);
    return [...targetBlockIds];
  }

  targetBlockIds.add(lifecycleIds.passiveBackgroundLearning);
  return [...targetBlockIds];
}

function buildRouterTrace(packId: string, event: NormalizedEventV1): RouterTraceV1 {
  const queryTokens = eventQueryTokens(event);
  const traceId = `trace-${stableHash(checksumJsonPayload({ packId, eventId: event.eventId, kind: event.kind, queryTokens }))}`;

  return {
    traceId,
    sourceEventId: event.eventId,
    sourceContract: event.contract,
    sourceKind: event.kind,
    supervisionKind: supervisionKindForEvent(event),
    targetBlockIds: targetBlockIdsForEvent(packId, event),
    reward: rewardForEvent(event),
    queryTokens,
    queryVector: countKeywordWeights(
      event.contract === CONTRACT_IDS.feedbackEvents ? `${event.content} ${summarizeEvent(event)}` : summarizeEvent(event)
    )
  };
}

function buildBlockTokenWeights(block: PackContextBlockRecordV1, vectorEntry: PackVectorEntryV1 | undefined): Map<string, number> {
  const weights = new Map<string, number>();
  const assign = (keyword: string, weight: number): void => {
    for (const token of keywordTokens(keyword)) {
      weights.set(token, Math.max(weights.get(token) ?? 0, weight));
    }
  };

  for (const keyword of block.keywords) {
    assign(keyword, 1);
  }

  if (vectorEntry !== undefined) {
    for (const keyword of vectorEntry.keywords) {
      assign(keyword, 2);
    }
    for (const [keyword, weight] of Object.entries(vectorEntry.weights ?? {})) {
      assign(keyword, Math.max(1, Math.round(weight)));
    }
  }

  return weights;
}

function summarizeVisibleLearnedDelta(router: RouterArtifactV1 | null): CandidatePackBuildResult["summary"]["learnedRouter"] {
  return {
    routerIdentity: router?.routerIdentity ?? null,
    refreshStatus: router?.training.status ?? null,
    updateCount: router?.training.updateCount ?? 0,
    supervisionCount: router?.training.supervisionCount ?? 0,
    weightsChecksum: router?.training.weightsChecksum ?? null,
    visibleDelta: router?.policyUpdates.slice(0, 3).map((update) => `${update.blockId}:${update.delta}`) ?? [],
    noOpReason: router?.training.noOpReason ?? null
  };
}

function createRouterArtifact(
  packId: string,
  builtAt: string,
  graph: PackGraphPayloadV1,
  vectors: PackVectorsPayloadV1,
  eventExport: NormalizedEventExportV1 | null
): RouterArtifactV1 {
  const traces = eventExport === null ? [] : sortNormalizedEvents([...eventExport.interactionEvents, ...eventExport.feedbackEvents]).map((event) => buildRouterTrace(packId, event));
  const blockIds = new Set(graph.blocks.map((block) => block.id));
  const vectorEntries = new Map(vectors.entries.map((entry) => [entry.blockId, entry] as const));
  const policyUpdatesByBlock = new Map<
    string,
    {
      blockId: string;
      delta: number;
      evidenceCount: number;
      rewardSum: number;
      tokenWeights: Map<string, number>;
      traceIds: Set<string>;
    }
  >();

  for (const trace of traces) {
    if (trace.reward === 0) {
      continue;
    }

    const targetIds = trace.targetBlockIds.filter((blockId) => blockIds.has(blockId));

    for (const block of graph.blocks) {
      const vectorEntry = vectorEntries.get(block.id);
      const blockWeights = buildBlockTokenWeights(block, vectorEntry);
      const overlap = trace.queryTokens.filter((token) => blockWeights.has(token));
      const targeted = targetIds.includes(block.id);
      const overlapScore = overlap.reduce((sum, token) => sum + Math.max(1, Math.min(3, blockWeights.get(token) ?? 1)), 0);
      let delta = 0;

      if (targeted) {
        delta += trace.reward * 3;
      }
      if (overlapScore > 0) {
        delta += trace.reward > 0 ? overlapScore : -Math.min(overlapScore, Math.abs(trace.reward) * 2);
      }

      if (delta === 0) {
        continue;
      }

      const current =
        policyUpdatesByBlock.get(block.id) ?? {
          blockId: block.id,
          delta: 0,
          evidenceCount: 0,
          rewardSum: 0,
          tokenWeights: new Map<string, number>(),
          traceIds: new Set<string>()
        };

      current.delta += delta;
      current.evidenceCount += 1;
      current.rewardSum += trace.reward;
      current.traceIds.add(trace.traceId);

      for (const token of targeted ? trace.queryTokens : overlap) {
        current.tokenWeights.set(token, (current.tokenWeights.get(token) ?? 0) + (trace.reward > 0 ? 1 : -1));
      }

      policyUpdatesByBlock.set(block.id, current);
    }
  }

  const policyUpdates: RouterPolicyUpdateV1[] = [...policyUpdatesByBlock.values()]
    .map((update) => ({
      blockId: update.blockId,
      delta: update.delta,
      evidenceCount: update.evidenceCount,
      rewardSum: update.rewardSum,
      tokenWeights: Object.fromEntries([...update.tokenWeights.entries()].sort(([left], [right]) => left.localeCompare(right))),
      traceIds: [...update.traceIds].sort()
    }))
    .sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta) || right.evidenceCount - left.evidenceCount || left.blockId.localeCompare(right.blockId));

  const supervisionCount = traces.filter((trace) => trace.supervisionKind !== "route_trace" && trace.reward !== 0).length;
  const status = policyUpdates.length > 0 ? "updated" : "no_supervision";
  const noOpReason =
    policyUpdates.length > 0
      ? null
      : eventExport === null
        ? "no normalized event export supplied for learned routing refresh"
        : supervisionCount === 0
          ? "no canonical supervision found in normalized event export"
          : "supervision produced no learned routing delta";

  return {
    routerIdentity: `${packId}:route_fn`,
    strategy: "learned_route_fn_v1",
    trainedAt: builtAt,
    requiresLearnedRouting: true,
    training: {
      status,
      eventExportDigest: eventExport?.provenance.exportDigest ?? null,
      routeTraceCount: traces.length,
      supervisionCount,
      updateCount: policyUpdates.length,
      queryChecksum: computeRouterQueryChecksum(traces),
      weightsChecksum: computeRouterWeightsChecksum(policyUpdates),
      freshnessChecksum: computeRouterFreshnessChecksum({
        trainedAt: builtAt,
        status,
        eventExportDigest: eventExport?.provenance.exportDigest ?? null,
        routeTraceCount: traces.length,
        supervisionCount,
        updateCount: policyUpdates.length
      }),
      noOpReason
    },
    traces,
    policyUpdates
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
    builtAt,
    offlineArtifacts,
    structuralOps: structuralOpsSummary(input),
    eventExportDigest: eventExport?.provenance.exportDigest ?? null,
    learningSurface
  });
  const packId = `pack-${stableHash(seed)}`;

  const graph = createGraphPayload(packId, input, workspace, eventExport, learningSurface, builtAt);
  const vectors = createVectorsPayload(graph);
  const router = input.learnedRouting ? createRouterArtifact(packId, builtAt, graph, vectors, eventExport) : null;

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
      bootstrapping: manifest.graphDynamics.bootstrapping,
      learnedRouter: summarizeVisibleLearnedDelta(router)
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
