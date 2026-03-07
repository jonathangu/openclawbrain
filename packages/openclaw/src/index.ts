import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";

import { compileRuntimeFromActivation } from "@openclawbrain/compiler";
import {
  CONTRACT_IDS,
  buildNormalizedEventExport,
  canonicalJson,
  checksumJsonPayload,
  createFeedbackEvent,
  createInteractionEvent,
  sortNormalizedEvents,
  type ArtifactManifestV1,
  type ActivationPointerRecordV1,
  type FeedbackEventKind,
  type FeedbackEventV1,
  type InteractionEventV1,
  type NormalizedEventExportV1,
  type NormalizedEventV1,
  type RouteMode,
  type RuntimeCompileResponseV1,
  type RuntimeCompileTargetV1,
  type TeacherSupervisionArtifactV1,
  validateNormalizedEventExport
} from "@openclawbrain/contracts";
import {
  DEFAULT_TEACHER_SUPERVISION_STALE_AFTER_MS,
  advanceAlwaysOnLearningRuntime,
  buildTeacherSupervisionArtifactsFromNormalizedEventExport,
  createAlwaysOnLearningRuntimeState,
  materializeAlwaysOnLearningCandidatePack,
  type AdvanceAlwaysOnLearningRuntimeInput,
  type AlwaysOnLearningCadenceV1,
  type AlwaysOnLearningMaterializationJobV1,
  type AlwaysOnLearningRuntimeStateV1
} from "@openclawbrain/learner";
import {
  describeActivationTarget,
  describePackCompileTarget,
  inspectActivationState,
  promoteCandidatePack,
  stageCandidatePack,
  type ActivationSlotInspection
} from "@openclawbrain/pack-format";

const DEFAULT_AGENT_ID = "openclaw-runtime";
const FEEDBACK_KINDS = new Set<FeedbackEventKind>(["correction", "teaching", "approval", "suppression"]);

export const DEFAULT_ASYNC_TEACHER_QUEUE_CAPACITY = 8;


const RUNTIME_EVENT_EXPORT_BUNDLE_CONTRACT = "normalized_event_export_bundle.v1" as const;

export const RUNTIME_EVENT_EXPORT_BUNDLE_LAYOUT = {
  manifest: "manifest.json",
  payload: "normalized-event-export.json"
} as const;

export interface RuntimeEventExportBundleSummaryV1 {
  runtimeOwner: "openclaw";
  sessionId: string | null;
  channel: string | null;
  eventRange: Pick<NormalizedEventExportV1["range"], "start" | "end" | "count">;
  interactionCount: number;
  feedbackCount: number;
  sourceStreams: string[];
  contracts: NormalizedEventExportV1["provenance"]["contracts"];
}

export interface RuntimeEventExportBundleManifestV1 {
  contract: typeof RUNTIME_EVENT_EXPORT_BUNDLE_CONTRACT;
  exportName: string;
  exportedAt: string;
  payloadPath: string;
  payloadDigest: string;
  summary: RuntimeEventExportBundleSummaryV1;
}

export interface RuntimeEventExportBundleDescriptor {
  rootDir: string;
  manifestPath: string;
  payloadPath: string;
  manifest: RuntimeEventExportBundleManifestV1;
  normalizedEventExport: NormalizedEventExportV1;
}

export interface CompileRuntimeContextInput {
  activationRoot: string;
  message: string;
  agentId?: string;
  maxContextBlocks?: number;
  mode?: RouteMode;
  runtimeHints?: readonly string[];
}

export interface ActiveCompileTarget {
  activationRoot: string;
  activePointer: ActivationPointerRecordV1;
  inspection: ActivationSlotInspection;
}

export interface RuntimeCompileSuccess {
  ok: true;
  fallbackToStaticContext: false;
  hardRequirementViolated: false;
  activationRoot: string;
  activePackId: string;
  packRootDir: string;
  compileResponse: RuntimeCompileResponseV1;
  brainContext: string;
}

export interface RuntimeCompileFailOpenFailure {
  ok: false;
  fallbackToStaticContext: true;
  hardRequirementViolated: false;
  activationRoot: string;
  error: string;
  brainContext: string;
}

export interface RuntimeCompileHardFailure {
  ok: false;
  fallbackToStaticContext: false;
  hardRequirementViolated: true;
  activationRoot: string;
  error: string;
  brainContext: string;
}

export type RuntimeCompileFailure = RuntimeCompileFailOpenFailure | RuntimeCompileHardFailure;

export type RuntimeCompileResult = RuntimeCompileSuccess | RuntimeCompileFailure;

export interface RuntimeTurnCompileInput {
  createdAt?: string | null;
  sequence?: number | null;
  eventId?: string | null;
}

export interface RuntimeTurnDeliveryInput {
  createdAt?: string | null;
  sequence?: number | null;
  eventId?: string | null;
  messageId?: string | null;
}

export interface RuntimeTurnFeedbackInput {
  content: string;
  createdAt?: string | null;
  sequence?: number | null;
  eventId?: string | null;
  kind?: FeedbackEventKind | null;
  messageId?: string | null;
  relatedInteractionId?: string | null;
}

export interface RuntimeTurnExportInput {
  rootDir: string;
  exportName?: string | null;
  exportedAt?: string | null;
}

export interface OpenClawRuntimeTurnInput {
  activationRoot?: string | null;
  agentId?: string | null;
  sessionId: string;
  channel: string;
  sourceStream?: string | null;
  userMessage: string;
  createdAt?: string | null;
  sequenceStart?: number | null;
  maxContextBlocks?: number;
  mode?: RouteMode;
  runtimeHints?: readonly string[];
  compile?: RuntimeTurnCompileInput | null;
  delivery?: boolean | RuntimeTurnDeliveryInput | null;
  feedback?: readonly (RuntimeTurnFeedbackInput | null)[] | null;
  export?: RuntimeTurnExportInput | null;
}

export interface RuntimeEventExportNoWrite {
  ok: true;
  wroteBundle: false;
  normalizedEventExport: NormalizedEventExportV1;
}

export interface RuntimeEventExportWriteSuccess {
  ok: true;
  wroteBundle: true;
  normalizedEventExport: NormalizedEventExportV1;
  rootDir: string;
  manifestPath: string;
  payloadPath: string;
  manifest: RuntimeEventExportBundleManifestV1;
}

export interface RuntimeEventExportFailure {
  ok: false;
  wroteBundle: false;
  error: string;
}

export type RuntimeEventExportResult =
  | RuntimeEventExportNoWrite
  | RuntimeEventExportWriteSuccess
  | RuntimeEventExportFailure;

export interface RunRuntimeTurnOptions {
  activationRoot?: string;
  failOpen?: boolean;
}

export type RuntimeTurnResult = RuntimeCompileResult & {
  eventExport: RuntimeEventExportResult;
  warnings: string[];
};

export type TeacherLoopNoOpReason = "none" | "duplicate_export" | "queue_full" | "no_teacher_artifacts";

export interface AsyncTeacherLiveLoopInput
  extends Pick<
    AdvanceAlwaysOnLearningRuntimeInput,
    | "packLabel"
    | "workspace"
    | "learnedRouting"
    | "builtAt"
    | "offlineArtifacts"
    | "structuralOps"
    | "liveSliceSize"
    | "backfillSliceSize"
    | "cadence"
  > {
  maxQueuedExports?: number;
  staleAfterMs?: number;
}

export interface AsyncTeacherQueuedExportJobV1 {
  jobId: string;
  exportDigest: string;
  observedAt: string;
  normalizedEventExport: NormalizedEventExportV1;
}

export interface AsyncTeacherLiveLoopDiagnosticsV1 {
  acceptedExportCount: number;
  processedExportCount: number;
  duplicateExportCount: number;
  droppedExportCount: number;
  emittedArtifactCount: number;
  dedupedArtifactCount: number;
  lastProcessedAt: string | null;
  latestFreshness: TeacherSupervisionArtifactV1["freshness"]["status"] | "none";
  lastNoOpReason: TeacherLoopNoOpReason;
  notes: string[];
}

export interface AsyncTeacherLiveLoopSnapshotV1 {
  runtimeOwner: "openclaw";
  queue: {
    capacity: number;
    depth: number;
    running: boolean;
  };
  teacher: {
    artifactCount: number;
    artifacts: TeacherSupervisionArtifactV1[];
    latestFreshness: TeacherSupervisionArtifactV1["freshness"]["status"] | "none";
  };
  learner: {
    state: AlwaysOnLearningRuntimeStateV1;
    lastMaterialization: AlwaysOnLearningMaterializationJobV1 | null;
  };
  diagnostics: AsyncTeacherLiveLoopDiagnosticsV1;
}

export interface AsyncTeacherEnqueueResultV1 {
  accepted: boolean;
  exportDigest: string;
  queueDepth: number;
  notes: string[];
  reason: Exclude<TeacherLoopNoOpReason, "none"> | null;
}

export interface CanonicalSupervisionFeedbackRecordV1 {
  eventId: string;
  kind: FeedbackEventKind;
  sequence: number;
  createdAt: string;
  content: string;
  relatedInteractionId: string | null;
}

export interface CanonicalSupervisionV1 {
  runtimeOwner: "openclaw";
  exportDigest: string;
  supervisionDigest: string;
  sessionId: string | null;
  channel: string | null;
  eventRange: Pick<NormalizedEventExportV1["range"], "start" | "end" | "count">;
  sourceStreams: string[];
  humanLabelCount: number;
  selfLabelCount: number;
  feedbackCounts: {
    corrections: number;
    teachings: number;
    approvals: number;
    suppressions: number;
  };
  compilePackIds: string[];
  relatedInteractionIds: string[];
  feedback: CanonicalSupervisionFeedbackRecordV1[];
}

export interface ContinuousProductLoopPackVersionV1 {
  version: number;
  packId: string;
  routePolicy: RuntimeCompileTargetV1["routePolicy"];
  routerIdentity: string | null;
  workspaceSnapshot: string;
  workspaceRevision: string | null;
  eventRange: RuntimeCompileTargetV1["eventRange"];
  eventExportDigest: string | null;
  builtAt: string;
}

export interface ContinuousProductLoopStateV1 {
  runtimeOwner: "openclaw";
  activationRoot: string;
  loopRoot: string;
  interactionEvents: InteractionEventV1[];
  feedbackEvents: FeedbackEventV1[];
  learner: AlwaysOnLearningRuntimeStateV1;
  activePackVersion: number;
  currentActivePack: ContinuousProductLoopPackVersionV1 | null;
  candidatePack: ContinuousProductLoopPackVersionV1 | null;
  packLineage: ContinuousProductLoopPackVersionV1[];
  nextPackVersion: number;
  promotionCount: number;
  lastSupervision: CanonicalSupervisionV1 | null;
}

export interface ContinuousProductLoopLearningUpdateV1 {
  warnings: string[];
  supervisionDigest: string | null;
  bridgeDigest: string | null;
  selectedSliceIds: string[];
  materializationJobId: string | null;
  materializationReason: AlwaysOnLearningMaterializationJobV1["reason"] | null;
  materializationLane: AlwaysOnLearningMaterializationJobV1["lane"] | null;
  candidateRootDir: string | null;
  candidatePack: ContinuousProductLoopPackVersionV1 | null;
  promotionAllowed: boolean;
  promotionFindings: string[];
  promoted: boolean;
}

export interface RunContinuousProductLoopTurnInput {
  activationRoot: string;
  loopRoot: string;
  packLabel: string;
  workspace: AdvanceAlwaysOnLearningRuntimeInput["workspace"];
  turn: OpenClawRuntimeTurnInput;
  state?: ContinuousProductLoopStateV1;
  learnedRouting?: boolean;
  failOpen?: boolean;
  autoPromote?: boolean;
  candidateBuiltAt?: string | null;
  stageUpdatedAt?: string | null;
  promoteUpdatedAt?: string | null;
  offlineArtifacts?: string[];
  structuralOps?: Partial<ArtifactManifestV1["graphDynamics"]["structuralOps"]>;
  liveSliceSize?: number;
  backfillSliceSize?: number;
  cadence?: Partial<AlwaysOnLearningCadenceV1>;
}

export interface ContinuousProductLoopTurnResultV1 {
  runtimeOwner: "openclaw";
  compileActiveVersion: number;
  compileActivePackId: string | null;
  turn: RuntimeTurnResult;
  supervision: CanonicalSupervisionV1 | null;
  learning: ContinuousProductLoopLearningUpdateV1;
  state: ContinuousProductLoopStateV1;
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function buildAsyncTeacherLoopNotes(input: {
  queueDepth: number;
  latestFreshness: TeacherSupervisionArtifactV1["freshness"]["status"] | "none";
  artifactCount: number;
  emittedArtifactCount: number;
  dedupedArtifactCount: number;
  noOpReason: TeacherLoopNoOpReason;
  materialization: AlwaysOnLearningMaterializationJobV1 | null;
}): string[] {
  return [
    `teacher_queue_depth=${input.queueDepth}`,
    `teacher_freshness=${input.latestFreshness}`,
    `teacher_artifacts_total=${input.artifactCount}`,
    `teacher_artifacts_emitted=${input.emittedArtifactCount}`,
    `teacher_artifacts_deduped=${input.dedupedArtifactCount}`,
    `teacher_noop=${input.noOpReason}`,
    input.materialization === null ? "teacher_materialization=noop" : `teacher_materialized_pack=${input.materialization.candidate.summary.packId}`
  ];
}

function cloneAlwaysOnLearningMaterializationJobOrNull(
  value: AlwaysOnLearningMaterializationJobV1 | null
): AlwaysOnLearningMaterializationJobV1 | null {
  return value === null ? null : structuredClone(value);
}

function cloneTeacherSupervisionArtifacts(value: readonly TeacherSupervisionArtifactV1[]): TeacherSupervisionArtifactV1[] {
  return [...structuredClone(value)];
}

function cloneCanonicalSupervision(value: CanonicalSupervisionV1): CanonicalSupervisionV1 {
  return structuredClone(value);
}

function cloneContinuousProductLoopPackVersion(
  value: ContinuousProductLoopPackVersionV1
): ContinuousProductLoopPackVersionV1 {
  return structuredClone(value);
}

function cloneContinuousProductLoopState(value: ContinuousProductLoopStateV1): ContinuousProductLoopStateV1 {
  return structuredClone(value);
}

function buildNormalizedEventDedupId(event: NormalizedEventV1): string {
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

function mergeRuntimeEventHistory(
  current: Pick<ContinuousProductLoopStateV1, "interactionEvents" | "feedbackEvents">,
  incoming: NormalizedEventExportV1
): Pick<ContinuousProductLoopStateV1, "interactionEvents" | "feedbackEvents"> {
  const merged = sortNormalizedEvents([
    ...current.interactionEvents,
    ...current.feedbackEvents,
    ...incoming.interactionEvents,
    ...incoming.feedbackEvents
  ]);
  const deduped: NormalizedEventV1[] = [];
  const seen = new Set<string>();

  for (const event of merged) {
    const dedupId = buildNormalizedEventDedupId(event);
    if (seen.has(dedupId)) {
      continue;
    }

    seen.add(dedupId);
    deduped.push(event);
  }

  return {
    interactionEvents: deduped.filter((event): event is InteractionEventV1 => event.contract === CONTRACT_IDS.interactionEvents),
    feedbackEvents: deduped.filter((event): event is FeedbackEventV1 => event.contract === CONTRACT_IDS.feedbackEvents)
  };
}

function buildContinuousTurnExport(turn: OpenClawRuntimeTurnInput, loopRoot: string): RuntimeTurnExportInput {
  const exportSeed = checksumJsonPayload({
    sessionId: turn.sessionId,
    channel: turn.channel,
    sourceStream: turn.sourceStream ?? null,
    userMessage: turn.userMessage,
    createdAt: turn.createdAt ?? null,
    sequenceStart: turn.sequenceStart ?? null,
    compileCreatedAt: turn.compile?.createdAt ?? null,
    delivery:
      turn.delivery === false
        ? false
        : turn.delivery === undefined || turn.delivery === null
          ? null
          : turn.delivery === true
            ? true
            : {
                createdAt: turn.delivery.createdAt ?? null,
                messageId: turn.delivery.messageId ?? null,
                sequence: turn.delivery.sequence ?? null
              },
    feedback: (turn.feedback ?? [])
      .filter((item): item is RuntimeTurnFeedbackInput => item !== null)
      .map((item) => ({
        content: item.content,
        createdAt: item.createdAt ?? null,
        sequence: item.sequence ?? null,
        kind: item.kind ?? null,
        messageId: item.messageId ?? null,
        relatedInteractionId: item.relatedInteractionId ?? null
      }))
  })
    .replace(/^sha256-/u, "")
    .slice(0, 12);
  const exportName = `${turn.sessionId}-${exportSeed}`;

  return {
    rootDir: path.join(loopRoot, "event-exports", exportName),
    exportName
  };
}

function withContinuousTurnExport(turn: OpenClawRuntimeTurnInput, loopRoot: string): OpenClawRuntimeTurnInput {
  if (turn.export !== undefined && turn.export !== null) {
    return {
      ...turn,
      export: {
        ...turn.export
      }
    };
  }

  return {
    ...turn,
    export: buildContinuousTurnExport(turn, loopRoot)
  };
}

function buildPackVersion(version: number, target: RuntimeCompileTargetV1): ContinuousProductLoopPackVersionV1 {
  return {
    version,
    packId: target.packId,
    routePolicy: target.routePolicy,
    routerIdentity: target.routerIdentity,
    workspaceSnapshot: target.workspaceSnapshot,
    workspaceRevision: target.workspaceRevision,
    eventRange: {
      start: target.eventRange.start,
      end: target.eventRange.end,
      count: target.eventRange.count
    },
    eventExportDigest: target.eventExportDigest,
    builtAt: target.builtAt
  };
}

function buildLearningCandidateTarget(
  candidate: AlwaysOnLearningMaterializationJobV1["candidate"]
): RuntimeCompileTargetV1 {
  return {
    packId: candidate.summary.packId,
    routePolicy: candidate.summary.routePolicy,
    routerIdentity: candidate.payloads.router?.routerIdentity ?? null,
    workspaceSnapshot: candidate.summary.workspaceSnapshot,
    workspaceRevision: candidate.manifest.provenance.workspace.revision,
    eventRange: {
      start: candidate.summary.eventRange.start,
      end: candidate.summary.eventRange.end,
      count: candidate.summary.eventRange.count
    },
    eventExportDigest: candidate.summary.eventExportDigest,
    builtAt: candidate.manifest.provenance.builtAt
  };
}

function registerPackVersion(
  state: ContinuousProductLoopStateV1,
  target: RuntimeCompileTargetV1
): ContinuousProductLoopPackVersionV1 {
  const existing = state.packLineage.find((entry) => entry.packId === target.packId);
  if (existing !== undefined) {
    return cloneContinuousProductLoopPackVersion(existing);
  }

  const created = buildPackVersion(state.nextPackVersion, target);
  state.packLineage.push(cloneContinuousProductLoopPackVersion(created));
  state.nextPackVersion += 1;
  return created;
}

function tryReadActivePackTarget(rootDir: string): RuntimeCompileTargetV1 | null {
  try {
    return describeActivationTarget(rootDir, "active", { requireActivationReady: true });
  } catch {
    return null;
  }
}

function syncContinuousActivePack(state: ContinuousProductLoopStateV1): ContinuousProductLoopPackVersionV1 | null {
  const activeTarget = tryReadActivePackTarget(state.activationRoot);
  if (activeTarget === null) {
    state.currentActivePack = null;
    state.activePackVersion = 0;
    return null;
  }

  const activePack = registerPackVersion(state, activeTarget);
  state.currentActivePack = cloneContinuousProductLoopPackVersion(activePack);
  state.activePackVersion = activePack.version;
  return activePack;
}

function buildContinuousPackRoot(loopRoot: string, packVersion: ContinuousProductLoopPackVersionV1): string {
  return path.join(loopRoot, "packs", `v${String(packVersion.version).padStart(4, "0")}-${packVersion.packId}`);
}

export function buildCanonicalSupervision(normalizedEventExport: NormalizedEventExportV1): CanonicalSupervisionV1 {
  const feedback = normalizedEventExport.feedbackEvents.map((event) => ({
    eventId: event.eventId,
    kind: event.kind,
    sequence: event.sequence,
    createdAt: event.createdAt,
    content: event.content,
    relatedInteractionId: event.relatedInteractionId ?? null
  } satisfies CanonicalSupervisionFeedbackRecordV1));
  const compilePackIds = [
    ...new Set(
      normalizedEventExport.interactionEvents.flatMap((event) =>
        event.kind === "memory_compiled" && event.packId ? [event.packId] : []
      )
    )
  ];
  const relatedInteractionIds = [...new Set(feedback.flatMap((event) => (event.relatedInteractionId ? [event.relatedInteractionId] : [])))];
  const feedbackCounts = {
    corrections: feedback.filter((event) => event.kind === "correction").length,
    teachings: feedback.filter((event) => event.kind === "teaching").length,
    approvals: feedback.filter((event) => event.kind === "approval").length,
    suppressions: feedback.filter((event) => event.kind === "suppression").length
  };
  const supervisionDigest = checksumJsonPayload({
    exportDigest: normalizedEventExport.provenance.exportDigest,
    eventRange: {
      start: normalizedEventExport.range.start,
      end: normalizedEventExport.range.end,
      count: normalizedEventExport.range.count
    },
    sourceStreams: normalizedEventExport.provenance.sourceStreams,
    humanLabelCount: normalizedEventExport.provenance.learningSurface.labelHarvest.humanLabels,
    selfLabelCount: normalizedEventExport.provenance.learningSurface.labelHarvest.selfLabels,
    feedback,
    compilePackIds,
    relatedInteractionIds
  });

  return {
    runtimeOwner: "openclaw",
    exportDigest: normalizedEventExport.provenance.exportDigest,
    supervisionDigest,
    sessionId: normalizedEventExport.provenance.sessionId,
    channel: normalizedEventExport.provenance.channel,
    eventRange: {
      start: normalizedEventExport.range.start,
      end: normalizedEventExport.range.end,
      count: normalizedEventExport.range.count
    },
    sourceStreams: [...normalizedEventExport.provenance.sourceStreams],
    humanLabelCount: normalizedEventExport.provenance.learningSurface.labelHarvest.humanLabels,
    selfLabelCount: normalizedEventExport.provenance.learningSurface.labelHarvest.selfLabels,
    feedbackCounts,
    compilePackIds,
    relatedInteractionIds,
    feedback
  };
}

export function createContinuousProductLoopState(input: {
  activationRoot: string;
  loopRoot: string;
}): ContinuousProductLoopStateV1 {
  const activationRoot = path.resolve(normalizeNonEmptyString(input.activationRoot, "activationRoot"));
  const loopRoot = path.resolve(normalizeNonEmptyString(input.loopRoot, "loopRoot"));
  const activeTarget = tryReadActivePackTarget(activationRoot);
  const activePack = activeTarget === null ? null : buildPackVersion(1, activeTarget);

  return {
    runtimeOwner: "openclaw",
    activationRoot,
    loopRoot,
    interactionEvents: [],
    feedbackEvents: [],
    learner: createAlwaysOnLearningRuntimeState(),
    activePackVersion: activePack?.version ?? 0,
    currentActivePack: activePack === null ? null : cloneContinuousProductLoopPackVersion(activePack),
    candidatePack: null,
    packLineage: activePack === null ? [] : [cloneContinuousProductLoopPackVersion(activePack)],
    nextPackVersion: activePack === null ? 1 : 2,
    promotionCount: 0,
    lastSupervision: null
  };
}


function mergeUniqueEvents<T extends InteractionEventV1 | FeedbackEventV1>(
  current: readonly T[],
  additions: readonly T[]
): T[] {
  const merged = new Map<string, T>();

  for (const event of [...current, ...additions]) {
    merged.set(buildNormalizedEventDedupId(event), structuredClone(event));
  }

  return [...merged.values()].sort((left, right) => left.sequence - right.sequence || left.createdAt.localeCompare(right.createdAt));
}

function mergeTeacherArtifacts(
  current: readonly TeacherSupervisionArtifactV1[],
  additions: readonly TeacherSupervisionArtifactV1[]
): TeacherSupervisionArtifactV1[] {
  const merged = new Map<string, TeacherSupervisionArtifactV1>();

  for (const artifact of [...current, ...additions]) {
    const existing = merged.get(artifact.dedupId);
    if (
      existing === undefined ||
      Date.parse(artifact.freshness.observedAt) > Date.parse(existing.freshness.observedAt) ||
      (artifact.freshness.observedAt === existing.freshness.observedAt && artifact.artifactId.localeCompare(existing.artifactId) < 0)
    ) {
      merged.set(artifact.dedupId, structuredClone(artifact));
    }
  }

  return [...merged.values()].sort((left, right) => {
    if (left.freshness.status !== right.freshness.status) {
      return left.freshness.status === "fresh" ? -1 : 1;
    }
    if (left.createdAt !== right.createdAt) {
      return Date.parse(right.createdAt) - Date.parse(left.createdAt);
    }

    return left.artifactId.localeCompare(right.artifactId);
  });
}

function latestTeacherFreshness(
  artifacts: readonly TeacherSupervisionArtifactV1[]
): TeacherSupervisionArtifactV1["freshness"]["status"] | "none" {
  return artifacts[0]?.freshness.status ?? "none";
}

export class AsyncTeacherLiveLoop {
  private readonly input: AsyncTeacherLiveLoopInput;
  private readonly queueCapacity: number;
  private readonly staleAfterMs: number;
  private readonly queuedExportDigests = new Set<string>();
  private readonly seenExportDigests = new Set<string>();
  private queue: AsyncTeacherQueuedExportJobV1[] = [];
  private drainPromise: Promise<void> | null = null;
  private interactionEvents: InteractionEventV1[] = [];
  private feedbackEvents: FeedbackEventV1[] = [];
  private teacherArtifacts: TeacherSupervisionArtifactV1[] = [];
  private learnerState: AlwaysOnLearningRuntimeStateV1 = createAlwaysOnLearningRuntimeState();
  private lastMaterialization: AlwaysOnLearningMaterializationJobV1 | null = null;
  private diagnostics: AsyncTeacherLiveLoopDiagnosticsV1 = {
    acceptedExportCount: 0,
    processedExportCount: 0,
    duplicateExportCount: 0,
    droppedExportCount: 0,
    emittedArtifactCount: 0,
    dedupedArtifactCount: 0,
    lastProcessedAt: null,
    latestFreshness: "none",
    lastNoOpReason: "none",
    notes: buildAsyncTeacherLoopNotes({
      queueDepth: 0,
      latestFreshness: "none",
      artifactCount: 0,
      emittedArtifactCount: 0,
      dedupedArtifactCount: 0,
      noOpReason: "none",
      materialization: null
    })
  };

  constructor(input: AsyncTeacherLiveLoopInput) {
    this.input = input;
    this.queueCapacity = input.maxQueuedExports ?? DEFAULT_ASYNC_TEACHER_QUEUE_CAPACITY;
    this.staleAfterMs = input.staleAfterMs ?? DEFAULT_TEACHER_SUPERVISION_STALE_AFTER_MS;

    if (!Number.isInteger(this.queueCapacity) || this.queueCapacity <= 0) {
      throw new Error("maxQueuedExports must be a positive integer");
    }
    if (!Number.isInteger(this.staleAfterMs) || this.staleAfterMs <= 0) {
      throw new Error("staleAfterMs must be a positive integer");
    }
  }

  enqueueNormalizedEventExport(
    normalizedEventExport: NormalizedEventExportV1,
    options: { observedAt?: string } = {}
  ): AsyncTeacherEnqueueResultV1 {
    const validationErrors = validateNormalizedEventExport(normalizedEventExport);
    if (validationErrors.length > 0) {
      throw new Error(`normalized event export is invalid: ${validationErrors.join("; ")}`);
    }

    const exportDigest = normalizedEventExport.provenance.exportDigest;
    if (this.seenExportDigests.has(exportDigest) || this.queuedExportDigests.has(exportDigest)) {
      this.diagnostics.duplicateExportCount += 1;
      this.diagnostics.lastNoOpReason = "duplicate_export";
      this.refreshNotes();
      return {
        accepted: false,
        exportDigest,
        queueDepth: this.queue.length,
        notes: [...this.diagnostics.notes],
        reason: "duplicate_export"
      };
    }

    if (this.queue.length >= this.queueCapacity) {
      this.diagnostics.droppedExportCount += 1;
      this.diagnostics.lastNoOpReason = "queue_full";
      this.refreshNotes();
      return {
        accepted: false,
        exportDigest,
        queueDepth: this.queue.length,
        notes: [...this.diagnostics.notes],
        reason: "queue_full"
      };
    }

    const observedAt =
      options.observedAt ?? normalizedEventExport.range.lastCreatedAt ?? normalizedEventExport.range.firstCreatedAt ?? new Date().toISOString();

    this.queue.push({
      jobId: `teacher-loop-${createHash("sha256").update(`${exportDigest}:${observedAt}`).digest("hex")}`,
      exportDigest,
      observedAt,
      normalizedEventExport: structuredClone(normalizedEventExport)
    });
    this.queuedExportDigests.add(exportDigest);
    this.diagnostics.acceptedExportCount += 1;
    this.refreshNotes();
    void this.ensureDrain();

    return {
      accepted: true,
      exportDigest,
      queueDepth: this.queue.length,
      notes: [...this.diagnostics.notes],
      reason: null
    };
  }

  async flush(): Promise<AsyncTeacherLiveLoopSnapshotV1> {
    await this.ensureDrain();
    return this.snapshot();
  }

  snapshot(): AsyncTeacherLiveLoopSnapshotV1 {
    return {
      runtimeOwner: "openclaw",
      queue: {
        capacity: this.queueCapacity,
        depth: this.queue.length,
        running: this.drainPromise !== null
      },
      teacher: {
        artifactCount: this.teacherArtifacts.length,
        artifacts: cloneTeacherSupervisionArtifacts(this.teacherArtifacts),
        latestFreshness: this.diagnostics.latestFreshness
      },
      learner: {
        state: structuredClone(this.learnerState),
        lastMaterialization: cloneAlwaysOnLearningMaterializationJobOrNull(this.lastMaterialization)
      },
      diagnostics: {
        ...this.diagnostics,
        notes: [...this.diagnostics.notes]
      }
    };
  }

  private async ensureDrain(): Promise<void> {
    if (this.drainPromise === null) {
      this.drainPromise = this.drain().finally(() => {
        this.drainPromise = null;
      });
    }

    await this.drainPromise;

    if (this.queue.length > 0) {
      await this.ensureDrain();
    }
  }

  private async drain(): Promise<void> {
    while (this.queue.length > 0) {
      const job = this.queue.shift() as AsyncTeacherQueuedExportJobV1;
      this.queuedExportDigests.delete(job.exportDigest);
      this.seenExportDigests.add(job.exportDigest);

      this.interactionEvents = mergeUniqueEvents(this.interactionEvents, job.normalizedEventExport.interactionEvents);
      this.feedbackEvents = mergeUniqueEvents(this.feedbackEvents, job.normalizedEventExport.feedbackEvents);

      const builtArtifacts = buildTeacherSupervisionArtifactsFromNormalizedEventExport({
        normalizedEventExport: job.normalizedEventExport,
        observedAt: job.observedAt,
        staleAfterMs: this.staleAfterMs
      });
      const nextTeacherArtifacts = mergeTeacherArtifacts(this.teacherArtifacts, builtArtifacts);
      const emittedArtifactCount = builtArtifacts.length;
      const dedupedArtifactCount = Math.max(0, this.teacherArtifacts.length + builtArtifacts.length - nextTeacherArtifacts.length);

      this.teacherArtifacts = nextTeacherArtifacts;

      const learnerResult = advanceAlwaysOnLearningRuntime({
        packLabel: this.input.packLabel,
        workspace: this.input.workspace,
        interactionEvents: this.interactionEvents,
        feedbackEvents: this.feedbackEvents,
        teacherSupervisionArtifacts: this.teacherArtifacts,
        learnedRouting: this.input.learnedRouting,
        state: this.learnerState,
        builtAt: this.input.builtAt ?? job.observedAt,
        ...(this.input.offlineArtifacts !== undefined ? { offlineArtifacts: this.input.offlineArtifacts } : {}),
        ...(this.input.structuralOps !== undefined ? { structuralOps: this.input.structuralOps } : {}),
        ...(this.input.liveSliceSize !== undefined ? { liveSliceSize: this.input.liveSliceSize } : {}),
        ...(this.input.backfillSliceSize !== undefined ? { backfillSliceSize: this.input.backfillSliceSize } : {}),
        ...(this.input.cadence !== undefined ? { cadence: this.input.cadence } : {})
      });

      this.learnerState = structuredClone(learnerResult.state);
      this.lastMaterialization = cloneAlwaysOnLearningMaterializationJobOrNull(learnerResult.materialization);
      this.diagnostics.processedExportCount += 1;
      this.diagnostics.emittedArtifactCount += emittedArtifactCount;
      this.diagnostics.dedupedArtifactCount += dedupedArtifactCount;
      this.diagnostics.lastProcessedAt = job.observedAt;
      this.diagnostics.latestFreshness = latestTeacherFreshness(this.teacherArtifacts);
      this.diagnostics.lastNoOpReason = emittedArtifactCount === 0 ? "no_teacher_artifacts" : "none";
      this.refreshNotes();
    }
  }

  private refreshNotes(): void {
    this.diagnostics.notes = buildAsyncTeacherLoopNotes({
      queueDepth: this.queue.length,
      latestFreshness: this.diagnostics.latestFreshness,
      artifactCount: this.teacherArtifacts.length,
      emittedArtifactCount: this.diagnostics.emittedArtifactCount,
      dedupedArtifactCount: this.diagnostics.dedupedArtifactCount,
      noOpReason: this.diagnostics.lastNoOpReason,
      materialization: this.lastMaterialization
    });
  }
}

export function createAsyncTeacherLiveLoop(input: AsyncTeacherLiveLoopInput): AsyncTeacherLiveLoop {
  return new AsyncTeacherLiveLoop(input);
}

function readJsonFile<T>(filePath: string): T {
  return JSON.parse(readFileSync(filePath, "utf8")) as T;
}

function resolveBundlePayloadPath(rootDir: string, payloadPath: string): string {
  const resolved = path.resolve(rootDir, payloadPath);
  const relative = path.relative(rootDir, resolved);

  if (path.isAbsolute(payloadPath) || relative.startsWith("..") || relative === "") {
    throw new Error("event export bundle payloadPath must stay within the bundle root");
  }

  return resolved;
}

export function buildRuntimeEventExportBundleManifest(input: {
  exportName: string;
  exportedAt: string;
  payloadPath: string;
  normalizedEventExport: NormalizedEventExportV1;
}): RuntimeEventExportBundleManifestV1 {
  return {
    contract: RUNTIME_EVENT_EXPORT_BUNDLE_CONTRACT,
    exportName: input.exportName,
    exportedAt: input.exportedAt,
    payloadPath: input.payloadPath,
    payloadDigest: checksumJsonPayload(input.normalizedEventExport),
    summary: {
      runtimeOwner: input.normalizedEventExport.provenance.runtimeOwner,
      sessionId: input.normalizedEventExport.provenance.sessionId,
      channel: input.normalizedEventExport.provenance.channel,
      eventRange: {
        start: input.normalizedEventExport.range.start,
        end: input.normalizedEventExport.range.end,
        count: input.normalizedEventExport.range.count
      },
      interactionCount: input.normalizedEventExport.provenance.interactionCount,
      feedbackCount: input.normalizedEventExport.provenance.feedbackCount,
      sourceStreams: [...input.normalizedEventExport.provenance.sourceStreams],
      contracts: [...input.normalizedEventExport.provenance.contracts]
    }
  };
}

export function validateRuntimeEventExportBundleManifest(
  value: RuntimeEventExportBundleManifestV1,
  normalizedEventExport?: NormalizedEventExportV1
): string[] {
  const errors: string[] = [];

  if (value.contract !== RUNTIME_EVENT_EXPORT_BUNDLE_CONTRACT) {
    errors.push("normalized_event_export_bundle.v1 contract is required");
  }
  if (value.exportName.length === 0) {
    errors.push("exportName is required");
  }
  if (Number.isNaN(Date.parse(value.exportedAt))) {
    errors.push("exportedAt must be an ISO timestamp");
  }
  if (value.payloadPath.length === 0) {
    errors.push("payloadPath is required");
  }
  if (value.payloadDigest.length === 0) {
    errors.push("payloadDigest is required");
  }

  if (normalizedEventExport !== undefined) {
    const exportErrors = validateNormalizedEventExport(normalizedEventExport);
    if (exportErrors.length > 0) {
      errors.push(...exportErrors);
    }

    const rebuilt = buildRuntimeEventExportBundleManifest({
      exportName: value.exportName,
      exportedAt: value.exportedAt,
      payloadPath: value.payloadPath,
      normalizedEventExport
    });

    if (rebuilt.payloadDigest !== value.payloadDigest) {
      errors.push("event export bundle payloadDigest does not match the supplied normalized event export");
    }
    if (canonicalJson(rebuilt.summary) !== canonicalJson(value.summary)) {
      errors.push("event export bundle summary does not match the supplied normalized event export");
    }
  }

  return errors;
}

export function loadRuntimeEventExportBundle(rootDir: string): RuntimeEventExportBundleDescriptor {
  const resolvedRoot = path.resolve(rootDir);
  const manifestPath = path.join(resolvedRoot, RUNTIME_EVENT_EXPORT_BUNDLE_LAYOUT.manifest);
  const manifest = readJsonFile<RuntimeEventExportBundleManifestV1>(manifestPath);
  const payloadPath = resolveBundlePayloadPath(resolvedRoot, manifest.payloadPath);
  const normalizedEventExport = readJsonFile<NormalizedEventExportV1>(payloadPath);
  const validationErrors = validateRuntimeEventExportBundleManifest(manifest, normalizedEventExport);

  if (validationErrors.length > 0) {
    throw new Error(`event export bundle is invalid: ${validationErrors.join("; ")}`);
  }

  return {
    rootDir: resolvedRoot,
    manifestPath,
    payloadPath,
    manifest,
    normalizedEventExport
  };
}

function normalizeNonEmptyString(value: string | null | undefined, fieldName: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${fieldName} is required`);
  }

  return value.trim();
}

function normalizeOptionalString(value: string | null | undefined): string | undefined {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function normalizeNonNegativeInteger(
  value: number | null | undefined,
  fieldName: string,
  fallbackValue: number
): number {
  if (value === undefined || value === null) {
    return fallbackValue;
  }

  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${fieldName} must be a non-negative integer`);
  }

  return value;
}

function normalizeIsoTimestamp(
  value: string | null | undefined,
  fieldName: string,
  fallbackValue?: string | null
): string {
  const candidate = value ?? fallbackValue;
  if (candidate === undefined || candidate === null || candidate === "") {
    throw new Error(`${fieldName} is required`);
  }

  if (typeof candidate !== "string" || Number.isNaN(Date.parse(candidate))) {
    throw new Error(`${fieldName} must be an ISO timestamp`);
  }

  return new Date(candidate).toISOString();
}

function normalizeMode(value: RouteMode | undefined): RouteMode {
  return value ?? "heuristic";
}

function normalizeRuntimeHints(value: readonly string[] | undefined): string[] {
  if (value === undefined) {
    return [];
  }

  if (value.some((item) => typeof item !== "string" || item.trim().length === 0)) {
    throw new Error("runtimeHints must be an array of non-empty strings");
  }

  return value.map((item) => item.trim());
}

function deterministicEventId(value: unknown): string {
  const digest = createHash("sha256").update(JSON.stringify(value)).digest("hex");
  return `evt-${digest.slice(0, 16)}`;
}

function nextSequenceFactory(startValue = 1): (explicitValue?: number | null) => number {
  let cursor = normalizeNonNegativeInteger(startValue, "sequenceStart", 1);

  return (explicitValue?: number | null) => {
    if (explicitValue !== undefined && explicitValue !== null) {
      const normalized = normalizeNonNegativeInteger(explicitValue, "sequence", explicitValue);
      cursor = Math.max(cursor, normalized + 1);
      return normalized;
    }

    const current = cursor;
    cursor += 1;
    return current;
  };
}

function isFeedbackKind(value: string): value is FeedbackEventKind {
  return FEEDBACK_KINDS.has(value as FeedbackEventKind);
}

function isPresent<T>(value: T | null): value is T {
  return value !== null;
}

export function classifyFeedbackKind(content: string): FeedbackEventKind {
  const normalized = content.trim().toLowerCase();
  if (normalized.length === 0) {
    return "teaching";
  }
  if (/\b(stop|suppress|mute|silence|pause|don't send|do not send)\b/u.test(normalized)) {
    return "suppression";
  }
  if (/\b(approved|approve|looks good|ship it|exactly right|that works)\b/u.test(normalized)) {
    return "approval";
  }
  if (
    /\b(wrong|incorrect|correction|not right|do this instead|not x[, ]+y)\b/u.test(normalized) ||
    normalized.startsWith("no") ||
    normalized.startsWith("wrong")
  ) {
    return "correction";
  }
  return "teaching";
}

export function formatPromptContext(compileResponse: RuntimeCompileResponseV1): string {
  const lines = [
    "[BRAIN_CONTEXT v1]",
    `PACK_ID: ${compileResponse.packId}`,
    `MODE: ${compileResponse.diagnostics.modeEffective}`
  ];

  if (compileResponse.diagnostics.routerIdentity !== null) {
    lines.push(`ROUTER: ${compileResponse.diagnostics.routerIdentity}`);
  }

  if (compileResponse.selectedContext.length > 0) {
    lines.push("");
  }

  for (const block of compileResponse.selectedContext) {
    lines.push(`SOURCE: ${block.source}`);
    lines.push(`BLOCK_ID: ${block.id}`);
    lines.push(block.text.trim());
    lines.push("");
  }

  if (lines[lines.length - 1] === "") {
    lines.pop();
  }

  lines.push("[/BRAIN_CONTEXT]");
  return `${lines.join("\n")}\n`;
}

function failOpenCompileResult(error: unknown, activationRoot: string): RuntimeCompileFailOpenFailure {
  return {
    ok: false,
    fallbackToStaticContext: true,
    hardRequirementViolated: false,
    activationRoot: path.resolve(activationRoot),
    error: toErrorMessage(error),
    brainContext: ""
  };
}

function classifyCompileFailure(error: unknown, activationRoot: string): RuntimeCompileFailure {
  const resolvedActivationRoot = path.resolve(activationRoot);

  try {
    const inspection = inspectActivationState(resolvedActivationRoot);
    const active = inspection.active;
    if (active !== null && active.routePolicy === "requires_learned_routing") {
      const failureReason = active.findings.length > 0 ? active.findings.join("; ") : toErrorMessage(error);
      return {
        ok: false,
        fallbackToStaticContext: false,
        hardRequirementViolated: true,
        activationRoot: resolvedActivationRoot,
        error: `Learned-routing hotpath hard requirement violated for active pack ${active.packId} (routerIdentity=${active.routerIdentity ?? "null"}): ${failureReason}`,
        brainContext: ""
      };
    }
  } catch {
    return failOpenCompileResult(error, resolvedActivationRoot);
  }

  return failOpenCompileResult(error, resolvedActivationRoot);
}

export function resolveActivePackForCompile(activationRoot: string): ActiveCompileTarget {
  const resolvedActivationRoot = path.resolve(normalizeNonEmptyString(activationRoot, "activationRoot"));
  const inspection = inspectActivationState(resolvedActivationRoot);
  const activePointer = inspection.pointers.active;

  if (inspection.active === null || activePointer === null) {
    throw new Error(`No active pack pointer found in ${resolvedActivationRoot}`);
  }

  if (!inspection.active.activationReady) {
    throw new Error(`Active pack is not activation-ready: ${inspection.active.findings.join("; ")}`);
  }

  return {
    activationRoot: resolvedActivationRoot,
    activePointer,
    inspection: inspection.active
  };
}

export function compileRuntimeContext(input: CompileRuntimeContextInput): RuntimeCompileResult {
  const activationRoot = path.resolve(normalizeNonEmptyString(input.activationRoot, "activationRoot"));
  const agentId = normalizeOptionalString(input.agentId) ?? process.env.OPENCLAWBRAIN_AGENT_ID ?? DEFAULT_AGENT_ID;
  const runtimeHints = normalizeRuntimeHints(input.runtimeHints);

  try {
    const target = resolveActivePackForCompile(activationRoot);
    const compile = compileRuntimeFromActivation(activationRoot, {
      contract: CONTRACT_IDS.runtimeCompile,
      agentId,
      userMessage: normalizeNonEmptyString(input.message, "message"),
      maxContextBlocks: normalizeNonNegativeInteger(input.maxContextBlocks, "maxContextBlocks", 4),
      modeRequested: normalizeMode(input.mode),
      activePackId: target.activePointer.packId,
      ...(runtimeHints.length > 0 ? { runtimeHints } : {})
    });

    const compileResponse = {
      ...compile.response,
      diagnostics: {
        ...compile.response.diagnostics,
        notes: [...compile.response.diagnostics.notes, "OpenClaw remains the runtime owner"]
      }
    };

    return {
      ok: true,
      fallbackToStaticContext: false,
      hardRequirementViolated: false,
      activationRoot,
      activePackId: compile.target.packId,
      packRootDir: path.resolve(target.activePointer.packRootDir),
      compileResponse,
      brainContext: formatPromptContext(compileResponse)
    };
  } catch (error) {
    return classifyCompileFailure(error, activationRoot);
  }
}

function buildCompileInteractionEvent(input: {
  turn: OpenClawRuntimeTurnInput;
  compileResult: RuntimeCompileResult;
  sourceStream: string;
  nextSequence: (explicitValue?: number | null) => number;
  createdAt: string;
  agentId: string;
}): InteractionEventV1 | null {
  if (!input.compileResult.ok) {
    return null;
  }

  const sequence = input.nextSequence(input.turn.compile?.sequence);
  const eventId = normalizeOptionalString(input.turn.compile?.eventId) ?? deterministicEventId({
    channel: input.turn.channel,
    createdAt: input.createdAt,
    kind: "memory_compiled",
    packId: input.compileResult.compileResponse.packId,
    sequence,
    sessionId: input.turn.sessionId,
    source: input.sourceStream
  });

  return createInteractionEvent({
    eventId,
    agentId: input.agentId,
    sessionId: input.turn.sessionId,
    channel: input.turn.channel,
    sequence,
    kind: "memory_compiled",
    createdAt: input.createdAt,
    source: {
      runtimeOwner: "openclaw",
      stream: input.sourceStream
    },
    packId: input.compileResult.compileResponse.packId
  });
}

function buildDeliveryInteractionEvent(input: {
  turn: OpenClawRuntimeTurnInput;
  compileResult: RuntimeCompileResult;
  sourceStream: string;
  nextSequence: (explicitValue?: number | null) => number;
  defaultCreatedAt: string;
  agentId: string;
}): InteractionEventV1 | null {
  if (input.turn.delivery === undefined || input.turn.delivery === null || input.turn.delivery === false) {
    return null;
  }

  const delivery = typeof input.turn.delivery === "object" ? input.turn.delivery : {};
  const createdAt = normalizeIsoTimestamp(delivery.createdAt, "delivery.createdAt", input.defaultCreatedAt);
  const sequence = input.nextSequence(delivery.sequence);
  const messageId = normalizeOptionalString(delivery.messageId);
  const eventId = normalizeOptionalString(delivery.eventId) ?? deterministicEventId({
    channel: input.turn.channel,
    createdAt,
    kind: "message_delivered",
    messageId: messageId ?? null,
    packId: input.compileResult.ok ? input.compileResult.compileResponse.packId : null,
    sequence,
    sessionId: input.turn.sessionId,
    source: input.sourceStream
  });

  return createInteractionEvent({
    eventId,
    agentId: input.agentId,
    sessionId: input.turn.sessionId,
    channel: input.turn.channel,
    sequence,
    kind: "message_delivered",
    createdAt,
    source: {
      runtimeOwner: "openclaw",
      stream: input.sourceStream
    },
    ...(input.compileResult.ok ? { packId: input.compileResult.compileResponse.packId } : {}),
    ...(messageId !== undefined ? { messageId } : {})
  });
}

function buildFeedbackEvents(input: {
  turn: OpenClawRuntimeTurnInput;
  sourceStream: string;
  nextSequence: (explicitValue?: number | null) => number;
  defaultCreatedAt: string;
  compileInteraction: InteractionEventV1 | null;
  agentId: string;
}): FeedbackEventV1[] {
  const feedbackItems = input.turn.feedback ?? [];

  return feedbackItems.map((item, index) => {
    if (item === null) {
      throw new Error(`feedback[${index}] must be an object`);
    }

    const content = normalizeNonEmptyString(item.content, `feedback[${index}].content`);
    const createdAt = normalizeIsoTimestamp(item.createdAt, `feedback[${index}].createdAt`, input.defaultCreatedAt);
    const sequence = input.nextSequence(item.sequence);
    const kind = item.kind === undefined || item.kind === null ? classifyFeedbackKind(content) : item.kind;

    if (!isFeedbackKind(kind)) {
      throw new Error(`feedback[${index}].kind must be correction, teaching, approval, or suppression`);
    }

    const messageId = normalizeOptionalString(item.messageId);
    const eventId = normalizeOptionalString(item.eventId) ?? deterministicEventId({
      channel: input.turn.channel,
      content,
      createdAt,
      kind,
      messageId: messageId ?? null,
      sequence,
      sessionId: input.turn.sessionId,
      source: input.sourceStream
    });
    const relatedInteractionId = normalizeOptionalString(item.relatedInteractionId) ?? input.compileInteraction?.eventId;

    return createFeedbackEvent({
      eventId,
      agentId: input.agentId,
      sessionId: input.turn.sessionId,
      channel: input.turn.channel,
      sequence,
      kind,
      createdAt,
      source: {
        runtimeOwner: "openclaw",
        stream: input.sourceStream
      },
      content,
      ...(messageId !== undefined ? { messageId } : {}),
      ...(relatedInteractionId !== undefined ? { relatedInteractionId } : {})
    });
  });
}

export function buildNormalizedRuntimeEventExport(
  turn: OpenClawRuntimeTurnInput,
  compileResult: RuntimeCompileResult
): NormalizedEventExportV1 {
  const agentId = normalizeOptionalString(turn.agentId) ?? process.env.OPENCLAWBRAIN_AGENT_ID ?? DEFAULT_AGENT_ID;
  const sessionId = normalizeNonEmptyString(turn.sessionId, "sessionId");
  const channel = normalizeNonEmptyString(turn.channel, "channel");
  const sourceStream = normalizeOptionalString(turn.sourceStream) ?? `openclaw/runtime/${channel}`;
  const compileCreatedAt = normalizeIsoTimestamp(turn.compile?.createdAt, "compile.createdAt", turn.createdAt);
  const nextSequence = nextSequenceFactory(turn.sequenceStart ?? 1);
  const normalizedTurn: OpenClawRuntimeTurnInput = {
    ...turn,
    agentId,
    channel,
    sessionId
  };

  const compileInteraction = buildCompileInteractionEvent({
    turn: normalizedTurn,
    compileResult,
    sourceStream,
    nextSequence,
    createdAt: compileCreatedAt,
    agentId
  });
  const feedbackEvents = buildFeedbackEvents({
    turn: normalizedTurn,
    sourceStream,
    nextSequence,
    defaultCreatedAt: compileCreatedAt,
    compileInteraction,
    agentId
  });
  const deliveryInteraction = buildDeliveryInteractionEvent({
    turn: normalizedTurn,
    compileResult,
    sourceStream,
    nextSequence,
    defaultCreatedAt: compileCreatedAt,
    agentId
  });
  const interactionEvents = [compileInteraction, deliveryInteraction].filter(isPresent);

  if (interactionEvents.length === 0 && feedbackEvents.length === 0) {
    throw new Error("runtime turn did not produce any normalized events");
  }

  const normalizedEventExport = buildNormalizedEventExport({
    interactionEvents,
    feedbackEvents
  });
  const validationErrors = validateNormalizedEventExport(normalizedEventExport);
  if (validationErrors.length > 0) {
    throw new Error(`normalized event export is invalid: ${validationErrors.join("; ")}`);
  }

  return normalizedEventExport;
}

export function writeRuntimeEventExportBundle(
  turn: OpenClawRuntimeTurnInput,
  normalizedEventExport: NormalizedEventExportV1
): RuntimeEventExportNoWrite | RuntimeEventExportWriteSuccess {
  if (turn.export === undefined || turn.export === null) {
    return {
      ok: true,
      wroteBundle: false,
      normalizedEventExport
    };
  }

  const rootDir = normalizeNonEmptyString(turn.export.rootDir, "export.rootDir");
  const exportName =
    normalizeOptionalString(turn.export.exportName) ??
    `${turn.sessionId}-${normalizedEventExport.range.start}-${normalizedEventExport.range.end}`;
  const exportedAt = normalizeIsoTimestamp(
    turn.export.exportedAt,
    "export.exportedAt",
    normalizedEventExport.range.lastCreatedAt ?? normalizedEventExport.range.firstCreatedAt ?? new Date().toISOString()
  );

  const resolvedRoot = path.resolve(rootDir);
  const manifest = buildRuntimeEventExportBundleManifest({
    exportName,
    exportedAt,
    payloadPath: RUNTIME_EVENT_EXPORT_BUNDLE_LAYOUT.payload,
    normalizedEventExport
  });
  const manifestPath = path.join(resolvedRoot, RUNTIME_EVENT_EXPORT_BUNDLE_LAYOUT.manifest);
  const payloadPath = path.join(resolvedRoot, manifest.payloadPath);

  mkdirSync(path.dirname(payloadPath), { recursive: true });
  writeFileSync(payloadPath, canonicalJson(normalizedEventExport), "utf8");
  writeFileSync(manifestPath, canonicalJson(manifest), "utf8");
  const descriptor = loadRuntimeEventExportBundle(resolvedRoot);

  return {
    ok: true,
    wroteBundle: true,
    normalizedEventExport,
    rootDir: descriptor.rootDir,
    manifestPath: descriptor.manifestPath,
    payloadPath: descriptor.payloadPath,
    manifest: descriptor.manifest
  };
}

export function runRuntimeTurn(turn: OpenClawRuntimeTurnInput, options: RunRuntimeTurnOptions = {}): RuntimeTurnResult {
  const agentId = normalizeOptionalString(turn.agentId);
  const compileInput: CompileRuntimeContextInput = {
    activationRoot: options.activationRoot ?? normalizeNonEmptyString(turn.activationRoot ?? undefined, "activationRoot"),
    message: normalizeNonEmptyString(turn.userMessage, "userMessage"),
    ...(agentId !== undefined ? { agentId } : {}),
    ...(turn.maxContextBlocks !== undefined ? { maxContextBlocks: turn.maxContextBlocks } : {}),
    ...(turn.mode !== undefined ? { mode: turn.mode } : {}),
    ...(turn.runtimeHints !== undefined ? { runtimeHints: turn.runtimeHints } : {})
  };
  const compileResult = compileRuntimeContext(compileInput);
  if (!compileResult.ok && compileResult.hardRequirementViolated) {
    throw new Error(compileResult.error);
  }
  const warnings: string[] = [];

  try {
    const normalizedEventExport = buildNormalizedRuntimeEventExport(turn, compileResult);
    const eventExport = writeRuntimeEventExportBundle(turn, normalizedEventExport);
    return {
      ...compileResult,
      eventExport,
      warnings
    };
  } catch (error) {
    if (options.failOpen === false) {
      throw error;
    }

    warnings.push(toErrorMessage(error));
    return {
      ...compileResult,
      eventExport: {
        ok: false,
        wroteBundle: false,
        error: toErrorMessage(error)
      },
      warnings
    };
  }
}

export function runContinuousProductLoopTurn(
  input: RunContinuousProductLoopTurnInput
): ContinuousProductLoopTurnResultV1 {
  const activationRoot = path.resolve(normalizeNonEmptyString(input.activationRoot, "activationRoot"));
  const loopRoot = path.resolve(normalizeNonEmptyString(input.loopRoot, "loopRoot"));
  const failOpen = input.failOpen !== false;
  const currentState = cloneContinuousProductLoopState(
    input.state ??
      createContinuousProductLoopState({
        activationRoot,
        loopRoot
      })
  );

  currentState.activationRoot = activationRoot;
  currentState.loopRoot = loopRoot;

  const activeBeforeTurn = syncContinuousActivePack(currentState);
  const compileActiveVersion = activeBeforeTurn?.version ?? 0;
  const compileActivePackId = activeBeforeTurn?.packId ?? null;
  const turn = withContinuousTurnExport(input.turn, loopRoot);
  const turnResult = runRuntimeTurn(turn, {
    activationRoot,
    failOpen
  });
  const learningWarnings: string[] = [];
  let supervision: CanonicalSupervisionV1 | null = null;
  const learning: ContinuousProductLoopLearningUpdateV1 = {
    warnings: learningWarnings,
    supervisionDigest: null,
    bridgeDigest: null,
    selectedSliceIds: [],
    materializationJobId: null,
    materializationReason: null,
    materializationLane: null,
    candidateRootDir: null,
    candidatePack: currentState.candidatePack === null ? null : cloneContinuousProductLoopPackVersion(currentState.candidatePack),
    promotionAllowed: false,
    promotionFindings: [],
    promoted: false
  };

  if (!turnResult.eventExport.ok) {
    learningWarnings.push(`continuous learner skipped: ${turnResult.eventExport.error}`);
    return {
      runtimeOwner: "openclaw",
      compileActiveVersion,
      compileActivePackId,
      turn: turnResult,
      supervision,
      learning,
      state: cloneContinuousProductLoopState(currentState)
    };
  }

  const normalizedEventExport = turnResult.eventExport.normalizedEventExport;
  supervision = buildCanonicalSupervision(normalizedEventExport);
  learning.supervisionDigest = supervision.supervisionDigest;
  currentState.lastSupervision = cloneCanonicalSupervision(supervision);

  const mergedHistory = mergeRuntimeEventHistory(currentState, normalizedEventExport);
  currentState.interactionEvents = mergedHistory.interactionEvents;
  currentState.feedbackEvents = mergedHistory.feedbackEvents;

  try {
    const learnerResult = advanceAlwaysOnLearningRuntime({
      packLabel: input.packLabel,
      workspace: input.workspace,
      interactionEvents: currentState.interactionEvents,
      feedbackEvents: currentState.feedbackEvents,
      learnedRouting: input.learnedRouting ?? true,
      state: currentState.learner,
      builtAt: normalizeIsoTimestamp(
        input.candidateBuiltAt,
        "candidateBuiltAt",
        normalizedEventExport.range.lastCreatedAt ?? normalizedEventExport.range.firstCreatedAt
      ),
      ...(input.offlineArtifacts !== undefined ? { offlineArtifacts: input.offlineArtifacts } : {}),
      ...(input.structuralOps !== undefined ? { structuralOps: input.structuralOps } : {}),
      ...(input.liveSliceSize !== undefined ? { liveSliceSize: input.liveSliceSize } : {}),
      ...(input.backfillSliceSize !== undefined ? { backfillSliceSize: input.backfillSliceSize } : {}),
      ...(input.cadence !== undefined ? { cadence: input.cadence } : {})
    });

    currentState.learner = structuredClone(learnerResult.state);
    learning.bridgeDigest = learnerResult.bridge.bridgeDigest;
    learning.selectedSliceIds = learnerResult.selectedSlices.map((slice) => slice.sliceId);
    learning.materializationJobId = learnerResult.materialization?.jobId ?? null;
    learning.materializationReason = learnerResult.materialization?.reason ?? null;
    learning.materializationLane = learnerResult.materialization?.lane ?? null;

    if (learnerResult.materialization !== null) {
      const candidatePack = registerPackVersion(currentState, buildLearningCandidateTarget(learnerResult.materialization.candidate));
      const candidateRootDir = buildContinuousPackRoot(loopRoot, candidatePack);
      const descriptor = materializeAlwaysOnLearningCandidatePack(candidateRootDir, learnerResult.materialization);
      const candidateTarget = describePackCompileTarget(descriptor);

      learning.candidateRootDir = candidateRootDir;
      learning.candidatePack = cloneContinuousProductLoopPackVersion(candidatePack);
      currentState.candidatePack = cloneContinuousProductLoopPackVersion(candidatePack);

      const stagedAt = normalizeIsoTimestamp(input.stageUpdatedAt, "stageUpdatedAt", descriptor.manifest.provenance.builtAt);

      stageCandidatePack(activationRoot, candidateRootDir, stagedAt);

      const stagedInspection = inspectActivationState(activationRoot, stagedAt);
      learning.promotionAllowed = stagedInspection.promotion.allowed;
      learning.promotionFindings = [...stagedInspection.promotion.findings];

      if ((input.autoPromote ?? true) && stagedInspection.promotion.allowed) {
        const promotedAt = normalizeIsoTimestamp(input.promoteUpdatedAt, "promoteUpdatedAt", stagedAt);

        promoteCandidatePack(activationRoot, promotedAt);
        currentState.promotionCount += 1;
        currentState.candidatePack = null;
        learning.promoted = true;

        const activePack = registerPackVersion(currentState, candidateTarget);
        currentState.currentActivePack = cloneContinuousProductLoopPackVersion(activePack);
        currentState.activePackVersion = activePack.version;
        syncContinuousActivePack(currentState);
      }
    }
  } catch (error) {
    if (!failOpen) {
      throw error;
    }

    learningWarnings.push(`continuous learner failed open: ${toErrorMessage(error)}`);
  }

  return {
    runtimeOwner: "openclaw",
    compileActiveVersion,
    compileActivePackId,
    turn: turnResult,
    supervision,
    learning,
    state: cloneContinuousProductLoopState(currentState)
  };
}
