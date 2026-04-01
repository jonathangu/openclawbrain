import type {
  BrainConfig,
  BrainEdge,
  BrainEvidence,
  BrainEvidenceResolution,
  BrainObservationBindingMode,
  BundleEvaluationVerdict,
  Episode,
  MutationProposal,
  ObservationBindingQuality,
  PolicyGradientCandidateUpdateArtifact,
  PolicyGradientCandidateUpdateArtifactV2,
  PolicyGradientEpisodeUpdateArtifact,
  PolicyGradientRouteUpdateArtifact,
  PolicyGradientRouteUpdateContribution,
  PolicyGradientSupervisionArtifact,
  PromotionRunVerdict,
  ReplayGateVerdict,
  RewardSource,
  TeacherFeedbackRichness,
} from "../brain-core/types.js";
import { classifyObservationBindingQuality, START_NODE_ID, trustRank } from "../brain-core/types.js";
import type { BrainStore } from "../brain-store/store.js";
import type { BrainGraph } from "../brain-core/graph.js";
import type { BrainTeacher } from "../brain-core/teacher.js";
import type { BrainMutator } from "../brain-core/mutator.js";
import type { PackManager } from "../brain-core/pack.js";
import {
  applyWeightUpdates,
  collectReinforceUpdateContributions,
  computeReinforceUpdates,
  policyWeightUpdateKey,
  updateBaseline,
} from "../brain-core/update.js";
import { decayAllWeights } from "../brain-core/decay.js";
import { computeHealth } from "../brain-core/health.js";
import { clusterMutationsIntoBundles, evaluateBundle, DEFAULT_BUNDLE_CONFIG } from "../brain-core/bundle-evaluator.js";
import type { BundleEvaluationConfig, MutationBundle } from "../brain-core/bundle-evaluator.js";
import { evaluateContextUsefulness } from "../brain-core/usefulness.js";

const TEACHER_OBSERVATION_BUDGET = 20;
const TEACHER_CYCLE_DECISION_LIMIT = 8;

function toOptionalString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function toStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string")
    : [];
}

function toRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function hasNonEmptyToolResults(value: unknown): boolean {
  return Array.isArray(value)
    ? value.length > 0
    : typeof value === "string" && value.trim().length > 0 && value.trim() !== "[]";
}

function classifyTeacherFeedbackRichness(params: {
  followUpText: unknown;
  toolResults: unknown;
}): TeacherFeedbackRichness {
  const hasFollowUp = toOptionalString(params.followUpText) !== null;
  const hasToolResults = hasNonEmptyToolResults(params.toolResults);
  if (hasFollowUp && hasToolResults) {
    return "followup_and_tool";
  }
  if (hasFollowUp) {
    return "followup_only";
  }
  if (hasToolResults) {
    return "tool_only";
  }
  return "sparse";
}

function toObservationBindingMode(value: unknown): BrainObservationBindingMode | null {
  switch (value) {
    case "exact_decision_id":
    case "exact_selection_digest":
    case "turn_compile_event_id":
    case "trace_id":
    case "legacy_heuristic":
    case "unbound":
      return value;
    default:
      return null;
  }
}

function readTeacherEvaluationRecord(metadata: Record<string, unknown> | null | undefined): Record<string, unknown> | null {
  return toRecord(metadata?.teacherEvaluation);
}

function readTeacherInputRecord(metadata: Record<string, unknown> | null | undefined): Record<string, unknown> | null {
  return toRecord(metadata?.teacherInput);
}

function readObservationId(metadata: Record<string, unknown> | null | undefined): string | null {
  return toOptionalString(metadata?.observationId)
    ?? toOptionalString(readTeacherEvaluationRecord(metadata)?.observationId);
}

function readTeacherTraceId(metadata: Record<string, unknown> | null | undefined): string | null {
  const teacherEvaluation = readTeacherEvaluationRecord(metadata);
  const traceId = toOptionalString(teacherEvaluation?.traceId);
  if (traceId) {
    return traceId;
  }

  const teacherLabel = toRecord(metadata?.teacherLabel);
  return toOptionalString(teacherLabel?.traceId) ?? toOptionalString(metadata?.traceId);
}

function readSupervisionProvenanceField(
  metadata: Record<string, unknown> | null | undefined,
  key: string,
): string | null {
  return toOptionalString(metadata?.[key]) ?? toOptionalString(readTeacherEvaluationRecord(metadata)?.[key]);
}

function readSupervisionBindingMode(
  metadata: Record<string, unknown> | null | undefined,
): BrainObservationBindingMode | null {
  return toObservationBindingMode(metadata?.bindingMode)
    ?? toObservationBindingMode(readTeacherEvaluationRecord(metadata)?.bindingMode);
}

function readSupervisionAttributionQuality(
  metadata: Record<string, unknown> | null | undefined,
): ObservationBindingQuality | null {
  const explicit = toOptionalString(metadata?.attributionQuality);
  if (explicit === "exact" || explicit === "fallback" || explicit === "unbound") {
    return explicit;
  }
  const bindingMode = readSupervisionBindingMode(metadata);
  return bindingMode ? classifyObservationBindingQuality(bindingMode) : null;
}

function readSupervisionFeedbackRichness(
  metadata: Record<string, unknown> | null | undefined,
): TeacherFeedbackRichness | null {
  const explicit = toOptionalString(metadata?.feedbackRichness);
  if (
    explicit === "followup_and_tool"
    || explicit === "followup_only"
    || explicit === "tool_only"
    || explicit === "sparse"
  ) {
    return explicit;
  }
  const teacherInput = readTeacherInputRecord(metadata);
  if (!teacherInput) {
    return null;
  }
  return classifyTeacherFeedbackRichness({
    followUpText: teacherInput.nextUserTurn,
    toolResults: teacherInput.toolResults,
  });
}

function combineAttributionQualities(
  supervision: PolicyGradientSupervisionArtifact[],
): ObservationBindingQuality | "mixed" | null {
  const values = new Set(
    supervision
      .map((entry) => entry.attributionQuality)
      .filter((entry): entry is ObservationBindingQuality => entry !== null),
  );
  if (values.size === 0) {
    return null;
  }
  if (values.size === 1) {
    return [...values][0] ?? null;
  }
  return "mixed";
}

function combineFeedbackRichness(
  supervision: PolicyGradientSupervisionArtifact[],
): TeacherFeedbackRichness | "mixed" | null {
  const values = new Set(
    supervision
      .map((entry) => entry.feedbackRichness)
      .filter((entry): entry is TeacherFeedbackRichness => entry !== null),
  );
  if (values.size === 0) {
    return null;
  }
  if (values.size === 1) {
    return [...values][0] ?? null;
  }
  return "mixed";
}

function describeTeacherReadyReason(feedbackRichness: TeacherFeedbackRichness): string {
  switch (feedbackRichness) {
    case "tool_only":
      return "tool results made the observation ready without waiting for follow-up";
    case "followup_only":
      return "operator follow-up made the observation ready";
    case "followup_and_tool":
      return "follow-up and tool results made the observation ready";
    case "sparse":
      return "teacher delay elapsed, so sparse feedback became eligible";
  }
}

function buildEpisodeUpdateReason(params: {
  episode: Episode;
  supervision: PolicyGradientSupervisionArtifact[];
  routeUpdateCount: number;
}): string {
  const primary =
    params.supervision.find((entry) => entry.source === params.episode.rewardSource)
    ?? params.supervision[0]
    ?? null;
  if (!primary) {
    return `episode reward ${params.episode.reward?.toFixed(2) ?? "n/a"} updated ${params.routeUpdateCount} route weight(s)`;
  }

  const score = Number.isFinite(primary.value) ? primary.value.toFixed(2) : "n/a";
  const confidence = Number.isFinite(primary.confidence) ? primary.confidence.toFixed(2) : "n/a";
  const binding =
    primary.attributionQuality === null
      ? "without observation attribution"
      : primary.attributionQuality === "exact"
        ? `${primary.bindingMode ?? "exact"} attribution`
        : `${primary.bindingMode ?? "unknown"} ${primary.attributionQuality} attribution`;
  const feedback =
    primary.feedbackRichness === null
      ? "no feedback richness detail"
      : primary.feedbackRichness.replaceAll("_", " ");
  return `${primary.source} ${primary.kind} ${score} (confidence ${confidence}, ${binding}, ${feedback}) updated ${params.routeUpdateCount} route weight(s)`;
}

function readPolicyWeightBeforeUpdate(
  graph: BrainGraph,
  update: Parameters<typeof policyWeightUpdateKey>[0],
): number | null {
  switch (update.kind) {
    case "seed":
      return graph.getSeedWeight(update.nodeId);
    case "stop_local":
      return graph.getStopLocalWeight(update.sourceNodeId);
    case "edge": {
      const edge = graph.getEdge(update.source, update.target);
      return edge ? edge.weight : null;
    }
  }
}

function buildRouteUpdateArtifact(
  update: Parameters<typeof policyWeightUpdateKey>[0],
  previousWeight: number | null,
  nextWeight: number,
  contributions: PolicyGradientRouteUpdateContribution[],
): PolicyGradientRouteUpdateArtifact {
  const firstContribution = contributions[0] ?? null;
  return {
    updateKey: policyWeightUpdateKey(update),
    kind: update.kind,
    sourceNodeId: firstContribution?.sourceNodeId
      ?? (update.kind === "seed" ? START_NODE_ID : update.kind === "stop_local" ? update.sourceNodeId : update.source),
    targetNodeId: firstContribution?.targetNodeId
      ?? (update.kind === "stop_local" ? null : update.kind === "seed" ? update.nodeId : update.target),
    delta: update.delta,
    previousWeight,
    nextWeight,
    contributionCount: contributions.length,
    contributions,
  };
}

function incrementRewardSourceCount(
  counts: Record<RewardSource, number>,
  source: RewardSource | null,
): void {
  if (!source) {
    return;
  }
  counts[source] = (counts[source] ?? 0) + 1;
}

export class BrainWorker {
  private interval: ReturnType<typeof setInterval> | null = null;
  private running = false;

  constructor(
    private store: BrainStore,
    private graph: BrainGraph,
    private teacher: BrainTeacher | null,
    private mutator: BrainMutator,
    private packManager: PackManager,
    private config: BrainConfig,
    private log: { info: (msg: string) => void; error: (msg: string) => void; warn: (msg: string) => void },
    private hooks: {
      isEnabled?: () => boolean;
      onPromotionReady?: (params: { healthJson: string; promotionVerdict: PromotionRunVerdict }) => Promise<void> | void;
      onTickResult?: (params: { ok: boolean; at: number; error?: string }) => void;
    } = {},
  ) {}

  start(): void {
    if (this.interval || !this.config.enabled || this.hooks.isEnabled?.() === false) {
      return;
    }

    this.interval = setInterval(() => {
      void this.tick()
        .then(() => {
          this.hooks.onTickResult?.({ ok: true, at: Date.now() });
        })
        .catch((error) => {
          const message = (error as Error).message;
          this.log.error(`[brain] Worker tick failed: ${message}`);
          this.hooks.onTickResult?.({ ok: false, at: Date.now(), error: message });
        });
    }, this.config.trainerIntervalMs);
    this.log.info(`[brain] Worker started (interval=${this.config.trainerIntervalMs}ms)`);
  }

  stop(): void {
    if (!this.interval) {
      return;
    }
    clearInterval(this.interval);
    this.interval = null;
    this.log.info("[brain] Worker stopped");
  }

  async tick(): Promise<void> {
    if (this.running) {
      return;
    }
    if (this.hooks.isEnabled?.() === false) {
      return;
    }

    this.running = true;
    try {
      this.store.setTrainingState("worker_last_tick_at", Date.now());
      await this.evaluatePendingObservations();
      await this.evaluatePendingShadowUsefulness();
      await this.processLabels();
      await this.applyUpdates();
      this.runDecay();
      this.proposeMutations();
      await this.checkPromotion();
    } finally {
      this.running = false;
    }
  }

  private resolveEvidenceWithTrace(params: {
    episode: Episode | null;
    evidence: BrainEvidence;
    resolution: BrainEvidenceResolution;
    labelId?: string | null;
    note?: string | null;
  }): void {
    this.store.resolveEvidence({
      evidenceId: params.evidence.id,
      episodeId: params.episode?.id ?? params.evidence.episodeId,
      source: params.evidence.source,
      value: params.evidence.value,
      confidence: params.evidence.confidence,
      resolution: params.resolution,
      labelId: params.labelId ?? null,
      note: params.note ?? null,
    });

    if (!params.episode) {
      return;
    }

    const metadataTraceId = typeof params.evidence.metadata?.traceId === "string" && params.evidence.metadata.traceId.length > 0
      ? params.evidence.metadata.traceId
      : null;
    const matchedTrace = metadataTraceId
      ? this.store.getTrace(metadataTraceId) ?? this.store.getTraceForEpisode(params.episode.id)
      : this.store.getTraceForEpisode(params.episode.id);
    if (!matchedTrace) {
      return;
    }

    this.store.insertTraceSupervision({
      traceId: matchedTrace.id,
      episodeId: params.episode.id,
      conversationId: params.episode.conversationId,
      source: params.evidence.source,
      kind: params.evidence.kind,
      value: params.evidence.value,
      confidence: params.evidence.confidence,
      reason: params.evidence.reason,
      contentSnippet: params.evidence.contentSnippet,
      resolution: params.resolution,
      labelId: params.labelId ?? null,
      evidenceId: params.evidence.id,
      metadata: {
        ...params.evidence.metadata,
        resolvedEpisodeId: params.episode.id,
        resolvedTraceId: matchedTrace.id,
        traceRequestDigest: matchedTrace.routeTrace?.requestDigest ?? params.evidence.metadata?.traceRequestDigest ?? null,
        tracePackVersion: matchedTrace.packVersion ?? params.evidence.metadata?.tracePackVersion ?? null,
        traceSelectedNodeIds: matchedTrace.routeTrace?.selectedNodeIds ?? matchedTrace.firedNodes,
        traceSelectedPathNodeIds: matchedTrace.routeTrace?.selectedPathNodeIds ?? params.evidence.metadata?.traceSelectedPathNodeIds ?? [],
        traceCandidateCount: matchedTrace.routeTrace?.candidateNodeIds.length ?? params.evidence.metadata?.traceCandidateCount ?? 0,
        note: params.note ?? null,
      },
    });
  }

  private observationReadyBefore(): number {
    return Date.now() - Math.max(1_000, this.config.trainerIntervalMs);
  }

  private async evaluatePendingObservations(): Promise<void> {
    if (!this.teacher || !this.config.teacherEnabled) {
      return;
    }

    const readyBefore = this.observationReadyBefore();
    const queue = this.store.getTeacherQueueSummary(readyBefore, TEACHER_OBSERVATION_BUDGET);
    const pending = this.store.getPendingObservations(TEACHER_OBSERVATION_BUDGET, readyBefore);
    const decisions: Array<{
      observationId: string;
      episodeId: string;
      traceId: string | null;
      decision: "delayed" | "budget_deferred" | "skipped" | "evaluated";
      reason: string;
      feedbackRichness: TeacherFeedbackRichness;
      bindingMode: BrainObservationBindingMode | null;
      attributionQuality: ObservationBindingQuality | null;
      finalScore: number | null;
      confidence: number | null;
      createdAt: number;
      evaluatedAt: number | null;
    }> = queue.sample
      .filter((entry) => entry.gate !== "ready")
      .map((entry) => ({
        observationId: entry.observationId,
        episodeId: entry.episodeId,
        traceId: entry.traceId,
        decision: entry.gate as "delayed" | "budget_deferred",
        reason: entry.reason,
        feedbackRichness: entry.feedbackRichness,
        bindingMode: null,
        attributionQuality: null,
        finalScore: null,
        confidence: null,
        createdAt: entry.createdAt,
        evaluatedAt: null,
      }));
    let evaluatedCount = 0;
    let skippedCount = 0;
    let exactAttributionCount = 0;
    let fallbackAttributionCount = 0;
    let unboundAttributionCount = 0;

    for (const observation of pending) {
      const feedbackRichness = classifyTeacherFeedbackRichness({
        followUpText: observation.followUpText,
        toolResults: observation.toolResults,
      });
      const episode = this.store.getEpisode(observation.episodeId);
      if (!episode) {
        this.store.completeObservationEvaluation({
          observationId: observation.id,
          phase1Score: 0,
          phase2Score: 0,
          finalScore: 0,
          confidence: 0,
          reason: "episode missing",
        });
        skippedCount += 1;
        decisions.push({
          observationId: observation.id,
          episodeId: observation.episodeId,
          traceId: observation.traceId,
          decision: "skipped",
          reason: "episode missing",
          feedbackRichness,
          bindingMode: null,
          attributionQuality: null,
          finalScore: 0,
          confidence: 0,
          createdAt: observation.createdAt,
          evaluatedAt: Date.now(),
        });
        continue;
      }

      const review = await this.teacher.evaluateObservation(observation);
      if (!review) {
        this.store.completeObservationEvaluation({
          observationId: observation.id,
          phase1Score: 0,
          phase2Score: 0,
          finalScore: 0,
          confidence: 0,
          reason: "teacher input unavailable",
        });
        skippedCount += 1;
        decisions.push({
          observationId: observation.id,
          episodeId: observation.episodeId,
          traceId: observation.traceId,
          decision: "skipped",
          reason: "teacher input unavailable",
          feedbackRichness,
          bindingMode: null,
          attributionQuality: null,
          finalScore: 0,
          confidence: 0,
          createdAt: observation.createdAt,
          evaluatedAt: Date.now(),
        });
        continue;
      }
      const attributionQuality = classifyObservationBindingQuality(review.bindingMode);
      if (attributionQuality === "exact") {
        exactAttributionCount += 1;
      } else if (attributionQuality === "fallback") {
        fallbackAttributionCount += 1;
      } else {
        unboundAttributionCount += 1;
      }
      const teacherEvaluation = {
        version: review.version,
        observationId: review.observationId,
        episodeId: review.episodeId,
        traceId: review.traceId,
        serveDecisionRecordId: review.serveDecisionRecordId,
        selectionDigest: review.selectionDigest,
        turnCompileEventId: review.turnCompileEventId,
        decisionRecordedAt: review.decisionRecordedAt,
        activePackId: review.activePackId,
        activePackEventExportDigest: review.activePackEventExportDigest,
        activePackGraphChecksum: review.activePackGraphChecksum,
        activePackRouterChecksum: review.activePackRouterChecksum,
        activePackBuiltAt: review.activePackBuiltAt,
        bindingMode: review.bindingMode,
        retrievalRelevance: review.retrievalRelevance,
        agentUsage: review.agentUsage,
        outcomeSupport: review.outcomeSupport,
        finalScore: review.finalScore,
        confidence: review.confidence,
        reason: review.reason,
      };

      const evidence = this.store.insertEvidence({
        episodeId: episode.id,
        conversationId: episode.conversationId,
        source: "teacher",
        kind: "teacher_review",
        value: review.finalScore,
        confidence: review.confidence,
        reason: review.reason,
        contentSnippet: (observation.assistantResponse || episode.queryText).slice(0, 240),
        metadata: {
          observationId: observation.id,
          traceId: review.traceId,
          serveDecisionRecordId: review.serveDecisionRecordId,
          selectionDigest: review.selectionDigest,
          turnCompileEventId: review.turnCompileEventId,
          activePackGraphChecksum: review.activePackGraphChecksum,
          bindingMode: review.bindingMode,
          attributionQuality,
          feedbackRichness,
          phase1Score: review.retrievalRelevance,
          phase2Score: review.outcomeSupport,
          agentUsage: review.agentUsage,
          teacherEvaluation,
          teacherInput: review.input,
        },
      });
      const label = this.store.insertLabel({
        episodeId: episode.id,
        source: "teacher",
        value: review.finalScore,
        confidence: review.confidence,
        reason: review.reason,
      });
      this.resolveEvidenceWithTrace({
        episode,
        evidence,
        resolution: "promoted_to_label",
        labelId: label.id,
        note: "teacher_observation_v2",
      });
      this.store.completeObservationEvaluation({
        observationId: observation.id,
        phase1Score: review.retrievalRelevance,
        phase2Score: review.outcomeSupport,
        finalScore: review.finalScore,
        confidence: review.confidence,
        reason: review.reason,
        teacherEvaluation,
      });
      evaluatedCount += 1;
      decisions.push({
        observationId: observation.id,
        episodeId: observation.episodeId,
        traceId: observation.traceId,
        decision: "evaluated",
        reason: `${review.reason}; ${describeTeacherReadyReason(feedbackRichness)}`,
        feedbackRichness,
        bindingMode: review.bindingMode,
        attributionQuality,
        finalScore: review.finalScore,
        confidence: review.confidence,
        createdAt: observation.createdAt,
        evaluatedAt: Date.now(),
      });
    }

    this.store.setTrainingStateJson("last_teacher_evaluation_cycle_json", {
      version: 1,
      generatedAt: Date.now(),
      budgetPerTick: TEACHER_OBSERVATION_BUDGET,
      delayMs: Math.max(0, Date.now() - readyBefore),
      pendingCount: queue.pendingCount,
      readyCount: queue.readyCount,
      delayedCount: queue.delayedCount,
      budgetDeferredCount: queue.budgetDeferredCount,
      sparseReadyCount: queue.sparseReadyCount,
      richReadyCount: queue.richReadyCount,
      evaluatedCount,
      skippedCount,
      exactAttributionCount,
      fallbackAttributionCount,
      unboundAttributionCount,
      decisions: decisions.slice(0, TEACHER_CYCLE_DECISION_LIMIT),
      detail:
        queue.pendingCount === 0
          ? "no teacher observations were pending this cycle"
          : `${evaluatedCount} evaluated, ${skippedCount} skipped; ${queue.readyCount} ready, ${queue.delayedCount} delayed, ${queue.budgetDeferredCount} deferred by teacher budget`,
    });
  }

  private async evaluatePendingShadowUsefulness(): Promise<void> {
    const pending = this.store.getPendingContextUsefulnessObservations(20);
    for (const observation of pending) {
      const evaluation = evaluateContextUsefulness(observation);
      this.store.insertContextUsefulnessEvaluation({
        observation,
        evaluation,
      });
    }
  }

  private async processLabels(): Promise<void> {
    const pending = this.store.getPendingLabels();
    for (const label of pending) {
      const episode = this.store.getEpisode(label.episodeId);
      if (!episode) {
        this.store.markLabelApplied(label.id);
        continue;
      }

      if (episode.reward !== null && episode.rewardSource !== null) {
        if (trustRank(episode.rewardSource) >= trustRank(label.source)) {
          this.store.markLabelApplied(label.id);
          continue;
        }
      }

      this.store.setEpisodeReward(episode.id, label.value, label.source);
      this.store.markLabelApplied(label.id);
    }
  }

  private collectPolicyGradientSupervision(episode: Episode): {
    traceIds: string[];
    observationIds: string[];
    supervisionIds: string[];
    teacherTraceIds: string[];
    supervision: PolicyGradientSupervisionArtifact[];
  } | null {
    const supervision = this.store
      .getTraceSupervisionForEpisode(episode.id, 50)
      .filter((record) => record.resolution === "promoted_to_label");
    if (supervision.length === 0) {
      return null;
    }

    const traceIds = new Set<string>();
    const observationIds = new Set<string>();
    const supervisionIds = new Set<string>();
    const teacherTraceIds = new Set<string>();
    const materialized: PolicyGradientSupervisionArtifact[] = [];

    for (const record of supervision) {
      traceIds.add(record.traceId);
      supervisionIds.add(record.id);
      const observationId = readObservationId(record.metadata);
      if (observationId) {
        observationIds.add(observationId);
      }
      const teacherTraceId = record.source === "teacher"
        ? (readTeacherTraceId(record.metadata) ?? record.traceId)
        : null;
      const bindingMode = readSupervisionBindingMode(record.metadata);
      if (record.source === "teacher") {
        teacherTraceIds.add(teacherTraceId ?? record.traceId);
      }
      materialized.push({
        supervisionId: record.id,
        traceId: record.traceId,
        source: record.source,
        kind: record.kind,
        value: record.value,
        confidence: record.confidence,
        reason: record.reason,
        labelId: record.labelId,
        evidenceId: record.evidenceId,
        observationId,
        teacherTraceId,
        serveDecisionRecordId: readSupervisionProvenanceField(record.metadata, "serveDecisionRecordId"),
        selectionDigest: readSupervisionProvenanceField(record.metadata, "selectionDigest"),
        turnCompileEventId: readSupervisionProvenanceField(record.metadata, "turnCompileEventId"),
        activePackGraphChecksum: readSupervisionProvenanceField(record.metadata, "activePackGraphChecksum"),
        bindingMode,
        attributionQuality: readSupervisionAttributionQuality(record.metadata),
        feedbackRichness: readSupervisionFeedbackRichness(record.metadata),
        traceRequestDigest: readSupervisionProvenanceField(record.metadata, "traceRequestDigest"),
        traceSelectedNodeIds: toStringArray(record.metadata?.traceSelectedNodeIds),
        traceSelectedPathNodeIds: toStringArray(record.metadata?.traceSelectedPathNodeIds),
      });
    }

    return {
      traceIds: [...traceIds].sort(),
      observationIds: [...observationIds].sort(),
      supervisionIds: [...supervisionIds].sort(),
      teacherTraceIds: [...teacherTraceIds].sort(),
      supervision: materialized,
    };
  }

  private materializePolicyGradientCandidateUpdate(params: {
    baselineBefore: number;
    baselineAfter: number;
    updatedEpisodes: Array<{
      episode: Episode;
      baselineBeforeEpisode: number;
      baselineAfterEpisode: number;
      routeUpdateCount: number;
      seedUpdateCount: number;
      stopLocalUpdateCount: number;
      edgeUpdateCount: number;
      routeUpdates: PolicyGradientRouteUpdateArtifact[];
    }>;
  }): void {
    const episodeIds = new Set<string>();
    const traceIds = new Set<string>();
    const observationIds = new Set<string>();
    const supervisionIds = new Set<string>();
    const teacherTraceIds = new Set<string>();
    const rewardSources: Record<RewardSource, number> = {
      human: 0,
      scanner: 0,
      teacher: 0,
      self: 0,
    };

    let routeUpdateCount = 0;
    let seedUpdateCount = 0;
    let stopLocalUpdateCount = 0;
    let edgeUpdateCount = 0;
    const episodeUpdates: PolicyGradientEpisodeUpdateArtifact[] = [];

    for (const entry of params.updatedEpisodes) {
      const consumed = this.collectPolicyGradientSupervision(entry.episode);
      if (!consumed) {
        continue;
      }

      episodeIds.add(entry.episode.id);
      incrementRewardSourceCount(rewardSources, entry.episode.rewardSource);
      routeUpdateCount += entry.routeUpdateCount;
      seedUpdateCount += entry.seedUpdateCount;
      stopLocalUpdateCount += entry.stopLocalUpdateCount;
      edgeUpdateCount += entry.edgeUpdateCount;

      for (const traceId of consumed.traceIds) {
        traceIds.add(traceId);
      }
      for (const observationId of consumed.observationIds) {
        observationIds.add(observationId);
      }
      for (const supervisionId of consumed.supervisionIds) {
        supervisionIds.add(supervisionId);
      }
      for (const teacherTraceId of consumed.teacherTraceIds) {
        teacherTraceIds.add(teacherTraceId);
      }
      const attributionQuality = combineAttributionQualities(consumed.supervision);
      const feedbackRichness = combineFeedbackRichness(consumed.supervision);
      if (entry.episode.reward !== null) {
        episodeUpdates.push({
          episodeId: entry.episode.id,
          observationIds: consumed.observationIds,
          traceIds: consumed.traceIds,
          supervisionIds: consumed.supervisionIds,
          reward: entry.episode.reward,
          rewardSource: entry.episode.rewardSource,
          attributionQuality,
          feedbackRichness,
          updateReason: buildEpisodeUpdateReason({
            episode: entry.episode,
            supervision: consumed.supervision,
            routeUpdateCount: entry.routeUpdateCount,
          }),
          baselineBefore: entry.baselineBeforeEpisode,
          baselineAfter: entry.baselineAfterEpisode,
          advantage: entry.episode.reward - entry.baselineBeforeEpisode,
          routeUpdateCount: entry.routeUpdateCount,
          seedUpdateCount: entry.seedUpdateCount,
          stopLocalUpdateCount: entry.stopLocalUpdateCount,
          edgeUpdateCount: entry.edgeUpdateCount,
          supervision: consumed.supervision,
          routeUpdates: entry.routeUpdates,
        });
      }
    }

    if (episodeIds.size === 0 || supervisionIds.size === 0 || routeUpdateCount === 0) {
      return;
    }

    const currentPackVersion = this.store.getCurrentPackVersion();
    const health = computeHealth(
      this.graph,
      this.store.getRecentEpisodes(this.config.replayEpisodeCount),
      currentPackVersion ?? 0,
    );
    const candidatePack = this.packManager.buildCandidate(health);
    const priorArtifact = this.store.getTrainingStateJson<PolicyGradientCandidateUpdateArtifact>(
      "last_pg_candidate_update_json",
    );
    const artifact: PolicyGradientCandidateUpdateArtifactV2 = {
      version: 2,
      updateCount: (priorArtifact?.updateCount ?? 0) + 1,
      candidatePackVersion: candidatePack.version,
      currentPackVersion,
      generatedAt: Date.now(),
      episodeIds: [...episodeIds].sort(),
      traceIds: [...traceIds].sort(),
      observationIds: [...observationIds].sort(),
      supervisionIds: [...supervisionIds].sort(),
      teacherTraceIds: [...teacherTraceIds].sort(),
      rewardSources,
      episodeCount: episodeIds.size,
      traceCount: traceIds.size,
      observationCount: observationIds.size,
      supervisionCount: supervisionIds.size,
      teacherLabelCount: teacherTraceIds.size,
      routeUpdateCount,
      seedUpdateCount,
      stopLocalUpdateCount,
      edgeUpdateCount,
      baselineBefore: params.baselineBefore,
      baselineAfter: params.baselineAfter,
      episodeUpdates,
    };

    this.store.setTrainingStateJson("last_pg_candidate_update_json", artifact);
    this.store.setTrainingState("last_pg_candidate_pack_version", candidatePack.version);
    this.store.setTrainingState("last_pg_candidate_update_at", artifact.generatedAt);

    try {
      this.store.writePackSnapshot({
        version: candidatePack.version,
        nodes: this.graph.getAllNodes(),
        edges: this.graph.getAllEdges(),
        seedWeights: this.graph.getAllSeedWeights().map((seedWeight) => ({
          nodeId: seedWeight.nodeId,
          weight: seedWeight.weight,
          updatedAt: artifact.generatedAt,
        })),
        stopLocalWeights: this.graph.getAllStopLocalWeights().map((stopLocalWeight) => ({
          sourceNodeId: stopLocalWeight.sourceNodeId,
          weight: stopLocalWeight.weight,
          updatedAt: artifact.generatedAt,
        })),
        metadata: {
          reason: "pg_update_candidate",
          pgCandidateUpdate: artifact,
        },
      });
    } catch (error) {
      this.log.warn(`[brain] Failed to write PG candidate snapshot for pack v${candidatePack.version}: ${String(error)}`);
    }
  }

  private async applyUpdates(): Promise<void> {
    const episodes = this.store.getEpisodesForUpdate(20);
    if (episodes.length === 0) {
      return;
    }

    const baselineStr = this.store.getTrainingState("baseline_reward");
    const baselineBefore = baselineStr ? Number.parseFloat(baselineStr) : 0;
    let baseline = baselineBefore;
    const updatedEpisodes: Array<{
      episode: Episode;
      baselineBeforeEpisode: number;
      baselineAfterEpisode: number;
      routeUpdateCount: number;
      seedUpdateCount: number;
      stopLocalUpdateCount: number;
      edgeUpdateCount: number;
      routeUpdates: PolicyGradientRouteUpdateArtifact[];
    }> = [];
    const decisions: Array<{
      episodeId: string;
      status: "applied" | "skipped";
      reason: string;
      reward: number | null;
      rewardSource: RewardSource | null;
      observationIds: string[];
      supervisionIds: string[];
      traceIds: string[];
      attributionQuality: ObservationBindingQuality | "mixed" | null;
      feedbackRichness: TeacherFeedbackRichness | "mixed" | null;
      routeUpdateCount: number;
      baselineBefore: number;
      baselineAfter: number;
    }> = [];
    const skippedReasons = {
      missing_reward: 0,
      missing_supervision: 0,
      zero_policy_delta: 0,
    };
    let appliedEpisodeCount = 0;
    let skippedEpisodeCount = 0;

    for (const episode of episodes) {
      if (episode.reward === null) {
        skippedEpisodeCount += 1;
        skippedReasons.missing_reward += 1;
        decisions.push({
          episodeId: episode.id,
          status: "skipped",
          reason: "episode has no reward yet",
          reward: null,
          rewardSource: episode.rewardSource,
          observationIds: [],
          supervisionIds: [],
          traceIds: [],
          attributionQuality: null,
          feedbackRichness: null,
          routeUpdateCount: 0,
          baselineBefore: baseline,
          baselineAfter: baseline,
        });
        continue;
      }

      const consumed = this.collectPolicyGradientSupervision(episode);
      if (!consumed) {
        skippedEpisodeCount += 1;
        skippedReasons.missing_supervision += 1;
        decisions.push({
          episodeId: episode.id,
          status: "skipped",
          reason: "reward is present, but no promoted supervision is bound to the route yet",
          reward: episode.reward,
          rewardSource: episode.rewardSource,
          observationIds: [],
          supervisionIds: [],
          traceIds: [],
          attributionQuality: null,
          feedbackRichness: null,
          routeUpdateCount: 0,
          baselineBefore: baseline,
          baselineAfter: baseline,
        });
        continue;
      }

      const baselineBeforeEpisode = baseline;
      const contributions = collectReinforceUpdateContributions(
        episode,
        this.config.learningRate,
        baselineBeforeEpisode,
      );
      const updates = computeReinforceUpdates(episode, this.config.learningRate, baselineBeforeEpisode);
      const attributionQuality = combineAttributionQualities(consumed.supervision);
      const feedbackRichness = combineFeedbackRichness(consumed.supervision);
      if (updates.length === 0) {
        const baselineAfterEpisode = updateBaseline(baseline, episode.reward, this.config.baselineAlpha);
        this.store.markEpisodeUpdated(episode.id);
        baseline = baselineAfterEpisode;
        skippedEpisodeCount += 1;
        skippedReasons.zero_policy_delta += 1;
        decisions.push({
          episodeId: episode.id,
          status: "skipped",
          reason: contributions.length === 0
            ? "reward matched the running baseline, so no policy delta was emitted"
            : "no policy updates were materialized from the routed trajectory",
          reward: episode.reward,
          rewardSource: episode.rewardSource,
          observationIds: consumed.observationIds,
          supervisionIds: consumed.supervisionIds,
          traceIds: consumed.traceIds,
          attributionQuality,
          feedbackRichness,
          routeUpdateCount: 0,
          baselineBefore: baselineBeforeEpisode,
          baselineAfter: baselineAfterEpisode,
        });
        continue;
      }
      const contributionsByKey = new Map<string, PolicyGradientRouteUpdateContribution[]>();
      for (const contribution of contributions) {
        const existing = contributionsByKey.get(contribution.updateKey);
        if (existing) {
          existing.push(contribution);
        } else {
          contributionsByKey.set(contribution.updateKey, [contribution]);
        }
      }
      const previousWeights = new Map<string, number | null>();
      for (const update of updates) {
        previousWeights.set(policyWeightUpdateKey(update), readPolicyWeightBeforeUpdate(this.graph, update));
      }
      applyWeightUpdates(this.graph, updates);

      let seedUpdateCount = 0;
      let stopLocalUpdateCount = 0;
      let edgeUpdateCount = 0;
      const routeUpdates: PolicyGradientRouteUpdateArtifact[] = [];
      for (const update of updates) {
        const updateKey = policyWeightUpdateKey(update);
        const previousWeight = previousWeights.get(updateKey) ?? null;
        const updateContributions = contributionsByKey.get(updateKey) ?? [];
        if (update.kind === "seed") {
          seedUpdateCount += 1;
          const nextWeight = this.graph.getSeedWeight(update.nodeId);
          this.store.setSeedWeight(update.nodeId, nextWeight);
          routeUpdates.push(buildRouteUpdateArtifact(update, previousWeight, nextWeight, updateContributions));
          continue;
        }

        if (update.kind === "stop_local") {
          stopLocalUpdateCount += 1;
          const nextWeight = this.graph.getStopLocalWeight(update.sourceNodeId);
          this.store.setStopLocalWeight(update.sourceNodeId, nextWeight);
          routeUpdates.push(buildRouteUpdateArtifact(update, previousWeight, nextWeight, updateContributions));
          continue;
        }

        edgeUpdateCount += 1;
        const edge = this.graph.getEdge(update.source, update.target);
        if (edge) {
          this.store.updateEdgeWeight(edge.source, edge.target, edge.kind, edge.weight);
          routeUpdates.push(buildRouteUpdateArtifact(update, previousWeight, edge.weight, updateContributions));
          continue;
        }

        const now = Date.now();
        const createdEdge: BrainEdge = {
          source: update.source,
          target: update.target,
          kind: "learned",
          weight: Math.max(-10, Math.min(10, update.delta)),
          prior: 0.5,
          metadata: { createdBy: "reinforce" },
          decayedAt: now,
          createdAt: now,
        };
        this.graph.addEdge(createdEdge);
        this.store.insertEdge(createdEdge);
        routeUpdates.push(buildRouteUpdateArtifact(update, previousWeight, createdEdge.weight, updateContributions));
      }

      const baselineAfterEpisode = updateBaseline(baseline, episode.reward, this.config.baselineAlpha);
      updatedEpisodes.push({
        episode,
        baselineBeforeEpisode,
        baselineAfterEpisode,
        routeUpdateCount: updates.length,
        seedUpdateCount,
        stopLocalUpdateCount,
        edgeUpdateCount,
        routeUpdates,
      });
      appliedEpisodeCount += 1;
      decisions.push({
        episodeId: episode.id,
        status: "applied",
        reason: buildEpisodeUpdateReason({
          episode,
          supervision: consumed.supervision,
          routeUpdateCount: updates.length,
        }),
        reward: episode.reward,
        rewardSource: episode.rewardSource,
        observationIds: consumed.observationIds,
        supervisionIds: consumed.supervisionIds,
        traceIds: consumed.traceIds,
        attributionQuality,
        feedbackRichness,
        routeUpdateCount: updates.length,
        baselineBefore: baselineBeforeEpisode,
        baselineAfter: baselineAfterEpisode,
      });
      this.store.markEpisodeUpdated(episode.id);
      baseline = baselineAfterEpisode;
    }

    this.store.setTrainingState("baseline_reward", baseline);
    this.store.setTrainingState("last_update_at", Date.now());
    this.store.setTrainingStateJson("last_teacher_update_cycle_json", {
      version: 1,
      generatedAt: Date.now(),
      eligibleEpisodeCount: episodes.length,
      appliedEpisodeCount,
      skippedEpisodeCount,
      skippedReasons,
      decisions: decisions.slice(0, TEACHER_CYCLE_DECISION_LIMIT),
      detail: `${appliedEpisodeCount} episode(s) updated, ${skippedEpisodeCount} skipped; missing supervision=${skippedReasons.missing_supervision}, zero policy delta=${skippedReasons.zero_policy_delta}`,
    });
    this.materializePolicyGradientCandidateUpdate({
      baselineBefore,
      baselineAfter: baseline,
      updatedEpisodes,
    });
  }

  private runDecay(): void {
    const lastDecay = Number.parseInt(this.store.getTrainingState("last_decay_at") ?? "0", 10);
    if (Date.now() - lastDecay < 60_000) {
      return;
    }

    decayAllWeights(this.graph, this.config.decayRate, Date.now());
    this.store.decayAllWeights(this.config.decayRate);
    this.store.setTrainingState("last_decay_at", Date.now());
  }

  private proposeMutations(): void {
    if (!this.config.mutationsEnabled) {
      return;
    }

    const proposals = this.mutator.proposeMutations(this.store.getRecentEpisodes(50));
    for (const proposal of proposals) {
      this.store.insertMutation(proposal);
      this.recordMutationProposed(proposal);
    }
  }

  private persistPromotionVerdict(verdict: PromotionRunVerdict): void {
    this.store.setTrainingState("last_promotion_reason", verdict.summary);
    this.store.setTrainingState("last_replay_failure_reason", verdict.replayGate && !verdict.replayGate.passed
      ? verdict.replayGate.reason.summary
      : "");
    this.store.setTrainingStateJson("last_promotion_verdict_json", verdict);
    this.store.setTrainingStateJson("last_replay_gate_verdict_json", verdict.replayGate);
  }

  private countBundleMutations(
    bundleVerdicts: BundleEvaluationVerdict[],
    status: BundleEvaluationVerdict["status"],
  ): number {
    return bundleVerdicts.reduce((sum, verdict) => (
      verdict.status === status ? sum + verdict.mutationIds.length : sum
    ), 0);
  }

  private async checkPromotion(): Promise<void> {
    const recentEpisodes = this.store.getRecentEpisodes(this.config.replayEpisodeCount);
    const pendingMutations = this.store.getMutationsByStatus("pending", 20); // Get more for bundling

    // Filter to supported mutation kinds
    const candidateMutations = pendingMutations.filter((proposal) =>
      proposal.kind === "connect" || proposal.kind === "prune" || proposal.kind === "inject",
    );

    // Cluster mutations into bundles
    const bundleConfig: BundleEvaluationConfig = {
      ...DEFAULT_BUNDLE_CONFIG,
      minBundleSize: Math.min(3, candidateMutations.length),
    };
    const bundles = clusterMutationsIntoBundles(candidateMutations, bundleConfig);

    if (bundles.length === 0) {
      // Fall back to old behavior if no bundles
      await this.checkPromotionLegacy(recentEpisodes, candidateMutations);
      return;
    }

    // Evaluate each bundle
    const bundleVerdicts: BundleEvaluationVerdict[] = [];
    for (const bundle of bundles) {
      this.store.insertMutationBundle({
        id: bundle.id,
        mutationIds: bundle.mutationIds,
        bundleSize: bundle.bundleSize,
        status: "evaluating",
        expectedGain: bundle.expectedGain,
        createdAt: bundle.createdAt,
      });
      this.recordBundleEvaluationStarted(bundle, recentEpisodes, candidateMutations.length, bundleConfig);

      const evalResult = await evaluateBundle(bundle, this.graph, recentEpisodes, bundleConfig);
      const qualifyingEpisodeIds = recentEpisodes
        .filter((episode) => episode.reward !== null && episode.reward >= bundleConfig.minRewardThreshold)
        .map((episode) => episode.id);

      bundle.status = evalResult.verdict.status;
      bundle.baseScore = evalResult.baseScore;
      bundle.candidateScore = evalResult.candidateScore;
      bundle.rejectionReason = evalResult.rejectionReason;
      bundle.resolvedAt = evalResult.verdict.resolvedAt;
      bundleVerdicts.push(evalResult.verdict);
      this.recordBundleEvaluationCompleted(bundle, qualifyingEpisodeIds, evalResult.shouldPromote, evalResult.rejectionReason);

      if (evalResult.shouldPromote) {
        for (const proposal of bundle.proposals) {
          this.mutator.applyMutation(proposal);
        }
        this.recordPromotionDecision("promotion_accepted", {
          gate: "bundle_evaluation",
          bundle,
          reason: evalResult.verdict.reason.summary,
          metadata: {
            verdict: evalResult.verdict,
          },
        });
        this.log.info(`[brain] Bundle ${bundle.id} promoted (${bundle.bundleSize} mutations, score ${evalResult.candidateScore.toFixed(3)} vs ${evalResult.baseScore.toFixed(3)})`);
      } else {
        for (const proposal of bundle.proposals) {
          this.store.resolveMutation(proposal.id, "rejected");
        }
        this.recordPromotionDecision("promotion_rejected", {
          gate: "bundle_evaluation",
          bundle,
          reason: evalResult.verdict.reason.summary,
          metadata: {
            verdict: evalResult.verdict,
          },
        });
        this.log.info(`[brain] Bundle ${bundle.id} rejected (${evalResult.verdict.reason.code}): ${evalResult.verdict.reason.summary}`);
      }

      this.store.resolveMutationBundle({
        id: bundle.id,
        status: bundle.status,
        baseScore: evalResult.baseScore,
        candidateScore: evalResult.candidateScore,
        rejectionReason: evalResult.rejectionReason,
        verdict: evalResult.verdict,
        resolvedAt: evalResult.verdict.resolvedAt,
      });
    }

    // Report health after bundle evaluation
    const health = computeHealth(this.graph, recentEpisodes, this.store.getCurrentPackVersion() ?? 0);
    const promotedBundleCount = bundleVerdicts.filter((verdict) => verdict.status === "promoted").length;
    const rejectedBundleCount = bundleVerdicts.length - promotedBundleCount;
    const verdict: PromotionRunVerdict = {
      mode: "bundle",
      status: "promoted",
      summary: promotedBundleCount > 0
        ? `candidate graph promoted after evaluating ${bundleVerdicts.length} bundle(s); ${promotedBundleCount} promoted, ${rejectedBundleCount} rejected`
        : `candidate graph promoted after bundle evaluation; no mutation bundles were accepted`,
      mutationCount: candidateMutations.length,
      promotedMutationCount: this.countBundleMutations(bundleVerdicts, "promoted"),
      rejectedMutationCount: this.countBundleMutations(bundleVerdicts, "rejected"),
      bundleCount: bundleVerdicts.length,
      promotedBundleCount,
      rejectedBundleCount,
      packPromotionTriggered: true,
      health,
      replayGate: null,
      bundleVerdicts,
      createdAt: Date.now(),
    };
    this.persistPromotionVerdict(verdict);
    await this.hooks.onPromotionReady?.({
      healthJson: JSON.stringify(health),
      promotionVerdict: verdict,
    });
  }

  /**
   * Legacy single-mutation promotion (fallback when bundling not applicable)
   */
  private async checkPromotionLegacy(recentEpisodes: Episode[], pendingMutations: MutationProposal[]): Promise<void> {
    const candidateGraph = this.graph.clone();
    for (const proposal of pendingMutations) {
      this.mutator.applyToCandidateGraph(candidateGraph, proposal);
    }

    const gate = this.packManager.replayGate(recentEpisodes, {
      minFiredPerQuery: this.config.minFiredPerQuery,
      maxDormantPercent: this.config.maxDormantPercent,
      maxOrphanCount: this.config.maxOrphanCount,
    }, candidateGraph);

    if (!gate.passed) {
      const verdict: PromotionRunVerdict = {
        mode: "legacy",
        status: "rejected",
        summary: `replay gate blocked promotion: ${gate.reason.summary}`,
        mutationCount: pendingMutations.length,
        promotedMutationCount: 0,
        rejectedMutationCount: pendingMutations.length,
        bundleCount: 0,
        promotedBundleCount: 0,
        rejectedBundleCount: 0,
        packPromotionTriggered: false,
        health: gate.health,
        replayGate: gate,
        bundleVerdicts: [],
        createdAt: Date.now(),
      };
      this.persistPromotionVerdict(verdict);
      this.log.warn(`[brain] Replay gate blocked promotion (${gate.reason.code}): ${gate.reason.summary}`);
      for (const proposal of pendingMutations) {
        this.store.resolveMutation(proposal.id, "rejected");
      }
      if (pendingMutations.length > 0) {
        this.recordLegacyPromotionDecision("promotion_rejected", pendingMutations, gate.reason.summary, gate.health);
      }
      return;
    }

    for (const proposal of pendingMutations) {
      this.mutator.applyMutation(proposal);
    }
    const health = computeHealth(this.graph, recentEpisodes, this.store.getCurrentPackVersion() ?? 0);
    const verdict: PromotionRunVerdict = {
      mode: "legacy",
      status: "promoted",
      summary: pendingMutations.length > 0
        ? `candidate graph promoted with ${pendingMutations.length} mutation(s)`
        : "weights and decay passed replay gate",
      mutationCount: pendingMutations.length,
      promotedMutationCount: pendingMutations.length,
      rejectedMutationCount: 0,
      bundleCount: 0,
      promotedBundleCount: 0,
      rejectedBundleCount: 0,
      packPromotionTriggered: true,
      health,
      replayGate: gate,
      bundleVerdicts: [],
      createdAt: Date.now(),
    };
    this.persistPromotionVerdict(verdict);
    if (pendingMutations.length > 0) {
      this.recordLegacyPromotionDecision("promotion_accepted", pendingMutations, gate.reason.summary, health);
    }
    await this.hooks.onPromotionReady?.({
      healthJson: JSON.stringify(health),
      promotionVerdict: verdict,
    });
  }

  private recordMutationProposed(proposal: MutationProposal): void {
    this.store.appendLearningJournal({
      eventType: "mutation_proposed",
      mutationId: proposal.id,
      mutationIds: [proposal.id],
      packVersion: this.store.getCurrentPackVersion(),
      payload: {
        mutationKind: proposal.kind,
        expectedGain: proposal.expectedGain,
        proposal: proposal.proposal,
        evidence: proposal.evidence,
      },
    });
  }

  private recordBundleEvaluationStarted(
    bundle: MutationBundle,
    recentEpisodes: Episode[],
    candidateMutationCount: number,
    config: BundleEvaluationConfig,
  ): void {
    this.store.appendLearningJournal({
      eventType: "bundle_evaluation_started",
      bundleId: bundle.id,
      mutationIds: bundle.mutationIds,
      packVersion: this.store.getCurrentPackVersion(),
      payload: {
        mutationKinds: bundle.proposals.map((proposal) => proposal.kind),
        bundleSize: bundle.bundleSize,
        expectedGain: bundle.expectedGain,
        candidateMutationCount,
        recentEpisodeIds: recentEpisodes.map((episode) => episode.id),
        config: {
          minBundleSize: config.minBundleSize,
          maxBundleSize: config.maxBundleSize,
          minRewardThreshold: config.minRewardThreshold,
          maxContextInflation: config.maxContextInflation,
          minImprovementRatio: config.minImprovementRatio,
        },
      },
    });
  }

  private recordBundleEvaluationCompleted(
    bundle: MutationBundle,
    qualifyingEpisodeIds: string[],
    shouldPromote: boolean,
    rejectionReason: string | null,
  ): void {
    this.store.appendLearningJournal({
      eventType: "bundle_evaluation_completed",
      bundleId: bundle.id,
      mutationIds: bundle.mutationIds,
      packVersion: this.store.getCurrentPackVersion(),
      payload: {
        mutationKinds: bundle.proposals.map((proposal) => proposal.kind),
        bundleSize: bundle.bundleSize,
        expectedGain: bundle.expectedGain,
        qualifyingEpisodeIds,
        baseScore: bundle.baseScore ?? 0,
        candidateScore: bundle.candidateScore ?? 0,
        shouldPromote,
        rejectionReason,
      },
    });
  }

  private recordPromotionDecision(
    eventType: "promotion_accepted" | "promotion_rejected",
    params: {
      gate: "bundle_evaluation";
      bundle: MutationBundle;
      reason: string | null;
      metadata: Record<string, unknown>;
    },
  ): void {
    this.store.appendLearningJournal({
      eventType,
      bundleId: params.bundle.id,
      mutationIds: params.bundle.mutationIds,
      packVersion: this.store.getCurrentPackVersion(),
      payload: {
        gate: params.gate,
        mutationKinds: params.bundle.proposals.map((proposal) => proposal.kind),
        mutationCount: params.bundle.bundleSize,
        reason: params.reason,
        baseScore: params.bundle.baseScore,
        candidateScore: params.bundle.candidateScore,
        metadata: params.metadata,
      },
    });
  }

  private recordLegacyPromotionDecision(
    eventType: "promotion_accepted" | "promotion_rejected",
    pendingMutations: MutationProposal[],
    reason: string | null,
    health: ReturnType<PackManager["replayGate"]>["health"],
  ): void {
    this.store.appendLearningJournal({
      eventType,
      mutationIds: pendingMutations.map((proposal) => proposal.id),
      packVersion: this.store.getCurrentPackVersion(),
      payload: {
        gate: "replay_gate",
        mutationKinds: pendingMutations.map((proposal) => proposal.kind),
        mutationCount: pendingMutations.length,
        reason,
        baseScore: null,
        candidateScore: null,
        metadata: {
          health,
        },
      },
    });
  }
}
