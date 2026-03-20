import type {
  BrainConfig,
  BrainEdge,
  BrainEvidence,
  BrainEvidenceResolution,
  BundleEvaluationVerdict,
  Episode,
  MutationProposal,
  PromotionRunVerdict,
  ReplayGateVerdict,
} from "../brain-core/types.js";
import { trustRank } from "../brain-core/types.js";
import type { BrainStore } from "../brain-store/store.js";
import type { BrainGraph } from "../brain-core/graph.js";
import type { BrainTeacher } from "../brain-core/teacher.js";
import type { BrainMutator } from "../brain-core/mutator.js";
import type { PackManager } from "../brain-core/pack.js";
import { computeReinforceUpdates, updateBaseline, applyWeightUpdates } from "../brain-core/update.js";
import { decayAllWeights } from "../brain-core/decay.js";
import { computeHealth } from "../brain-core/health.js";
import { isTeacherEligibleTrace } from "../brain-core/teacher.js";
import { clusterMutationsIntoBundles, evaluateBundle, DEFAULT_BUNDLE_CONFIG } from "../brain-core/bundle-evaluator.js";
import type { BundleEvaluationConfig, MutationBundle } from "../brain-core/bundle-evaluator.js";

function readExtractor(evidence: BrainEvidence): string | null {
  const extractor = evidence.metadata?.extractor;
  return typeof extractor === "string" && extractor.length > 0 ? extractor : null;
}

function evidenceSpecificityRank(evidence: BrainEvidence): number {
  const extractor = readExtractor(evidence);
  if (evidence.source === "scanner") {
    switch (extractor) {
      case "structured_guidance_parts":
        return 3;
      case "structured_tool_chain":
        return 2;
      case "scanner_marker":
        return 1;
      case "scanner_heuristic":
      default:
        return 0;
    }
  }

  return 0;
}

function compareEvidencePriority(left: BrainEvidence, right: BrainEvidence): number {
  const trustDelta = trustRank(left.source) - trustRank(right.source);
  if (trustDelta !== 0) {
    return trustDelta;
  }

  if (left.value !== right.value) {
    const specificityDelta = evidenceSpecificityRank(left) - evidenceSpecificityRank(right);
    if (specificityDelta !== 0) {
      return specificityDelta;
    }
  }

  const confidenceDelta = left.confidence - right.confidence;
  if (confidenceDelta !== 0) {
    return confidenceDelta;
  }

  return left.createdAt - right.createdAt;
}

function losingEvidenceResolution(
  winner: BrainEvidence,
  loser: BrainEvidence,
): { resolution: "discarded_lower_trust" | "discarded_duplicate"; note: string } {
  const winnerTrust = trustRank(winner.source);
  const loserTrust = trustRank(loser.source);
  if (winnerTrust > loserTrust) {
    return {
      resolution: "discarded_lower_trust",
      note: `pending evidence from ${winner.source} outranks ${loser.source}`,
    };
  }

  if (winner.value === loser.value) {
    return {
      resolution: "discarded_duplicate",
      note: `matching ${loser.source} evidence already queued`,
    };
  }

  const winnerSpecificity = evidenceSpecificityRank(winner);
  const loserSpecificity = evidenceSpecificityRank(loser);
  if (winnerSpecificity !== loserSpecificity) {
    return {
      resolution: "discarded_duplicate",
      note: `same-trust evidence superseded by more-structured ${winner.source} evidence`,
    };
  }

  if (winner.confidence !== loser.confidence) {
    return {
      resolution: "discarded_duplicate",
      note: `same-trust evidence superseded by higher-confidence ${winner.source} evidence`,
    };
  }

  return {
    resolution: "discarded_duplicate",
    note: `same-trust evidence superseded by newer ${winner.source} evidence`,
  };
}

function classifyEvidenceAgainstEpisode(
  episode: Episode,
  evidence: BrainEvidence,
): { resolution: "discarded_lower_trust" | "discarded_duplicate"; note: string } | null {
  if (episode.reward === null || episode.rewardSource === null) {
    return null;
  }

  const existingTrust = trustRank(episode.rewardSource);
  const newTrust = trustRank(evidence.source);
  if (existingTrust > newTrust) {
    return {
      resolution: "discarded_lower_trust",
      note: `existing reward from ${episode.rewardSource} outranks ${evidence.source}`,
    };
  }

  if (existingTrust === newTrust) {
    if (episode.reward === evidence.value) {
      return {
        resolution: "discarded_duplicate",
        note: "matching reward already present",
      };
    }

    return {
      resolution: "discarded_duplicate",
      note: `existing ${episode.rewardSource} reward already present; equal-trust override is not applied automatically`,
    };
  }

  return null;
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
      await this.processEvidence();
      await this.processLabels();
      await this.runTeacher();
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

  private async processEvidence(): Promise<void> {
    const pending = this.store.getPendingEvidence(100);
    const candidatesByEpisode = new Map<string, { episode: Episode; evidence: BrainEvidence[] }>();

    for (const evidence of pending) {
      const episode = this.store.getEpisode(evidence.episodeId);
      if (!episode) {
        this.resolveEvidenceWithTrace({
          episode: null,
          evidence,
          resolution: "discarded_missing_episode",
          note: evidence.reason ?? "episode missing",
        });
        continue;
      }

      const episodeClassification = classifyEvidenceAgainstEpisode(episode, evidence);
      if (episodeClassification) {
        this.resolveEvidenceWithTrace({
          episode,
          evidence,
          resolution: episodeClassification.resolution,
          note: episodeClassification.note,
        });
        continue;
      }

      const staged = candidatesByEpisode.get(episode.id);
      if (staged) {
        staged.evidence.push(evidence);
      } else {
        candidatesByEpisode.set(episode.id, { episode, evidence: [evidence] });
      }
    }

    for (const { episode, evidence } of candidatesByEpisode.values()) {
      let winner: BrainEvidence | null = null;
      const losers: Array<{ evidence: BrainEvidence; resolution: "discarded_lower_trust" | "discarded_duplicate"; note: string }> = [];

      for (const candidate of evidence) {
        if (!winner) {
          winner = candidate;
          continue;
        }

        if (compareEvidencePriority(candidate, winner) > 0) {
          losers.push({
            evidence: winner,
            ...losingEvidenceResolution(candidate, winner),
          });
          winner = candidate;
          continue;
        }

        losers.push({
          evidence: candidate,
          ...losingEvidenceResolution(winner, candidate),
        });
      }

      for (const loser of losers) {
        this.resolveEvidenceWithTrace({
          episode,
          evidence: loser.evidence,
          resolution: loser.resolution,
          note: loser.note,
        });
      }

      if (!winner) {
        continue;
      }

      const label = this.store.insertLabel({
        episodeId: episode.id,
        source: winner.source,
        value: winner.value,
        confidence: winner.confidence,
        reason: winner.reason ?? undefined,
      });
      this.resolveEvidenceWithTrace({
        episode,
        evidence: winner,
        resolution: "promoted_to_label",
        labelId: label.id,
        note: winner.kind,
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

  private async runTeacher(): Promise<void> {
    if (!this.teacher || !this.config.teacherEnabled) {
      return;
    }

    const unlabeled = this.store.getUnlabeledEpisodes(3);
    for (const episode of unlabeled) {
      const trace = this.store.getLatestTraceForEpisode(episode.id);
      if (!trace || !isTeacherEligibleTrace(trace)) {
        continue;
      }

      const review = await this.teacher.evaluateTrace(trace);
      if (!review || Math.abs(review.score) <= 0.05) {
        continue;
      }

      this.store.insertEvidence({
        episodeId: episode.id,
        conversationId: episode.conversationId,
        source: "teacher",
        kind: "teacher_review",
        value: review.score,
        confidence: 0.6,
        reason: review.reason,
        contentSnippet: episode.queryText.slice(0, 240),
        metadata: {
          queryText: episode.queryText,
          teacherLabel: {
            version: review.version,
            traceId: review.traceId,
            episodeId: review.episodeId,
            requestDigest: review.requestDigest,
            input: review.input,
          },
        },
      });
    }
  }

  private async applyUpdates(): Promise<void> {
    const episodes = this.store.getEpisodesForUpdate(20);
    if (episodes.length === 0) {
      return;
    }

    const baselineStr = this.store.getTrainingState("baseline_reward");
    let baseline = baselineStr ? Number.parseFloat(baselineStr) : 0;

    for (const episode of episodes) {
      if (episode.reward === null) {
        continue;
      }

      const updates = computeReinforceUpdates(episode, this.config.learningRate, baseline);
      applyWeightUpdates(this.graph, updates);

      for (const update of updates) {
        if (update.kind === "seed") {
          this.store.setSeedWeight(update.nodeId, this.graph.getSeedWeight(update.nodeId));
          continue;
        }

        const edge = this.graph.getEdge(update.source, update.target);
        if (edge) {
          this.store.updateEdgeWeight(edge.source, edge.target, edge.kind, edge.weight);
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
      }

      this.store.markEpisodeUpdated(episode.id);
      baseline = updateBaseline(baseline, episode.reward, this.config.baselineAlpha);
    }

    this.store.setTrainingState("baseline_reward", baseline);
    this.store.setTrainingState("last_update_at", Date.now());
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
