import type { BrainConfig, BrainEdge } from "../brain-core/types.js";
import { START_NODE_ID, trustRank } from "../brain-core/types.js";
import type { BrainStore } from "../brain-store/store.js";
import type { BrainGraph } from "../brain-core/graph.js";
import type { BrainTeacher } from "../brain-core/teacher.js";
import type { BrainMutator } from "../brain-core/mutator.js";
import type { PackManager } from "../brain-core/pack.js";
import { computeReinforceUpdates, updateBaseline, applyWeightUpdates } from "../brain-core/update.js";
import { decayAllWeights } from "../brain-core/decay.js";
import { computeHealth } from "../brain-core/health.js";

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
      onPromotionReady?: (params: { healthJson: string }) => Promise<void> | void;
    } = {},
  ) {}

  start(): void {
    if (this.interval || !this.config.enabled || this.hooks.isEnabled?.() === false) {
      return;
    }

    this.interval = setInterval(() => {
      void this.tick().catch((error) => {
        this.log.error(`[brain] Worker tick failed: ${(error as Error).message}`);
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
      const { score, reason } = await this.teacher.evaluate(episode);
      if (Math.abs(score) > 0.05) {
        this.store.insertLabel({
          episodeId: episode.id,
          source: "teacher",
          value: score,
          reason,
        });
      }
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
        const edge = this.graph.getEdge(update.source, update.target);
        if (edge) {
          this.store.updateEdgeWeight(edge.source, edge.target, edge.kind, edge.weight);
          continue;
        }

        const now = Date.now();
        const createdEdge: BrainEdge = {
          source: update.source,
          target: update.target,
          kind: update.source === START_NODE_ID ? "seed" : "learned",
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
    }
  }

  private async checkPromotion(): Promise<void> {
    const recentEpisodes = this.store.getRecentEpisodes(this.config.replayEpisodeCount);
    const pendingMutations = this.store.getMutationsByStatus("pending", 5);
    const candidateGraph = this.graph.clone();
    const candidateMutations = pendingMutations.filter((proposal) =>
      proposal.kind === "connect" || proposal.kind === "prune" || proposal.kind === "inject",
    );
    for (const proposal of candidateMutations) {
      this.mutator.applyToCandidateGraph(candidateGraph, proposal);
    }

    const gate = this.packManager.replayGate(recentEpisodes, {
      minFiredPerQuery: this.config.minFiredPerQuery,
      maxDormantPercent: this.config.maxDormantPercent,
      maxOrphanCount: this.config.maxOrphanCount,
    }, candidateGraph);

    if (!gate.passed) {
      this.store.setTrainingState("last_replay_failure_reason", gate.reason);
      this.log.warn(`[brain] Replay gate blocked promotion: ${gate.reason}`);
      for (const proposal of candidateMutations) {
        this.store.resolveMutation(proposal.id, "rejected");
      }
      return;
    }

    for (const proposal of candidateMutations) {
      this.mutator.applyMutation(proposal);
    }
    const health = computeHealth(this.graph, recentEpisodes, this.store.getCurrentPackVersion() ?? 0);
    this.store.setTrainingState("last_promotion_reason", candidateMutations.length > 0
      ? `candidate graph promoted with ${candidateMutations.length} mutation(s)`
      : "weights and decay passed replay gate");
    this.store.setTrainingState("last_replay_failure_reason", "");
    await this.hooks.onPromotionReady?.({
      healthJson: JSON.stringify(health),
    });
  }
}
