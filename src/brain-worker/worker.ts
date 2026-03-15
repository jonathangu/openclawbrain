import type { BrainConfig } from "../brain-core/types.js";
import { trustRank } from "../brain-core/types.js";
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
      onPromotionReady?: (params: { healthJson: string }) => Promise<void> | void;
    } = {},
  ) {}

  start(): void {
    if (this.interval || !this.config.enabled) {
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

    this.running = true;
    try {
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
        }
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
    const health = computeHealth(this.graph, recentEpisodes, this.store.getCurrentPackVersion() ?? 0);
    const gate = this.packManager.replayGate(recentEpisodes, {
      minFiredPerQuery: this.config.minFiredPerQuery,
      maxDormantPercent: this.config.maxDormantPercent,
      maxOrphanCount: this.config.maxOrphanCount,
    });

    if (!gate.passed) {
      this.log.warn(`[brain] Replay gate blocked promotion: ${gate.reason}`);
      return;
    }

    await this.hooks.onPromotionReady?.({
      healthJson: JSON.stringify(health),
    });
  }
}
