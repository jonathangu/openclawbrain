/**
 * Background trainer: batch update loop.
 *
 * Runs on a timer, executing:
 * 1. Process pending labels → assign rewards to episodes
 * 2. Run teacher on unlabeled episodes (off-path)
 * 3. Apply REINFORCE updates (Lemma 6.1)
 * 4. Decay all weights toward priors
 * 5. Propose structural mutations
 * 6. Check pack promotion
 */

import type { BrainConfig } from "./types.js";
import { trustRank } from "./types.js";
import type { BrainStore } from "./store.js";
import type { BrainGraph } from "./graph.js";
import type { BrainTeacher } from "./teacher.js";
import type { BrainMutator } from "./mutator.js";
import type { PackManager } from "./pack.js";
import { computeReinforceUpdates, updateBaseline, applyWeightUpdates } from "./update.js";
import { decayAllWeights } from "./decay.js";
import { computeHealth } from "./health.js";

export class BrainTrainer {
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
  ) {}

  start(): void {
    if (this.interval) return;
    this.interval = setInterval(() => this.tick().catch((err) => {
      this.log.error(`[brain] Trainer tick failed: ${(err as Error).message}`);
    }), this.config.trainerIntervalMs);
    this.log.info(`[brain] Trainer started (interval: ${this.config.trainerIntervalMs}ms)`);
  }

  stop(): void {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
      this.log.info("[brain] Trainer stopped");
    }
  }

  async tick(): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      await this.processLabels();
      if (this.config.teacherEnabled && this.teacher) {
        await this.runTeacher();
      }
      await this.applyUpdates();
      this.runDecay();
      if (this.config.mutationsEnabled) {
        await this.proposeMutations();
      }
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

      // If episode already has a higher-trust reward, skip
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
    if (!this.teacher) return;
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
    if (episodes.length === 0) return;

    const baselineStr = this.store.getTrainingState("baseline_reward");
    let baseline = baselineStr ? parseFloat(baselineStr) : 0;

    for (const episode of episodes) {
      if (episode.reward === null) continue;

      const updates = computeReinforceUpdates(episode, this.config.learningRate, baseline);
      applyWeightUpdates(this.graph, updates);

      // Persist updated weights
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
    const lastDecayStr = this.store.getTrainingState("last_decay_at");
    const lastDecay = lastDecayStr ? parseInt(lastDecayStr, 10) : 0;
    if (Date.now() - lastDecay < 60_000) return; // Decay at most once per minute

    decayAllWeights(this.graph, this.config.decayRate, Date.now());
    this.store.decayAllWeights(this.config.decayRate);
    this.store.setTrainingState("last_decay_at", Date.now());
  }

  private async proposeMutations(): Promise<void> {
    const recentEpisodes = this.store.getRecentEpisodes(50);
    const proposals = this.mutator.proposeMutations(recentEpisodes);

    for (const proposal of proposals) {
      this.store.insertMutation(proposal);

      // Simple validation: apply if expected gain > 0
      if ((proposal.expectedGain ?? 0) > 0) {
        this.mutator.applyMutation(proposal);
      } else {
        this.store.resolveMutation(proposal.id, "rejected");
      }
    }
  }

  private async checkPromotion(): Promise<void> {
    const recentEpisodes = this.store.getRecentEpisodes(this.config.replayEpisodeCount);
    const health = computeHealth(this.graph, recentEpisodes, 0);

    const gate = this.packManager.replayGate(recentEpisodes, {
      minFiredPerQuery: this.config.minFiredPerQuery,
      maxDormantPercent: this.config.maxDormantPercent,
      maxOrphanCount: this.config.maxOrphanCount,
    });

    if (gate.passed) {
      const candidate = this.packManager.buildCandidate(health);
      this.packManager.promote(candidate.version);
    }
  }
}
