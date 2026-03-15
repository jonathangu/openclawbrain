/**
 * Immutable pack management.
 *
 * Gateway reads only promoted pack. Daemon writes only mutable state.
 * Promotion requires passing replay gate and health bounds.
 */

import type { Pack, HealthMetrics, Episode } from "./types.js";
import type { BrainGraph } from "./graph.js";
import { computeHealth } from "./health.js";
import { replayEpisode } from "./episode.js";

export interface BrainPackPersistence {
  insertPack(params: { nodeCount: number; edgeCount: number; healthJson: string }): Pack;
  promotePack(version: number): void;
  rollbackPack(version: number): void;
}

export class PackManager {
  constructor(
    private persistence: BrainPackPersistence,
    private graph: BrainGraph,
    private log: { info: (msg: string) => void; warn: (msg: string) => void },
  ) {}

  /**
   * Build a candidate pack from current graph state.
   */
  buildCandidate(health: HealthMetrics): Pack {
    return this.persistence.insertPack({
      nodeCount: health.nodeCount,
      edgeCount: health.edgeCount,
      healthJson: JSON.stringify(health),
    });
  }

  /**
   * Replay gate: test candidate against recent episodes.
   * Returns whether the candidate should be promoted.
   */
  replayGate(
    recentEpisodes: Episode[],
    config: { minFiredPerQuery: number; maxDormantPercent: number; maxOrphanCount: number },
    candidateGraph: BrainGraph = this.graph,
  ): { passed: boolean; reason: string; health: HealthMetrics } {
    if (recentEpisodes.length === 0) {
      return {
        passed: true,
        reason: "no episodes to replay",
        health: computeHealth(candidateGraph, recentEpisodes, 0),
      };
    }

    const health = computeHealth(candidateGraph, recentEpisodes, 0);

    if (health.firedPerQuery < config.minFiredPerQuery) {
      return {
        passed: false,
        reason: `firedPerQuery ${health.firedPerQuery.toFixed(2)} < ${config.minFiredPerQuery}`,
        health,
      };
    }
    if (health.dormantPercent > config.maxDormantPercent) {
      return {
        passed: false,
        reason: `dormantPercent ${(health.dormantPercent * 100).toFixed(1)}% > ${(config.maxDormantPercent * 100).toFixed(1)}%`,
        health,
      };
    }
    if (health.orphanCount > config.maxOrphanCount) {
      return {
        passed: false,
        reason: `orphanCount ${health.orphanCount} > ${config.maxOrphanCount}`,
        health,
      };
    }

    // Check no human-labeled episodes regressed
    const humanEpisodes = recentEpisodes.filter((ep) => ep.rewardSource === "human" && ep.reward !== null);
    for (const ep of humanEpisodes) {
      const replay = replayEpisode(ep, candidateGraph);
      if (replay.wouldChange && ep.reward! > 0) {
        return {
          passed: false,
          reason: `human-positive episode ${ep.id} would change routing`,
          health,
        };
      }
    }

    const selfNegativeEpisodes = recentEpisodes.filter(
      (ep) => ep.rewardSource === "self" && ep.reward !== null && ep.reward < 0,
    );
    for (const ep of selfNegativeEpisodes) {
      const replay = replayEpisode(ep, candidateGraph);
      if (!replay.wouldChange) {
        return {
          passed: false,
          reason: `self-negative episode ${ep.id} did not change routing`,
          health,
        };
      }
    }

    return { passed: true, reason: "all gates passed", health };
  }

  promote(version: number): void {
    this.persistence.promotePack(version);
    this.log.info(`[brain] Pack v${version} promoted`);
  }

  rollback(version: number): void {
    this.persistence.rollbackPack(version);
    this.log.warn(`[brain] Pack v${version} rolled back`);
  }
}
