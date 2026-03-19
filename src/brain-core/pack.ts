/**
 * Immutable pack management.
 *
 * Gateway reads only promoted pack. Daemon writes only mutable state.
 * Promotion requires passing replay gate and health bounds.
 */

import type {
  Pack,
  HealthMetrics,
  Episode,
  ReplayGateReason,
  ReplayGateVerdict,
} from "./types.js";
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
  ): ReplayGateVerdict {
    const buildVerdict = (
      passed: boolean,
      reason: ReplayGateReason,
      health: HealthMetrics,
      humanPositiveEpisodeCount: number,
      selfNegativeEpisodeCount: number,
    ): ReplayGateVerdict => ({
      passed,
      reason,
      health,
      evaluatedEpisodeCount: recentEpisodes.length,
      humanPositiveEpisodeCount,
      selfNegativeEpisodeCount,
    });

    if (recentEpisodes.length === 0) {
      return {
        passed: true,
        reason: {
          code: "no_episodes_to_replay",
          summary: "no episodes to replay",
          details: {},
        },
        health: computeHealth(candidateGraph, recentEpisodes, 0),
        evaluatedEpisodeCount: 0,
        humanPositiveEpisodeCount: 0,
        selfNegativeEpisodeCount: 0,
      };
    }

    const health = computeHealth(candidateGraph, recentEpisodes, 0);
    const humanEpisodes = recentEpisodes.filter((ep) => ep.rewardSource === "human" && ep.reward !== null);
    const selfNegativeEpisodes = recentEpisodes.filter(
      (ep) => ep.rewardSource === "self" && ep.reward !== null && ep.reward < 0,
    );

    if (health.firedPerQuery < config.minFiredPerQuery) {
      return buildVerdict(false, {
        code: "fired_per_query_below_min",
        summary: `firedPerQuery ${health.firedPerQuery.toFixed(2)} < ${config.minFiredPerQuery}`,
        details: {
          metric: "firedPerQuery",
          actual: health.firedPerQuery,
          minimum: config.minFiredPerQuery,
        },
      }, health, humanEpisodes.length, selfNegativeEpisodes.length);
    }
    if (health.dormantPercent > config.maxDormantPercent) {
      return buildVerdict(false, {
        code: "dormant_percent_above_max",
        summary: `dormantPercent ${(health.dormantPercent * 100).toFixed(1)}% > ${(config.maxDormantPercent * 100).toFixed(1)}%`,
        details: {
          metric: "dormantPercent",
          actual: health.dormantPercent,
          maximum: config.maxDormantPercent,
        },
      }, health, humanEpisodes.length, selfNegativeEpisodes.length);
    }
    if (health.orphanCount > config.maxOrphanCount) {
      return buildVerdict(false, {
        code: "orphan_count_above_max",
        summary: `orphanCount ${health.orphanCount} > ${config.maxOrphanCount}`,
        details: {
          metric: "orphanCount",
          actual: health.orphanCount,
          maximum: config.maxOrphanCount,
        },
      }, health, humanEpisodes.length, selfNegativeEpisodes.length);
    }

    // Check no human-labeled episodes regressed
    for (const ep of humanEpisodes) {
      const replay = replayEpisode(ep, candidateGraph);
      if (replay.wouldChange && ep.reward! > 0) {
        return buildVerdict(false, {
          code: "human_positive_route_regression",
          summary: `human-positive episode ${ep.id} would change routing`,
          details: {
            episodeId: ep.id,
            reward: ep.reward,
            rewardSource: ep.rewardSource,
          },
        }, health, humanEpisodes.length, selfNegativeEpisodes.length);
      }
    }

    for (const ep of selfNegativeEpisodes) {
      const replay = replayEpisode(ep, candidateGraph);
      if (!replay.wouldChange) {
        return buildVerdict(false, {
          code: "self_negative_route_unchanged",
          summary: `self-negative episode ${ep.id} did not change routing`,
          details: {
            episodeId: ep.id,
            reward: ep.reward,
            rewardSource: ep.rewardSource,
          },
        }, health, humanEpisodes.length, selfNegativeEpisodes.length);
      }
    }

    return buildVerdict(true, {
      code: "all_gates_passed",
      summary: "all gates passed",
      details: {},
    }, health, humanEpisodes.length, selfNegativeEpisodes.length);
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
