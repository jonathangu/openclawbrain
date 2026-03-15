import { homedir } from "os";
import { dirname, join } from "path";

export type OpenClawBrainRuntimeConfig = {
  enabled: boolean;
  root: string;
  budgetFraction: number;
  maxHops: number;
  maxSeeds: number;
  semanticThreshold: number;
  servingTemperature: number;
  learningTemperature: number;
  learningRate: number;
  baselineAlpha: number;
  decayRate: number;
  trainerIntervalMs: number;
  teacherEnabled: boolean;
  teacherProvider: string;
  teacherModel: string;
  mutationsEnabled: boolean;
  replayEpisodeCount: number;
  minFiredPerQuery: number;
  maxDormantPercent: number;
  maxOrphanCount: number;
  embeddingProvider: string;
  embeddingModel: string;
  embeddingBaseUrl: string;
};

export type LcmConfig = {
  enabled: boolean;
  databasePath: string;
  contextThreshold: number;
  freshTailCount: number;
  leafMinFanout: number;
  condensedMinFanout: number;
  condensedMinFanoutHard: number;
  incrementalMaxDepth: number;
  leafChunkTokens: number;
  leafTargetTokens: number;
  condensedTargetTokens: number;
  maxExpandTokens: number;
  largeFileTokenThreshold: number;
  /** Provider override for large-file text summarization. */
  largeFileSummaryProvider: string;
  /** Model override for large-file text summarization. */
  largeFileSummaryModel: string;
  autocompactDisabled: boolean;
  /** IANA timezone for timestamps in summaries (from TZ env or system default) */
  timezone: string;
  /** When true, retroactively delete HEARTBEAT_OK turn cycles from LCM storage. */
  pruneHeartbeatOk: boolean;
  /** OpenClawBrain v2 runtime settings. */
  brain?: OpenClawBrainRuntimeConfig;
};

/** Safely coerce an unknown value to a finite number, or return undefined. */
function toNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

/** Safely coerce an unknown value to a boolean, or return undefined. */
function toBool(value: unknown): boolean | undefined {
  if (typeof value === "boolean") return value;
  if (value === "true") return true;
  if (value === "false") return false;
  return undefined;
}

/** Safely coerce an unknown value to a trimmed non-empty string, or return undefined. */
function toStr(value: unknown): string | undefined {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  }
  return undefined;
}

/**
 * Resolve LCM configuration with three-tier precedence:
 *   1. Environment variables (highest — backward compat)
 *   2. Plugin config object (from plugins.entries.lossless-claw.config)
 *   3. Hardcoded defaults (lowest)
 */
export function resolveLcmConfig(
  env: NodeJS.ProcessEnv = process.env,
  pluginConfig?: Record<string, unknown>,
): LcmConfig {
  const pc = pluginConfig ?? {};
  const databasePath =
    env.LCM_DATABASE_PATH
    ?? toStr(pc.dbPath)
    ?? toStr(pc.databasePath)
    ?? join(homedir(), ".openclaw", "lcm.db");
  const brainRoot =
    env.OPENCLAWBRAIN_ROOT?.trim()
    ?? toStr(pc.brainRoot)
    ?? join(dirname(databasePath), "openclawbrain");

  return {
    enabled:
      env.LCM_ENABLED !== undefined
        ? env.LCM_ENABLED !== "false"
        : toBool(pc.enabled) ?? true,
    databasePath,
    contextThreshold:
      (env.LCM_CONTEXT_THRESHOLD !== undefined ? parseFloat(env.LCM_CONTEXT_THRESHOLD) : undefined)
        ?? toNumber(pc.contextThreshold) ?? 0.75,
    freshTailCount:
      (env.LCM_FRESH_TAIL_COUNT !== undefined ? parseInt(env.LCM_FRESH_TAIL_COUNT, 10) : undefined)
        ?? toNumber(pc.freshTailCount) ?? 32,
    leafMinFanout:
      (env.LCM_LEAF_MIN_FANOUT !== undefined ? parseInt(env.LCM_LEAF_MIN_FANOUT, 10) : undefined)
        ?? toNumber(pc.leafMinFanout) ?? 8,
    condensedMinFanout:
      (env.LCM_CONDENSED_MIN_FANOUT !== undefined ? parseInt(env.LCM_CONDENSED_MIN_FANOUT, 10) : undefined)
        ?? toNumber(pc.condensedMinFanout) ?? 4,
    condensedMinFanoutHard:
      (env.LCM_CONDENSED_MIN_FANOUT_HARD !== undefined ? parseInt(env.LCM_CONDENSED_MIN_FANOUT_HARD, 10) : undefined)
        ?? toNumber(pc.condensedMinFanoutHard) ?? 2,
    incrementalMaxDepth:
      (env.LCM_INCREMENTAL_MAX_DEPTH !== undefined ? parseInt(env.LCM_INCREMENTAL_MAX_DEPTH, 10) : undefined)
        ?? toNumber(pc.incrementalMaxDepth) ?? 0,
    leafChunkTokens:
      (env.LCM_LEAF_CHUNK_TOKENS !== undefined ? parseInt(env.LCM_LEAF_CHUNK_TOKENS, 10) : undefined)
        ?? toNumber(pc.leafChunkTokens) ?? 20000,
    leafTargetTokens:
      (env.LCM_LEAF_TARGET_TOKENS !== undefined ? parseInt(env.LCM_LEAF_TARGET_TOKENS, 10) : undefined)
        ?? toNumber(pc.leafTargetTokens) ?? 1200,
    condensedTargetTokens:
      (env.LCM_CONDENSED_TARGET_TOKENS !== undefined ? parseInt(env.LCM_CONDENSED_TARGET_TOKENS, 10) : undefined)
        ?? toNumber(pc.condensedTargetTokens) ?? 2000,
    maxExpandTokens:
      (env.LCM_MAX_EXPAND_TOKENS !== undefined ? parseInt(env.LCM_MAX_EXPAND_TOKENS, 10) : undefined)
        ?? toNumber(pc.maxExpandTokens) ?? 4000,
    largeFileTokenThreshold:
      (env.LCM_LARGE_FILE_TOKEN_THRESHOLD !== undefined ? parseInt(env.LCM_LARGE_FILE_TOKEN_THRESHOLD, 10) : undefined)
        ?? toNumber(pc.largeFileThresholdTokens)
        ?? toNumber(pc.largeFileTokenThreshold)
        ?? 25000,
    largeFileSummaryProvider:
      env.LCM_LARGE_FILE_SUMMARY_PROVIDER?.trim() ?? toStr(pc.largeFileSummaryProvider) ?? "",
    largeFileSummaryModel:
      env.LCM_LARGE_FILE_SUMMARY_MODEL?.trim() ?? toStr(pc.largeFileSummaryModel) ?? "",
    autocompactDisabled:
      env.LCM_AUTOCOMPACT_DISABLED !== undefined
        ? env.LCM_AUTOCOMPACT_DISABLED === "true"
        : toBool(pc.autocompactDisabled) ?? false,
    timezone: env.TZ ?? toStr(pc.timezone) ?? Intl.DateTimeFormat().resolvedOptions().timeZone,
    pruneHeartbeatOk:
      env.LCM_PRUNE_HEARTBEAT_OK !== undefined
        ? env.LCM_PRUNE_HEARTBEAT_OK === "true"
        : toBool(pc.pruneHeartbeatOk) ?? false,
    brain: {
      enabled:
        env.OPENCLAWBRAIN_ENABLED !== undefined
          ? env.OPENCLAWBRAIN_ENABLED !== "false"
          : toBool(pc.brainEnabled) ?? true,
      root: brainRoot,
      budgetFraction:
        (env.OPENCLAWBRAIN_BUDGET_FRACTION !== undefined
          ? parseFloat(env.OPENCLAWBRAIN_BUDGET_FRACTION)
          : undefined) ?? toNumber(pc.brainBudgetFraction) ?? 0.3,
      maxHops:
        (env.OPENCLAWBRAIN_MAX_HOPS !== undefined
          ? parseInt(env.OPENCLAWBRAIN_MAX_HOPS, 10)
          : undefined) ?? toNumber(pc.brainMaxHops) ?? 8,
      maxSeeds:
        (env.OPENCLAWBRAIN_MAX_SEEDS !== undefined
          ? parseInt(env.OPENCLAWBRAIN_MAX_SEEDS, 10)
          : undefined) ?? toNumber(pc.brainMaxSeeds) ?? 10,
      semanticThreshold:
        (env.OPENCLAWBRAIN_SEMANTIC_THRESHOLD !== undefined
          ? parseFloat(env.OPENCLAWBRAIN_SEMANTIC_THRESHOLD)
          : undefined) ?? toNumber(pc.brainSemanticThreshold) ?? 0.7,
      servingTemperature:
        (env.OPENCLAWBRAIN_SERVING_TEMPERATURE !== undefined
          ? parseFloat(env.OPENCLAWBRAIN_SERVING_TEMPERATURE)
          : undefined) ?? toNumber(pc.brainServingTemperature) ?? 0.1,
      learningTemperature:
        (env.OPENCLAWBRAIN_LEARNING_TEMPERATURE !== undefined
          ? parseFloat(env.OPENCLAWBRAIN_LEARNING_TEMPERATURE)
          : undefined) ?? toNumber(pc.brainLearningTemperature) ?? 1.0,
      learningRate:
        (env.OPENCLAWBRAIN_LEARNING_RATE !== undefined
          ? parseFloat(env.OPENCLAWBRAIN_LEARNING_RATE)
          : undefined) ?? toNumber(pc.brainLearningRate) ?? 0.01,
      baselineAlpha:
        (env.OPENCLAWBRAIN_BASELINE_ALPHA !== undefined
          ? parseFloat(env.OPENCLAWBRAIN_BASELINE_ALPHA)
          : undefined) ?? toNumber(pc.brainBaselineAlpha) ?? 0.1,
      decayRate:
        (env.OPENCLAWBRAIN_DECAY_RATE !== undefined
          ? parseFloat(env.OPENCLAWBRAIN_DECAY_RATE)
          : undefined) ?? toNumber(pc.brainDecayRate) ?? 0.995,
      trainerIntervalMs:
        (env.OPENCLAWBRAIN_TRAINER_INTERVAL_MS !== undefined
          ? parseInt(env.OPENCLAWBRAIN_TRAINER_INTERVAL_MS, 10)
          : undefined) ?? toNumber(pc.brainTrainerIntervalMs) ?? 30_000,
      teacherEnabled:
        env.OPENCLAWBRAIN_TEACHER_ENABLED !== undefined
          ? env.OPENCLAWBRAIN_TEACHER_ENABLED !== "false"
          : toBool(pc.brainTeacherEnabled) ?? true,
      teacherProvider:
        env.OPENCLAWBRAIN_TEACHER_PROVIDER?.trim() ?? toStr(pc.brainTeacherProvider) ?? "",
      teacherModel:
        env.OPENCLAWBRAIN_TEACHER_MODEL?.trim() ?? toStr(pc.brainTeacherModel) ?? "",
      mutationsEnabled:
        env.OPENCLAWBRAIN_MUTATIONS_ENABLED !== undefined
          ? env.OPENCLAWBRAIN_MUTATIONS_ENABLED !== "false"
          : toBool(pc.brainMutationsEnabled) ?? true,
      replayEpisodeCount:
        (env.OPENCLAWBRAIN_REPLAY_EPISODE_COUNT !== undefined
          ? parseInt(env.OPENCLAWBRAIN_REPLAY_EPISODE_COUNT, 10)
          : undefined) ?? toNumber(pc.brainReplayEpisodeCount) ?? 100,
      minFiredPerQuery:
        (env.OPENCLAWBRAIN_MIN_FIRED_PER_QUERY !== undefined
          ? parseFloat(env.OPENCLAWBRAIN_MIN_FIRED_PER_QUERY)
          : undefined) ?? toNumber(pc.brainMinFiredPerQuery) ?? 1.0,
      maxDormantPercent:
        (env.OPENCLAWBRAIN_MAX_DORMANT_PERCENT !== undefined
          ? parseFloat(env.OPENCLAWBRAIN_MAX_DORMANT_PERCENT)
          : undefined) ?? toNumber(pc.brainMaxDormantPercent) ?? 0.3,
      maxOrphanCount:
        (env.OPENCLAWBRAIN_MAX_ORPHAN_COUNT !== undefined
          ? parseInt(env.OPENCLAWBRAIN_MAX_ORPHAN_COUNT, 10)
          : undefined) ?? toNumber(pc.brainMaxOrphanCount) ?? 10,
      embeddingProvider:
        env.OPENCLAWBRAIN_EMBEDDING_PROVIDER?.trim()
        ?? toStr(pc.brainEmbeddingProvider)
        ?? "openai",
      embeddingModel:
        env.OPENCLAWBRAIN_EMBEDDING_MODEL?.trim() ?? toStr(pc.brainEmbeddingModel) ?? "",
      embeddingBaseUrl:
        env.OPENCLAWBRAIN_EMBEDDING_BASE_URL?.trim()
        ?? toStr(pc.brainEmbeddingBaseUrl)
        ?? "",
    },
  };
}
