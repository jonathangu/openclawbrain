/**
 * Public API surface for the brain learning layer.
 */

export { BrainService } from "./service.js";
export { BrainStore } from "./store.js";
export { BrainGraph, cosineSimilarity } from "./graph.js";
export { runBrainMigrations } from "./migration.js";
export { traverse } from "./traverse.js";
export { softmaxPolicy, sampleAction, scoreAction } from "./policy.js";
export { computeReinforceUpdates, updateBaseline, applyWeightUpdates } from "./update.js";
export { decayWeight, decayAllWeights } from "./decay.js";
export { computeHealth } from "./health.js";
export { recordEpisode, replayEpisode } from "./episode.js";
export { recordTrace, generateFooter } from "./trace.js";
export { LabelHarvester } from "./harvester.js";
export { BrainTeacher } from "./teacher.js";
export { BrainTrainer } from "./trainer.js";
export { BrainMutator } from "./mutator.js";
export { PackManager } from "./pack.js";
export { initBrain, discoverSources, chunkSources, createNodesFromChunks } from "./ingest.js";
export { createBrainTeachTool, createBrainStatusTool, createBrainTraceTool } from "./tools.js";

export type {
  BrainNode, BrainEdge, NodeKind, EdgeKind, TrustLevel, RewardSource,
  TraversalState, TraversalAction, TrajectoryStep,
  Episode, Label, Pack, MutationProposal, MutationKind, MutationStatus,
  HealthMetrics, DecisionTrace, BrainConfig, PolicyParams,
  TraversalResult, RouteFn,
} from "./types.js";

export { DEFAULT_BRAIN_CONFIG, DEFAULT_POLICY_PARAMS, trustRank } from "./types.js";
