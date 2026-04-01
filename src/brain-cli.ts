import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import process from "node:process";
import { DatabaseSync } from "node:sqlite";
import { BrainGraph } from "./brain-core/graph.js";
import { computeHealth } from "./brain-core/health.js";
import { PackManager } from "./brain-core/pack.js";
import { BrainStore } from "./brain-store/store.js";
import { runBrainMigrations } from "./brain-store/migrations.js";
import { initBrain } from "./brain-store/init.js";
import {
  createEmbeddingClient,
  describeEmbeddingConfig,
} from "./brain-store/embedding.js";
import { resolveLcmConfig } from "./db/config.js";
import {
  flattenSeedWeights,
  flattenStopLocalWeights,
  populateGraph,
} from "./brain-runtime/graph-io.js";
import { buildPromotionStory } from "./brain-runtime/promotion-story.js";
import { readWorkerRuntimeState } from "./brain-runtime/worker-state.js";

function printJson(payload: unknown): void {
  process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
}

function buildInitLog(): { info: (msg: string) => void; warn: (msg: string) => void } {
  const verbose = /^(1|true|yes)$/i.test(process.env.OPENCLAWBRAIN_INIT_VERBOSE ?? "");
  if (!verbose) {
    return { info: () => {}, warn: () => {} };
  }
  return {
    info: (msg: string) => process.stderr.write(`${msg}\n`),
    warn: (msg: string) => process.stderr.write(`${msg}\n`),
  };
}

function usage(): never {
  process.stderr.write(
    "Usage: openclawbrain <init|status|trace|replay|promote|rollback|disable|enable|doctor> [args]\n",
  );
  process.exit(1);
}

function loadStore() {
  const config = resolveLcmConfig(process.env, {});
  const brainConfig = config.brain;
  if (!brainConfig) {
    throw new Error("OpenClawBrain configuration is unavailable");
  }

  mkdirSync(brainConfig.root, { recursive: true });
  const dbPath = join(brainConfig.root, "state.db");
  const db = new DatabaseSync(dbPath);
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA busy_timeout = 5000");
  db.exec("PRAGMA foreign_keys = ON");
  runBrainMigrations(db);
  const store = new BrainStore(db, { brainRoot: brainConfig.root });
  const graph = new BrainGraph();
  populateGraph(
    graph,
    store.getAllNodes(),
    store.loadAllEdges(),
    store.loadAllSeedWeights(),
    store.loadAllStopLocalWeights(),
  );

  return { config, brainConfig, store, graph };
}

function flattenEdges(graph: BrainGraph) {
  return graph.getAllEdges();
}

async function commandInit(workspaceArg?: string): Promise<void> {
  const { brainConfig, store, graph } = loadStore();
  const embedFn = createEmbeddingClient({ config: brainConfig });
  if (!embedFn) {
    throw new Error("OPENCLAWBRAIN_EMBEDDING_MODEL is required for init");
  }

  const workspaceRoot = resolve(workspaceArg ?? process.cwd());
  const result = await initBrain({
    workspaceRoot,
    embedFn,
    semanticThreshold: brainConfig.semanticThreshold,
    log: buildInitLog(),
  });

  store.clearGraph();
  graph.clear();
  for (const node of result.nodes) {
    graph.addNode(node);
    store.insertNode(node);
  }
  for (const edge of result.edges) {
    graph.addEdge(edge);
    store.insertEdge(edge);
  }

  const health = computeHealth(graph, [], 0);
  const pack = store.insertPack({
    nodeCount: health.nodeCount,
    edgeCount: health.edgeCount,
    healthJson: JSON.stringify(health),
  });
  store.writePackSnapshot({
    version: pack.version,
    nodes: graph.getAllNodes(),
    edges: flattenEdges(graph),
    seedWeights: flattenSeedWeights(graph),
    stopLocalWeights: flattenStopLocalWeights(graph),
    metadata: { reason: "cli-init", workspaceRoot, summary: result.summary },
  });
  store.promotePack(pack.version);

  printJson({
    command: "init",
    workspaceRoot,
    summary: result.summary,
    packVersion: pack.version,
  });
}

function commandStatus(): void {
  const { store, graph, brainConfig } = loadStore();
  const recentEpisodes = store.getRecentEpisodes(100);
  const currentPack = store.getCurrentPackVersion();
  const health = computeHealth(graph, recentEpisodes, currentPack ?? 0);
  const embeddingConfig = describeEmbeddingConfig(brainConfig);
  const workerState = readWorkerRuntimeState(store, brainConfig);
  const contextFeedback = store.getContextFeedbackSummary();
  const promotionStory = buildPromotionStory(store, { contextFeedback });
  const observationAttribution = store.getObservationAttributionSummary();
  const recentDecisionSummary = store.getRecentDecisionSummary(25);
  const recentTrace = store.getRecentTraces(1)[0] ?? null;
  const lastAssemblyDecision = store.getTrainingStateJson<Record<string, unknown>>("last_assembly_decision_json")
    ?? {
      mode: store.getTrainingState("last_assembly_mode"),
      footer: store.getTrainingState("last_assembly_footer"),
      episodeId: store.getTrainingState("last_assembly_episode_id"),
      traceId: store.getTrainingState("last_assembly_trace_id"),
    };

  printJson({
    command: "status",
    brainRoot: brainConfig.root,
    disabled: existsSync(join(brainConfig.root, "DISABLED")),
    shadowMode: brainConfig.shadowMode,
    maxCompileMs: brainConfig.maxCompileMs,
    budgetFraction: brainConfig.budgetFraction,
    maxHops: brainConfig.maxHops,
    maxFanoutPerNode: brainConfig.maxFanoutPerNode,
    maxFrontierSize: brainConfig.maxFrontierSize,
    maxSeeds: brainConfig.maxSeeds,
    semanticThreshold: brainConfig.semanticThreshold,
    workerMode: brainConfig.workerMode,
    workerHeartbeatTimeoutMs: brainConfig.workerHeartbeatTimeoutMs,
    workerRestartDelayMs: brainConfig.workerRestartDelayMs,
    embeddingProvider: brainConfig.embeddingProvider,
    embeddingModel: brainConfig.embeddingModel,
    embeddingBaseUrl: brainConfig.embeddingModel ? embeddingConfig.baseUrl : "",
    embeddingAuthMode: embeddingConfig.authMode,
    embeddingConfigError: embeddingConfig.error,
    ...workerState,
    currentPackVersion: currentPack,
    currentPackMetadata: promotionStory.currentPack?.metadata ?? null,
    pendingObservations: store.countPendingObservations(),
    pendingObservationsByStatus: store.countObservationsByStatus(),
    observationAttribution,
    contextFeedback,
    pendingLabels: store.getPendingLabels().length,
    pendingLabelsBySource: store.countPendingLabelsBySource(),
    mutationBacklog: store.countMutationsByStatus(),
    recentMutationBundles: store.getRecentMutationBundles(5),
    lastPromotionReason: store.getTrainingState("last_promotion_reason"),
    lastPromotionVerdict: store.getTrainingStateJson("last_promotion_verdict_json"),
    lastReplayFailureReason: store.getTrainingState("last_replay_failure_reason"),
    lastReplayGateVerdict: store.getTrainingStateJson("last_replay_gate_verdict_json"),
    promotionStory,
    recentDecisionSummary,
    lastTraceSelectionMetadata: recentTrace?.routeTrace?.selectionMetadata ?? null,
    lastAssemblyDecision,
    seedLearningEnabled: graph.hasSeedWeights(),
    recentTraceCount: store.getRecentTraces(5).length,
    ...health,
  });
}

function commandTrace(traceId?: string): void {
  const { store } = loadStore();
  const trace = traceId ? store.getTrace(traceId) : store.getRecentTraces(1)[0] ?? null;
  const chosenSeeds = trace?.seedScores.filter((seed) => seed.selected) ?? [];
  printJson({
    command: "trace",
    trace,
    chosenSeeds,
    chosenSeed: chosenSeeds.length === 1 ? chosenSeeds[0] : null,
    finalSectionOrder: [
      "correction_cards",
      "route_selected_evidence",
      "toolcards_and_workflows",
      "transcript_support",
    ],
  });
}

function commandReplay(): void {
  const { store, graph, brainConfig } = loadStore();
  const gate = new PackManager(
    {
      insertPack: (params) => store.insertPack(params),
      promotePack: (version) => store.promotePack(version),
      rollbackPack: (version) => store.rollbackPack(version),
    },
    graph,
    { info: () => {}, warn: () => {} },
  ).replayGate(store.getRecentEpisodes(brainConfig.replayEpisodeCount), {
    minFiredPerQuery: brainConfig.minFiredPerQuery,
    maxDormantPercent: brainConfig.maxDormantPercent,
    maxOrphanCount: brainConfig.maxOrphanCount,
  });
  printJson({
    command: "replay",
    passed: gate.passed,
    reason: gate.reason.summary,
    reasonCode: gate.reason.code,
    verdict: gate,
    health: gate.health,
  });
}

function commandPromote(): void {
  const { store, graph } = loadStore();
  const health = computeHealth(graph, store.getRecentEpisodes(100), store.getCurrentPackVersion() ?? 0);
  const pack = store.insertPack({
    nodeCount: health.nodeCount,
    edgeCount: health.edgeCount,
    healthJson: JSON.stringify(health),
  });
  store.writePackSnapshot({
    version: pack.version,
    nodes: graph.getAllNodes(),
    edges: flattenEdges(graph),
    seedWeights: flattenSeedWeights(graph),
    stopLocalWeights: flattenStopLocalWeights(graph),
    metadata: { reason: "cli-promote" },
  });
  store.promotePack(pack.version);
  printJson({
    command: "promote",
    version: pack.version,
  });
}

function commandRollback(versionArg?: string): void {
  const { store } = loadStore();
  const version = versionArg ? Number.parseInt(versionArg, 10) : store.getCurrentPackVersion();
  if (!version) {
    throw new Error("No pack version available to roll back");
  }
  store.rollbackPack(version);
  printJson({
    command: "rollback",
    version,
    currentPackVersion: store.getCurrentPackVersion(),
  });
}

function commandDisable(): void {
  const { brainConfig } = loadStore();
  const disabledFile = join(brainConfig.root, "DISABLED");
  writeFileSync(disabledFile, "disabled\n", "utf8");
  printJson({
    command: "disable",
    disabledFile,
  });
}

function commandEnable(): void {
  const { brainConfig } = loadStore();
  const disabledFile = join(brainConfig.root, "DISABLED");
  if (existsSync(disabledFile)) {
    rmSync(disabledFile);
  }
  printJson({
    command: "enable",
    disabledFile,
    enabled: true,
  });
}

function commandDoctor(): void {
  const { brainConfig, store, graph } = loadStore();
  const currentPackVersion = store.getCurrentPackVersion();
  const snapshot = currentPackVersion !== null ? store.readPackSnapshot(currentPackVersion) : null;
  const embeddingConfig = describeEmbeddingConfig(brainConfig);
  const workerState = readWorkerRuntimeState(store, brainConfig);
  printJson({
    command: "doctor",
    brainRoot: brainConfig.root,
    stateDbExists: existsSync(join(brainConfig.root, "state.db")),
    currentPackVersion,
    currentPackSnapshotExists: snapshot !== null,
    embeddingConfigured: brainConfig.embeddingModel.trim().length > 0,
    embeddingProvider: brainConfig.embeddingProvider,
    embeddingModel: brainConfig.embeddingModel,
    embeddingBaseUrl: brainConfig.embeddingModel.trim().length > 0 ? embeddingConfig.baseUrl : "",
    embeddingAuthMode: embeddingConfig.authMode,
    embeddingConfigError: embeddingConfig.error,
    shadowMode: brainConfig.shadowMode,
    ...workerState,
    disabled: existsSync(join(brainConfig.root, "DISABLED")),
    pendingObservations: store.countPendingObservations(),
    mutationBacklog: store.countMutationsByStatus(),
    orphanedTraceRows: store.countOrphanedTraceRows(),
    nodeCount: graph.nodeCount(),
    edgeCount: graph.edgeCount(),
    lastTraceId: store.getRecentTraces(1)[0]?.id ?? null,
  });
}

async function main(): Promise<void> {
  const [command, arg] = process.argv.slice(2);
  switch (command) {
    case "init":
      await commandInit(arg);
      return;
    case "status":
      commandStatus();
      return;
    case "trace":
      commandTrace(arg);
      return;
    case "replay":
      commandReplay();
      return;
    case "promote":
      commandPromote();
      return;
    case "rollback":
      commandRollback(arg);
      return;
    case "disable":
      commandDisable();
      return;
    case "enable":
      commandEnable();
      return;
    case "doctor":
      commandDoctor();
      return;
    default:
      usage();
  }
}

void main().catch((error) => {
  process.stderr.write(`${(error as Error).message}\n`);
  process.exit(1);
});
