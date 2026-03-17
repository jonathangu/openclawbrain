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
import { flattenSeedWeights } from "./brain-runtime/graph-io.js";
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
  for (const node of store.getAllNodes()) {
    graph.addNode(node);
  }
  for (const edge of store.loadAllEdges()) {
    graph.addEdge(edge);
  }

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
  const currentSnapshot = currentPack !== null ? store.readPackSnapshot(currentPack) : null;
  const embeddingConfig = describeEmbeddingConfig(brainConfig);
  const workerState = readWorkerRuntimeState(store, brainConfig);

  printJson({
    command: "status",
    brainRoot: brainConfig.root,
    disabled: existsSync(join(brainConfig.root, "DISABLED")),
    shadowMode: brainConfig.shadowMode,
    embeddingProvider: brainConfig.embeddingProvider,
    embeddingModel: brainConfig.embeddingModel,
    embeddingBaseUrl: brainConfig.embeddingModel ? embeddingConfig.baseUrl : "",
    embeddingAuthMode: embeddingConfig.authMode,
    embeddingConfigError: embeddingConfig.error,
    ...workerState,
    currentPackVersion: currentPack,
    currentPackMetadata: currentSnapshot?.metadata ?? null,
    pendingEvidence: store.getPendingEvidence(100).length,
    pendingEvidenceBySource: store.countPendingEvidenceBySource(),
    pendingLabels: store.getPendingLabels().length,
    pendingLabelsBySource: store.countPendingLabelsBySource(),
    mutationBacklog: store.countMutationsByStatus(),
    lastPromotionReason: store.getTrainingState("last_promotion_reason"),
    lastReplayFailureReason: store.getTrainingState("last_replay_failure_reason"),
    lastAssemblyDecision: {
      mode: store.getTrainingState("last_assembly_mode"),
      footer: store.getTrainingState("last_assembly_footer"),
      episodeId: store.getTrainingState("last_assembly_episode_id"),
      traceId: store.getTrainingState("last_assembly_trace_id"),
    },
    seedLearningEnabled: graph.hasSeedWeights(),
    recentTraceCount: store.getRecentTraces(5).length,
    ...health,
  });
}

function commandTrace(traceId?: string): void {
  const { store } = loadStore();
  const trace = traceId ? store.getTrace(traceId) : store.getRecentTraces(1)[0] ?? null;
  const chosenSeed = trace?.seedScores.find((seed) => seed.chosen) ?? null;
  printJson({
    command: "trace",
    trace,
    chosenSeed,
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
    reason: gate.reason,
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
    pendingEvidence: store.getPendingEvidence(100).length,
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
