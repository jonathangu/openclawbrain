import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { OpenClawBrainRuntimeConfig } from "../db/config.js";
import type {
  BrainConfig,
  BrainNode,
  DecisionTrace,
  NodeKind,
  TraversalResult,
} from "../brain-core/types.js";
import { DEFAULT_BRAIN_CONFIG } from "../brain-core/types.js";
import { BrainGraph } from "../brain-core/graph.js";
import { traverse } from "../brain-core/traverse.js";
import { recordEpisode } from "../brain-core/episode.js";
import { recordTrace } from "../brain-core/trace.js";
import { computeHealth } from "../brain-core/health.js";
import { BrainTeacher } from "../brain-core/teacher.js";
import { BrainMutator } from "../brain-core/mutator.js";
import { PackManager } from "../brain-core/pack.js";
import { BrainStore } from "../brain-store/store.js";
import { runBrainMigrations } from "../brain-store/migrations.js";
import { initBrain as runInit } from "../brain-store/init.js";
import { createEmbeddingClient, type BrainEmbeddingFn } from "../brain-store/embedding.js";
import { LabelHarvester } from "./harvester-extension.js";
import { BrainWorker } from "../brain-worker/worker.js";
import type { LcmDependencies } from "../types.js";

function flattenEdges(graph: BrainGraph) {
  return graph
    .getAllNodes()
    .flatMap((node) => graph.getOutgoingEdges(node.id));
}

function buildBrainConfig(
  runtimeConfig: OpenClawBrainRuntimeConfig,
  overrides?: Partial<BrainConfig>,
): BrainConfig {
  return {
    ...DEFAULT_BRAIN_CONFIG,
    ...runtimeConfig,
    ...overrides,
  };
}

function populateGraph(graph: BrainGraph, nodes: BrainNode[], edges: ReturnType<typeof flattenEdges>): void {
  graph.clear();
  for (const node of nodes) {
    graph.addNode(node);
  }
  for (const edge of edges) {
    graph.addEdge(edge);
  }
}

export class BrainService {
  private store: BrainStore;
  private mutableGraph = new BrainGraph();
  private servingGraph = new BrainGraph();
  private worker: BrainWorker;
  private harvesterImpl: LabelHarvester;
  private packManager: PackManager;
  private embeddingClient: BrainEmbeddingFn | null;
  private config: BrainConfig;
  private initialized = false;

  constructor(params: {
    deps: LcmDependencies;
    config?: Partial<BrainConfig>;
    runtimeConfig?: OpenClawBrainRuntimeConfig;
  }) {
    const runtimeConfig = params.runtimeConfig ?? params.deps.config.brain;
    if (!runtimeConfig) {
      throw new Error("OpenClawBrain runtime configuration is missing");
    }

    this.config = buildBrainConfig(runtimeConfig, params.config);
    mkdirSync(this.config.root, { recursive: true });

    const db = new DatabaseSync(join(this.config.root, "state.db"));
    db.exec("PRAGMA journal_mode = WAL");
    db.exec("PRAGMA foreign_keys = ON");
    runBrainMigrations(db);

    this.store = new BrainStore(db, { brainRoot: this.config.root });
    this.harvesterImpl = new LabelHarvester(this.store, params.deps.log);
    this.embeddingClient = createEmbeddingClient({
      config: runtimeConfig,
      getApiKey: (provider, model) => params.deps.getApiKey(provider, model),
      log: params.deps.log,
    });

    populateGraph(this.mutableGraph, this.store.getAllNodes(), this.store.loadAllEdges());
    this.reloadServingGraph();

    const teacher =
      this.config.teacherEnabled
        ? new BrainTeacher(
            async (request) =>
              params.deps.complete({
                provider: request.provider,
                model: request.model,
                apiKey: request.apiKey,
                messages: request.messages,
                system: request.system,
                maxTokens: request.maxTokens,
                temperature: request.temperature,
              }),
            () =>
              params.deps.resolveModel(
                this.config.teacherModel || undefined,
                this.config.teacherProvider || undefined,
              ),
            (provider, model) => params.deps.getApiKey(provider, model),
            this.mutableGraph,
            params.deps.log,
          )
        : null;

    const persistence = {
      insertEdge: (edge: Parameters<BrainStore["insertEdge"]>[0]) => this.store.insertEdge(edge),
      deleteNode: (id: string) => this.store.deleteNode(id),
      deleteEdge: (source: string, target: string, kind: string) =>
        this.store.deleteEdge(source, target, kind as never),
      resolveMutation: (id: string, status: "promoted" | "rejected") =>
        this.store.resolveMutation(id, status),
    };
    const mutator = new BrainMutator(persistence, this.mutableGraph, params.deps.log);
    this.packManager = new PackManager(
      {
        insertPack: (pack) => this.store.insertPack(pack),
        promotePack: (version) => this.store.promotePack(version),
        rollbackPack: (version) => this.store.rollbackPack(version),
      },
      this.mutableGraph,
      params.deps.log,
    );
    this.worker = new BrainWorker(
      this.store,
      this.mutableGraph,
      teacher,
      mutator,
      this.packManager,
      this.config,
      params.deps.log,
      {
        isEnabled: () => this.isEnabled(),
        onPromotionReady: async ({ healthJson }) => {
          await this.promoteMutableGraph("worker", { healthJson });
        },
      },
    );
  }

  startWorker(): void {
    if (!this.isEnabled()) {
      return;
    }
    this.worker.start();
  }

  stopWorker(): void {
    this.worker.stop();
  }

  get harvester(): LabelHarvester {
    return this.harvesterImpl;
  }

  isEnabled(): boolean {
    return this.config.enabled && !existsSync(join(this.config.root, "DISABLED"));
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  async query(params: {
    conversationId: number;
    queryText: string;
    budgetChars: number;
    queryEmbedding?: Float32Array;
  }): Promise<TraversalResult | null> {
    if (!this.isEnabled() || this.servingGraph.nodeCount() === 0) {
      return null;
    }

    const embedding =
      params.queryEmbedding
      ?? (this.embeddingClient ? await this.embeddingClient(params.queryText) : null);
    if (!embedding || embedding.length === 0) {
      return null;
    }

    const traversalResult = traverse({
      graph: this.servingGraph,
      queryEmbedding: embedding,
      queryText: params.queryText,
      maxHops: this.config.maxHops,
      budgetChars: params.budgetChars,
      temperature: this.config.servingTemperature,
      maxSeeds: this.config.maxSeeds,
      semanticThreshold: this.config.semanticThreshold,
    });
    if (traversalResult.firedNodes.length === 0) {
      return null;
    }

    const episode = recordEpisode({
      traversalResult,
      queryText: params.queryText,
      queryEmbedding: embedding,
      conversationId: params.conversationId,
      packVersion: this.store.getCurrentPackVersion(),
    });
    this.store.insertEpisode(episode);

    const trace = recordTrace({
      traversalResult,
      queryText: params.queryText,
      episodeId: episode.id,
      packVersion: episode.packVersion,
    });
    this.store.insertTrace(trace);

    return {
      fired: traversalResult.firedNodes,
      vetoed: traversalResult.vetoedNodes,
      episode,
      trace,
    };
  }

  async teach(params: {
    instruction: string;
    conversationId?: number;
    kind?: string;
    tags?: string[];
  }): Promise<{ nodeId: string; packVersion: number | null }> {
    if (!this.embeddingClient) {
      throw new Error("Embedding model is required before brain_teach can make knowledge retrievable");
    }

    const nodeKind = (params.kind ?? "correction") as NodeKind;
    const now = Date.now();
    const node: BrainNode = {
      id: `bn_${randomUUID().slice(0, 12)}`,
      kind: nodeKind,
      content: params.instruction,
      embedding: await this.embeddingClient(params.instruction),
      sourceUri: null,
      trust: "human",
      tags: params.tags ?? [],
      tokenCount: Math.ceil(params.instruction.length / 4),
      metadata: { taught: true },
      createdAt: now,
      updatedAt: now,
    };

    this.mutableGraph.addNode(node);
    this.store.insertNode(node);

    const recentEpisodes = this.store
      .getRecentEpisodes(10)
      .filter((episode) => (
        params.conversationId === undefined
          ? true
          : episode.conversationId === params.conversationId
      ));

    const recentEpisode = recentEpisodes[0];
    const connectedNodes = new Set<string>();
    for (const firedNodeId of recentEpisode?.firedNodes ?? []) {
      if (connectedNodes.has(firedNodeId)) {
        continue;
      }
      connectedNodes.add(firedNodeId);
      const edge = {
        source: firedNodeId,
        target: node.id,
        kind: "learned" as const,
        weight: 1.0,
        prior: 1.0,
        metadata: { taught: true, conversationId: params.conversationId ?? null },
        decayedAt: now,
        createdAt: now,
      };
      const reverse = {
        ...edge,
        source: node.id,
        target: firedNodeId,
      };
      this.mutableGraph.addEdge(edge);
      this.mutableGraph.addEdge(reverse);
      this.store.insertEdge(edge);
      this.store.insertEdge(reverse);
    }

    for (const episode of recentEpisodes.slice(0, 3)) {
      if (episode.reward === null) {
        this.store.insertLabel({
          episodeId: episode.id,
          source: "human",
          value: -0.5,
          reason: `correction taught: "${params.instruction.slice(0, 80)}"`,
        });
      }
    }

    const packVersion = await this.promoteMutableGraph("teach", {
      taughtNodeId: node.id,
      conversationId: params.conversationId ?? null,
    });
    return { nodeId: node.id, packVersion };
  }

  async status(): Promise<Record<string, unknown>> {
    const recentEpisodes = this.store.getRecentEpisodes(100);
    const currentPack = this.store.getCurrentPack();
    const health = computeHealth(
      this.mutableGraph,
      recentEpisodes,
      currentPack?.version ?? this.store.getCurrentPackVersion() ?? 0,
    );
    const recentTraces = this.store.getRecentTraces(5);

    return {
      initialized: this.initialized,
      enabled: this.isEnabled(),
      embeddingConfigured: Boolean(this.embeddingClient),
      currentPackVersion: this.store.getCurrentPackVersion(),
      currentPackPromotedAt: currentPack?.promotedAt ?? null,
      pendingLabels: this.store.getPendingLabels().length,
      recentTraceCount: recentTraces.length,
      lastTraceFooter: recentTraces[0]?.footer ?? null,
      brainRoot: this.config.root,
      ...health,
    };
  }

  async getTrace(traceId?: string): Promise<DecisionTrace | null> {
    if (traceId) {
      return this.store.getTrace(traceId);
    }
    return this.store.getRecentTraces(1)[0] ?? null;
  }

  async init(params: {
    workspaceRoot: string;
    embedFn?: BrainEmbeddingFn;
  }): Promise<string> {
    const embedFn = params.embedFn ?? this.embeddingClient;
    if (!embedFn) {
      throw new Error("OpenClawBrain init requires OPENCLAWBRAIN_EMBEDDING_MODEL or an explicit embedFn");
    }

    const result = await runInit({
      workspaceRoot: params.workspaceRoot,
      embedFn,
      semanticThreshold: this.config.semanticThreshold,
      log: { info: () => {}, warn: () => {} },
    });

    this.store.clearGraph();
    this.mutableGraph.clear();
    for (const node of result.nodes) {
      this.mutableGraph.addNode(node);
      this.store.insertNode(node);
    }
    for (const edge of result.edges) {
      this.mutableGraph.addEdge(edge);
      this.store.insertEdge(edge);
    }

    await this.promoteMutableGraph("init", {
      workspaceRoot: params.workspaceRoot,
      summary: result.summary,
    });
    return result.summary;
  }

  async harvestFromMessage(params: {
    conversationId: number;
    role: string;
    content: string;
  }): Promise<void> {
    await this.harvesterImpl.harvestFromMessage(params);
  }

  async promoteLatestCandidate(): Promise<number | null> {
    return this.promoteMutableGraph("manual-promote", {});
  }

  rollback(version: number): void {
    this.packManager.rollback(version);
    this.reloadServingGraph();
  }

  private reloadServingGraph(): void {
    const currentVersion = this.store.getCurrentPackVersion();
    const snapshot = currentVersion !== null ? this.store.readPackSnapshot(currentVersion) : null;
    if (!snapshot) {
      this.servingGraph.clear();
      this.initialized = false;
      return;
    }

    populateGraph(this.servingGraph, snapshot.nodes, snapshot.edges);
    this.initialized = true;
  }

  private async promoteMutableGraph(
    reason: string,
    metadata: Record<string, unknown>,
  ): Promise<number | null> {
    if (this.mutableGraph.nodeCount() === 0) {
      return null;
    }

    const health = computeHealth(
      this.mutableGraph,
      this.store.getRecentEpisodes(this.config.replayEpisodeCount),
      this.store.getCurrentPackVersion() ?? 0,
    );
    const pack = this.packManager.buildCandidate(health);
    this.store.writePackSnapshot({
      version: pack.version,
      nodes: this.mutableGraph.getAllNodes(),
      edges: flattenEdges(this.mutableGraph),
      metadata: {
        reason,
        ...metadata,
      },
    });
    this.packManager.promote(pack.version);
    this.reloadServingGraph();
    return pack.version;
  }
}
