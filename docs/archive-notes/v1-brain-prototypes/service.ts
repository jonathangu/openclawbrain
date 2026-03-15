/**
 * BrainService: the orchestrator.
 *
 * Wires together all brain components and exposes the API surface
 * used by index.ts (plugin registration), assembler.ts (context injection),
 * and engine.ts (label harvesting).
 */

import type { DatabaseSync } from "node:sqlite";
import { randomUUID } from "node:crypto";
import type {
  BrainConfig,
  BrainNode,
  NodeKind,
  TraversalResult,
  HealthMetrics,
  DecisionTrace,
} from "./types.js";
import { DEFAULT_BRAIN_CONFIG } from "./types.js";
import { BrainGraph } from "./graph.js";
import { BrainStore } from "./store.js";
import { runBrainMigrations } from "./migration.js";
import { traverse } from "./traverse.js";
import { recordEpisode } from "./episode.js";
import { recordTrace } from "./trace.js";
import { computeHealth } from "./health.js";
import { LabelHarvester } from "./harvester.js";
import { BrainTeacher } from "./teacher.js";
import { BrainTrainer } from "./trainer.js";
import { BrainMutator } from "./mutator.js";
import { PackManager } from "./pack.js";
import { initBrain as runInit } from "./ingest.js";
import type { LcmDependencies } from "../types.js";

export class BrainService {
  private store: BrainStore;
  private graph: BrainGraph;
  private trainer: BrainTrainer;
  private _harvester: LabelHarvester;
  private packManager: PackManager;
  private config: BrainConfig;
  private initialized = false;

  constructor(params: {
    db: DatabaseSync;
    deps: LcmDependencies;
    config?: Partial<BrainConfig>;
  }) {
    const { db, deps } = params;
    this.config = { ...DEFAULT_BRAIN_CONFIG, ...params.config };

    // Run migrations
    runBrainMigrations(db);

    // Core components
    this.store = new BrainStore(db);
    this.graph = new BrainGraph();
    this._harvester = new LabelHarvester(this.store, deps.log);

    // Load existing graph from store
    const nodes = this.store.getAllNodes();
    for (const node of nodes) {
      this.graph.addNode(node);
    }
    const edges = this.store.loadAllEdges();
    for (const edge of edges) {
      this.graph.addEdge(edge);
    }

    if (nodes.length > 0) {
      this.initialized = true;
      deps.log.info(`[brain] Loaded graph: ${nodes.length} nodes, ${edges.length} edges`);
    }

    // Teacher (needs LLM access)
    const teacher = this.config.teacherEnabled
      ? new BrainTeacher(deps.complete, deps.resolveModel, deps.getApiKey, this.graph, deps.log)
      : null;

    // Mutation + Pack
    const mutator = new BrainMutator(this.store, this.graph, deps.log);
    this.packManager = new PackManager(this.store, this.graph, deps.log);

    // Trainer
    this.trainer = new BrainTrainer(
      this.store, this.graph, teacher, mutator, this.packManager,
      this.config, deps.log,
    );
  }

  // ─── Lifecycle ───

  startTrainer(): void {
    if (!this.config.enabled) return;
    this.trainer.start();
  }

  stopTrainer(): void {
    this.trainer.stop();
  }

  get harvester(): LabelHarvester {
    return this._harvester;
  }

  // ─── Query (called from assembler) ───

  async query(queryText: string, budgetChars: number, queryEmbedding?: Float32Array): Promise<TraversalResult | null> {
    if (!this.config.enabled || !this.initialized) return null;
    if (this.graph.nodeCount() === 0) return null;

    // Use provided embedding or create a zero embedding (degraded mode)
    const embedding = queryEmbedding ?? new Float32Array(0);
    if (embedding.length === 0) return null;

    const traversalResult = traverse({
      graph: this.graph,
      queryEmbedding: embedding,
      queryText,
      maxHops: this.config.maxHops,
      budgetChars,
      temperature: this.config.servingTemperature,
      maxSeeds: this.config.maxSeeds,
      semanticThreshold: this.config.semanticThreshold,
    });

    if (traversalResult.firedNodes.length === 0) return null;

    // Record episode
    const episode = recordEpisode({
      traversalResult,
      queryText,
      queryEmbedding: embedding,
      conversationId: null,
      packVersion: this.store.getCurrentPack()?.version ?? null,
    });
    this.store.insertEpisode(episode);

    // Record trace
    const trace = recordTrace({
      traversalResult,
      queryText,
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

  // ─── Teach (called from brain_teach tool) ───

  async teach(instruction: string, kind?: string, tags?: string[]): Promise<{ nodeId: string }> {
    const nodeKind = (kind ?? "correction") as NodeKind;
    const now = Date.now();
    const node: BrainNode = {
      id: `bn_${randomUUID().slice(0, 12)}`,
      kind: nodeKind,
      content: instruction,
      embedding: null, // Will be computed by trainer or on next init
      sourceUri: null,
      trust: "human",
      tags: tags ?? [],
      tokenCount: Math.ceil(instruction.length / 4),
      metadata: { taught: true },
      createdAt: now,
      updatedAt: now,
    };

    this.graph.addNode(node);
    this.store.insertNode(node);

    // Label recent episodes as negative (this correction implies previous context was wrong)
    const recentEpisodes = this.store.getRecentEpisodes(3);
    for (const ep of recentEpisodes) {
      if (ep.reward === null) {
        this.store.insertLabel({
          episodeId: ep.id,
          source: "human",
          value: -0.5,
          reason: `correction taught: "${instruction.slice(0, 50)}..."`,
        });
      }
    }

    return { nodeId: node.id };
  }

  // ─── Status (called from brain_status tool) ───

  async status(): Promise<Record<string, unknown>> {
    const recentEpisodes = this.store.getRecentEpisodes(100);
    const currentPack = this.store.getCurrentPack();
    const health = computeHealth(
      this.graph,
      recentEpisodes,
      currentPack?.version ?? 0,
    );
    const pendingLabels = this.store.getPendingLabels();
    const recentTraces = this.store.getRecentTraces(5);

    return {
      initialized: this.initialized,
      enabled: this.config.enabled,
      ...health,
      pendingLabels: pendingLabels.length,
      currentPackVersion: currentPack?.version ?? null,
      currentPackPromotedAt: currentPack?.promotedAt ?? null,
      recentTraceCount: recentTraces.length,
      lastTraceFooter: recentTraces[0]?.footer ?? null,
    };
  }

  // ─── Trace (called from brain_trace tool) ───

  async getTrace(traceId?: string): Promise<DecisionTrace | null> {
    if (traceId) {
      return this.store.getTrace(traceId);
    }
    const recent = this.store.getRecentTraces(1);
    return recent[0] ?? null;
  }

  // ─── Init (called during brain initialization) ───

  async init(params: {
    workspaceRoot: string;
    embedFn: (text: string) => Promise<Float32Array>;
  }): Promise<string> {
    const result = await runInit({
      workspaceRoot: params.workspaceRoot,
      embedFn: params.embedFn,
      semanticThreshold: this.config.semanticThreshold,
      log: { info: () => {}, warn: () => {} }, // Quiet during init
    });

    // Persist to store
    for (const node of result.nodes) {
      this.graph.addNode(node);
      this.store.insertNode(node);
    }
    for (const edge of result.edges) {
      this.graph.addEdge(edge);
      this.store.insertEdge(edge);
    }

    // Build and promote pack v0
    const health = computeHealth(this.graph, [], 0);
    const pack = this.packManager.buildCandidate(health);
    this.packManager.promote(pack.version);

    this.initialized = true;
    return result.summary;
  }
}
