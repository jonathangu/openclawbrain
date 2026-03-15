import { randomUUID } from "node:crypto";
import { fork, type ChildProcess } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { fileURLToPath } from "node:url";
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
import type { CompletionContentBlock, LcmDependencies } from "../types.js";
import { flattenEdges, populateGraph, promoteGraphSnapshot, reloadGraphFromStore } from "./graph-io.js";

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

export class BrainService {
  private deps: LcmDependencies;
  private store: BrainStore;
  private mutableGraph = new BrainGraph();
  private servingGraph = new BrainGraph();
  private worker: BrainWorker | null;
  private workerChild: ChildProcess | null = null;
  private workerShouldRun = false;
  private workerRestartTimer: ReturnType<typeof setTimeout> | null = null;
  private workerLastHeartbeatAt: number | null = null;
  private workerLastReadyAt: number | null = null;
  private workerLastExit:
    | { code: number | null; signal: NodeJS.Signals | null; at: number }
    | null = null;
  private harvesterImpl: LabelHarvester;
  private packManager: PackManager;
  private embeddingClient: BrainEmbeddingFn | null;
  private config: BrainConfig;
  private resolvedTeacherModel: { provider: string; model: string } | null;
  private initialized = false;
  private latestEpisodeByConversation = new Map<number, string>();
  private lastAssemblyDecision:
    | {
        mode: "use_brain" | "shadow" | "skip_no_query" | "skip_short_static_lookup" | "skip_no_embedding" | "skip_uninitialized" | "skip_budget_too_small";
        conversationId?: number;
        episodeId?: string | null;
        traceId?: string | null;
        footer?: string | null;
      }
    | null = null;

  constructor(params: {
    deps: LcmDependencies;
    config?: Partial<BrainConfig>;
    runtimeConfig?: OpenClawBrainRuntimeConfig;
  }) {
    this.deps = params.deps;
    const runtimeConfig = params.runtimeConfig ?? params.deps.config.brain;
    if (!runtimeConfig) {
      throw new Error("OpenClawBrain runtime configuration is missing");
    }

    this.config = buildBrainConfig(runtimeConfig, params.config);
    this.resolvedTeacherModel = this.config.teacherEnabled
      ? params.deps.resolveModel(
          this.config.teacherModel || undefined,
          this.config.teacherProvider || undefined,
        )
      : null;
    mkdirSync(this.config.root, { recursive: true });

    const db = new DatabaseSync(join(this.config.root, "state.db"));
    db.exec("PRAGMA journal_mode = WAL");
    db.exec("PRAGMA foreign_keys = ON");
    runBrainMigrations(db);

    this.store = new BrainStore(db, { brainRoot: this.config.root });
    this.harvesterImpl = new LabelHarvester(
      this.store,
      params.deps.log,
      (conversationId) => this.latestEpisodeByConversation.get(conversationId) ?? null,
    );
    this.embeddingClient = createEmbeddingClient({
      config: runtimeConfig,
      getApiKey: (provider, model) => params.deps.getApiKey(provider, model),
      log: params.deps.log,
    });

    populateGraph(this.mutableGraph, this.store.getAllNodes(), this.store.loadAllEdges(), this.store.loadAllSeedWeights());
    this.reloadServingGraph();

    const persistence = {
      insertNode: (node: Parameters<BrainStore["insertNode"]>[0]) => this.store.insertNode(node),
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

    if (this.config.workerMode === "in_process") {
      const teacher =
        this.config.teacherEnabled && this.resolvedTeacherModel
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
              () => this.resolvedTeacherModel as { provider: string; model: string },
              (provider, model) => params.deps.getApiKey(provider, model),
              this.mutableGraph,
              params.deps.log,
            )
          : null;

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
    } else {
      this.worker = null;
    }
  }

  startWorker(): void {
    if (!this.isEnabled()) {
      return;
    }
    this.workerShouldRun = true;
    if (this.config.workerMode === "in_process") {
      this.store.setTrainingState("worker_mode", "in_process");
      this.store.setTrainingState("worker_status", "running");
      this.worker?.start();
      return;
    }
    this.ensureChildWorker();
  }

  stopWorker(): void {
    this.workerShouldRun = false;
    if (this.workerRestartTimer) {
      clearTimeout(this.workerRestartTimer);
      this.workerRestartTimer = null;
    }
    if (this.config.workerMode === "in_process") {
      this.store.setTrainingState("worker_status", "stopped");
      this.worker?.stop();
      return;
    }
    if (this.workerChild) {
      this.workerChild.send({ type: "shutdown" });
      const child = this.workerChild;
      setTimeout(() => {
        if (this.workerChild === child) {
          this.workerChild.kill("SIGTERM");
        }
      }, 2_000);
    }
  }

  private ensureChildWorker(): void {
    if (this.workerChild || !this.isEnabled()) {
      return;
    }

    const child = fork(
      fileURLToPath(new URL("../brain-worker/child-runner.ts", import.meta.url)),
      [],
      {
        execArgv: ["--import", "tsx/esm"],
        stdio: ["ignore", "pipe", "pipe", "ipc"],
        env: {
          ...process.env,
          OPENCLAWBRAIN_CHILD_CONFIG_JSON: JSON.stringify(this.config),
          OPENCLAWBRAIN_CHILD_TEACHER_MODEL_JSON: this.resolvedTeacherModel
            ? JSON.stringify(this.resolvedTeacherModel)
            : "",
        },
      },
    );
    this.workerChild = child;
    this.store.setTrainingState("worker_mode", "child");

    child.stdout?.on("data", (chunk) => {
      const text = String(chunk).trim();
      if (text) {
        this.deps.log.info(text);
      }
    });
    child.stderr?.on("data", (chunk) => {
      const text = String(chunk).trim();
      if (text) {
        this.deps.log.warn(text);
      }
    });
    child.on("message", (message) => {
      void this.handleChildWorkerMessage(message as Record<string, unknown>, child);
    });
    child.on("exit", (code, signal) => {
      this.workerLastExit = {
        code,
        signal: signal as NodeJS.Signals | null,
        at: Date.now(),
      };
      if (this.workerChild === child) {
        this.workerChild = null;
      }
      if (this.workerShouldRun && this.isEnabled()) {
        this.workerRestartTimer = setTimeout(() => {
          this.workerRestartTimer = null;
          this.ensureChildWorker();
        }, this.config.workerRestartDelayMs);
      }
    });
  }

  private async handleChildWorkerMessage(message: Record<string, unknown>, child: ChildProcess): Promise<void> {
    switch (message.type) {
      case "worker-ready": {
        this.workerLastReadyAt = Date.now();
        this.workerLastHeartbeatAt = Date.now();
        return;
      }
      case "worker-heartbeat": {
        const at = Number(message.at ?? Date.now());
        this.workerLastHeartbeatAt = at;
        return;
      }
      case "pack-promoted": {
        this.reloadMutableGraphFromStore();
        this.reloadServingGraph();
        return;
      }
      case "teacher-complete": {
        const provider = typeof message.provider === "string"
          ? message.provider
          : this.resolvedTeacherModel?.provider;
        const model = typeof message.model === "string"
          ? message.model
          : this.resolvedTeacherModel?.model;
        const requestId = String(message.requestId ?? "");
        if (!provider || !model || !requestId) {
          child.send?.({
            type: "teacher-complete-result",
            requestId,
            ok: false,
            error: "teacher completion request missing provider/model/requestId",
          });
          return;
        }
        try {
          const apiKey = await this.deps.getApiKey(provider, model);
          const result = await this.deps.complete({
            provider,
            model,
            apiKey,
            messages: Array.isArray(message.messages)
              ? message.messages as Array<{ role: string; content: unknown }>
              : [],
            system: typeof message.system === "string" ? message.system : undefined,
            maxTokens: Number(message.maxTokens ?? 200),
            temperature: typeof message.temperature === "number" ? message.temperature : undefined,
          });
          child.send?.({
            type: "teacher-complete-result",
            requestId,
            ok: true,
            content: (result.content ?? []) as CompletionContentBlock[],
          });
        } catch (error) {
          child.send?.({
            type: "teacher-complete-result",
            requestId,
            ok: false,
            error: (error as Error).message,
          });
        }
        return;
      }
      case "worker-error": {
        this.deps.log.error(`[brain] child worker error: ${String(message.error ?? "unknown error")}`);
        return;
      }
      default:
        return;
    }
  }

  private notifyWorkerGraphReload(): void {
    if (this.workerChild) {
      this.workerChild.send({ type: "reload-graph" });
    }
  }

  private reloadMutableGraphFromStore(): void {
    reloadGraphFromStore(this.store, this.mutableGraph);
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

  isEmbeddingConfigured(): boolean {
    return Boolean(this.embeddingClient);
  }

  isShadowMode(): boolean {
    return this.config.shadowMode;
  }

  noteAssemblyDecision(decision: NonNullable<BrainService["lastAssemblyDecision"]>): void {
    this.lastAssemblyDecision = decision;
    this.store.setTrainingState("last_assembly_mode", decision.mode);
    this.store.setTrainingState("last_assembly_footer", decision.footer ?? "");
    this.store.setTrainingState("last_assembly_episode_id", decision.episodeId ?? "");
    this.store.setTrainingState("last_assembly_trace_id", decision.traceId ?? "");
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
    this.latestEpisodeByConversation.set(params.conversationId, episode.id);

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
    this.reloadMutableGraphFromStore();
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
    const exactEpisode =
      typeof params.conversationId === "number"
        ? this.store.getEpisode(this.latestEpisodeByConversation.get(params.conversationId) ?? "")
        : null;
    const recentEpisode = exactEpisode ?? recentEpisodes[0] ?? null;
    const connectedNodes = new Set<string>();
    const firstTraversalStep = recentEpisode?.trajectory.find(
      (step) => step.chosenAction.type === "traverse",
    );
    const chosenSeedNodeId =
      firstTraversalStep?.chosenAction.type === "traverse"
        ? firstTraversalStep.chosenAction.targetNodeId
        : null;
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
    if (chosenSeedNodeId && !connectedNodes.has(chosenSeedNodeId)) {
      const now = Date.now();
      const seedEdge = {
        source: chosenSeedNodeId,
        target: node.id,
        kind: "learned" as const,
        weight: 1.0,
        prior: 1.0,
        metadata: { taught: true, seedRegion: true, conversationId: params.conversationId ?? null },
        decayedAt: now,
        createdAt: now,
      };
      const reverseSeedEdge = {
        ...seedEdge,
        source: node.id,
        target: chosenSeedNodeId,
      };
      this.mutableGraph.addEdge(seedEdge);
      this.mutableGraph.addEdge(reverseSeedEdge);
      this.store.insertEdge(seedEdge);
      this.store.insertEdge(reverseSeedEdge);
    }

    const misroutedTargetId = recentEpisode?.firedNodes.at(-1) ?? null;
    if (chosenSeedNodeId && misroutedTargetId && misroutedTargetId !== node.id) {
      const inhibitoryEdge = {
        source: chosenSeedNodeId,
        target: misroutedTargetId,
        kind: "inhibitory" as const,
        weight: -1.0,
        prior: -1.0,
        metadata: { taught: true, reason: "human correction", conversationId: params.conversationId ?? null },
        decayedAt: Date.now(),
        createdAt: Date.now(),
      };
      this.mutableGraph.addEdge(inhibitoryEdge);
      this.store.insertEdge(inhibitoryEdge);
    }

    const targetEpisodes = exactEpisode ? [exactEpisode] : recentEpisodes.slice(0, 1);
    for (const episode of targetEpisodes) {
      if (episode && episode.reward === null) {
        const reason = `correction taught: "${params.instruction.slice(0, 80)}"`;
        this.store.insertEvidence({
          episodeId: episode.id,
          conversationId: episode.conversationId,
          source: "human",
          kind: "teach_correction",
          value: -0.5,
          confidence: 1.0,
          reason,
          contentSnippet: params.instruction.slice(0, 240),
          metadata: {
            taughtNodeId: node.id,
            correctedEpisodeId: episode.id,
            extractor: "brain_teach",
            via: "brain_teach",
          },
        });
        this.store.insertLabel({
          episodeId: episode.id,
          source: "human",
          value: -0.5,
          reason,
        });
      }
    }

    const packVersion = await this.promoteMutableGraph("teach", {
      taughtNodeId: node.id,
      conversationId: params.conversationId ?? null,
    });
    this.notifyWorkerGraphReload();
    return { nodeId: node.id, packVersion };
  }

  async status(): Promise<Record<string, unknown>> {
    this.reloadMutableGraphFromStore();
    const recentEpisodes = this.store.getRecentEpisodes(100);
    const currentPack = this.store.getCurrentPack();
    const health = computeHealth(
      this.mutableGraph,
      recentEpisodes,
      currentPack?.version ?? this.store.getCurrentPackVersion() ?? 0,
    );
    const recentTraces = this.store.getRecentTraces(5);

    const workerPid = Number.parseInt(this.store.getTrainingState("worker_pid") ?? "0", 10) || null;
    const workerHeartbeatAt = Number.parseInt(this.store.getTrainingState("worker_last_heartbeat_at") ?? "0", 10) || this.workerLastHeartbeatAt;
    const workerStatus = this.store.getTrainingState("worker_status") ?? (this.config.workerMode === "child" ? "unknown" : "running");

    return {
      initialized: this.initialized,
      enabled: this.isEnabled(),
      embeddingConfigured: Boolean(this.embeddingClient),
      currentPackVersion: this.store.getCurrentPackVersion(),
      currentPackPromotedAt: currentPack?.promotedAt ?? null,
      shadowMode: this.config.shadowMode,
      workerMode: this.config.workerMode,
      workerPid,
      workerStatus,
      workerLastHeartbeatAt: workerHeartbeatAt,
      workerLastReadyAt: this.workerLastReadyAt,
      workerHealthy: this.config.workerMode === "child"
        ? Boolean(workerHeartbeatAt && (Date.now() - workerHeartbeatAt) < this.config.workerHeartbeatTimeoutMs)
        : true,
      workerLastExit: this.workerLastExit,
      pendingEvidence: this.store.getPendingEvidence(100).length,
      pendingEvidenceBySource: this.store.countPendingEvidenceBySource(),
      pendingLabels: this.store.getPendingLabels().length,
      pendingLabelsBySource: this.store.countPendingLabelsBySource(),
      mutationBacklog: this.store.countMutationsByStatus(),
      seedLearningEnabled: this.mutableGraph.hasSeedWeights(),
      recentTraceCount: recentTraces.length,
      lastTraceFooter: recentTraces[0]?.footer ?? null,
      lastAssemblyDecision: this.lastAssemblyDecision,
      lastPromotionReason: this.store.getTrainingState("last_promotion_reason"),
      lastReplayFailureReason: this.store.getTrainingState("last_replay_failure_reason"),
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
    this.notifyWorkerGraphReload();
    return result.summary;
  }

  async harvestFromMessage(params: {
    conversationId: number;
    episodeId?: string;
    role: string;
    content: string;
  }): Promise<void> {
    await this.harvesterImpl.harvestFromMessage(params);
  }

  async promoteLatestCandidate(): Promise<number | null> {
    this.reloadMutableGraphFromStore();
    const version = await this.promoteMutableGraph("manual-promote", {});
    this.notifyWorkerGraphReload();
    return version;
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

    populateGraph(this.servingGraph, snapshot.nodes, snapshot.edges, snapshot.seedWeights);
    this.initialized = true;
  }

  private async promoteMutableGraph(
    reason: string,
    metadata: Record<string, unknown>,
  ): Promise<number | null> {
    this.reloadMutableGraphFromStore();
    const version = promoteGraphSnapshot({
      store: this.store,
      graph: this.mutableGraph,
      packManager: this.packManager,
      config: this.config,
      reason,
      metadata,
    });
    this.reloadServingGraph();
    return version;
  }
}
