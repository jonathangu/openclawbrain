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
import {
  createEmbeddingClient,
  describeEmbeddingConfig,
  type BrainEmbeddingFn,
} from "../brain-store/embedding.js";
import { LabelHarvester } from "./harvester-extension.js";
import { BrainWorker } from "../brain-worker/worker.js";
import type { LcmDependencies } from "../types.js";
import type { WorkerTeacherCompleteRequestMessage } from "../brain-worker/protocol.js";
import { flattenEdges, populateGraph, promoteGraphSnapshot, reloadGraphFromStore } from "./graph-io.js";
import { buildPromotionStory, buildWorkerPromotionSnapshotMetadata } from "./promotion-story.js";
import { readWorkerRuntimeState } from "./worker-state.js";
import { WorkerSupervisor } from "./worker-supervisor.js";
import {
  proposeUserCorrectionFast,
  proposeUserCorrectionWithModel,
  type UserMemoryObservation,
  type UserMemoryProposal,
} from "./user-memory-proposals.js";

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
  private childSupervisor: WorkerSupervisor | null = null;
  private harvesterImpl: LabelHarvester;
  private packManager: PackManager;
  private embeddingClient: BrainEmbeddingFn | null;
  private config: BrainConfig;
  private resolvedTeacherModel: { provider: string; model: string } | null;
  private teacherConfigError: string | null = null;
  private resolvedAutoUserCorrectionsModel: { provider: string; model: string } | null = null;
  private autoUserCorrectionsConfigError: string | null = null;
  private initialized = false;
  private userObservationQueue: Promise<void> = Promise.resolve();
  private pendingUserObservationCount = 0;
  private committedUserCorrectionMessageIds = new Set<number>();
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
    if (this.config.teacherEnabled) {
      try {
        this.resolvedTeacherModel = params.deps.resolveModel(
          this.config.teacherModel || undefined,
          this.config.teacherProvider || undefined,
        );
      } catch (error) {
        this.resolvedTeacherModel = null;
        this.teacherConfigError = (error as Error).message;
        params.deps.log.warn(
          `[brain] Teacher disabled: ${this.teacherConfigError}`,
        );
      }
    } else {
      this.resolvedTeacherModel = null;
    }

    if (this.config.autoUserCorrectionsEnabled) {
      try {
        this.resolvedAutoUserCorrectionsModel = params.deps.resolveModel(
          this.config.autoUserCorrectionsModel || undefined,
          this.config.autoUserCorrectionsProvider || undefined,
        );
      } catch (error) {
        this.resolvedAutoUserCorrectionsModel = null;
        this.autoUserCorrectionsConfigError = (error as Error).message;
        params.deps.log.warn(
          `[brain] Auto user corrections disabled: ${this.autoUserCorrectionsConfigError}`,
        );
      }
    } else {
      this.resolvedAutoUserCorrectionsModel = null;
    }
    mkdirSync(this.config.root, { recursive: true });

    const db = new DatabaseSync(join(this.config.root, "state.db"));
    db.exec("PRAGMA journal_mode = WAL");
    db.exec("PRAGMA busy_timeout = 5000");
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
      this.deps.log.warn("[brain] in_process worker mode is dev-only; use child mode for production operator truth");
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
          onPromotionReady: async ({ healthJson, promotionVerdict }) => {
            await this.promoteMutableGraph(
              "worker",
              buildWorkerPromotionSnapshotMetadata(this.store, { healthJson, promotionVerdict }),
            );
          },
        },
      );
    } else {
      this.worker = null;
      this.childSupervisor = new WorkerSupervisor({
        config: this.config,
        store: this.store,
        log: params.deps.log,
        teacherModel: this.resolvedTeacherModel,
        isEnabled: () => this.isEnabled(),
        onPackPromoted: () => {
          this.reloadMutableGraphFromStore();
          this.reloadServingGraph();
        },
        onTeacherComplete: async (
          message: WorkerTeacherCompleteRequestMessage,
          teacherModel,
        ) => {
          const provider = typeof message.provider === "string"
            ? message.provider
            : teacherModel?.provider;
          const model = typeof message.model === "string"
            ? message.model
            : teacherModel?.model;
          const requestId = String(message.requestId ?? "");
          if (!provider || !model || !requestId) {
            return {
              type: "teacher-complete-result",
              requestId,
              ok: false,
              error: "teacher completion request missing provider/model/requestId",
            };
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
            return {
              type: "teacher-complete-result",
              requestId,
              ok: true,
              content: result.content ?? [],
            };
          } catch (error) {
            return {
              type: "teacher-complete-result",
              requestId,
              ok: false,
              error: (error as Error).message,
            };
          }
        },
      });
    }
  }

  startWorker(): void {
    if (!this.isEnabled()) {
      return;
    }
    if (this.config.workerMode === "in_process") {
      this.store.setTrainingState("worker_mode", "in_process");
      this.store.setTrainingState("worker_status", "running");
      this.worker?.start();
      return;
    }
    this.childSupervisor?.start();
  }

  stopWorker(): void {
    if (this.config.workerMode === "in_process") {
      this.store.setTrainingState("worker_status", "stopped");
      this.worker?.stop();
      return;
    }
    this.childSupervisor?.stop();
  }

  private notifyWorkerGraphReload(): void {
    this.childSupervisor?.requestGraphReload();
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

    const queryStartedAt = Date.now();
    const embeddingStartedAt = Date.now();
    const embedding =
      params.queryEmbedding
      ?? (this.embeddingClient ? await this.embeddingClient(params.queryText) : null);
    const embeddingMs = Date.now() - embeddingStartedAt;
    if (!embedding || embedding.length === 0) {
      return null;
    }

    const routeSelectionStartedAt = Date.now();
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
    const routeSelectionMs = Date.now() - routeSelectionStartedAt;
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

    const selectedNodes = traversalResult.firedNodes
      .map((node) => this.servingGraph.getNode(node.nodeId))
      .filter((node): node is BrainNode => !!node);
    const trace = recordTrace({
      traversalResult,
      queryText: params.queryText,
      episodeId: episode.id,
      conversationId: params.conversationId,
      packVersion: episode.packVersion,
      budgetChars: params.budgetChars,
      maxHops: this.config.maxHops,
      embeddingMs,
      routeSelectionMs,
      totalQueryMs: Date.now() - queryStartedAt,
      queryEmbeddingSource: params.queryEmbedding ? "provided" : "runtime",
      selectedNodes,
    });
    this.store.insertTrace(trace);

    return {
      fired: traversalResult.firedNodes,
      vetoed: traversalResult.vetoedNodes,
      episode,
      trace,
    };
  }

  async teachUserCorrection(params: {
    canonicalInstruction: string;
    sourceQuote: string;
    conversationId?: number;
    sourceMessageId?: number;
    tags?: string[];
    metadata?: Record<string, unknown>;
    via?: string;
  }): Promise<{ nodeId: string; packVersion: number | null }> {
    return this.teach({
      instruction: params.canonicalInstruction,
      conversationId: params.conversationId,
      kind: "correction",
      tags: params.tags,
      metadata: {
        sourceAuthority: "user_explicit",
        sourceQuote: params.sourceQuote,
        ...(typeof params.sourceMessageId === "number" ? { sourceMessageId: params.sourceMessageId } : {}),
        ...(params.metadata ?? {}),
      },
      via: params.via ?? "brain_teach_user_correction",
    });
  }

  private shouldRunAutoUserCorrectionProposal(): boolean {
    return this.config.autoUserCorrectionsEnabled && !!this.resolvedAutoUserCorrectionsModel;
  }

  private hasCommittedUserCorrectionForMessage(messageId: number): boolean {
    if (this.committedUserCorrectionMessageIds.has(messageId)) {
      return true;
    }
    return this.store.getAllNodes().some((node) => node.metadata?.sourceMessageId === messageId);
  }

  private async commitObservedUserCorrection(params: {
    observation: UserMemoryObservation;
    proposal: Extract<UserMemoryProposal, { kind: "explicit_correction" }>;
    via: string;
    extraMetadata?: Record<string, unknown>;
  }): Promise<{ nodeId: string; packVersion: number | null } | null> {
    if (this.hasCommittedUserCorrectionForMessage(params.observation.messageId)) {
      return null;
    }

    const committed = await this.teachUserCorrection({
      canonicalInstruction: params.proposal.canonicalInstruction,
      sourceQuote: params.observation.userText,
      conversationId: params.observation.conversationId,
      sourceMessageId: params.observation.messageId,
      tags: ["user-correction", "auto"],
      metadata: {
        proposalConfidence: params.proposal.confidence,
        proposalReason: params.proposal.reason,
        ...(params.extraMetadata ?? {}),
      },
      via: params.via,
    });
    this.committedUserCorrectionMessageIds.add(params.observation.messageId);
    return committed;
  }

  private enqueueUserObservation(observation: UserMemoryObservation): void {
    this.pendingUserObservationCount += 1;
    this.userObservationQueue = this.userObservationQueue
      .catch(() => {})
      .then(async () => {
        try {
          if (this.hasCommittedUserCorrectionForMessage(observation.messageId)) {
            return;
          }
          const model = this.resolvedAutoUserCorrectionsModel;
          if (!model) {
            return;
          }
          const apiKey = await this.deps.getApiKey(model.provider, model.model);
          const proposal = await proposeUserCorrectionWithModel({
            complete: this.deps.complete,
            provider: model.provider,
            model: model.model,
            apiKey,
            observation,
          });
          if (proposal.kind !== "explicit_correction") {
            return;
          }
          if (proposal.confidence < this.config.autoUserCorrectionsMinConfidence) {
            return;
          }
          await this.commitObservedUserCorrection({
            observation,
            proposal,
            via: "brain_auto_user_correction_async",
            extraMetadata: { proposalLane: "async_model" },
          });
        } catch (error) {
          this.deps.log.warn(`[brain] Auto user correction proposal failed: ${(error as Error).message}`);
        } finally {
          this.pendingUserObservationCount = Math.max(0, this.pendingUserObservationCount - 1);
        }
      });
  }

  async observeUserTurn(observation: UserMemoryObservation): Promise<void> {
    if (!this.embeddingClient) {
      return;
    }
    if (this.hasCommittedUserCorrectionForMessage(observation.messageId)) {
      return;
    }

    const fastProposal = proposeUserCorrectionFast(observation);
    if (fastProposal.kind === "explicit_correction") {
      await this.commitObservedUserCorrection({
        observation,
        proposal: fastProposal,
        via: "brain_auto_user_correction_fast",
        extraMetadata: { proposalLane: "fast_deterministic" },
      });
    }

    if (!this.shouldRunAutoUserCorrectionProposal()) {
      return;
    }
    if (this.hasCommittedUserCorrectionForMessage(observation.messageId)) {
      return;
    }
    this.enqueueUserObservation(observation);
  }

  async teach(params: {
    instruction: string;
    conversationId?: number;
    kind?: string;
    tags?: string[];
    metadata?: Record<string, unknown>;
    via?: string;
  }): Promise<{ nodeId: string; packVersion: number | null }> {
    this.reloadMutableGraphFromStore();
    if (!this.embeddingClient) {
      throw new Error("Embedding model is required before brain_teach can make knowledge retrievable");
    }

    const teachVia = typeof params.via === "string" && params.via.trim().length > 0
      ? params.via.trim()
      : undefined;
    const provenanceMetadata = params.metadata ?? {};
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
      metadata: {
        taught: true,
        ...provenanceMetadata,
        ...(teachVia ? { via: teachVia } : {}),
      },
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
        const matchedTrace = this.store.getTraceForEpisode(episode.id);
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
            ...provenanceMetadata,
            taughtNodeId: node.id,
            correctedEpisodeId: episode.id,
            extractor: teachVia ?? "brain_teach",
            via: teachVia ?? "brain_teach",
            traceId: matchedTrace?.id ?? null,
            tracePackVersion: matchedTrace?.packVersion ?? null,
            traceRequestDigest: matchedTrace?.routeTrace?.requestDigest ?? null,
            traceSelectedNodeIds: matchedTrace?.routeTrace?.selectedNodeIds ?? matchedTrace?.firedNodes ?? [],
            traceSelectedPathNodeIds: matchedTrace?.routeTrace?.selectedPathNodeIds ?? [],
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
    const workerState = readWorkerRuntimeState(this.store, this.config);
    const promotionStory = buildPromotionStory(this.store);
    const routeTraceCount = this.store.countTraces();
    const supervisionCount = this.store.countTraceSupervision();
    const lastPgCandidateUpdate = this.store.getTrainingStateJson("last_pg_candidate_update_json");
    const lastPgCandidatePackVersionRaw = this.store.getTrainingState("last_pg_candidate_pack_version");
    const lastPgCandidatePackVersion = lastPgCandidatePackVersionRaw
      ? Number.parseInt(lastPgCandidatePackVersionRaw, 10)
      : null;

    const embeddingConfig = describeEmbeddingConfig(this.config);

    return {
      initialized: this.initialized,
      enabled: this.isEnabled(),
      embeddingConfigured: Boolean(this.embeddingClient),
      embeddingProvider: this.config.embeddingProvider,
      embeddingModel: this.config.embeddingModel,
      embeddingBaseUrl: this.config.embeddingModel ? embeddingConfig.baseUrl : "",
      embeddingAuthMode: embeddingConfig.authMode,
      embeddingConfigError: embeddingConfig.error,
      currentPackVersion: this.store.getCurrentPackVersion(),
      currentPackPromotedAt: currentPack?.promotedAt ?? null,
      currentPackMetadata: promotionStory.currentPack?.metadata ?? null,
      shadowMode: this.config.shadowMode,
      teacherEnabled: this.config.teacherEnabled,
      teacherConfigured: Boolean(this.resolvedTeacherModel),
      teacherProvider: this.resolvedTeacherModel?.provider ?? this.config.teacherProvider,
      teacherModel: this.resolvedTeacherModel?.model ?? this.config.teacherModel,
      teacherConfigError: this.teacherConfigError,
      autoUserCorrectionsEnabled: this.config.autoUserCorrectionsEnabled,
      autoUserCorrectionsConfigured: Boolean(this.resolvedAutoUserCorrectionsModel),
      autoUserCorrectionsProvider:
        this.resolvedAutoUserCorrectionsModel?.provider ?? this.config.autoUserCorrectionsProvider,
      autoUserCorrectionsModel:
        this.resolvedAutoUserCorrectionsModel?.model ?? this.config.autoUserCorrectionsModel,
      autoUserCorrectionsMinConfidence: this.config.autoUserCorrectionsMinConfidence,
      autoUserCorrectionsConfigError: this.autoUserCorrectionsConfigError,
      pendingUserObservationCount: this.pendingUserObservationCount,
      ...workerState,
      pendingEvidence: this.store.getPendingEvidence(100).length,
      pendingEvidenceBySource: this.store.countPendingEvidenceBySource(),
      pendingLabels: this.store.getPendingLabels().length,
      pendingLabelsBySource: this.store.countPendingLabelsBySource(),
      mutationBacklog: this.store.countMutationsByStatus(),
      recentMutationBundles: this.store.getRecentMutationBundles(5),
      seedLearningEnabled: this.mutableGraph.hasSeedWeights(),
      routeTraceCount,
      supervisionCount,
      lastPgCandidatePackVersion: Number.isFinite(lastPgCandidatePackVersion ?? NaN)
        ? lastPgCandidatePackVersion
        : null,
      lastPgCandidateUpdate,
      recentTraceCount: recentTraces.length,
      lastTraceFooter: recentTraces[0]?.footer ?? null,
      lastAssemblyDecision: this.lastAssemblyDecision,
      lastPromotionReason: this.store.getTrainingState("last_promotion_reason"),
      lastPromotionVerdict: this.store.getTrainingStateJson("last_promotion_verdict_json"),
      lastReplayFailureReason: this.store.getTrainingState("last_replay_failure_reason"),
      lastReplayGateVerdict: this.store.getTrainingStateJson("last_replay_gate_verdict_json"),
      promotionStory,
      brainRoot: this.config.root,
      ...health,
    };
  }

  async getTrace(traceId?: string): Promise<DecisionTrace | null> {
    const trace = traceId
      ? this.store.getTrace(traceId)
      : this.store.getRecentTraces(1)[0] ?? null;
    if (!trace) {
      return null;
    }
    return {
      ...trace,
      supervision: this.store.getTraceSupervision(trace.id, 20),
    };
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
    messageId?: number;
    episodeId?: string;
    role: string;
    content: string;
    messageParts?: Array<{
      partType: string;
      ordinal?: number;
      textContent?: string | null;
      toolCallId?: string | null;
      toolName?: string | null;
      toolInput?: string | null;
      toolOutput?: string | null;
      metadata?: string | null;
    }>;
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
