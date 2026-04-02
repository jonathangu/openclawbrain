import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { OpenClawBrainRuntimeConfig } from "../db/config.js";
import type {
  BrainAgentIdentity,
  BrainConfig,
  BrainCompileReportV1,
  ContextFeedbackSummary,
  ContextUsefulnessSummary,
  BrainDropReason,
  BrainDropStage,
  BrainFitStrategy,
  BrainFittingDropReason,
  BrainInterruptionMetadata,
  BrainNode,
  BrainObservationBindingMode,
  BrainObservationRouteMetadata,
  BrainObservationToolResult,
  BrainPrefetchBudgetClass,
  BrainPrefetchDecision,
  BrainPrefetchState,
  DecisionTraceBranchOutcomeSummary,
  DecisionRouteTrace,
  DecisionTrace,
  InterruptionAccounting,
  MutationBundleRecord,
  NodeKind,
  RecentPrefetchSummary,
  ReplayGateVerdict,
  TraversalResult,
} from "../brain-core/types.js";
import { DEFAULT_BRAIN_CONFIG, resolveObservationBindingMode } from "../brain-core/types.js";
import { BrainGraph } from "../brain-core/graph.js";
import { traverse } from "../brain-core/traverse.js";
import type { TraverseResult } from "../brain-core/traverse.js";
import { recordEpisode } from "../brain-core/episode.js";
import { buildBrainCompileReport, recordTrace, redactDecisionTrace, redactInjectedNodeSummary, redactRouteTrace, redactTextSurface, redactToolResult, rewriteBrainCompileReportSummary, summarizeRecentPrefetchDecisions } from "../brain-core/trace.js";
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
import { BrainWorker } from "../brain-worker/worker.js";
import type { LcmDependencies } from "../types.js";
import type { WorkerTeacherCompleteRequestMessage } from "../brain-worker/protocol.js";
import { flattenEdges, populateGraph, promoteGraphSnapshot, reloadGraphFromStore } from "./graph-io.js";
import { buildPromotionStory, buildWorkerPromotionSnapshotMetadata } from "./promotion-story.js";
import { readWorkerRuntimeState } from "./worker-state.js";
import { WorkerSupervisor } from "./worker-supervisor.js";
import { isSystemMessage } from "../brain-harvest/system-filter.js";
import { buildContextManagementModel } from "../context-management-model.js";
import { summarizeAttributionTruth, summarizeOperatorHealth } from "../live-runtime-audit.js";
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

function cloneObservationServedArtifact(
  artifact: BrainObservationRouteMetadata["servedArtifact"] | undefined,
): BrainObservationRouteMetadata["servedArtifact"] {
  return artifact ? JSON.parse(JSON.stringify(artifact)) as BrainObservationRouteMetadata["servedArtifact"] : null;
}

function cloneBranchOutcomeSummary(
  summary: DecisionTraceBranchOutcomeSummary | null | undefined,
): DecisionTraceBranchOutcomeSummary | null {
  return summary
    ? {
        ...summary,
        terminationReasons: summary.terminationReasons ? { ...summary.terminationReasons } : null,
      }
    : null;
}

function cloneInterruptionAccounting(
  accounting: InterruptionAccounting | null | undefined,
): InterruptionAccounting | null {
  return accounting
    ? {
        ...accounting,
        droppedFrontierNodeIds: [...accounting.droppedFrontierNodeIds],
        droppedProposalNodeIds: [...accounting.droppedProposalNodeIds],
        droppedProposalReasons: { ...accounting.droppedProposalReasons },
      }
    : null;
}

type BrainAssemblyDecisionMode =
  | "use_brain"
  | "shadow"
  | "skip_no_query"
  | "skip_short_static_lookup"
  | "skip_no_embedding"
  | "skip_uninitialized"
  | "skip_budget_too_small"
  | "skip_query_returned_no_nodes"
  | "skip_deadline_before_query"
  | "skip_deadline_after_query"
  | "skip_deadline_before_injection"
  | "partial_query_interruption"
  | "partial_deadline_after_query"
  | "partial_deadline_before_injection";

type BrainAssemblyDecisionSelectionSurface = Pick<
  DecisionRouteTrace["selectionMetadata"],
  | "interruption"
  | "queryInterrupted"
  | "interruptionStage"
  | "interruptionReason"
  | "servedPartial"
  | "compileElapsedMs"
  | "compileDeadlineMs"
  | "compileDeadlineHit"
  | "brainDropReason"
  | "brainDropStage"
  | "chosenStopCount"
  | "forcedStopCount"
  | "branchOutcomeSummary"
  | "droppedProposalCount"
  | "droppedProposalReasons"
  | "budgetFraction"
  | "maxContextChars"
  | "queryBudgetChars"
  | "injectedChars"
  | "droppedChars"
  | "contextClipped"
  | "fitStrategy"
  | "retrievedNodeCount"
  | "fittedNodeCount"
  | "droppedNodeCount"
  | "fittingDropReasons"
  | "interruptionAccounting"
>;

type BrainAssemblyDecisionSnapshot = {
  mode: BrainAssemblyDecisionMode;
  conversationId?: number;
  agentIdentity?: BrainAgentIdentity | null;
  episodeId?: string | null;
  traceId?: string | null;
  footer?: string | null;
  bindingMode?: BrainObservationBindingMode | null;
  serveDecisionRecordId?: string | null;
  selectionDigest?: string | null;
  turnCompileEventId?: string | null;
  decisionRecordedAt?: string | null;
  activePackId?: string | null;
  activePackEventExportDigest?: string | null;
  activePackGraphChecksum?: string | null;
  activePackRouterChecksum?: string | null;
  activePackBuiltAt?: string | null;
  prefetch?: BrainPrefetchDecision | null;
  servedArtifact?: BrainObservationRouteMetadata["servedArtifact"];
  compileReport?: BrainCompileReportV1 | null;
  compileReportSummary?: string | null;
} & BrainAssemblyDecisionSelectionSurface;

type LearningHealthStatus =
  | "idle"
  | "review_harmful_context"
  | "changing_without_feedback"
  | "needs_feedback_coverage"
  | "learning_backed_by_feedback"
  | "monitor";

type LearningHealthSummary = {
  status: LearningHealthStatus;
  summary: string;
  detail: string;
  focus: {
    action: string;
    detail: string;
  };
  signals: {
    routeTraceCount: number;
    supervisedTraceCount: number;
    supervisionCoverage: number;
    helpfulCount: number;
    irrelevantCount: number;
    harmfulCount: number;
    scoredObservationCount: number;
    recentBundleCount: number;
    promotedBundleCount: number;
    rejectedBundleCount: number;
    pendingBundleCount: number;
    replayGatePassed: boolean | null;
  };
};

type BrainPrefetchCacheEntry = {
  key: string;
  queryDigest: string;
  budgetClass: BrainPrefetchBudgetClass;
  summaryRoutingMode: string | null;
  activePackId: string | null;
  activePackVersion: number | null;
  state: BrainPrefetchState;
  traversalResult: TraverseResult | null;
  queryEmbedding: Float32Array | null;
  queryEmbeddingSource: "provided" | "runtime";
  createdAt: number;
  updatedAt: number;
  readyAt: number | null;
  consumedAt: number | null;
  invalidatedReason: string | null;
  prefetchMs: number | null;
  cacheAgeMs: number | null;
  reusedNodeCount: number | null;
  reusedChars: number | null;
  savingsChars: number | null;
  promise: Promise<BrainPrefetchCacheEntry> | null;
};

type TraversalCompileResult = {
  traversalResult: TraverseResult | null;
  queryEmbedding: Float32Array | null;
  queryEmbeddingSource: "provided" | "runtime";
  embeddingMs: number;
  routeSelectionMs: number;
  totalQueryMs: number;
  queryInterruption: BrainInterruptionMetadata | null;
};

function normalizeAssemblyDecision(
  decision: BrainAssemblyDecisionSnapshot,
): BrainAssemblyDecisionSnapshot {
  const bindingMode = resolveObservationBindingMode({
    bindingMode: decision.bindingMode,
    serveDecisionRecordId: decision.serveDecisionRecordId,
    selectionDigest: decision.selectionDigest,
    activePackGraphChecksum: decision.activePackGraphChecksum,
    turnCompileEventId: decision.turnCompileEventId,
    traceId: decision.traceId ?? null,
  });
  const compileReportSource = decision.compileReport ?? (decision.servedArtifact?.compileReport as BrainCompileReportV1 | null | undefined) ?? null;
  const compileReportSummarySource = decision.compileReportSummary ?? compileReportSource?.summary ?? null;
  const compileReport = compileReportSource
    ? rewriteBrainCompileReportSummary(compileReportSource, { bindingMode })
    : null;
  const compileReportSummary = compileReport?.summary ?? compileReportSummarySource;
  const servedArtifact = cloneObservationServedArtifact(decision.servedArtifact);
  const normalizedServedArtifact = servedArtifact
    ? { ...servedArtifact }
    : null;
  if (compileReport || compileReportSummary !== null) {
    const artifact = normalizedServedArtifact ?? {};
    (artifact as Record<string, unknown>).compileReport = compileReport;
    (artifact as Record<string, unknown>).compileReportSummary = compileReportSummary;
    return {
      ...decision,
      agentIdentity: cloneAgentIdentity(decision.agentIdentity),
      bindingMode,
      servedArtifact: artifact,
      compileReport,
      compileReportSummary,
      branchOutcomeSummary: cloneBranchOutcomeSummary(decision.branchOutcomeSummary),
      droppedProposalReasons: decision.droppedProposalReasons
        ? { ...decision.droppedProposalReasons }
        : decision.droppedProposalReasons ?? null,
      fittingDropReasons: decision.fittingDropReasons
        ? { ...decision.fittingDropReasons } as Partial<Record<BrainFittingDropReason, number>>
        : decision.fittingDropReasons ?? null,
      interruptionAccounting: cloneInterruptionAccounting(decision.interruptionAccounting),
      fitStrategy: decision.fitStrategy ?? null as BrainFitStrategy | null,
      prefetch: decision.prefetch ? normalizePrefetchDecision(decision.prefetch) : null,
    };
  }
  return {
    ...decision,
    agentIdentity: cloneAgentIdentity(decision.agentIdentity),
    bindingMode,
    servedArtifact: normalizedServedArtifact,
    compileReport,
    compileReportSummary,
    branchOutcomeSummary: cloneBranchOutcomeSummary(decision.branchOutcomeSummary),
    droppedProposalReasons: decision.droppedProposalReasons
      ? { ...decision.droppedProposalReasons }
      : decision.droppedProposalReasons ?? null,
    fittingDropReasons: decision.fittingDropReasons
      ? { ...decision.fittingDropReasons } as Partial<Record<BrainFittingDropReason, number>>
      : decision.fittingDropReasons ?? null,
    interruptionAccounting: cloneInterruptionAccounting(decision.interruptionAccounting),
    fitStrategy: decision.fitStrategy ?? null as BrainFitStrategy | null,
    prefetch: decision.prefetch ? normalizePrefetchDecision(decision.prefetch) : null,
  };
}

function normalizePrefetchDecision(decision: BrainPrefetchDecision): BrainPrefetchDecision {
  return {
    enabled: decision.enabled,
    state: decision.state,
    kind: decision.kind ?? null,
    budgetClass: decision.budgetClass ?? null,
    key: decision.key ?? null,
    queryDigest: decision.queryDigest ?? null,
    activePackId: decision.activePackId ?? null,
    activePackVersion: decision.activePackVersion ?? null,
    summaryRoutingMode: decision.summaryRoutingMode ?? null,
    prefetchMs: decision.prefetchMs ?? null,
    cacheAgeMs: decision.cacheAgeMs ?? null,
    invalidatedReason: decision.invalidatedReason ?? null,
    reusedNodeCount: decision.reusedNodeCount ?? null,
    reusedChars: decision.reusedChars ?? null,
    savingsChars: decision.savingsChars ?? null,
  };
}

function clonePrefetchDecision(decision: BrainPrefetchDecision | null | undefined): BrainPrefetchDecision | null {
  return decision ? normalizePrefetchDecision(decision) : null;
}

function cloneAgentIdentity(identity: BrainAgentIdentity | null | undefined): BrainAgentIdentity | null {
  return identity
    ? {
        agentId: identity.agentId,
        lane: identity.lane,
      }
    : null;
}

function buildLearningHealthSummary(params: {
  contextFeedback: ContextFeedbackSummary;
  contextUsefulness: ContextUsefulnessSummary;
  recentMutationBundles: MutationBundleRecord[];
  lastReplayGateVerdict: ReplayGateVerdict | null;
}): LearningHealthSummary {
  const recentBundleCount = params.recentMutationBundles.length;
  const promotedBundleCount = params.recentMutationBundles.filter((bundle) => bundle.status === "promoted").length;
  const rejectedBundleCount = params.recentMutationBundles.filter((bundle) => bundle.status === "rejected").length;
  const pendingBundleCount = params.recentMutationBundles.filter((bundle) => bundle.status === "pending").length;
  const replayGatePassed =
    typeof params.lastReplayGateVerdict?.passed === "boolean"
      ? params.lastReplayGateVerdict.passed
      : null;

  const signals = {
    routeTraceCount: params.contextFeedback.coverage.routeTraceCount,
    supervisedTraceCount: params.contextFeedback.coverage.supervisedTraceCount,
    supervisionCoverage: params.contextFeedback.coverage.supervisionCoverage,
    helpfulCount: params.contextFeedback.verdictCounts.helpful,
    irrelevantCount: params.contextFeedback.verdictCounts.irrelevant,
    harmfulCount: params.contextFeedback.verdictCounts.harmful,
    scoredObservationCount: params.contextUsefulness.coverage.scoredObservationCount,
    recentBundleCount,
    promotedBundleCount,
    rejectedBundleCount,
    pendingBundleCount,
    replayGatePassed,
  };

  if (
    signals.routeTraceCount === 0
    && signals.recentBundleCount === 0
    && signals.scoredObservationCount === 0
  ) {
    return {
      status: "idle",
      summary: "no learning activity recorded yet",
      detail: "No traced routes, shadow usefulness scores, or mutation bundles are present yet.",
      focus: {
        action: "monitor",
        detail: "wait for traced routes before judging learning health",
      },
      signals,
    };
  }

  if (signals.harmfulCount > 0) {
    return {
      status: "review_harmful_context",
      summary: `${signals.harmfulCount} harmful traced route verdict(s) need review`,
      detail: "Feedback is landing, but at least one traced route was harmful. Investigate the latest harmful context before trusting further change.",
      focus: {
        action: params.contextFeedback.focus.action,
        detail: params.contextFeedback.focus.detail,
      },
      signals,
    };
  }

  if (signals.promotedBundleCount > 0 && signals.supervisedTraceCount === 0) {
    return {
      status: "changing_without_feedback",
      summary: `${signals.promotedBundleCount} promoted bundle(s) recorded without traced-route feedback`,
      detail: "The system is changing, but no traced routes have been closed with helpful/irrelevant/harmful supervision yet. Treat this as unverified change, not demonstrated learning.",
      focus: {
        action: "increase_feedback_coverage",
        detail: "capture operator or teacher feedback on recent traced routes before trusting promoted changes",
      },
      signals,
    };
  }

  if (signals.promotedBundleCount > 0 && signals.supervisedTraceCount < signals.routeTraceCount) {
    return {
      status: "changing_without_feedback",
      summary: `${signals.promotedBundleCount} promoted bundle(s); feedback covers ${signals.supervisedTraceCount}/${signals.routeTraceCount} traced route(s)`,
      detail: "Recent change is visible, but supervision still trails routed behavior. The system may be changing faster than operators can tell whether that change is useful.",
      focus: {
        action: params.contextFeedback.focus.action,
        detail: params.contextFeedback.focus.detail,
      },
      signals,
    };
  }

  if (signals.routeTraceCount > 0 && signals.supervisedTraceCount < signals.routeTraceCount) {
    return {
      status: "needs_feedback_coverage",
      summary: `feedback covers ${signals.supervisedTraceCount}/${signals.routeTraceCount} traced route(s)`,
      detail: "Learning-health truth is incomplete until more traced routes are closed with operator or teacher verdicts.",
      focus: {
        action: params.contextFeedback.focus.action,
        detail: params.contextFeedback.focus.detail,
      },
      signals,
    };
  }

  if (
    signals.supervisedTraceCount > 0
    && signals.helpfulCount > signals.harmfulCount
    && (signals.promotedBundleCount > 0 || signals.scoredObservationCount > 0)
  ) {
    return {
      status: "learning_backed_by_feedback",
      summary: `${signals.helpfulCount} helpful traced route verdict(s) back recent learning activity`,
      detail: "Recent change is being evaluated by traced-route feedback. That is stronger than mere churn, but it is still local evidence rather than proof of broad answer-quality improvement.",
      focus: {
        action: "monitor",
        detail: "keep sampling traced routes to confirm the current helpful trend holds",
      },
      signals,
    };
  }

  return {
    status: "monitor",
    summary: "learning surfaces are active but mixed",
    detail: "There is learning-related activity, but the current evidence is not yet strong enough to call it clearly useful or clearly harmful.",
    focus: {
      action: params.contextFeedback.focus.action,
      detail: params.contextFeedback.focus.detail,
    },
    signals,
  };
}

function hashQueryDigest(queryText: string): string {
  return createHash("sha256").update(queryText).digest("hex").slice(0, 16);
}

function derivePrefetchBudgetClass(params: {
  budgetChars: number;
  deadlineAtMs?: number | null;
}): BrainPrefetchBudgetClass {
  if (typeof params.deadlineAtMs === "number" && Number.isFinite(params.deadlineAtMs)) {
    return "deadline_pressure";
  }
  if (params.budgetChars <= 512) {
    return "tiny";
  }
  if (params.budgetChars <= 1200) {
    return "small";
  }
  if (params.budgetChars <= 2600) {
    return "standard";
  }
  return "large";
}

function buildPrefetchKey(params: {
  queryDigest: string;
  activePackVersion: number | null;
  budgetClass: BrainPrefetchBudgetClass;
  summaryRoutingMode: string | null;
}): string {
  return [
    params.queryDigest,
    params.activePackVersion ?? "no-pack",
    params.budgetClass,
    params.summaryRoutingMode ?? "ignore",
    "traversal",
  ].join("::");
}

const PREFETCH_CACHE_LIMIT = 32;

export class BrainService {
  private deps: LcmDependencies;
  private store: BrainStore;
  private mutableGraph = new BrainGraph();
  private servingGraph = new BrainGraph();
  private worker: BrainWorker | null;
  private childSupervisor: WorkerSupervisor | null = null;
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
  private lastQueryInterruption: BrainInterruptionMetadata | null = null;
  private lastAssemblyDecision: BrainAssemblyDecisionSnapshot | null = null;
  private lastPrefetchDecision: BrainPrefetchDecision | null = null;
  private recentPrefetchDecisions: BrainPrefetchDecision[] = [];
  private prefetchCache = new Map<string, BrainPrefetchCacheEntry>();
  private prefetchCacheByQueryDigest = new Map<string, string>();
  private lastObservedPrefetchPackVersion: number | null = null;

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
    this.lastAssemblyDecision = this.getStoredLastAssemblyDecision();
    this.lastPrefetchDecision = this.getStoredLastPrefetchDecision();
    this.recentPrefetchDecisions = this.getStoredRecentPrefetchDecisions();
    this.embeddingClient = createEmbeddingClient({
      config: runtimeConfig,
      getApiKey: (provider, model) => params.deps.getApiKey(provider, model),
      log: params.deps.log,
    });

    populateGraph(
      this.mutableGraph,
      this.store.getAllNodes(),
      this.store.loadAllEdges(),
      this.store.loadAllSeedWeights(),
      this.store.loadAllStopLocalWeights(),
    );
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

  private buildObservationRouteMetadata(
    trace: DecisionTrace,
    assemblyDecision: NonNullable<BrainService["lastAssemblyDecision"]> | null = null,
  ): BrainObservationRouteMetadata {
    const routeTrace = trace.routeTrace ?? null;
    const bindingMode = resolveObservationBindingMode({
      bindingMode: assemblyDecision?.bindingMode,
      serveDecisionRecordId: assemblyDecision?.serveDecisionRecordId,
      selectionDigest: assemblyDecision?.selectionDigest,
      activePackGraphChecksum: assemblyDecision?.activePackGraphChecksum,
      turnCompileEventId: assemblyDecision?.turnCompileEventId,
      traceId: trace.id,
    });
    return {
      requestDigest: routeTrace?.requestDigest ?? null,
      agentIdentity: cloneAgentIdentity(routeTrace?.agentIdentity ?? assemblyDecision?.agentIdentity),
      activePackId: assemblyDecision?.activePackId ?? routeTrace?.activePackId ?? null,
      routerIdentity: routeTrace?.routerIdentity ?? null,
      persistenceMode: routeTrace?.persistenceMode ?? null,
      bindingMode,
      serveDecisionRecordId: assemblyDecision?.serveDecisionRecordId ?? null,
      selectionDigest: assemblyDecision?.selectionDigest ?? null,
      turnCompileEventId: assemblyDecision?.turnCompileEventId ?? null,
      decisionRecordedAt: assemblyDecision?.decisionRecordedAt ?? null,
      activePackEventExportDigest: assemblyDecision?.activePackEventExportDigest ?? null,
      activePackGraphChecksum: assemblyDecision?.activePackGraphChecksum ?? null,
      activePackRouterChecksum: assemblyDecision?.activePackRouterChecksum ?? null,
      activePackBuiltAt: assemblyDecision?.activePackBuiltAt ?? null,
      servedArtifact: cloneObservationServedArtifact(assemblyDecision?.servedArtifact),
      candidateNodeIds: [...(routeTrace?.candidateNodeIds ?? [])],
      selectedNodeIds: [...(routeTrace?.selectedNodeIds ?? trace.firedNodes)],
      selectedTraversalNodeIds: [...(routeTrace?.selectedTraversalNodeIds ?? [])],
      selectedPathNodeIds: [...(routeTrace?.selectedPathNodeIds ?? [])],
      selectedSeedNodeIds: [...(routeTrace?.selectedSeedNodeIds ?? [])],
      sourceSummary: routeTrace?.sourceSummary
        ? {
            injectedCount: routeTrace.sourceSummary.injectedCount,
            kinds: { ...routeTrace.sourceSummary.kinds },
            trusts: { ...routeTrace.sourceSummary.trusts },
            sourceUris: [...routeTrace.sourceSummary.sourceUris],
            sourceRefs: [...routeTrace.sourceSummary.sourceRefs],
          }
        : null,
      operatorAudit: routeTrace?.operatorAudit
        ? JSON.parse(JSON.stringify(routeTrace.operatorAudit)) as BrainObservationRouteMetadata["operatorAudit"]
        : null,
      selectionMetadata: routeTrace?.selectionMetadata
        ? {
            ...routeTrace.selectionMetadata,
            branchOutcomeSummary: cloneBranchOutcomeSummary(routeTrace.selectionMetadata.branchOutcomeSummary),
            droppedProposalReasons: routeTrace.selectionMetadata.droppedProposalReasons
              ? { ...routeTrace.selectionMetadata.droppedProposalReasons }
              : routeTrace.selectionMetadata.droppedProposalReasons ?? null,
            fittingDropReasons: routeTrace.selectionMetadata.fittingDropReasons
              ? { ...routeTrace.selectionMetadata.fittingDropReasons }
              : routeTrace.selectionMetadata.fittingDropReasons ?? null,
            interruptionAccounting: cloneInterruptionAccounting(routeTrace.selectionMetadata.interruptionAccounting),
          }
        : null,
    };
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

  getCompileDeadlineMs(): number | null {
    return this.config.maxCompileMs;
  }

  getBudgetFraction(): number {
    return this.config.budgetFraction;
  }

  getLastQueryInterruption(): BrainInterruptionMetadata | null {
    return this.lastQueryInterruption ? { ...this.lastQueryInterruption } : null;
  }

  private getStoredLastAssemblyDecision(): BrainAssemblyDecisionSnapshot | null {
    const stored = this.store.getTrainingStateJson<BrainAssemblyDecisionSnapshot>("last_assembly_decision_json");
    return stored ? normalizeAssemblyDecision(stored) : null;
  }

  private getStoredLastPrefetchDecision(): BrainPrefetchDecision | null {
    const stored = this.store.getTrainingStateJson<BrainPrefetchDecision>("last_prefetch_decision_json");
    return stored ? normalizePrefetchDecision(stored) : null;
  }

  private getStoredRecentPrefetchDecisions(): BrainPrefetchDecision[] {
    const stored = this.store.getTrainingStateJson<BrainPrefetchDecision[]>("recent_prefetch_decisions_json");
    if (!Array.isArray(stored)) {
      return [];
    }
    return stored
      .filter((decision): decision is BrainPrefetchDecision => !!decision && typeof decision === "object")
      .map((decision) => normalizePrefetchDecision(decision))
      .slice(-25);
  }

  private persistPrefetchDecisionHistory(): void {
    this.store.setTrainingStateJson("last_prefetch_decision_json", this.lastPrefetchDecision);
    this.store.setTrainingStateJson("recent_prefetch_decisions_json", this.recentPrefetchDecisions.slice(-25));
  }

  private notePrefetchDecision(decision: BrainPrefetchDecision): BrainPrefetchDecision {
    const normalizedDecision = normalizePrefetchDecision(decision);
    this.lastPrefetchDecision = normalizedDecision;
    this.recentPrefetchDecisions = [...this.recentPrefetchDecisions.slice(-24), normalizedDecision];
    this.persistPrefetchDecisionHistory();
    return normalizedDecision;
  }

  getLastPrefetchDecision(): BrainPrefetchDecision | null {
    return clonePrefetchDecision(this.lastPrefetchDecision);
  }

  getRecentPrefetchSummary(limit = 25): RecentPrefetchSummary {
    return summarizeRecentPrefetchDecisions(this.recentPrefetchDecisions.slice(-limit), limit);
  }

  getPrefetchCacheSize(): number {
    return this.prefetchCache.size;
  }

  getPrefetchInFlightCount(): number {
    let count = 0;
    for (const entry of this.prefetchCache.values()) {
      if (entry.state === "scheduled" && entry.promise) {
        count += 1;
      }
    }
    return count;
  }

  private syncPrefetchCacheToPackVersion(currentPackVersion: number | null): void {
    if (
      this.lastObservedPrefetchPackVersion !== null
      && this.lastObservedPrefetchPackVersion !== currentPackVersion
    ) {
      this.prefetchCache.clear();
    }
    this.lastObservedPrefetchPackVersion = currentPackVersion;
  }

  private trimPrefetchCache(limit = PREFETCH_CACHE_LIMIT): void {
    if (this.prefetchCache.size <= limit) {
      return;
    }

    const evictableEntries = [...this.prefetchCache.values()]
      .filter((entry) => !(entry.state === "scheduled" && entry.promise))
      .sort((left, right) => left.updatedAt - right.updatedAt || left.createdAt - right.createdAt);

    while (this.prefetchCache.size > limit && evictableEntries.length > 0) {
      const entry = evictableEntries.shift();
      if (!entry) {
        break;
      }
      if (this.prefetchCacheByQueryDigest.get(entry.queryDigest) === entry.key) {
        this.prefetchCacheByQueryDigest.delete(entry.queryDigest);
      }
      this.prefetchCache.delete(entry.key);
    }
  }

  noteAssemblyDecision(decision: BrainAssemblyDecisionSnapshot): void {
    const normalizedDecision = normalizeAssemblyDecision(decision);
    this.lastAssemblyDecision = normalizedDecision;
    this.store.setTrainingState("last_assembly_mode", normalizedDecision.mode);
    this.store.setTrainingState("last_assembly_footer", normalizedDecision.footer ?? "");
    this.store.setTrainingState("last_assembly_episode_id", normalizedDecision.episodeId ?? "");
    this.store.setTrainingState("last_assembly_trace_id", normalizedDecision.traceId ?? "");
    this.store.setTrainingStateJson("last_assembly_decision_json", normalizedDecision);
  }

  recordTraceSelectionMetadata(
    trace: DecisionTrace | null | undefined,
    selectionMetadata: Partial<DecisionRouteTrace["selectionMetadata"]>,
  ): void {
    if (!trace?.routeTrace?.selectionMetadata) {
      return;
    }

    const mergedSelectionMetadata = {
      ...trace.routeTrace.selectionMetadata,
      ...selectionMetadata,
    };
    const compileReportDecision = {
      mode:
        selectionMetadata.compileReport?.decision.mode
        ?? ((this.lastAssemblyDecision?.traceId === trace.id ? this.lastAssemblyDecision.mode : null) ?? null)
        ?? trace.routeTrace.selectionMetadata.compileReport?.decision.mode
        ?? null,
      bindingMode:
        selectionMetadata.compileReport?.bindingMode
        ?? ((this.lastAssemblyDecision?.traceId === trace.id ? this.lastAssemblyDecision.bindingMode : null) ?? null)
        ?? trace.routeTrace.selectionMetadata.compileReport?.bindingMode
        ?? null,
      traceId: trace.id,
      episodeId: trace.episodeId,
    };
    const compileReport = buildBrainCompileReport({
      routeTrace: {
        ...trace.routeTrace,
        selectionMetadata: mergedSelectionMetadata,
      },
      decision: compileReportDecision,
      lookupNode: (nodeId) => this.servingGraph.getNode(nodeId) ?? null,
    });
    const persistedSelectionMetadata = compileReport
      ? {
          ...mergedSelectionMetadata,
          compileReport,
          compileReportSummary: compileReport.summary,
        }
      : mergedSelectionMetadata;

    trace.routeTrace = {
      ...trace.routeTrace,
      selectionMetadata: persistedSelectionMetadata,
    };
    this.store.updateTraceSelectionMetadata(trace.id, persistedSelectionMetadata);
  }

  private buildPrefetchDecision(params: {
    state: BrainPrefetchState;
    kind?: BrainPrefetchDecision["kind"];
    budgetClass?: BrainPrefetchBudgetClass | null;
    key?: string | null;
    queryDigest?: string | null;
    activePackId?: string | null;
    activePackVersion?: number | null;
    summaryRoutingMode?: string | null;
    prefetchMs?: number | null;
    cacheAgeMs?: number | null;
    invalidatedReason?: string | null;
    reusedNodeCount?: number | null;
    reusedChars?: number | null;
    savingsChars?: number | null;
  }): BrainPrefetchDecision {
    return normalizePrefetchDecision({
      enabled: true,
      state: params.state,
      kind: params.kind ?? "traversal",
      budgetClass: params.budgetClass ?? null,
      key: params.key ?? null,
      queryDigest: params.queryDigest ?? null,
      activePackId: params.activePackId ?? null,
      activePackVersion: params.activePackVersion ?? null,
      summaryRoutingMode: params.summaryRoutingMode ?? null,
      prefetchMs: params.prefetchMs ?? null,
      cacheAgeMs: params.cacheAgeMs ?? null,
      invalidatedReason: params.invalidatedReason ?? null,
      reusedNodeCount: params.reusedNodeCount ?? null,
      reusedChars: params.reusedChars ?? null,
      savingsChars: params.savingsChars ?? null,
    });
  }

  private async performTraversalCompile(params: {
    queryText: string;
    budgetChars: number;
    queryEmbedding?: Float32Array;
    deadlineAtMs?: number | null;
    recordInterruption?: (interruption: BrainInterruptionMetadata | null) => void;
  }): Promise<TraversalCompileResult> {
    const queryStartedAt = Date.now();
    const deadlineAtMs =
      typeof params.deadlineAtMs === "number" && Number.isFinite(params.deadlineAtMs)
        ? params.deadlineAtMs
        : null;
    const deadlineExceeded = () => deadlineAtMs !== null && Date.now() >= deadlineAtMs;
    const setQueryInterruption = (interruption: BrainInterruptionMetadata) => {
      params.recordInterruption?.({ ...interruption });
    };
    const embeddingStartedAt = Date.now();
    const usingProvidedEmbedding = !!params.queryEmbedding;
    if (deadlineExceeded()) {
      const interruption = {
        interrupted: true,
        stage: usingProvidedEmbedding ? "query" : "embedding",
        reason: usingProvidedEmbedding ? "deadline_before_traversal" : "deadline_before_embedding",
        servedPartial: false,
      } as BrainInterruptionMetadata;
      setQueryInterruption(interruption);
      return {
        traversalResult: null,
        queryEmbedding: null,
        queryEmbeddingSource: usingProvidedEmbedding ? "provided" : "runtime",
        embeddingMs: Date.now() - embeddingStartedAt,
        routeSelectionMs: 0,
        totalQueryMs: Date.now() - queryStartedAt,
        queryInterruption: interruption,
      };
    }

    const embedding =
      params.queryEmbedding
      ?? (this.embeddingClient ? await this.embeddingClient(params.queryText) : null);
    const embeddingMs = Date.now() - embeddingStartedAt;
    if (deadlineExceeded()) {
      const interruption = {
        interrupted: true,
        stage: usingProvidedEmbedding ? "query" : "embedding",
        reason: usingProvidedEmbedding ? "deadline_before_traversal" : "deadline_during_embedding",
        servedPartial: false,
      } as BrainInterruptionMetadata;
      setQueryInterruption(interruption);
      return {
        traversalResult: null,
        queryEmbedding: embedding,
        queryEmbeddingSource: params.queryEmbedding ? "provided" : "runtime",
        embeddingMs,
        routeSelectionMs: 0,
        totalQueryMs: Date.now() - queryStartedAt,
        queryInterruption: interruption,
      };
    }
    if (!embedding || embedding.length === 0) {
      return {
        traversalResult: null,
        queryEmbedding: embedding,
        queryEmbeddingSource: params.queryEmbedding ? "provided" : "runtime",
        embeddingMs,
        routeSelectionMs: 0,
        totalQueryMs: Date.now() - queryStartedAt,
        queryInterruption: null,
      };
    }

    const routeSelectionStartedAt = Date.now();
    const traversalResult = traverse({
      graph: this.servingGraph,
      queryEmbedding: embedding,
      queryText: params.queryText,
      maxHops: this.config.maxHops,
      maxFanoutPerNode: this.config.maxFanoutPerNode,
      maxFrontierSize: this.config.maxFrontierSize,
      budgetChars: params.budgetChars,
      temperature: this.config.servingTemperature,
      maxSeeds: this.config.maxSeeds,
      semanticThreshold: this.config.semanticThreshold,
      deadlineAtMs,
    });
    const routeSelectionMs = Date.now() - routeSelectionStartedAt;
    if (traversalResult.interruption) {
      setQueryInterruption(traversalResult.interruption);
    }
    return {
      traversalResult: traversalResult.firedNodes.length === 0 ? null : traversalResult,
      queryEmbedding: embedding,
      queryEmbeddingSource: params.queryEmbedding ? "provided" : "runtime",
      embeddingMs,
      routeSelectionMs,
      totalQueryMs: Date.now() - queryStartedAt,
      queryInterruption: traversalResult.interruption ?? null,
    };
  }

  private async persistTraversalCompileResult(params: {
    conversationId: number;
    queryText: string;
    budgetChars: number;
    agentIdentity?: BrainAgentIdentity | null;
    compileResult: TraversalCompileResult;
  }): Promise<TraversalResult | null> {
    const { compileResult } = params;
    this.lastQueryInterruption = compileResult.queryInterruption ? { ...compileResult.queryInterruption } : null;
    if (!compileResult.traversalResult) {
      return null;
    }

    const traversalResult = compileResult.traversalResult;
    const episode = recordEpisode({
      traversalResult,
      queryText: params.queryText,
      queryEmbedding: compileResult.queryEmbedding,
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
      agentIdentity: params.agentIdentity,
      packVersion: episode.packVersion,
      budgetChars: params.budgetChars,
      maxHops: this.config.maxHops,
      maxFanoutPerNode: this.config.maxFanoutPerNode,
      maxFrontierSize: this.config.maxFrontierSize,
      embeddingMs: compileResult.embeddingMs,
      routeSelectionMs: compileResult.routeSelectionMs,
      totalQueryMs: compileResult.totalQueryMs,
      queryEmbeddingSource: compileResult.queryEmbeddingSource,
      selectedNodes,
      persistRawSurfaces: this.config.persistRawSurfaces,
    });

    const compileReport = buildBrainCompileReport({
      routeTrace: redactRouteTrace(trace.routeTrace, params.queryText, false) ?? trace.routeTrace,
      decision: {
        traceId: trace.id,
        episodeId: episode.id,
      },
      lookupNode: (nodeId: string) => this.servingGraph.getNode(nodeId) ?? null,
    });
    if (compileReport && trace.routeTrace?.selectionMetadata) {
      trace.routeTrace = {
        ...trace.routeTrace,
        selectionMetadata: {
          ...trace.routeTrace.selectionMetadata,
          compileReport,
          compileReportSummary: compileReport.summary,
        },
      };
    }
    this.store.insertTrace(trace);

    return {
      fired: traversalResult.firedNodes,
      vetoed: traversalResult.vetoedNodes,
      episode,
      trace,
      interruption: traversalResult.interruption ?? null,
    };
  }

  async query(params: {
    conversationId: number;
    queryText: string;
    budgetChars: number;
    agentIdentity?: BrainAgentIdentity | null;
    queryEmbedding?: Float32Array;
    deadlineAtMs?: number | null;
    summaryRoutingMode?: string | null;
  }): Promise<TraversalResult | null> {
    this.lastQueryInterruption = null;
    if (!this.isEnabled() || this.servingGraph.nodeCount() === 0) {
      return null;
    }

    const currentPackVersion = this.store.getCurrentPackVersion();
    this.syncPrefetchCacheToPackVersion(currentPackVersion);

    const queryDigest = hashQueryDigest(params.queryText);
    const summaryRoutingMode = params.summaryRoutingMode ?? "ignore";
    const budgetClass = derivePrefetchBudgetClass({
      budgetChars: params.budgetChars,
      deadlineAtMs: params.deadlineAtMs,
    });
    const prefetchKey = buildPrefetchKey({
      queryDigest,
      activePackVersion: currentPackVersion,
      budgetClass,
      summaryRoutingMode,
    });

    let recordedCacheOutcome = false;
    const cachedEntry = this.prefetchCache.get(prefetchKey) ?? null;
    if (cachedEntry) {
      const resolvedEntry = cachedEntry.promise
        ? await cachedEntry.promise.catch(() => cachedEntry)
        : cachedEntry;
      if (
        resolvedEntry
        && resolvedEntry.traversalResult
        && resolvedEntry.state !== "dropped"
        && resolvedEntry.state !== "invalidated"
      ) {
        const now = Date.now();
        const cacheAgeMs = resolvedEntry.readyAt !== null
          ? now - resolvedEntry.readyAt
          : resolvedEntry.cacheAgeMs;
        resolvedEntry.state = "hit";
        resolvedEntry.consumedAt = now;
        resolvedEntry.cacheAgeMs = cacheAgeMs ?? null;
        resolvedEntry.reusedNodeCount = resolvedEntry.traversalResult.firedNodes.length;
        resolvedEntry.reusedChars = resolvedEntry.traversalResult.contextChars;
        resolvedEntry.savingsChars = resolvedEntry.traversalResult.contextChars;
        resolvedEntry.updatedAt = now;
        this.notePrefetchDecision(this.buildPrefetchDecision({
          state: "hit",
          kind: "traversal",
          budgetClass: resolvedEntry.budgetClass,
          key: resolvedEntry.key,
          queryDigest: resolvedEntry.queryDigest,
          activePackId: resolvedEntry.activePackId,
          activePackVersion: resolvedEntry.activePackVersion,
          summaryRoutingMode: resolvedEntry.summaryRoutingMode,
          prefetchMs: resolvedEntry.prefetchMs,
          cacheAgeMs: resolvedEntry.cacheAgeMs,
          reusedNodeCount: resolvedEntry.reusedNodeCount,
          reusedChars: resolvedEntry.reusedChars,
          savingsChars: resolvedEntry.savingsChars,
        }));
        this.prefetchCacheByQueryDigest.set(queryDigest, prefetchKey);
        const result = await this.persistTraversalCompileResult({
          conversationId: params.conversationId,
          queryText: params.queryText,
          budgetChars: params.budgetChars,
          agentIdentity: params.agentIdentity,
          compileResult: {
            traversalResult: resolvedEntry.traversalResult,
            queryEmbedding: resolvedEntry.queryEmbedding,
            queryEmbeddingSource: resolvedEntry.queryEmbeddingSource,
            embeddingMs: resolvedEntry.prefetchMs ?? 0,
            routeSelectionMs: 0,
            totalQueryMs: resolvedEntry.prefetchMs ?? 0,
            queryInterruption: null,
          },
        });
        return result;
      }
    }

    const knownPrefetchKey = this.prefetchCacheByQueryDigest.get(queryDigest) ?? null;
    let knownPrefetchState: BrainPrefetchState | null = null;
    let knownPrefetchReason: string | null = null;
    if (knownPrefetchKey && knownPrefetchKey !== prefetchKey) {
      const knownEntry = this.prefetchCache.get(knownPrefetchKey) ?? null;
      const prefetchState: BrainPrefetchState = knownEntry?.activePackVersion !== currentPackVersion
        ? "invalidated"
        : "stale";
      knownPrefetchState = prefetchState;
      knownPrefetchReason = knownEntry?.activePackVersion !== currentPackVersion
        ? "pack_version_changed"
        : knownEntry?.budgetClass !== budgetClass
          ? "budget_class_changed"
          : knownEntry?.summaryRoutingMode !== summaryRoutingMode
            ? "summary_routing_changed"
            : "prefetch_key_mismatch";
      if (knownEntry) {
        const now = Date.now();
        knownEntry.state = knownPrefetchState;
        knownEntry.invalidatedReason = knownPrefetchReason;
        knownEntry.cacheAgeMs = knownEntry.readyAt !== null ? now - knownEntry.readyAt : knownEntry.cacheAgeMs;
        knownEntry.updatedAt = now;
        this.notePrefetchDecision(this.buildPrefetchDecision({
          state: prefetchState,
          kind: "traversal",
          budgetClass: knownEntry.budgetClass,
          key: knownEntry.key,
          queryDigest: knownEntry.queryDigest,
          activePackId: knownEntry.activePackId,
          activePackVersion: knownEntry.activePackVersion,
          summaryRoutingMode: knownEntry.summaryRoutingMode,
          prefetchMs: knownEntry.prefetchMs,
          cacheAgeMs: knownEntry.cacheAgeMs,
          invalidatedReason: knownPrefetchReason,
        }));
      } else {
        const prefetchState: BrainPrefetchState = knownPrefetchState ?? "stale";
        this.notePrefetchDecision(this.buildPrefetchDecision({
          state: prefetchState,
          kind: "traversal",
          budgetClass,
          key: knownPrefetchKey,
          queryDigest,
          activePackId: currentPackVersion === null ? null : `brain-pack-v${currentPackVersion}`,
          activePackVersion: currentPackVersion,
          summaryRoutingMode,
          invalidatedReason: knownPrefetchReason,
        }));
      }
      recordedCacheOutcome = true;
    }

    const liveCompile = await this.performTraversalCompile({
      queryText: params.queryText,
      budgetChars: params.budgetChars,
      queryEmbedding: params.queryEmbedding,
      deadlineAtMs: params.deadlineAtMs,
      recordInterruption: (interruption) => {
        this.lastQueryInterruption = interruption ? { ...interruption } : null;
      },
    });
    if (!liveCompile.traversalResult) {
      if (!recordedCacheOutcome && (cachedEntry || knownPrefetchKey)) {
        this.notePrefetchDecision(this.buildPrefetchDecision({
          state: "miss",
          kind: "traversal",
          budgetClass,
          key: prefetchKey,
          queryDigest,
          activePackId: currentPackVersion === null ? null : `brain-pack-v${currentPackVersion}`,
          activePackVersion: currentPackVersion,
          summaryRoutingMode,
          invalidatedReason: cachedEntry?.state === "dropped" ? "prefetch_dropped" : "prefetch_unavailable",
        }));
      }
      return null;
    }
    if (!recordedCacheOutcome && (cachedEntry || knownPrefetchKey)) {
      this.notePrefetchDecision(this.buildPrefetchDecision({
        state: "miss",
        kind: "traversal",
        budgetClass,
        key: prefetchKey,
        queryDigest,
        activePackId: currentPackVersion === null ? null : `brain-pack-v${currentPackVersion}`,
        activePackVersion: currentPackVersion,
        summaryRoutingMode,
        invalidatedReason: cachedEntry?.state === "dropped" ? "prefetch_dropped" : "prefetch_unavailable",
      }));
    }
    return this.persistTraversalCompileResult({
      conversationId: params.conversationId,
      queryText: params.queryText,
      budgetChars: params.budgetChars,
      agentIdentity: params.agentIdentity,
      compileResult: liveCompile,
    });
  }

  async schedulePrefetch(params: {
    conversationId: number;
    queryText: string;
    budgetChars: number;
    queryEmbedding?: Float32Array;
    deadlineAtMs?: number | null;
    summaryRoutingMode?: string | null;
  }): Promise<BrainPrefetchDecision | null> {
    if (!this.isEnabled() || !this.isInitialized() || this.servingGraph.nodeCount() === 0) {
      return null;
    }

    const currentPackVersion = this.store.getCurrentPackVersion();
    this.syncPrefetchCacheToPackVersion(currentPackVersion);

    const queryDigest = hashQueryDigest(params.queryText);
    const summaryRoutingMode = params.summaryRoutingMode ?? "ignore";
    const budgetClass = derivePrefetchBudgetClass({
      budgetChars: params.budgetChars,
      deadlineAtMs: params.deadlineAtMs,
    });
    const key = buildPrefetchKey({
      queryDigest,
      activePackVersion: currentPackVersion,
      budgetClass,
      summaryRoutingMode,
    });

    const existingEntry = this.prefetchCache.get(key) ?? null;
    if (existingEntry) {
      return existingEntry.promise
        ? existingEntry.promise.then(() => this.getLastPrefetchDecision())
        : Promise.resolve(this.getLastPrefetchDecision());
    }

    const now = Date.now();
    const scheduledDecision = this.notePrefetchDecision(this.buildPrefetchDecision({
      state: "scheduled",
      kind: "traversal",
      budgetClass,
      key,
      queryDigest,
      activePackId: currentPackVersion === null ? null : `brain-pack-v${currentPackVersion}`,
      activePackVersion: currentPackVersion,
      summaryRoutingMode,
    }));

    const cacheEntry: BrainPrefetchCacheEntry = {
      key,
      queryDigest,
      budgetClass,
      summaryRoutingMode,
      activePackId: scheduledDecision.activePackId,
      activePackVersion: currentPackVersion,
      state: "scheduled",
      traversalResult: null,
      queryEmbedding: null,
      queryEmbeddingSource: params.queryEmbedding ? "provided" : "runtime",
      createdAt: now,
      updatedAt: now,
      readyAt: null,
      consumedAt: null,
      invalidatedReason: null,
      prefetchMs: null,
      cacheAgeMs: null,
      reusedNodeCount: null,
      reusedChars: null,
      savingsChars: null,
      promise: null,
    };

    const promise = (async () => {
      try {
        const compileResult = await this.performTraversalCompile({
          queryText: params.queryText,
          budgetChars: params.budgetChars,
          queryEmbedding: params.queryEmbedding,
          deadlineAtMs: params.deadlineAtMs,
        });
        const finishedAt = Date.now();
        cacheEntry.updatedAt = finishedAt;
        cacheEntry.prefetchMs = finishedAt - now;
        cacheEntry.queryEmbedding = compileResult.queryEmbedding;
        cacheEntry.queryEmbeddingSource = compileResult.queryEmbeddingSource;
        if (!compileResult.traversalResult) {
          cacheEntry.state = compileResult.queryInterruption ? "dropped" : "miss";
          cacheEntry.invalidatedReason = compileResult.queryInterruption?.reason ?? "no_nodes";
          cacheEntry.cacheAgeMs = finishedAt - now;
          this.notePrefetchDecision(this.buildPrefetchDecision({
            state: cacheEntry.state,
            kind: "traversal",
            budgetClass: cacheEntry.budgetClass,
            key: cacheEntry.key,
            queryDigest: cacheEntry.queryDigest,
            activePackId: cacheEntry.activePackId,
            activePackVersion: cacheEntry.activePackVersion,
            summaryRoutingMode: cacheEntry.summaryRoutingMode,
            prefetchMs: cacheEntry.prefetchMs,
            cacheAgeMs: cacheEntry.cacheAgeMs,
            invalidatedReason: cacheEntry.invalidatedReason,
          }));
          return cacheEntry;
        }

        const livePackVersion = this.store.getCurrentPackVersion();
        if (livePackVersion !== currentPackVersion) {
          cacheEntry.state = "invalidated";
          cacheEntry.invalidatedReason = "pack_version_changed";
          cacheEntry.cacheAgeMs = finishedAt - now;
          this.notePrefetchDecision(this.buildPrefetchDecision({
            state: cacheEntry.state,
            kind: "traversal",
            budgetClass: cacheEntry.budgetClass,
            key: cacheEntry.key,
            queryDigest: cacheEntry.queryDigest,
            activePackId: cacheEntry.activePackId,
            activePackVersion: cacheEntry.activePackVersion,
            summaryRoutingMode: cacheEntry.summaryRoutingMode,
            prefetchMs: cacheEntry.prefetchMs,
            cacheAgeMs: cacheEntry.cacheAgeMs,
            invalidatedReason: cacheEntry.invalidatedReason,
          }));
          return cacheEntry;
        }

        cacheEntry.state = "materialized";
        cacheEntry.readyAt = finishedAt;
        cacheEntry.traversalResult = compileResult.traversalResult;
        cacheEntry.cacheAgeMs = 0;
        this.prefetchCache.set(cacheEntry.key, cacheEntry);
        this.trimPrefetchCache();
        this.prefetchCacheByQueryDigest.set(cacheEntry.queryDigest, cacheEntry.key);
        this.notePrefetchDecision(this.buildPrefetchDecision({
          state: cacheEntry.state,
          kind: "traversal",
          budgetClass: cacheEntry.budgetClass,
          key: cacheEntry.key,
          queryDigest: cacheEntry.queryDigest,
          activePackId: cacheEntry.activePackId,
          activePackVersion: cacheEntry.activePackVersion,
          summaryRoutingMode: cacheEntry.summaryRoutingMode,
          prefetchMs: cacheEntry.prefetchMs,
          cacheAgeMs: cacheEntry.cacheAgeMs,
        }));
        return cacheEntry;
      } catch (error) {
        const finishedAt = Date.now();
        cacheEntry.state = "dropped";
        cacheEntry.invalidatedReason = (error as Error).message || "prefetch_error";
        cacheEntry.updatedAt = finishedAt;
        cacheEntry.cacheAgeMs = finishedAt - now;
        this.notePrefetchDecision(this.buildPrefetchDecision({
          state: cacheEntry.state,
          kind: "traversal",
          budgetClass: cacheEntry.budgetClass,
          key: cacheEntry.key,
          queryDigest: cacheEntry.queryDigest,
          activePackId: cacheEntry.activePackId,
          activePackVersion: cacheEntry.activePackVersion,
          summaryRoutingMode: cacheEntry.summaryRoutingMode,
          prefetchMs: cacheEntry.prefetchMs,
          cacheAgeMs: cacheEntry.cacheAgeMs,
          invalidatedReason: cacheEntry.invalidatedReason,
        }));
        return cacheEntry;
      } finally {
        cacheEntry.promise = null;
      }
    })();

    cacheEntry.promise = promise;
    this.prefetchCache.set(key, cacheEntry);
    this.trimPrefetchCache();
    this.prefetchCacheByQueryDigest.set(queryDigest, key);
    await promise;
    return this.getLastPrefetchDecision();
  }
  async recordTurnObservation(params: {
    episodeId?: string | null;
    assistantResponse: string;
    toolResults?: BrainObservationToolResult[];
  }): Promise<void> {
    const episodeId = typeof params.episodeId === "string" ? params.episodeId.trim() : "";
    if (!episodeId) {
      return;
    }

    const episode = this.store.getEpisode(episodeId);
    const trace = this.store.getTraceForEpisode(episodeId);
    if (!episode || !trace) {
      return;
    }
    const assemblyDecision =
      this.lastAssemblyDecision
      && this.lastAssemblyDecision.traceId === trace.id
      && this.lastAssemblyDecision.episodeId === episode.id
        ? this.lastAssemblyDecision
        : null;

    this.store.insertObservation({
      episodeId: episode.id,
      conversationId: episode.conversationId,
      traceId: trace.id,
      queryText: this.config.persistRawSurfaces
        ? episode.queryText
        : (redactTextSurface("query", episode.queryText) ?? ""),
      retrievedContext: (trace.routeTrace?.injectedNodeSummaries ?? []).map((summary) =>
        this.config.persistRawSurfaces ? summary : redactInjectedNodeSummary(summary)
      ),
      routeMetadata: this.buildObservationRouteMetadata(trace, assemblyDecision),
      assistantResponse: this.config.persistRawSurfaces
        ? params.assistantResponse
        : (redactTextSurface("assistant_response", params.assistantResponse) ?? ""),
      toolResults: (params.toolResults ?? []).map((result) =>
        this.config.persistRawSurfaces ? result : redactToolResult(result)
      ),
    });
  }

  async teachUserCorrection(params: {
    canonicalInstruction: string;
    sourceQuote: string;
    conversationId?: number;
    episodeId?: string;
    sourceMessageId?: number;
    tags?: string[];
    metadata?: Record<string, unknown>;
    via?: string;
  }): Promise<{ nodeId: string; packVersion: number | null }> {
    return this.teach({
      instruction: params.canonicalInstruction,
      conversationId: params.conversationId,
      episodeId: params.episodeId,
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
      episodeId: params.observation.episodeId,
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
    if (!isSystemMessage(observation.userText)) {
      this.store.attachObservationFollowUp(
        observation.conversationId,
        this.config.persistRawSurfaces
          ? observation.userText
          : (redactTextSurface("follow_up", observation.userText) ?? ""),
        observation.episodeId,
      );
    }
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
    episodeId?: string;
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
    const requestedEpisodeId = typeof params.episodeId === "string" ? params.episodeId.trim() : "";
    const explicitEpisode = requestedEpisodeId.length > 0
      ? this.store.getEpisode(requestedEpisodeId)
      : null;
    const exactEpisode =
      explicitEpisode && (
        params.conversationId === undefined
          || explicitEpisode.conversationId === params.conversationId
      )
        ? explicitEpisode
        : typeof params.conversationId === "number"
          ? this.store.getEpisode(this.latestEpisodeByConversation.get(params.conversationId) ?? "")
          : null;
    const recentEpisode = exactEpisode ?? recentEpisodes[0] ?? null;
    const episodeAttributionMode =
      explicitEpisode && exactEpisode?.id === explicitEpisode.id
        ? "explicit_episode"
        : exactEpisode
          ? "latest_conversation_episode"
          : recentEpisode
            ? "recent_conversation_fallback"
            : "no_episode";
    const connectedNodes = new Set<string>();
    const selectedSeedNodeIds = recentEpisode?.trajectory.find(
      (expansion) => expansion.sourceNodeId === null,
    )?.acceptedTargets ?? [];
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
    for (const selectedSeedNodeId of selectedSeedNodeIds) {
      if (connectedNodes.has(selectedSeedNodeId)) {
        continue;
      }
      const now = Date.now();
      const seedEdge = {
        source: selectedSeedNodeId,
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
        target: selectedSeedNodeId,
      };
      this.mutableGraph.addEdge(seedEdge);
      this.mutableGraph.addEdge(reverseSeedEdge);
      this.store.insertEdge(seedEdge);
      this.store.insertEdge(reverseSeedEdge);
    }

    const misroutedTargetId = recentEpisode?.firedNodes.at(-1) ?? null;
    if (misroutedTargetId && misroutedTargetId !== node.id) {
      for (const selectedSeedNodeId of selectedSeedNodeIds) {
        const inhibitoryEdge = {
          source: selectedSeedNodeId,
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
    }

    const targetEpisodes = exactEpisode ? [exactEpisode] : recentEpisodes.slice(0, 1);
    for (const episode of targetEpisodes) {
      if (episode && episode.reward === null) {
        const reason = `correction taught: "${params.instruction.slice(0, 80)}"`;
        const matchedTrace = this.store.getTraceForEpisode(episode.id);
        const label = this.store.insertLabel({
          episodeId: episode.id,
          source: "human",
          value: -0.5,
          reason,
        });
        if (matchedTrace) {
          this.store.insertTraceSupervision({
            traceId: matchedTrace.id,
            episodeId: episode.id,
            conversationId: episode.conversationId,
            source: "human",
            kind: "teach_correction",
            value: -0.5,
            confidence: 1.0,
            reason,
            contentSnippet: params.instruction.slice(0, 240),
            resolution: "promoted_to_label",
            labelId: label.id,
            metadata: {
              ...provenanceMetadata,
              taughtNodeId: node.id,
              correctedEpisodeId: episode.id,
              episodeAttributionMode,
              episodeAttributionRequestedId: requestedEpisodeId || null,
              extractor: teachVia ?? "brain_teach",
              via: teachVia ?? "brain_teach",
              traceId: matchedTrace.id,
              tracePackVersion: matchedTrace.packVersion ?? null,
              traceRequestDigest: matchedTrace.routeTrace?.requestDigest ?? null,
              traceSelectedNodeIds: matchedTrace.routeTrace?.selectedNodeIds ?? matchedTrace.firedNodes,
              traceSelectedPathNodeIds: matchedTrace.routeTrace?.selectedPathNodeIds ?? [],
            },
          });
        }
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
    const recentDecisionSummary = this.store.getRecentDecisionSummary(25);
    const recentPrefetchSummary = this.getRecentPrefetchSummary(25);
    const workerState = readWorkerRuntimeState(this.store, this.config);
    const contextFeedback = this.store.getContextFeedbackSummary();
    const contextUsefulness = this.store.getContextUsefulnessSummary();
    const promotionStory = buildPromotionStory(this.store, { contextFeedback });
    const routeTraceCount = this.store.countTraces();
    const supervisionCount = this.store.countTraceSupervision();
    const observationAttribution = this.store.getObservationAttributionSummary();
    const teacherReadyBefore = Date.now() - Math.max(1_000, this.config.trainerIntervalMs);
    const teacherTruth = {
      queue: this.store.getTeacherQueueSummary(teacherReadyBefore, 20),
      lastEvaluationCycle: this.store.getTrainingStateJson("last_teacher_evaluation_cycle_json"),
      lastUpdateCycle: this.store.getTrainingStateJson("last_teacher_update_cycle_json"),
    };
    const attributionTruth = summarizeAttributionTruth({
      observationAttribution,
      teacherTruth,
    });
    const operatorHealth = summarizeOperatorHealth({
      workerHealthy: workerState.workerHealthy,
      workerMode: workerState.workerMode,
      workerStatus: workerState.workerStatus,
      watchState: null,
      proofState: null,
      teacherArtifactCount: null,
    });
    const lastPgCandidateUpdate = this.store.getTrainingStateJson("last_pg_candidate_update_json");
    const lastPgCandidatePackVersionRaw = this.store.getTrainingState("last_pg_candidate_pack_version");
    const lastPgCandidatePackVersion = lastPgCandidatePackVersionRaw
      ? Number.parseInt(lastPgCandidatePackVersionRaw, 10)
      : null;
    const lastAssemblyDecision = this.lastAssemblyDecision ?? this.getStoredLastAssemblyDecision();
    const lastPrefetchDecision = this.getLastPrefetchDecision();
    const recentMutationBundles = this.store.getRecentMutationBundles(5);
    const lastReplayGateVerdict = this.store.getTrainingStateJson<ReplayGateVerdict>("last_replay_gate_verdict_json") ?? null;
    const learningHealth = buildLearningHealthSummary({
      contextFeedback,
      contextUsefulness,
      recentMutationBundles,
      lastReplayGateVerdict,
    });
    const lastCompileReportSummary = lastAssemblyDecision?.compileReportSummary
      ?? (lastAssemblyDecision?.servedArtifact as { compileReportSummary?: string | null } | null | undefined)?.compileReportSummary
      ?? recentTraces[0]?.routeTrace?.selectionMetadata?.compileReportSummary
      ?? recentTraces[0]?.routeTrace?.selectionMetadata?.compileReport?.summary
      ?? null;

    const embeddingConfig = describeEmbeddingConfig(this.config);
    const contextManagement = buildContextManagementModel({
      lcmConfig: this.deps.config,
      brainConfig: this.config,
    });

    return {
      initialized: this.initialized,
      enabled: this.isEnabled(),
      embeddingConfigured: Boolean(this.embeddingClient),
      embeddingProvider: this.config.embeddingProvider,
      embeddingModel: this.config.embeddingModel,
      embeddingBaseUrl: this.config.embeddingModel ? embeddingConfig.baseUrl : "",
      embeddingAuthMode: embeddingConfig.authMode,
      embeddingConfigError: embeddingConfig.error,
      contextManagement,
      maxCompileMs: this.config.maxCompileMs,
      budgetFraction: this.config.budgetFraction,
      maxHops: this.config.maxHops,
      maxFanoutPerNode: this.config.maxFanoutPerNode,
      maxFrontierSize: this.config.maxFrontierSize,
      maxSeeds: this.config.maxSeeds,
      semanticThreshold: this.config.semanticThreshold,
      workerHeartbeatTimeoutMs: this.config.workerHeartbeatTimeoutMs,
      workerRestartDelayMs: this.config.workerRestartDelayMs,
      currentPackVersion: this.store.getCurrentPackVersion(),
      currentPackPromotedAt: currentPack?.promotedAt ?? null,
      currentPackMetadata: promotionStory.currentPack?.metadata ?? null,
      shadowMode: this.config.shadowMode,
      teacherEnabled: this.config.teacherEnabled,
      rawPersistenceEnabled: this.config.persistRawSurfaces,
      modelTraceSurface: "redacted",
      teacherInputSurface: "redacted",
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
      pendingObservations: this.store.countPendingObservations(),
      pendingObservationsByStatus: this.store.countObservationsByStatus(),
      observationAttribution,
      attributionTruth,
      teacherTruth,
      operatorHealth,
      contextFeedback,
      contextUsefulness,
      learningHealth,
      ...workerState,
      pendingLabels: this.store.getPendingLabels().length,
      pendingLabelsBySource: this.store.countPendingLabelsBySource(),
      mutationBacklog: this.store.countMutationsByStatus(),
      recentMutationBundles,
      seedLearningEnabled: this.mutableGraph.hasSeedWeights(),
      routeTraceCount,
      supervisionCount,
      lastPgCandidatePackVersion: Number.isFinite(lastPgCandidatePackVersion ?? NaN)
        ? lastPgCandidatePackVersion
        : null,
      lastPgCandidateUpdate,
      recentTraceCount: recentTraces.length,
      recentDecisionSummary,
      recentPrefetchSummary,
      lastTraceFooter: recentTraces[0]?.footer ?? null,
      lastTraceContextChars: recentTraces[0]?.contextChars ?? null,
      lastTraceSelectionMetadata: recentTraces[0]?.routeTrace?.selectionMetadata ?? null,
      lastCompileReportSummary,
      lastAssemblyDecision,
      lastPrefetchDecision,
      prefetchCacheSize: this.getPrefetchCacheSize(),
      prefetchInFlightCount: this.getPrefetchInFlightCount(),
      lastPromotionReason: this.store.getTrainingState("last_promotion_reason"),
      lastPromotionVerdict: this.store.getTrainingStateJson("last_promotion_verdict_json"),
      lastReplayFailureReason: this.store.getTrainingState("last_replay_failure_reason"),
      lastReplayGateVerdict,
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
      ...redactDecisionTrace(trace),
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

    populateGraph(
      this.servingGraph,
      snapshot.nodes,
      snapshot.edges,
      snapshot.seedWeights,
      snapshot.stopLocalWeights,
    );
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
