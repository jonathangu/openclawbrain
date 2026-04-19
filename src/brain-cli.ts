import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import process from "node:process";
import { DatabaseSync } from "node:sqlite";
import { BrainGraph } from "./brain-core/graph.js";
import { computeHealth } from "./brain-core/health.js";
import { PackManager } from "./brain-core/pack.js";
import type {
  BrainPrefetchDecision,
  ContextFeedbackSummary,
  ContextUsefulnessSummary,
  MutationBundleRecord,
  ReplayGateVerdict,
} from "./brain-core/types.js";
import {
  buildRecentRouteDecisionSummaryV1,
  DEFAULT_ROUTE_DECISION_SUMMARY_WINDOW_SIZE_V1,
} from "./brain-core/route-decision-event.js";
import { summarizeRecentPrefetchDecisions } from "./brain-core/trace.js";
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
  flattenToolActionPriors,
  populateGraph,
} from "./brain-runtime/graph-io.js";
import {
  buildContinuousLearningOperatorStatus,
  clearContinuousLearningControl,
  continuousLearningControlDir,
  writeContinuousLearningControl,
} from "./brain-runtime/continuous-learning-status.js";
import { buildPromotionStory } from "./brain-runtime/promotion-story.js";
import { buildRouteQualitySummaryV1, type RouteQualitySummaryRoutingModeV1 } from "./brain-runtime/route-quality-summary.js";
import type { AssembledSummaryMetadata } from "./assembler.js";
import { readWorkerRuntimeState } from "./brain-runtime/worker-state.js";
import { buildContextManagementModel } from "./context-management-model.js";
import { summarizeAttributionTruth, summarizeOperatorHealth } from "./live-runtime-audit.js";
import { summarizeBoundedAnytimeStatus } from "./brain-core/trace.js";

function printJson(payload: unknown): void {
  process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
}

function buildLearningHealthSummary(params: {
  contextFeedback: ContextFeedbackSummary;
  contextUsefulness: ContextUsefulnessSummary;
  recentMutationBundles: MutationBundleRecord[];
  lastReplayGateVerdict: ReplayGateVerdict | null;
}) {
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
      focus: params.contextFeedback.focus,
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
      focus: params.contextFeedback.focus,
      signals,
    };
  }

  if (signals.routeTraceCount > 0 && signals.supervisedTraceCount < signals.routeTraceCount) {
    return {
      status: "needs_feedback_coverage",
      summary: `feedback covers ${signals.supervisedTraceCount}/${signals.routeTraceCount} traced route(s)`,
      detail: "Learning-health truth is incomplete until more traced routes are closed with operator or teacher verdicts.",
      focus: params.contextFeedback.focus,
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
    focus: params.contextFeedback.focus,
    signals,
  };
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
    "Usage: openclawbrain <init|status|trace|replay|promote|rollback|disable|enable|doctor|control> [args]\n",
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
    store.loadAllToolActionPriors(),
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
    toolActionPriors: flattenToolActionPriors(graph),
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
  const { config, store, graph, brainConfig } = loadStore();
  const workspaceRoot = resolve(process.env.OPENCLAWBRAIN_WORKSPACE_ROOT ?? process.cwd());
  const recentEpisodes = store.getRecentEpisodes(100);
  const currentPack = store.getCurrentPackVersion();
  const health = computeHealth(graph, recentEpisodes, currentPack ?? 0);
  const embeddingConfig = describeEmbeddingConfig(brainConfig);
  const workerState = readWorkerRuntimeState(store, brainConfig);
  const contextFeedback = store.getContextFeedbackSummary();
  const contextUsefulness = store.getContextUsefulnessSummary();
  const routeOutcomeTruthStored = store.getTrainingStateJson<Record<string, unknown>>("route_outcome_truth_summary_json");
  const promotionStory = buildPromotionStory(store, { contextFeedback });
  const routeTraceCount = store.countTraces();
  const observationAttribution = store.getObservationAttributionSummary();
  const routeOutcomeTruth = routeOutcomeTruthStored
    ? {
        ...routeOutcomeTruthStored,
        coverage: {
          ...(typeof routeOutcomeTruthStored.coverage === "object" && routeOutcomeTruthStored.coverage !== null
            ? routeOutcomeTruthStored.coverage as Record<string, unknown>
            : {}),
          routeTraceCount,
          observationCount: observationAttribution.totalObservationCount,
          servedCoverage: routeTraceCount > 0
            ? Number((routeOutcomeTruthStored.activation as Record<string, unknown> | undefined)?.totalServedCount ?? 0) / routeTraceCount
            : 0,
          outcomeCoverage: observationAttribution.totalObservationCount > 0
            ? Number((routeOutcomeTruthStored.coverage as Record<string, unknown> | undefined)?.followUpCount ?? 0) / observationAttribution.totalObservationCount
            : 0,
          resolutionCoverage: observationAttribution.totalObservationCount > 0
            ? Number((routeOutcomeTruthStored.coverage as Record<string, unknown> | undefined)?.followUpCount ?? 0) / observationAttribution.totalObservationCount
            : 0,
        },
        latest: {
          routeServed: store.getTrainingStateJson("last_route_served_event_json"),
          turnOutcome: store.getTrainingStateJson("last_turn_outcome_event_json"),
          retryOrIntervention: store.getTrainingStateJson("last_retry_or_intervention_event_json"),
          episodeResolution: store.getTrainingStateJson("last_episode_resolution_event_json"),
        },
        detail: routeTraceCount === 0
          ? "no traced routes recorded yet"
          : `served=${Number((routeOutcomeTruthStored.activation as Record<string, unknown> | undefined)?.totalServedCount ?? 0)}/${routeTraceCount} traced turns, followups=${Number((routeOutcomeTruthStored.coverage as Record<string, unknown> | undefined)?.followUpCount ?? 0)}, confirmations=${Number((routeOutcomeTruthStored.coverage as Record<string, unknown> | undefined)?.confirmationCount ?? 0)}`,
      }
    : store.getRouteOutcomeTruthSummary();
  const teacherReadyBefore = Date.now() - Math.max(1_000, brainConfig.trainerIntervalMs);
  const teacherTruth = {
    queue: store.getTeacherQueueSummary(teacherReadyBefore, 20),
    lastEvaluationCycle: store.getTrainingStateJson("last_teacher_evaluation_cycle_json"),
    lastUpdateCycle: store.getTrainingStateJson("last_teacher_update_cycle_json"),
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
  const recentDecisionSummary = store.getRecentDecisionSummary(25);
  const recentRouteDecisionEvents = store.getTrainingStateJson<unknown[]>("recent_route_decision_events_json") ?? [];
  const routeDecisionSummary = buildRecentRouteDecisionSummaryV1(
    recentRouteDecisionEvents,
    DEFAULT_ROUTE_DECISION_SUMMARY_WINDOW_SIZE_V1,
  );
  const recentMutationBundles = store.getRecentMutationBundles(5);
  const lastReplayGateVerdict = store.getTrainingStateJson<ReplayGateVerdict>("last_replay_gate_verdict_json") ?? null;
  const learningHealth = buildLearningHealthSummary({
    contextFeedback,
    contextUsefulness,
    recentMutationBundles,
    lastReplayGateVerdict,
  });
  const continuousLearning = buildContinuousLearningOperatorStatus({
    store,
    graph,
    workspaceRoot,
    brainRoot: brainConfig.root,
    controlRoot: continuousLearningControlDir(workspaceRoot),
    now: Date.now(),
  });
  const recentPrefetchDecisions = store.getTrainingStateJson<BrainPrefetchDecision[]>("recent_prefetch_decisions_json") ?? [];
  const recentPrefetchSummary = summarizeRecentPrefetchDecisions(recentPrefetchDecisions, 25);
  const recentTrace = store.getRecentTraces(1)[0] ?? null;
  const lastAssemblyDecision = store.getTrainingStateJson<Record<string, unknown>>("last_assembly_decision_json")
    ?? {
      mode: store.getTrainingState("last_assembly_mode"),
      footer: store.getTrainingState("last_assembly_footer"),
      episodeId: store.getTrainingState("last_assembly_episode_id"),
      traceId: store.getTrainingState("last_assembly_trace_id"),
    };
  const lastCompileReportSummary = (lastAssemblyDecision?.compileReportSummary as string | null | undefined)
    ?? (lastAssemblyDecision?.servedArtifact as { compileReportSummary?: string | null } | null | undefined)?.compileReportSummary
    ?? recentTrace?.routeTrace?.selectionMetadata?.compileReportSummary
    ?? recentTrace?.routeTrace?.selectionMetadata?.compileReport?.summary
    ?? null;
  const boundedAnytimeSummary = summarizeBoundedAnytimeStatus({
    recentDecisionSummary,
    latestSelectionMetadata: recentTrace?.routeTrace?.selectionMetadata ?? null,
    configuredCompileDeadlineMs: brainConfig.maxCompileMs,
  });
  const lastPrefetchDecision = store.getTrainingStateJson<BrainPrefetchDecision>("last_prefetch_decision_json");
  const routeQuality = buildRouteQualitySummaryV1({
    surface: "status",
    activePackVersion: currentPack,
    activePackId: recentTrace?.routeTrace?.activePackId ?? null,
    routerIdentity: recentTrace?.routeTrace?.routerIdentity ?? null,
    summaryRoutingMode: (lastAssemblyDecision?.summaryRoutingMode as RouteQualitySummaryRoutingModeV1 | null | undefined) ?? null,
    summaryMetadata: (lastAssemblyDecision?.summaryMetadata as AssembledSummaryMetadata | null | undefined) ?? null,
    replayVerdict: {
      passed: lastReplayGateVerdict?.passed ?? null,
      verdict: lastReplayGateVerdict?.passed === true
        ? "pass"
        : lastReplayGateVerdict?.passed === false
          ? "fail"
          : "unknown",
      summary: lastReplayGateVerdict?.reason?.summary ?? null,
    },
    stopLocalWeights: graph.getAllStopLocalWeights(),
    toolActionPriors: graph.getAllToolActionPriors(),
    disabled: existsSync(join(brainConfig.root, "DISABLED")),
    shadowMode: brainConfig.shadowMode,
    rolledBack: promotionStory.currentPack?.rolledBack ?? false,
    rollbackKey: null,
    proofBundleId: null,
  });
  const contextManagement = buildContextManagementModel({
    lcmConfig: config,
    brainConfig,
  });

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
    workerHeartbeatTimeoutMs: brainConfig.workerHeartbeatTimeoutMs,
    workerRestartDelayMs: brainConfig.workerRestartDelayMs,
    embeddingProvider: brainConfig.embeddingProvider,
    embeddingModel: brainConfig.embeddingModel,
    embeddingBaseUrl: brainConfig.embeddingModel ? embeddingConfig.baseUrl : "",
    embeddingAuthMode: embeddingConfig.authMode,
    embeddingConfigError: embeddingConfig.error,
    contextManagement,
    ...workerState,
    currentPackVersion: currentPack,
    currentPackMetadata: promotionStory.currentPack?.metadata ?? null,
    pendingObservations: store.countPendingObservations(),
    pendingObservationsByStatus: store.countObservationsByStatus(),
    routeTraceCount,
    observationAttribution,
    attributionTruth,
    teacherTruth,
    operatorHealth,
    contextFeedback,
    contextUsefulness,
    routeOutcomeTruth,
    learningHealth,
    routeDecisionSummary,
    routeQuality,
    continuousLearning,
    pendingLabels: store.getPendingLabels().length,
    pendingLabelsBySource: store.countPendingLabelsBySource(),
    mutationBacklog: store.countMutationsByStatus(),
    recentMutationBundles,
    lastPromotionReason: store.getTrainingState("last_promotion_reason"),
    lastPromotionVerdict: store.getTrainingStateJson("last_promotion_verdict_json"),
    lastReplayFailureReason: store.getTrainingState("last_replay_failure_reason"),
    lastReplayGateVerdict,
    promotionStory,
    recentDecisionSummary,
    boundedAnytimeSummary,
    recentPrefetchSummary,
    lastTraceSelectionMetadata: recentTrace?.routeTrace?.selectionMetadata ?? null,
    lastCompileReportSummary,
    lastAssemblyDecision,
    lastPrefetchDecision,
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
    compileReportSummary: trace?.routeTrace?.selectionMetadata?.compileReportSummary ?? trace?.routeTrace?.selectionMetadata?.compileReport?.summary ?? null,
    compileReport: trace?.routeTrace?.selectionMetadata?.compileReport ?? null,
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
    toolActionPriors: flattenToolActionPriors(graph),
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

function commandControl(scopeArg?: string, actionArg?: string, reasonArg?: string): void {
  const { brainConfig } = loadStore();
  const workspaceRoot = resolve(process.env.OPENCLAWBRAIN_WORKSPACE_ROOT ?? process.cwd());
  const scope = scopeArg === "graphify-import" || scopeArg === "retrain" ? scopeArg : null;
  const action = actionArg === "status" || actionArg === "pause" || actionArg === "resume" ? actionArg : null;

  if (!scope || !action) {
    usage();
  }

  const controlRoot = continuousLearningControlDir(workspaceRoot);
  if (action === "status") {
    printJson({
      command: "control",
      brainRoot: brainConfig.root,
      workspaceRoot,
      controlRoot,
      scope,
      control: buildContinuousLearningOperatorStatus({
        store: null,
        workspaceRoot,
        brainRoot: brainConfig.root,
        controlRoot,
        now: Date.now(),
      }).controls,
    });
    return;
  }

  if (action === "pause") {
    const control = writeContinuousLearningControl(workspaceRoot, scope, true, reasonArg ?? null, "openclawbrain control");
    printJson({
      command: "control",
      brainRoot: brainConfig.root,
      workspaceRoot,
      controlRoot,
      scope,
      control,
    });
    return;
  }

  clearContinuousLearningControl(workspaceRoot, scope);
  const control = writeContinuousLearningControl(workspaceRoot, scope, false, null, "openclawbrain control");
  printJson({
    command: "control",
    brainRoot: brainConfig.root,
    workspaceRoot,
    controlRoot,
    scope,
    control,
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
  const [command, arg1, arg2, arg3] = process.argv.slice(2);
  switch (command) {
    case "init":
      await commandInit(arg1);
      return;
    case "status":
      commandStatus();
      return;
    case "trace":
      commandTrace(arg1);
      return;
    case "replay":
      commandReplay();
      return;
    case "promote":
      commandPromote();
      return;
    case "rollback":
      commandRollback(arg1);
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
    case "control":
      commandControl(arg1, arg2, arg3);
      return;
    default:
      usage();
  }
}

void main().catch((error) => {
  process.stderr.write(`${(error as Error).message}\n`);
  process.exit(1);
});
