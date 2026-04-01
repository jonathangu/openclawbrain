import path from "node:path";
import process from "node:process";

import { compileRuntimeFromActivation } from "@openclawbrain/compiler";
import { CONTRACT_IDS } from "@openclawbrain/contracts";
import { inspectActivationState, loadPackFromActivation } from "@openclawbrain/pack-format";

import { appendServeTimeRouteDecisionLog } from "./learning-spine.js";

const DEFAULT_AGENT_ID = "openclaw-runtime";
const BRAIN_SERVE_HOT_PATH_TIMING_DETAIL =
  "Measured inside compileRuntimeContext before serve-route logging; includes serve-path normalization, active-pack lookup, structural-budget resolution, route/candidate selection, and prompt assembly when run; excludes background scanner/embedder/teacher work, promotion, and runtime event-export writes.";

function toErrorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function normalizeNonEmptyString(value, fieldName) {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${fieldName} is required`);
  }

  return value.trim();
}

function normalizeOptionalString(value) {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : undefined;
}

function normalizeNonNegativeInteger(value, fieldName, fallbackValue) {
  if (value === undefined || value === null) {
    return fallbackValue;
  }

  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${fieldName} must be a non-negative integer`);
  }

  return value;
}

const RUNTIME_COMPARATIVE_REPLAY_MODE_CONFIG = {
  vector_only: {
    routeMode: "heuristic",
    selectionMode: "flat_rank_v1",
  },
  graph_prior_only: {
    routeMode: "heuristic",
    selectionMode: "graph_walk_v1",
  },
  learned_route: {
    routeMode: "learned",
    selectionMode: "graph_walk_v1",
  },
};

function resolveRuntimeComparativeReplayMode(value) {
  return Object.prototype.hasOwnProperty.call(RUNTIME_COMPARATIVE_REPLAY_MODE_CONFIG, value)
    ? value
    : null;
}

function resolveCompileModePlan(modeValue, selectionModeValue) {
  const requestedSelectionMode = normalizeCompileSelectionMode(selectionModeValue);
  const comparativeMode = resolveRuntimeComparativeReplayMode(modeValue);

  if (comparativeMode === null) {
    if (modeValue === undefined) {
      return {
        comparativeMode: null,
        routeMode: "heuristic",
        selectionMode: requestedSelectionMode,
      };
    }

    if (modeValue === "heuristic" || modeValue === "learned") {
      return {
        comparativeMode: null,
        routeMode: modeValue,
        selectionMode: requestedSelectionMode,
      };
    }

    throw new Error(
      "mode must be heuristic, learned, vector_only, graph_prior_only, or learned_route",
    );
  }

  const plan = RUNTIME_COMPARATIVE_REPLAY_MODE_CONFIG[comparativeMode];

  if (requestedSelectionMode !== undefined && requestedSelectionMode !== plan.selectionMode) {
    throw new Error(
      `selectionMode ${requestedSelectionMode} conflicts with comparative mode ${comparativeMode}, expected ${plan.selectionMode}`,
    );
  }

  return {
    comparativeMode,
    routeMode: plan.routeMode,
    selectionMode: plan.selectionMode,
  };
}

function resolveSyntheticTurnMode(value) {
  const comparativeMode = resolveRuntimeComparativeReplayMode(value);
  if (comparativeMode !== null) {
    return RUNTIME_COMPARATIVE_REPLAY_MODE_CONFIG[comparativeMode].routeMode;
  }

  return value === "heuristic" || value === "learned" ? value : undefined;
}

function normalizeCompileSelectionMode(value) {
  if (value === undefined) {
    return undefined;
  }

  if (value === "flat_rank_v1" || value === "graph_walk_v1") {
    return value;
  }

  throw new Error(`selectionMode must be flat_rank_v1 or graph_walk_v1, received ${String(value)}`);
}

function normalizeRuntimeHints(value) {
  if (value === undefined) {
    return [];
  }

  if (!Array.isArray(value) || value.some((item) => typeof item !== "string" || item.trim().length === 0)) {
    throw new Error("runtimeHints must be an array of non-empty strings");
  }

  return value.map((item) => item.trim());
}

function isPlainObject(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }

  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function normalizeOptionalBoolean(value) {
  return typeof value === "boolean" ? value : undefined;
}

function normalizeOptionalFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function buildPromptTruthSummaryLines(compileResponse) {
  const lines = [];
  const structuralSignals = isPlainObject(compileResponse.structuralSignals)
    ? compileResponse.structuralSignals
    : null;
  const footer = normalizeOptionalString(compileResponse.footer);

  if (structuralSignals !== null) {
    const chosenStopCount = normalizeOptionalFiniteNumber(structuralSignals.chosenStopCount);
    const forcedStopCount = normalizeOptionalFiniteNumber(structuralSignals.forcedStopCount);
    const droppedProposalCount = normalizeOptionalFiniteNumber(structuralSignals.droppedProposalCount);

    if (
      chosenStopCount !== undefined ||
      forcedStopCount !== undefined ||
      droppedProposalCount !== undefined
    ) {
      lines.push(
        `STOP_TRUTH: chosen=${chosenStopCount ?? 0} forced=${forcedStopCount ?? 0} dropped=${droppedProposalCount ?? 0}`,
      );
    }

    const queryInterrupted = normalizeOptionalBoolean(structuralSignals.queryInterrupted);
    const interruptionStage = normalizeOptionalString(structuralSignals.interruptionStage);
    const interruptionReason = normalizeOptionalString(structuralSignals.interruptionReason);
    const servedPartial = normalizeOptionalBoolean(structuralSignals.servedPartial);

    if (
      queryInterrupted !== undefined ||
      interruptionStage !== undefined ||
      interruptionReason !== undefined ||
      servedPartial !== undefined
    ) {
      lines.push(
        `INTERRUPTION: interrupted=${queryInterrupted === true ? "true" : "false"} stage=${interruptionStage ?? "unknown"} reason=${interruptionReason ?? "unknown"} servedPartial=${servedPartial === true ? "true" : "false"}`,
      );
    }

    const interruptionAccounting = isPlainObject(structuralSignals.interruptionAccounting)
      ? structuralSignals.interruptionAccounting
      : null;
    if (interruptionAccounting !== null) {
      const droppedFrontierCount = Array.isArray(interruptionAccounting.droppedFrontierNodeIds)
        ? interruptionAccounting.droppedFrontierNodeIds.filter((value) => typeof value === "string").length
        : 0;
      const completedExpansionCount = normalizeOptionalFiniteNumber(
        interruptionAccounting.completedExpansionCount,
      );
      const maxExpansions = normalizeOptionalFiniteNumber(interruptionAccounting.maxExpansions);
      const budgetUsed = normalizeOptionalFiniteNumber(interruptionAccounting.budgetUsed);
      const budgetTotal = normalizeOptionalFiniteNumber(interruptionAccounting.budgetTotal);
      const budgetUtilization = normalizeOptionalFiniteNumber(interruptionAccounting.budgetUtilization);
      const droppedProposalTotal = normalizeOptionalFiniteNumber(
        interruptionAccounting.droppedProposalCount,
      );

      lines.push(
        `INTERRUPTION_ACCOUNTING: frontierDropped=${droppedFrontierCount} completedExpansions=${completedExpansionCount ?? 0}/${maxExpansions ?? 0} proposalsDropped=${droppedProposalTotal ?? 0} budgetUsed=${budgetUsed ?? 0}/${budgetTotal ?? 0} utilization=${budgetUtilization === undefined ? "unknown" : `${Math.round(budgetUtilization * 100)}%`}`,
      );
    }
  }

  if (footer !== undefined) {
    lines.push(`TRACE: ${footer}`);
  }

  return lines;
}

function formatPromptContext(compileResponse) {
  const lines = [
    "[BRAIN_CONTEXT v1]",
    `PACK_ID: ${compileResponse.packId}`,
    `MODE: ${compileResponse.diagnostics.modeEffective}`,
  ];

  if (compileResponse.diagnostics.routerIdentity !== null) {
    lines.push(`ROUTER: ${compileResponse.diagnostics.routerIdentity}`);
  }

  const truthSummaryLines = buildPromptTruthSummaryLines(compileResponse);
  if (truthSummaryLines.length > 0) {
    lines.push(...truthSummaryLines);
  }

  if (compileResponse.selectedContext.length > 0) {
    lines.push("");
  }

  for (const block of compileResponse.selectedContext) {
    lines.push(`PROVENANCE_REF: ctx_${block.id}`);
    lines.push(`BLOCK_ID: ${block.id}`);
    lines.push(block.text.trim());
    lines.push("");
  }

  if (lines.at(-1) === "") {
    lines.pop();
  }

  lines.push("[/BRAIN_CONTEXT]");

  return `${lines.join("\n")}\n`;
}

function resolveActivationRootForFailure(value) {
  return path.resolve(normalizeOptionalString(value) ?? ".");
}

function monotonicClockNs() {
  return process.hrtime.bigint();
}

function elapsedMsFrom(startedAtNs, endedAtNs = monotonicClockNs()) {
  return Number(endedAtNs - startedAtNs) / 1_000_000;
}

function roundHotPathTimingMs(value) {
  return Math.round(value * 1_000) / 1_000;
}

function buildUnavailableBrainServeHotPathTiming(detail) {
  return {
    scope: "brain_serve_hot_path_only",
    totalMs: null,
    routeSelectionMs: null,
    promptAssemblyMs: null,
    otherMs: null,
    backgroundWorkIncluded: false,
    detail,
  };
}

function buildBrainServeHotPathTiming(input) {
  const totalMs = roundHotPathTimingMs(input.totalMs);
  const routeSelectionMs =
    input.routeSelectionMs === null ? null : roundHotPathTimingMs(input.routeSelectionMs);
  const promptAssemblyMs =
    input.promptAssemblyMs === null ? null : roundHotPathTimingMs(input.promptAssemblyMs);
  const otherMs = roundHotPathTimingMs(
    Math.max(0, input.totalMs - (input.routeSelectionMs ?? 0) - (input.promptAssemblyMs ?? 0)),
  );

  return {
    scope: "brain_serve_hot_path_only",
    totalMs,
    routeSelectionMs,
    promptAssemblyMs,
    otherMs,
    backgroundWorkIncluded: false,
    detail: BRAIN_SERVE_HOT_PATH_TIMING_DETAIL,
  };
}

function failOpenCompileResult(
  error,
  activationRoot,
  timing = buildUnavailableBrainServeHotPathTiming("serve-path timing was unavailable"),
) {
  return {
    ok: false,
    fallbackToStaticContext: true,
    hardRequirementViolated: false,
    activationRoot: path.resolve(activationRoot),
    error: toErrorMessage(error),
    brainContext: "",
    timing,
  };
}

function classifyCompileFailure(
  error,
  activationRoot,
  timing = buildUnavailableBrainServeHotPathTiming("serve-path timing was unavailable"),
) {
  const resolvedActivationRoot = path.resolve(activationRoot);

  try {
    const inspection = inspectActivationState(resolvedActivationRoot);
    const active = inspection.active;

    if (active !== null && active.routePolicy === "requires_learned_routing") {
      const failureReason = active.findings.length > 0 ? active.findings.join("; ") : toErrorMessage(error);

      return {
        ok: false,
        fallbackToStaticContext: false,
        hardRequirementViolated: true,
        activationRoot: resolvedActivationRoot,
        error: `Learned-routing hotpath hard requirement violated for active pack ${active.packId} (routerIdentity=${active.routerIdentity ?? "null"}): ${failureReason}`,
        brainContext: "",
        timing,
      };
    }
  } catch {
    return failOpenCompileResult(error, resolvedActivationRoot, timing);
  }

  return failOpenCompileResult(error, resolvedActivationRoot, timing);
}

function uniqueNotes(notes) {
  return [...new Set(notes.filter((note) => note.length > 0))];
}

function buildServeRouteLogFailOpenWarning(scope, error) {
  return `learning spine serve route log failed open (${scope}): ${toErrorMessage(error)}`;
}

function buildServeRouteLogFailOpenNotes(scope, error) {
  return [
    "serve_route_log_status=fail_open",
    `serve_route_log_scope=${scope}`,
    `serve_route_log_error=${toErrorMessage(error)}`,
  ];
}

function normalizeServeRouteChannel(value) {
  if (typeof value !== "string") {
    return undefined;
  }

  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function normalizeServeRouteMessage(value) {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeServeRouteInstalledEntryPath(value) {
  if (typeof value !== "string") {
    return null;
  }

  const trimmed = value.trim();
  return trimmed.length === 0 ? null : path.resolve(trimmed);
}

function buildCompileServeRouteBreadcrumbs(input) {
  return {
    entrypoint: "compileRuntimeContext",
    invocationSurface: input._serveRouteBreadcrumbs?.invocationSurface ?? "direct_compile_call",
    hostEvent: input._serveRouteBreadcrumbs?.hostEvent ?? null,
    installedEntryPath: normalizeServeRouteInstalledEntryPath(input._serveRouteBreadcrumbs?.installedEntryPath),
    syntheticTurn: true,
  };
}

function appendCompileServeRouteDecisionLog(input) {
  if (input.compileInput._suppressServeLog) {
    return;
  }

  if (!input.compileResult.ok && input.userMessage.length === 0) {
    return;
  }

  const recordedAt = new Date().toISOString();
  const sessionId = normalizeServeRouteChannel(input.compileInput.sessionId) ?? `ext-compile-${Date.now()}`;
  const channel = normalizeServeRouteChannel(input.compileInput.channel) ?? "extension";
  const syntheticTurn = {
    sessionId,
    channel,
    userMessage: input.userMessage,
    createdAt: recordedAt,
  };

  if (input.compileInput.maxContextBlocks !== undefined) {
    syntheticTurn.maxContextBlocks = input.compileInput.maxContextBlocks;
  }

  if (input.compileInput.maxContextChars !== undefined) {
    syntheticTurn.maxContextChars = input.compileInput.maxContextChars;
  }

  if (input.compileInput.budgetStrategy === "fixed_v1" || input.compileInput.budgetStrategy === "empirical_v1") {
    syntheticTurn.budgetStrategy = input.compileInput.budgetStrategy;
  }

  const syntheticTurnMode = resolveSyntheticTurnMode(input.compileInput.mode);

  if (syntheticTurnMode !== undefined) {
    syntheticTurn.mode = syntheticTurnMode;
  }

  if (input.compileInput.runtimeHints !== undefined) {
    syntheticTurn.runtimeHints = input.compileInput.runtimeHints;
  }

  try {
    appendServeTimeRouteDecisionLog({
      activationRoot: input.activationRoot,
      turn: syntheticTurn,
      compileResult: input.compileResult,
      recordedAt,
      breadcrumbs: buildCompileServeRouteBreadcrumbs(input.compileInput),
    });
  } catch (error) {
    if (input.compileResult.ok) {
      input.compileResult.compileResponse.diagnostics.notes = uniqueNotes([
        ...input.compileResult.compileResponse.diagnostics.notes,
        ...buildServeRouteLogFailOpenNotes("compileRuntimeContext", error),
      ]);
    }

    console.warn(
      `[openclawbrain] ${buildServeRouteLogFailOpenWarning("compileRuntimeContext", error)} ` +
        `(activationRoot=${input.activationRoot}, sessionId=${sessionId}, channel=${channel})`,
    );
  }
}

function clampInteger(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, Math.round(value)));
}

function deriveEmpiricalStructuralBudget(input) {
  const requestedStrategy = input.requestedStrategy ?? "fixed_v1";
  const defaultMaxContextBlocks = input.defaultMaxContextBlocks ?? 4;
  const minimumMaxContextBlocks = input.minimumMaxContextBlocks ?? 2;
  const maximumMaxContextBlocks = input.maximumMaxContextBlocks ?? 6;

  if (input.requestedMaxContextBlocks !== undefined) {
    const maxContextBlocks = clampInteger(input.requestedMaxContextBlocks, 0, Number.MAX_SAFE_INTEGER);

    return {
      requestedStrategy,
      effectiveStrategy: requestedStrategy,
      maxContextBlocks,
      defaultMaxContextBlocks,
      evidence: { split: 0, merge: 0, prune: 0, connect: 0 },
      evidenceTotal: 0,
      tendencies: { split: 0, merge: 0, prune: 0, connect: 0 },
      notes: [
        `requested_budget_strategy=${requestedStrategy}`,
        `requested_max_context_blocks=${maxContextBlocks}`,
        `resolved_budget_strategy=${requestedStrategy}`,
        `resolved_max_context_blocks=${maxContextBlocks}`,
        "structural_budget_source=caller_override",
      ],
    };
  }

  if (requestedStrategy !== "empirical_v1") {
    return {
      requestedStrategy,
      effectiveStrategy: requestedStrategy,
      maxContextBlocks: defaultMaxContextBlocks,
      defaultMaxContextBlocks,
      evidence: { split: 0, merge: 0, prune: 0, connect: 0 },
      evidenceTotal: 0,
      tendencies: { split: 0, merge: 0, prune: 0, connect: 0 },
      notes: [
        `requested_budget_strategy=${requestedStrategy}`,
        `resolved_budget_strategy=${requestedStrategy}`,
        `resolved_max_context_blocks=${defaultMaxContextBlocks}`,
        `structural_budget_source=${requestedStrategy === "fixed_v1" ? "fixed_default" : "fixed_fallback"}`,
      ],
    };
  }

  const evidence = {
    split: Math.max(0, input.evolution?.structuralOps.split ?? 0),
    merge: Math.max(0, input.evolution?.structuralOps.merge ?? 0),
    prune: Math.max(
      0,
      Math.max(input.evolution?.structuralOps.prune ?? 0, input.evolution?.prunedBlockIds.length ?? 0),
    ),
    connect: Math.max(0, input.evolution?.structuralOps.connect ?? 0),
  };
  const evidenceTotal = evidence.split + evidence.merge + evidence.prune + evidence.connect;

  if (evidenceTotal === 0) {
    return {
      requestedStrategy,
      effectiveStrategy: "fixed_v1",
      maxContextBlocks: defaultMaxContextBlocks,
      defaultMaxContextBlocks,
      evidence,
      evidenceTotal,
      tendencies: { split: 0, merge: 0, prune: 0, connect: 0 },
      notes: [
        `requested_budget_strategy=${requestedStrategy}`,
        "resolved_budget_strategy=fixed_v1",
        `resolved_max_context_blocks=${defaultMaxContextBlocks}`,
        "structural_budget_source=no_evidence_fallback",
      ],
    };
  }

  const tendencies = {
    split: evidence.split / evidenceTotal,
    merge: evidence.merge / evidenceTotal,
    prune: evidence.prune / evidenceTotal,
    connect: evidence.connect / evidenceTotal,
  };
  const expansionPressure = tendencies.split + tendencies.connect;
  const contractionPressure = tendencies.merge + tendencies.prune;
  const directionalPressure = expansionPressure - contractionPressure;
  const maxContextBlocks = clampInteger(
    defaultMaxContextBlocks + directionalPressure * 2,
    minimumMaxContextBlocks,
    maximumMaxContextBlocks,
  );

  return {
    requestedStrategy,
    effectiveStrategy: requestedStrategy,
    maxContextBlocks,
    defaultMaxContextBlocks,
    evidence,
    evidenceTotal,
    tendencies,
    notes: [
      `requested_budget_strategy=${requestedStrategy}`,
      `resolved_budget_strategy=${requestedStrategy}`,
      `resolved_max_context_blocks=${maxContextBlocks}`,
      "structural_budget_source=graph_evolution_empirical_v1",
      `structural_budget_evidence=split:${evidence.split},merge:${evidence.merge},prune:${evidence.prune},connect:${evidence.connect},total:${evidenceTotal}`,
      `structural_budget_tendencies=split:${tendencies.split.toFixed(4)},merge:${tendencies.merge.toFixed(4)},prune:${tendencies.prune.toFixed(4)},connect:${tendencies.connect.toFixed(4)}`,
      `structural_budget_pressures=expand:${expansionPressure.toFixed(4)},contract:${contractionPressure.toFixed(4)},directional:${directionalPressure.toFixed(4)}`,
    ],
  };
}

function resolveCompileBudget(target, input) {
  const pack = loadPackFromActivation(target.activationRoot, "active");

  return deriveEmpiricalStructuralBudget({
    requestedStrategy: input.budgetStrategy ?? "empirical_v1",
    ...(input.maxContextBlocks !== undefined ? { requestedMaxContextBlocks: input.maxContextBlocks } : {}),
    ...(pack?.graph.evolution !== undefined ? { evolution: pack.graph.evolution } : {}),
    defaultMaxContextBlocks: 4,
    minimumMaxContextBlocks: 2,
    maximumMaxContextBlocks: 6,
  });
}

function resolveActivePackForCompile(activationRoot) {
  const resolvedActivationRoot = path.resolve(normalizeNonEmptyString(activationRoot, "activationRoot"));
  const inspection = inspectActivationState(resolvedActivationRoot);
  const activePointer = inspection.pointers.active;

  if (inspection.active === null || activePointer === null) {
    throw new Error(`No active pack pointer found in ${resolvedActivationRoot}`);
  }

  if (!inspection.active.activationReady) {
    throw new Error(`Active pack is not activation-ready: ${inspection.active.findings.join("; ")}`);
  }

  return {
    activationRoot: resolvedActivationRoot,
    activePointer,
    inspection: inspection.active,
  };
}

function normalizeFrozenReplayEvalIdentity(value) {
  if (value === undefined || value === null) {
    return null;
  }

  if (typeof value !== "object") {
    throw new Error("_frozenReplayEvalIdentity must be an object");
  }

  const frozenIdentity = value;

  return {
    packId: normalizeNonEmptyString(frozenIdentity.packId, "_frozenReplayEvalIdentity.packId"),
    routerIdentity: normalizeOptionalString(frozenIdentity.routerIdentity) ?? null,
  };
}

function validateFrozenReplayEvalIdentity(target, frozenReplayEvalIdentity) {
  if (frozenReplayEvalIdentity === null) {
    return;
  }

  const actualRouterIdentity = target.inspection.routerIdentity ?? null;

  if (
    target.activePointer.packId === frozenReplayEvalIdentity.packId &&
    actualRouterIdentity === frozenReplayEvalIdentity.routerIdentity
  ) {
    return;
  }

  throw new Error(
    `Frozen replay eval identity mismatch: expected pack ${frozenReplayEvalIdentity.packId} ` +
      `(routerIdentity=${frozenReplayEvalIdentity.routerIdentity ?? "null"}), received pack ${target.activePointer.packId} ` +
      `(routerIdentity=${actualRouterIdentity ?? "null"})`,
  );
}

export function compileRuntimeContext(input) {
  const totalStartedAtNs = monotonicClockNs();
  const fallbackActivationRoot = resolveActivationRootForFailure(input.activationRoot);
  let activationRoot = fallbackActivationRoot;
  let agentId = process.env.OPENCLAWBRAIN_AGENT_ID ?? DEFAULT_AGENT_ID;
  let runtimeHints = [];
  let comparativeMode = null;
  let selectionMode;
  let userMessage = "";
  let maxContextChars;
  let mode = "heuristic";
  let frozenReplayEvalIdentity = null;
  let routeSelectionStartedAtNs = null;
  let routeSelectionMs = null;
  let promptAssemblyStartedAtNs = null;
  let promptAssemblyMs = null;
  let result;

  try {
    activationRoot = path.resolve(normalizeNonEmptyString(input.activationRoot, "activationRoot"));
    agentId = normalizeOptionalString(input.agentId) ?? process.env.OPENCLAWBRAIN_AGENT_ID ?? DEFAULT_AGENT_ID;
    runtimeHints = normalizeRuntimeHints(input.runtimeHints);
    userMessage = normalizeNonEmptyString(input.message, "message");
    maxContextChars =
      input.maxContextChars !== undefined
        ? normalizeNonNegativeInteger(input.maxContextChars, "maxContextChars", input.maxContextChars)
        : undefined;
    const compileModePlan = resolveCompileModePlan(input.mode, input.selectionMode);
    comparativeMode = compileModePlan.comparativeMode;
    mode = compileModePlan.routeMode;
    selectionMode = compileModePlan.selectionMode;
    frozenReplayEvalIdentity = normalizeFrozenReplayEvalIdentity(input._frozenReplayEvalIdentity);
  } catch (error) {
    result = failOpenCompileResult(
      error,
      fallbackActivationRoot,
      buildBrainServeHotPathTiming({
        totalMs: elapsedMsFrom(totalStartedAtNs),
        routeSelectionMs,
        promptAssemblyMs,
      }),
    );
    appendCompileServeRouteDecisionLog({
      compileInput: input,
      activationRoot: result.activationRoot,
      compileResult: result,
      userMessage: normalizeServeRouteMessage(input.message),
    });
    return result;
  }

  try {
    const target = resolveActivePackForCompile(activationRoot);
    validateFrozenReplayEvalIdentity(target, frozenReplayEvalIdentity);
    const resolvedBudget = resolveCompileBudget(target, input);
    routeSelectionStartedAtNs = monotonicClockNs();
    const compile = compileRuntimeFromActivation(
      activationRoot,
      {
        contract: CONTRACT_IDS.runtimeCompile,
        agentId,
        userMessage,
        maxContextBlocks: resolvedBudget.maxContextBlocks,
        ...(maxContextChars !== undefined ? { maxContextChars } : {}),
        modeRequested: mode,
        activePackId: target.activePointer.packId,
        ...(input.compactionMode !== undefined ? { compactionMode: input.compactionMode } : {}),
        ...(runtimeHints.length > 0 ? { runtimeHints } : {}),
      },
      {
        ...(selectionMode !== undefined ? { selectionMode } : {}),
      },
    );
    routeSelectionMs = elapsedMsFrom(routeSelectionStartedAtNs);
    const selectionEngine = selectionMode ?? "flat_rank_v1";
    const compileResponse = {
      ...compile.response,
      diagnostics: {
        ...compile.response.diagnostics,
        notes: uniqueNotes([
          ...compile.response.diagnostics.notes,
          ...resolvedBudget.notes,
          `selection_engine=${selectionEngine}`,
          ...(comparativeMode === null
            ? []
            : [
                `comparative_mode=${comparativeMode}`,
                `comparative_mode_plan=${mode}+${selectionEngine}`,
              ]),
          ...(frozenReplayEvalIdentity === null
            ? []
            : [
                `replay_eval_pack_id=${frozenReplayEvalIdentity.packId}`,
                `replay_eval_router_identity=${frozenReplayEvalIdentity.routerIdentity ?? "null"}`,
              ]),
          "OpenClaw remains the runtime owner",
        ]),
      },
    };
    promptAssemblyStartedAtNs = monotonicClockNs();
    const brainContext = formatPromptContext(compileResponse);
    promptAssemblyMs = elapsedMsFrom(promptAssemblyStartedAtNs);
    result = {
      ok: true,
      fallbackToStaticContext: false,
      hardRequirementViolated: false,
      activationRoot,
      activePackId: compile.target.packId,
      packRootDir: path.resolve(target.activePointer.packRootDir),
      compileResponse,
      brainContext,
      timing: buildBrainServeHotPathTiming({
        totalMs: elapsedMsFrom(totalStartedAtNs),
        routeSelectionMs,
        promptAssemblyMs,
      }),
    };
  } catch (error) {
    if (routeSelectionStartedAtNs !== null && routeSelectionMs === null) {
      routeSelectionMs = elapsedMsFrom(routeSelectionStartedAtNs);
    }

    if (promptAssemblyStartedAtNs !== null && promptAssemblyMs === null) {
      promptAssemblyMs = elapsedMsFrom(promptAssemblyStartedAtNs);
    }

    result = classifyCompileFailure(
      error,
      activationRoot,
      buildBrainServeHotPathTiming({
        totalMs: elapsedMsFrom(totalStartedAtNs),
        routeSelectionMs,
        promptAssemblyMs,
      }),
    );
  }

  appendCompileServeRouteDecisionLog({
    compileInput: input,
    activationRoot: result.activationRoot,
    compileResult: result,
    userMessage,
  });

  return result;
}
