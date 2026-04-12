import type { AssembledSummaryMetadata } from "../assembler.js";

type RouteQualitySurfaceV1 = "status" | "proof";

export type RouteQualityPostureV1 = "promotable" | "held" | "quarantined";

export interface RouteQualityReplayVerdictV1 {
  passed: boolean | null;
  verdict: "pass" | "fail" | "unknown";
  summary: string | null;
}

export interface RouteQualityControlStateV1 {
  disabled: boolean;
  shadowMode: boolean;
  rolledBack: boolean;
  summary: string;
}

export interface RouteQualityHealthSummaryV1 {
  status: "healthy" | "missing";
  count: number;
  summary: string;
}

export interface RouteQualityCompactHealthSummaryV1 extends RouteQualityHealthSummaryV1 {
  freshCount: number;
  nonFreshCount: number;
  branchCount: number;
  snapshotCount: number;
  condensedCount: number;
  nonFreshPrevalence: number | null;
  snapshotShare: number | null;
}

export interface RouteQualityToolPriorHealthSummaryV1 extends RouteQualityHealthSummaryV1 {
  sourceCount: number;
  toolCount: number;
}

export type RouteQualitySummaryRoutingModeV1 = "ignore" | "summary_suffices" | "expand_to_source" | "prefer_typed_memory";

export interface RouteQualityRollbackLinkageV1 {
  rollbackKey: string | null;
  proofBundleId: string | null;
  activePackId: string | null;
  activePackVersion: number | null;
  routerIdentity: string | null;
  bound: boolean;
  summary: string;
}

export interface RouteQualitySummaryV1 {
  contract: "openclawbrain_route_quality_summary.v1";
  surface: RouteQualitySurfaceV1;
  activePackVersion: number | null;
  activePackId: string | null;
  routerIdentity: string | null;
  summaryRoutingMode: RouteQualitySummaryRoutingModeV1 | null;
  replayVerdict: RouteQualityReplayVerdictV1;
  controlState: RouteQualityControlStateV1;
  stopLocalHealth: RouteQualityHealthSummaryV1;
  compactHealth: RouteQualityCompactHealthSummaryV1;
  toolActionPriorsHealth: RouteQualityToolPriorHealthSummaryV1;
  rollbackLinkage: RouteQualityRollbackLinkageV1;
  posture: RouteQualityPostureV1;
  explainability: string;
  summary: string;
}

export interface RouteQualityWeightInputV1 {
  sourceNodeId: string;
  weight: number;
}

export interface RouteQualityToolPriorInputV1 {
  sourceNodeId: string;
  toolNodeId: string;
  weight: number;
}

export interface RouteQualitySummaryInputV1 {
  surface: RouteQualitySurfaceV1;
  activePackVersion: number | null;
  activePackId?: string | null;
  routerIdentity?: string | null;
  summaryRoutingMode?: RouteQualitySummaryRoutingModeV1 | null;
  replayVerdict?: {
    passed?: boolean | null;
    verdict?: string | null;
    summary?: string | null;
  } | null;
  stopLocalWeights?: RouteQualityWeightInputV1[] | null;
  toolActionPriors?: RouteQualityToolPriorInputV1[] | null;
  summaryMetadata?: Pick<AssembledSummaryMetadata, "totalCount" | "condensedCount" | "snapshotCount" | "branchCount" | "freshnessStateCounts" | "hasNonFreshSummaries"> | null;
  disabled?: boolean | null;
  shadowMode?: boolean | null;
  rolledBack?: boolean | null;
  rollbackKey?: string | null;
  proofBundleId?: string | null;
}

function normalizeText(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeBoolean(value: unknown): boolean | null {
  return value === true ? true : value === false ? false : null;
}

function normalizeNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  return null;
}

function normalizeSummaryRoutingMode(value: unknown): RouteQualitySummaryRoutingModeV1 | null {
  return value === "ignore" || value === "summary_suffices" || value === "expand_to_source" || value === "prefer_typed_memory"
    ? value
    : null;
}

function formatWeight(weight: number | null): string {
  if (weight === null) {
    return "n/a";
  }
  return weight.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function summarizeStopLocalHealth(weights: RouteQualityWeightInputV1[] | null | undefined): RouteQualityHealthSummaryV1 {
  const entries = Array.isArray(weights) ? weights : [];
  const count = entries.length;
  if (count === 0) {
    return {
      status: "missing",
      count: 0,
      summary: "STOP_LOCAL weights missing",
    };
  }

  const maxWeight = Math.max(...entries.map((entry) => entry.weight));
  return {
    status: "healthy",
    count,
    summary: `STOP_LOCAL weights: ${count} source(s); max=${formatWeight(maxWeight)}`,
  };
}

function summarizeToolActionPriorsHealth(weights: RouteQualityToolPriorInputV1[] | null | undefined): RouteQualityToolPriorHealthSummaryV1 {
  const entries = Array.isArray(weights) ? weights : [];
  const count = entries.length;
  if (count === 0) {
    return {
      status: "missing",
      count: 0,
      sourceCount: 0,
      toolCount: 0,
      summary: "toolActionPriors missing",
    };
  }

  const sourceCount = new Set(entries.map((entry) => entry.sourceNodeId)).size;
  const toolCount = new Set(entries.map((entry) => entry.toolNodeId)).size;
  const maxWeight = Math.max(...entries.map((entry) => entry.weight));
  return {
    status: "healthy",
    count,
    sourceCount,
    toolCount,
    summary: `toolActionPriors: ${count} prior(s) across ${sourceCount} source node(s) / ${toolCount} tool node(s); max=${formatWeight(maxWeight)}`,
  };
}

function summarizeCompactHealth(summaryMetadata: RouteQualitySummaryInputV1["summaryMetadata"]): RouteQualityCompactHealthSummaryV1 {
  const totalCount = summaryMetadata?.totalCount ?? 0;
  const condensedCount = summaryMetadata?.condensedCount ?? 0;
  const snapshotCount = summaryMetadata?.snapshotCount ?? 0;
  const branchCount = summaryMetadata?.branchCount ?? 0;
  const freshnessStateCounts = summaryMetadata?.freshnessStateCounts ?? {};
  const freshCount = freshnessStateCounts.fresh ?? Math.max(0, totalCount - Object.entries(freshnessStateCounts).reduce((sum, [freshnessState, value]) => sum + (freshnessState === "fresh" ? 0 : (value ?? 0)), 0));
  const nonFreshCount = Math.max(0, totalCount - freshCount);
  const nonFreshPrevalence = totalCount > 0 ? nonFreshCount / totalCount : null;
  const compactionPassCount = condensedCount + snapshotCount;
  const snapshotShare = compactionPassCount > 0 ? snapshotCount / compactionPassCount : null;

  if (totalCount === 0) {
    return {
      status: "missing",
      count: 0,
      freshCount: 0,
      nonFreshCount: 0,
      branchCount: 0,
      snapshotCount: 0,
      condensedCount: 0,
      nonFreshPrevalence: null,
      snapshotShare: null,
      summary: "compact health missing",
    };
  }

  const summaryBits = [
    `compact health: ${totalCount} summary item(s)`,
    branchCount > 1 ? `${branchCount} branches` : null,
    snapshotCount > 0 ? `${snapshotCount} snapshot(s)` : null,
    condensedCount > 0 ? `${condensedCount} condensed summary(s)` : null,
    nonFreshCount > 0 ? `${nonFreshCount} stale/superseded` : null,
  ].filter((bit): bit is string => bit !== null);

  return {
    status: "healthy",
    count: totalCount,
    freshCount,
    nonFreshCount,
    branchCount,
    snapshotCount,
    condensedCount,
    nonFreshPrevalence,
    snapshotShare,
    summary: `${summaryBits.join("; ")}${nonFreshPrevalence !== null ? `; non-fresh ${formatWeight(nonFreshPrevalence)}` : ""}${snapshotShare !== null ? `; snapshot share ${formatWeight(snapshotShare)}` : ""}`,
  };
}

function buildControlState(input: RouteQualitySummaryInputV1): RouteQualityControlStateV1 {
  const disabled = input.disabled === true;
  const shadowMode = input.shadowMode === true;
  const rolledBack = input.rolledBack === true;
  const flags = [
    disabled ? "disabled" : null,
    shadowMode ? "shadow-mode" : null,
    rolledBack ? "rolled-back" : null,
  ].filter((item): item is string => item !== null);

  return {
    disabled,
    shadowMode,
    rolledBack,
    summary: flags.length > 0 ? `controls: ${flags.join(", ")}` : "controls: live",
  };
}

function resolvePosture(params: {
  controlState: RouteQualityControlStateV1;
  replayPassed: boolean | null;
  stopLocalHealth: RouteQualityHealthSummaryV1;
  toolActionPriorsHealth: RouteQualityToolPriorHealthSummaryV1;
  activePackVersion: number | null;
}): RouteQualityPostureV1 {
  if (params.controlState.disabled || params.controlState.shadowMode || params.controlState.rolledBack) {
    return "quarantined";
  }
  if (params.activePackVersion === null) {
    return "held";
  }
  if (params.replayPassed !== true) {
    return "held";
  }
  if (params.stopLocalHealth.status !== "healthy" || params.toolActionPriorsHealth.status !== "healthy") {
    return "held";
  }
  return "promotable";
}

function buildExplainability(params: {
  surface: RouteQualitySurfaceV1;
  activePackId: string | null;
  routerIdentity: string | null;
  summaryRoutingMode: RouteQualitySummaryRoutingModeV1 | null;
  replayVerdict: RouteQualityReplayVerdictV1;
  stopLocalHealth: RouteQualityHealthSummaryV1;
  compactHealth: RouteQualityCompactHealthSummaryV1;
  toolActionPriorsHealth: RouteQualityToolPriorHealthSummaryV1;
  controlState: RouteQualityControlStateV1;
  posture: RouteQualityPostureV1;
}): string {
  const replayText = params.replayVerdict.passed === true
    ? "replay passed"
    : params.replayVerdict.passed === false
      ? "replay did not pass"
      : "replay verdict missing";
  const routingText = params.summaryRoutingMode ? `summary routing ${params.summaryRoutingMode}` : "summary routing unavailable";
  return `${params.surface} route quality: pack ${params.activePackId ?? "unbound"} / router ${params.routerIdentity ?? "unbound"}; ${replayText}; ${routingText}; ${params.stopLocalHealth.summary}; ${params.compactHealth.summary}; ${params.toolActionPriorsHealth.summary}; ${params.controlState.summary}; posture ${params.posture}.`;
}

function buildRollbackLinkage(params: {
  input: RouteQualitySummaryInputV1;
  activePackId: string | null;
  activePackVersion: number | null;
  routerIdentity: string | null;
}): RouteQualityRollbackLinkageV1 {
  const rollbackKey = normalizeText(params.input.rollbackKey);
  const proofBundleId = normalizeText(params.input.proofBundleId);
  const bound = rollbackKey !== null && (params.activePackId !== null || params.activePackVersion !== null || params.routerIdentity !== null);
  const linkageBits = [
    rollbackKey ? `rollback key ${rollbackKey}` : "rollback key missing",
    params.activePackId ?? (params.activePackVersion !== null ? `pack v${params.activePackVersion}` : "pack unbound"),
    params.routerIdentity ?? "router unbound",
  ];
  if (proofBundleId) {
    linkageBits.push(`proof bundle ${proofBundleId}`);
  }

  return {
    rollbackKey,
    proofBundleId,
    activePackId: params.activePackId,
    activePackVersion: params.activePackVersion,
    routerIdentity: params.routerIdentity,
    bound,
    summary: `${linkageBits.join(" · ")}${bound ? " (bound)" : " (not fully bound)"}`,
  };
}

export function buildRouteQualitySummaryV1(input: RouteQualitySummaryInputV1): RouteQualitySummaryV1 {
  const activePackVersion = normalizeNumber(input.activePackVersion);
  const activePackId = normalizeText(input.activePackId) ?? (activePackVersion !== null ? `brain-pack-v${activePackVersion}` : null);
  const routerIdentity = normalizeText(input.routerIdentity);
  const summaryRoutingMode = normalizeSummaryRoutingMode(input.summaryRoutingMode);
  const replayPassed = normalizeBoolean(input.replayVerdict?.passed);
  const replayVerdict: RouteQualityReplayVerdictV1 = {
    passed: replayPassed,
    verdict: replayPassed === true ? "pass" : replayPassed === false ? "fail" : "unknown",
    summary: normalizeText(input.replayVerdict?.summary)
      ?? (replayPassed === true ? "replay gate passed" : replayPassed === false ? "replay gate blocked" : "replay verdict unavailable"),
  };
  const stopLocalHealth = summarizeStopLocalHealth(input.stopLocalWeights);
  const compactHealth = summarizeCompactHealth(input.summaryMetadata);
  const toolActionPriorsHealth = summarizeToolActionPriorsHealth(input.toolActionPriors);
  const controlState = buildControlState(input);
  const posture = resolvePosture({
    controlState,
    replayPassed,
    stopLocalHealth,
    toolActionPriorsHealth,
    activePackVersion,
  });
  const rollbackLinkage = buildRollbackLinkage({
    input,
    activePackId,
    activePackVersion,
    routerIdentity,
  });
  const explainability = buildExplainability({
    surface: input.surface,
    activePackId,
    routerIdentity,
    summaryRoutingMode,
    replayVerdict,
    stopLocalHealth,
    compactHealth,
    toolActionPriorsHealth,
    controlState,
    posture,
  });

  return {
    contract: "openclawbrain_route_quality_summary.v1",
    surface: input.surface,
    activePackVersion,
    activePackId,
    routerIdentity,
    summaryRoutingMode,
    replayVerdict,
    controlState,
    stopLocalHealth,
    compactHealth,
    toolActionPriorsHealth,
    rollbackLinkage,
    posture,
    explainability,
    summary: `${input.surface} route quality: pack ${activePackId ?? "unbound"} / router ${routerIdentity ?? "unbound"}; replay ${replayVerdict.verdict}; summary routing ${summaryRoutingMode ?? "unavailable"}; STOP_LOCAL ${stopLocalHealth.status}; compact health ${compactHealth.status}; toolActionPriors ${toolActionPriorsHealth.status}; posture ${posture}.`,
  };
}

export function normalizeRouteQualitySummaryV1(value: unknown): RouteQualitySummaryV1 | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const summary = value as Partial<RouteQualitySummaryV1>;
  const contract = normalizeText(summary.contract);
  const surface = normalizeText(summary.surface);
  if (contract !== "openclawbrain_route_quality_summary.v1" || (surface !== "status" && surface !== "proof")) {
    return null;
  }

  return {
    contract: "openclawbrain_route_quality_summary.v1",
    surface,
    activePackVersion: normalizeNumber(summary.activePackVersion),
    activePackId: normalizeText(summary.activePackId),
    routerIdentity: normalizeText(summary.routerIdentity),
    summaryRoutingMode: normalizeSummaryRoutingMode((summary as { summaryRoutingMode?: unknown }).summaryRoutingMode),
    replayVerdict: {
      passed: normalizeBoolean(summary.replayVerdict?.passed),
      verdict: summary.replayVerdict?.verdict === "pass" || summary.replayVerdict?.verdict === "fail"
        ? summary.replayVerdict.verdict
        : "unknown",
      summary: normalizeText(summary.replayVerdict?.summary),
    },
    controlState: {
      disabled: summary.controlState?.disabled === true,
      shadowMode: summary.controlState?.shadowMode === true,
      rolledBack: summary.controlState?.rolledBack === true,
      summary: normalizeText(summary.controlState?.summary) ?? "controls: live",
    },
    stopLocalHealth: {
      status: summary.stopLocalHealth?.status === "healthy" ? "healthy" : "missing",
      count: normalizeNumber(summary.stopLocalHealth?.count) ?? 0,
      summary: normalizeText(summary.stopLocalHealth?.summary) ?? "STOP_LOCAL weights missing",
    },
    compactHealth: {
      status: summary.compactHealth?.status === "healthy" ? "healthy" : "missing",
      count: normalizeNumber(summary.compactHealth?.count) ?? 0,
      freshCount: normalizeNumber(summary.compactHealth?.freshCount) ?? 0,
      nonFreshCount: normalizeNumber(summary.compactHealth?.nonFreshCount) ?? 0,
      branchCount: normalizeNumber(summary.compactHealth?.branchCount) ?? 0,
      snapshotCount: normalizeNumber(summary.compactHealth?.snapshotCount) ?? 0,
      condensedCount: normalizeNumber(summary.compactHealth?.condensedCount) ?? 0,
      nonFreshPrevalence: normalizeNumber(summary.compactHealth?.nonFreshPrevalence),
      snapshotShare: normalizeNumber(summary.compactHealth?.snapshotShare),
      summary: normalizeText(summary.compactHealth?.summary) ?? "compact health missing",
    },
    toolActionPriorsHealth: {
      status: summary.toolActionPriorsHealth?.status === "healthy" ? "healthy" : "missing",
      count: normalizeNumber(summary.toolActionPriorsHealth?.count) ?? 0,
      sourceCount: normalizeNumber(summary.toolActionPriorsHealth?.sourceCount) ?? 0,
      toolCount: normalizeNumber(summary.toolActionPriorsHealth?.toolCount) ?? 0,
      summary: normalizeText(summary.toolActionPriorsHealth?.summary) ?? "toolActionPriors missing",
    },
    rollbackLinkage: {
      rollbackKey: normalizeText(summary.rollbackLinkage?.rollbackKey),
      proofBundleId: normalizeText(summary.rollbackLinkage?.proofBundleId),
      activePackId: normalizeText(summary.rollbackLinkage?.activePackId),
      activePackVersion: normalizeNumber(summary.rollbackLinkage?.activePackVersion),
      routerIdentity: normalizeText(summary.rollbackLinkage?.routerIdentity),
      bound: summary.rollbackLinkage?.bound === true,
      summary: normalizeText(summary.rollbackLinkage?.summary) ?? "rollback linkage unavailable",
    },
    posture: summary.posture === "promotable" || summary.posture === "held" || summary.posture === "quarantined"
      ? summary.posture
      : "held",
    explainability: normalizeText(summary.explainability) ?? "route quality explainability unavailable",
    summary: normalizeText(summary.summary)
      ?? `${surface} route quality: pack ${normalizeText(summary.activePackId) ?? "unbound"} / router ${normalizeText(summary.routerIdentity) ?? "unbound"}; posture ${summary.posture ?? "held"}.`,
  };
}
