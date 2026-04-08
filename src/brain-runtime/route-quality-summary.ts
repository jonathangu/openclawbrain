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

export interface RouteQualityToolPriorHealthSummaryV1 extends RouteQualityHealthSummaryV1 {
  sourceCount: number;
  toolCount: number;
}

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
  replayVerdict: RouteQualityReplayVerdictV1;
  controlState: RouteQualityControlStateV1;
  stopLocalHealth: RouteQualityHealthSummaryV1;
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
  replayVerdict?: {
    passed?: boolean | null;
    verdict?: string | null;
    summary?: string | null;
  } | null;
  stopLocalWeights?: RouteQualityWeightInputV1[] | null;
  toolActionPriors?: RouteQualityToolPriorInputV1[] | null;
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
  replayVerdict: RouteQualityReplayVerdictV1;
  stopLocalHealth: RouteQualityHealthSummaryV1;
  toolActionPriorsHealth: RouteQualityToolPriorHealthSummaryV1;
  controlState: RouteQualityControlStateV1;
  posture: RouteQualityPostureV1;
}): string {
  const replayText = params.replayVerdict.passed === true
    ? "replay passed"
    : params.replayVerdict.passed === false
      ? "replay did not pass"
      : "replay verdict missing";
  return `${params.surface} route quality: pack ${params.activePackId ?? "unbound"} / router ${params.routerIdentity ?? "unbound"}; ${replayText}; ${params.stopLocalHealth.summary}; ${params.toolActionPriorsHealth.summary}; ${params.controlState.summary}; posture ${params.posture}.`;
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
  const replayPassed = normalizeBoolean(input.replayVerdict?.passed);
  const replayVerdict: RouteQualityReplayVerdictV1 = {
    passed: replayPassed,
    verdict: replayPassed === true ? "pass" : replayPassed === false ? "fail" : "unknown",
    summary: normalizeText(input.replayVerdict?.summary)
      ?? (replayPassed === true ? "replay gate passed" : replayPassed === false ? "replay gate blocked" : "replay verdict unavailable"),
  };
  const stopLocalHealth = summarizeStopLocalHealth(input.stopLocalWeights);
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
    replayVerdict,
    stopLocalHealth,
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
    replayVerdict,
    controlState,
    stopLocalHealth,
    toolActionPriorsHealth,
    rollbackLinkage,
    posture,
    explainability,
    summary: `${input.surface} route quality: pack ${activePackId ?? "unbound"} / router ${routerIdentity ?? "unbound"}; replay ${replayVerdict.verdict}; STOP_LOCAL ${stopLocalHealth.status}; toolActionPriors ${toolActionPriorsHealth.status}; posture ${posture}.`,
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
