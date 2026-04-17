import { createHash } from "node:crypto";
import { Type, type Static } from "@sinclair/typebox";
import { Value } from "@sinclair/typebox/value";
import type { ColdStartStopLabelV1 } from "./cold-start-router-contracts.ts";
import type { RouteDecisionRowV1 } from "./route-rows.js";
import type { GraphifyRouteObjectiveTraceLabelLikeV1 } from "./graphify-training-bridge.js";

export const POLICY_SUPERVISION_ROW_VERSION_V1 = 1 as const;
export const POLICY_SUPERVISION_ROW_CONTRACT_V1 = "policy_supervision_row.v1";

export const POLICY_SUPERVISION_ROW_TYPES = ["activate", "abstain", "stop_local", "select_context", "continue_expand"] as const;
export const POLICY_SUPERVISION_PROJECTION_STATUSES = ["trace_projected", "decision_point_reviewed", "owner_labeled"] as const;
export const HARD_NEGATIVE_CLASSES = [
  "unnecessary_activation",
  "tie_with_cost",
  "graph_prior_preferred",
  "stale_memory",
  "wrapper_heavy",
  "no_outcome_change",
] as const;
const NET_UTILITY_DELTA_SOURCES = ["reviewed", "replay_inferred"] as const;
const CHOSEN_ACTION_KINDS = ["traverse", "tool_capability", "tool_instance", "stop_local", "stop"] as const;
const STOP_LABELS = ["CONTINUE", "STOP_LOCAL", "STOP"] as const;

export type PolicySupervisionRowType = (typeof POLICY_SUPERVISION_ROW_TYPES)[number];
export type PolicySupervisionProjectionStatus = (typeof POLICY_SUPERVISION_PROJECTION_STATUSES)[number];
export type HardNegativeClass = (typeof HARD_NEGATIVE_CLASSES)[number];

function hashStableParts(parts: Array<string | number | null | undefined>): string {
  return createHash("sha256").update(parts.map((part) => String(part ?? "")).join("\u001f")).digest("hex").slice(0, 16);
}

function normalizeString(value: string | null | undefined): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

export const PolicySupervisionTraceSliceSchemaV1 = Type.Object(
  {
    route_row_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    route_fn_version: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    chosen_action_kind: Type.Union([
      ...CHOSEN_ACTION_KINDS.map((kind) => Type.Literal(kind)),
      Type.Null(),
    ]),
    stop_label: Type.Union([
      ...STOP_LABELS.map((label) => Type.Literal(label)),
      Type.Null(),
    ]),
    query_text_hash: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
  },
  { additionalProperties: false },
);

export type PolicySupervisionTraceSliceV1 = Static<typeof PolicySupervisionTraceSliceSchemaV1>;

export const PolicySupervisionRowSchemaV1 = Type.Object(
  {
    schema_version: Type.Literal(POLICY_SUPERVISION_ROW_VERSION_V1),
    contract: Type.Literal(POLICY_SUPERVISION_ROW_CONTRACT_V1),
    row_id: Type.String({ minLength: 1 }),
    trace_id: Type.String({ minLength: 1 }),
    episode_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    decision_point_id: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    row_type: Type.Union(POLICY_SUPERVISION_ROW_TYPES.map((v) => Type.Literal(v))),
    focus_lane: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    trace_slice: PolicySupervisionTraceSliceSchemaV1,
    row_weight: Type.Number({ minimum: 0 }),
    confidence_target: Type.Union([Type.Number({ minimum: 0, maximum: 1 }), Type.Null()]),
    hard_negative_class: Type.Union([
      ...HARD_NEGATIVE_CLASSES.map((v) => Type.Literal(v)),
      Type.Null(),
    ]),
    net_utility_delta: Type.Union([Type.Number(), Type.Null()]),
    net_utility_delta_source: Type.Union([
      ...NET_UTILITY_DELTA_SOURCES.map((v) => Type.Literal(v)),
      Type.Null(),
    ]),
    projection_status: Type.Union(POLICY_SUPERVISION_PROJECTION_STATUSES.map((v) => Type.Literal(v))),
    oracle_best_mode: Type.Union([
      Type.Literal("learned_route"),
      Type.Literal("graph_prior_only"),
      Type.Literal("tie"),
      Type.Literal("unclear"),
      Type.Null(),
    ]),
    notes: Type.Array(Type.String({ minLength: 1 })),
  },
  { additionalProperties: false },
);

export type PolicySupervisionRowV1 = Static<typeof PolicySupervisionRowSchemaV1>;

export interface ContractValidationResultV1 {
  contract: string;
  valid: boolean;
  issues: string[];
}

export function validatePolicySupervisionRowV1(value: unknown): ContractValidationResultV1 {
  const contract = POLICY_SUPERVISION_ROW_CONTRACT_V1;
  const valid = Value.Check(PolicySupervisionRowSchemaV1, value);
  if (!valid) {
    return { contract, valid: false, issues: [`${contract}/schema_validation_failed`] };
  }

  const row = value as PolicySupervisionRowV1;
  const issues: string[] = [];

  if (row.row_type === "abstain" && row.hard_negative_class === null && row.oracle_best_mode !== "graph_prior_only" && row.oracle_best_mode !== "tie") {
    issues.push(`${contract}/abstain rows should have a hard_negative_class or oracle_best_mode indicating graph_prior_only or tie`);
  }

  if (row.net_utility_delta !== null && row.net_utility_delta_source === null) {
    issues.push(`${contract}/net_utility_delta_source is required when net_utility_delta is present`);
  }

  if (row.projection_status === "trace_projected" && row.decision_point_id !== null) {
    issues.push(`${contract}/decision_point_id should be null for trace_projected rows (trace-level truth only)`);
  }

  return issues.length === 0
    ? { contract, valid: true, issues: [] }
    : { contract, valid: false, issues };
}

function sha256Text(text: string): string {
  return `sha256:${createHash("sha256").update(text, "utf8").digest("hex")}`;
}

function rowTypeFromTraceLabel(
  oracleBestMode: string | null | undefined,
  routeRow: RouteDecisionRowV1,
): PolicySupervisionRowType {
  if (oracleBestMode === "learned_route") {
    if (routeRow.chosen_action_kind === "stop_local" || routeRow.chosen_action_kind === "stop") {
      return "stop_local";
    }
    return "activate";
  }
  if (oracleBestMode === "graph_prior_only" || oracleBestMode === "tie") {
    return "abstain";
  }
  if (routeRow.stop_label === "STOP_LOCAL") {
    return "stop_local";
  }
  if (routeRow.stop_label === "CONTINUE" && routeRow.chosen_action_kind === "traverse") {
    return "continue_expand";
  }
  return "abstain";
}

function normalizeHardNegativeClass(value: string | null | undefined): HardNegativeClass | null {
  return HARD_NEGATIVE_CLASSES.includes(value as HardNegativeClass)
    ? value as HardNegativeClass
    : null;
}

function hardNegativeClassFromTraceLabel(
  oracleBestMode: string | null | undefined,
  costSensitive: string | null | undefined,
  explicitHardNegativeClass: string | null | undefined,
): HardNegativeClass | null {
  const explicit = normalizeHardNegativeClass(explicitHardNegativeClass);
  if (explicit) {
    return explicit;
  }
  if (oracleBestMode === "graph_prior_only") {
    return "graph_prior_preferred";
  }
  if (oracleBestMode === "tie" && (costSensitive === "high" || costSensitive === "medium")) {
    return "tie_with_cost";
  }
  if (oracleBestMode === "tie") {
    return "unnecessary_activation";
  }
  return null;
}

function rowWeightFromInputs(params: {
  netUtilityDelta: number | null;
  costSensitive: string | null | undefined;
  oracleBestMode: string | null | undefined;
  hardNegativeClass: HardNegativeClass | null;
}): number {
  if (params.netUtilityDelta !== null && Number.isFinite(params.netUtilityDelta)) {
    return Number((1 + Math.abs(params.netUtilityDelta)).toFixed(6));
  }
  const hardNegativeFloor = params.hardNegativeClass === "tie_with_cost"
    ? 2
    : params.hardNegativeClass === "graph_prior_preferred" || params.hardNegativeClass === "stale_memory"
      ? 1.75
      : params.hardNegativeClass === "wrapper_heavy" || params.hardNegativeClass === "no_outcome_change"
        ? 1.5
        : params.hardNegativeClass === "unnecessary_activation"
          ? 1.25
          : 1;
  if (params.costSensitive === "high") {
    return Math.max(2, hardNegativeFloor);
  }
  if (params.costSensitive === "medium") {
    return Math.max(1.5, hardNegativeFloor);
  }
  if (params.oracleBestMode === "learned_route") {
    return Math.max(1.5, hardNegativeFloor);
  }
  return hardNegativeFloor;
}

export function buildPolicySupervisionRowV1(params: {
  routeRow: RouteDecisionRowV1;
  traceLabel: GraphifyRouteObjectiveTraceLabelLikeV1;
}): PolicySupervisionRowV1 {
  const { routeRow, traceLabel } = params;
  const oracleBestMode = traceLabel.oracleBestMode ?? null;
  const rowType = rowTypeFromTraceLabel(oracleBestMode, routeRow);
  const hardNegativeClass = hardNegativeClassFromTraceLabel(
    oracleBestMode,
    traceLabel.costSensitive,
    traceLabel.hardNegativeClass ?? null,
  );

  const netUtilityDelta = typeof traceLabel.netUtilityDelta === "number" && Number.isFinite(traceLabel.netUtilityDelta)
    ? traceLabel.netUtilityDelta
    : typeof traceLabel.utilityDeltaVsBaseline === "number" && Number.isFinite(traceLabel.utilityDeltaVsBaseline)
      ? traceLabel.utilityDeltaVsBaseline
      : typeof traceLabel.utilityDelta === "number" && Number.isFinite(traceLabel.utilityDelta)
        ? traceLabel.utilityDelta
        : null;

  const netUtilityDeltaSource = netUtilityDelta !== null
    ? (oracleBestMode === "learned_route" || oracleBestMode === "graph_prior_only" ? "reviewed" as const : "replay_inferred" as const)
    : null;

  const rowWeight = rowWeightFromInputs({
    netUtilityDelta,
    costSensitive: traceLabel.costSensitive,
    oracleBestMode,
    hardNegativeClass,
  });

  const confidenceTarget = oracleBestMode === "learned_route" || oracleBestMode === "graph_prior_only"
    ? 0.9
    : oracleBestMode === "tie"
      ? 0.7
      : null;

  const traceSlice: PolicySupervisionTraceSliceV1 = {
    route_row_id: routeRow.row_id,
    route_fn_version: normalizeString(routeRow.route_fn_version),
    chosen_action_kind: routeRow.chosen_action_kind,
    stop_label: routeRow.stop_label,
    query_text_hash: sha256Text(routeRow.query_text),
  };

  const notes: string[] = [
    "Policy supervision row projected from trace-level oracle label. Does not represent row-level teacher truth.",
  ];
  if (rowType === "activate") {
    notes.push("Activation row: learned_route judged oracle-best for this trace. Router should have activated.");
  } else if (rowType === "abstain") {
    notes.push("Abstention row: graph_prior_only or tie judged oracle-best. Router should have stayed off or stopped early.");
  } else if (rowType === "stop_local") {
    notes.push("Stop-local row: early stopping was the preferred policy for this trace slice.");
  } else if (rowType === "continue_expand") {
    notes.push("Continue-expand row: oracle mode unclear but trace chose traversal. Conservative continue label.");
  }

  return {
    schema_version: POLICY_SUPERVISION_ROW_VERSION_V1,
    contract: POLICY_SUPERVISION_ROW_CONTRACT_V1,
    row_id: `ps_${hashStableParts([routeRow.row_id, routeRow.trace_id, rowType, oracleBestMode])}`,
    trace_id: routeRow.trace_id,
    episode_id: routeRow.episode_id,
    decision_point_id: null,
    row_type: rowType,
    focus_lane: normalizeString(traceLabel.focusLane),
    trace_slice: traceSlice,
    row_weight: rowWeight,
    confidence_target: confidenceTarget,
    hard_negative_class: hardNegativeClass,
    net_utility_delta: netUtilityDelta,
    net_utility_delta_source: netUtilityDeltaSource,
    projection_status: "trace_projected",
    oracle_best_mode: oracleBestMode as PolicySupervisionRowV1["oracle_best_mode"],
    notes,
  };
}

export interface PolicySupervisionSummaryV1 {
  totalRows: number;
  byRowType: Record<PolicySupervisionRowType, number>;
  byProjectionStatus: Record<PolicySupervisionProjectionStatus, number>;
  byFocusLane: Record<string, number>;
  traceProjectedCount: number;
  withNetUtilityDelta: number;
  withHardNegativeClass: number;
}

export interface PolicySupervisionTrainerNormalizationOptionsV1 {
  focusLaneWeights?: Record<string, number>;
  rowTypeWeights?: Partial<Record<PolicySupervisionRowType, number>>;
}

export interface NormalizedPolicySupervisionTrainerRowV1 {
  policyRowId: string;
  routeRowId: string | null;
  rowType: PolicySupervisionRowType;
  stopLabel: ColdStartStopLabelV1;
  weight: number;
  rowWeight: number;
  focusLane: string | null;
  hardNegativeClass: HardNegativeClass | null;
  oracleBestMode: PolicySupervisionRowV1["oracle_best_mode"];
}

function trainerStopLabelForRowType(rowType: PolicySupervisionRowType): ColdStartStopLabelV1 | null {
  switch (rowType) {
    case "activate":
    case "continue_expand":
      return "CONTINUE";
    case "stop_local":
      return "STOP_LOCAL";
    case "abstain":
      return "STOP";
    case "select_context":
      return null;
  }
}

export function normalizePolicySupervisionRowsForTrainerV1(
  rows: PolicySupervisionRowV1[],
  options: PolicySupervisionTrainerNormalizationOptionsV1 = {},
): NormalizedPolicySupervisionTrainerRowV1[] {
  const normalized: NormalizedPolicySupervisionTrainerRowV1[] = [];

  for (const row of rows) {
    const stopLabel = trainerStopLabelForRowType(row.row_type);
    if (stopLabel === null) {
      continue;
    }

    const focusLaneWeight = row.focus_lane ? options.focusLaneWeights?.[row.focus_lane] ?? 1 : 1;
    const rowTypeWeight = options.rowTypeWeights?.[row.row_type] ?? 1;
    const weight = Number((row.row_weight * focusLaneWeight * rowTypeWeight).toFixed(6));
    if (!Number.isFinite(weight) || weight <= 0) {
      continue;
    }

    normalized.push({
      policyRowId: row.row_id,
      routeRowId: normalizeString(row.trace_slice.route_row_id),
      rowType: row.row_type,
      stopLabel,
      weight,
      rowWeight: row.row_weight,
      focusLane: row.focus_lane,
      hardNegativeClass: row.hard_negative_class,
      oracleBestMode: row.oracle_best_mode,
    });
  }

  return normalized;
}

export function summarizePolicySupervisionRowsV1(rows: PolicySupervisionRowV1[]): PolicySupervisionSummaryV1 {
  const byRowType: Record<PolicySupervisionRowType, number> = {
    activate: 0,
    abstain: 0,
    stop_local: 0,
    select_context: 0,
    continue_expand: 0,
  };
  const byProjectionStatus: Record<PolicySupervisionProjectionStatus, number> = {
    trace_projected: 0,
    decision_point_reviewed: 0,
    owner_labeled: 0,
  };
  const byFocusLane: Record<string, number> = {};
  let withNetUtilityDelta = 0;
  let withHardNegativeClass = 0;

  for (const row of rows) {
    byRowType[row.row_type] += 1;
    byProjectionStatus[row.projection_status] += 1;
    if (row.focus_lane) {
      byFocusLane[row.focus_lane] = (byFocusLane[row.focus_lane] ?? 0) + 1;
    }
    if (row.net_utility_delta !== null) {
      withNetUtilityDelta += 1;
    }
    if (row.hard_negative_class !== null) {
      withHardNegativeClass += 1;
    }
  }

  return {
    totalRows: rows.length,
    byRowType,
    byProjectionStatus,
    byFocusLane,
    traceProjectedCount: byProjectionStatus.trace_projected,
    withNetUtilityDelta,
    withHardNegativeClass,
  };
}
