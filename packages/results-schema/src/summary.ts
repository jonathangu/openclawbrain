import { BACKENDS, PRIMARY_SLICES, SUMMARY_SCHEMA_VERSION, type Backend, type TraceSlice, roundUtility } from "./rubric.ts";
import { parseLedgerRow, type LedgerRow, type LedgerRowInput } from "./ledger.ts";
import { describeUncertainty, type Uncertainty } from "./uncertainty.ts";

export type BackendSliceSummary = { slice: TraceSlice; n: number; meanNetTaskUtility: number; };
export type BackendSummary = {
  backend: Backend;
  n: number;
  productEvidenceN: number;
  fireRate: number;
  falseFireRate: number;
  averageHarmDelta: number;
  averageCostPenalty: number;
  averageActivationUtilityWhenFired: number;
  averageNetTaskUtility: number;
  primaryMeanNetTaskUtility: number;
  staleMemoryConflictMeanNetTaskUtility: number | null;
  correctionFollowupMeanNetTaskUtility: number | null;
  costPerUtilityPoint: number | null;
  slices: BackendSliceSummary[];
  netTaskUtilityUncertainty: Uncertainty;
};
export type ResultsSummary = {
  schema_version: typeof SUMMARY_SCHEMA_VERSION;
  run_id: string;
  row_count: number;
  trace_count: number;
  product_evidence_count: number;
  engineering_e2e_complete: boolean;
  evidence_e2e_complete: boolean;
  synthetic_pipeline_validation_only: boolean;
  backends: BackendSummary[];
  blockers: string[];
};

export function summarizeLedgerRows(inputRows: readonly (LedgerRowInput | LedgerRow)[]): ResultsSummary {
  const rows = inputRows.map((row) => parseLedgerRow(stripDerivedFields(row)));
  if (rows.length === 0) throw new Error("cannot summarize an empty ledger");
  const runIds = new Set(rows.map((row) => row.trace_id.includes(":") ? row.trace_id.split(":")[0] : "run"));
  void runIds;
  const backends = BACKENDS.map((backend) => summarizeBackend(backend, rows.filter((row) => row.backend === backend)));
  const productEvidenceCount = rows.filter((row) => row.counts_as_product_evidence).length;
  const syntheticOnly = rows.every((row) => row.mode === "smoke" || row.provenance_type !== "real");
  const traceIds = new Set(rows.map((row) => row.trace_id));
  const blockers: string[] = [];
  if (productEvidenceCount < 40) blockers.push("fewer than 40 admitted real product-evidence rows");
  for (const slice of ["direct-answer", "continuation", "correction-follow-up", "retrieval-heavy", "tool-heavy", "stale-memory-conflict"] as const) {
    const required = slice === "correction-follow-up" || slice === "stale-memory-conflict" ? 8 : 6;
    const count = uniqueTraces(rows.filter((row) => row.counts_as_product_evidence && row.slice === slice)).size;
    if (count < required) blockers.push(`slice ${slice} has ${count}/${required} admitted real traces`);
  }
  if (rows.some((row) => row.mode === "production" && row.judge_mode === "synthetic_smoke")) blockers.push("production rows cannot use synthetic_smoke judgments");
  if (syntheticOnly) blockers.push("synthetic/smoke data is not product evidence");
  return {
    schema_version: SUMMARY_SCHEMA_VERSION,
    run_id: "summary",
    row_count: rows.length,
    trace_count: traceIds.size,
    product_evidence_count: productEvidenceCount,
    engineering_e2e_complete: true,
    evidence_e2e_complete: blockers.length === 0,
    synthetic_pipeline_validation_only: syntheticOnly,
    backends,
    blockers,
  };
}

function stripDerivedFields(row: LedgerRowInput | LedgerRow): LedgerRowInput {
  const { schema_version: _schema, activation_utility: _a, abstention_regret_penalty: _p, net_task_utility: _n, ...input } = row as LedgerRow;
  return input;
}
function summarizeBackend(backend: Backend, rows: LedgerRow[]): BackendSummary {
  const firedRows = rows.filter((row) => row.memory_fired);
  const primaryRows = rows.filter((row) => (PRIMARY_SLICES as readonly string[]).includes(row.slice));
  const sumCost = rows.reduce((sum, row) => sum + row.estimated_cost_usd, 0);
  const positiveUtility = rows.reduce((sum, row) => sum + Math.max(0, row.net_task_utility), 0);
  return {
    backend,
    n: rows.length,
    productEvidenceN: rows.filter((row) => row.counts_as_product_evidence).length,
    fireRate: rate(firedRows.length, rows.length),
    falseFireRate: rate(rows.filter((row) => row.false_fire).length, rows.length),
    averageHarmDelta: average(rows.map((row) => row.harm_delta)),
    averageCostPenalty: average(rows.map((row) => row.cost_penalty)),
    averageActivationUtilityWhenFired: average(firedRows.map((row) => row.activation_utility)),
    averageNetTaskUtility: average(rows.map((row) => row.net_task_utility)),
    primaryMeanNetTaskUtility: average(primaryRows.map((row) => row.net_task_utility)),
    staleMemoryConflictMeanNetTaskUtility: meanForSlice(rows, "stale-memory-conflict"),
    correctionFollowupMeanNetTaskUtility: meanForSlice(rows, "correction-follow-up"),
    costPerUtilityPoint: positiveUtility > 0 ? roundUtility(sumCost / positiveUtility) : null,
    slices: [...new Set(rows.map((row) => row.slice))].sort().map((slice) => ({ slice, n: rows.filter((row) => row.slice === slice).length, meanNetTaskUtility: meanForSlice(rows, slice) ?? 0 })),
    netTaskUtilityUncertainty: describeUncertainty(rows.map((row) => row.net_task_utility)),
  };
}
function meanForSlice(rows: LedgerRow[], slice: TraceSlice): number | null { const matching=rows.filter((row)=>row.slice===slice); return matching.length ? average(matching.map((row)=>row.net_task_utility)) : null; }
function average(values: number[]): number { return values.length ? roundUtility(values.reduce((sum, value) => sum + value, 0) / values.length) : 0; }
function rate(n: number, d: number): number { return d ? roundUtility(n/d) : 0; }
function uniqueTraces(rows: LedgerRow[]): Set<string> { return new Set(rows.map((row)=>row.trace_id)); }
