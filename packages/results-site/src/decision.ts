import { PRODUCT_TRACE_THRESHOLD, REQUIRED_SLICES } from "./examples.ts";
import type { NormalizedRow } from "./examples.ts";
import { rate } from "./uncertainty.ts";

export type DecisionSummary = {
  decision: string;
  decisionKind: "blocked" | "pause" | "continue_gated" | "continue";
  engineering_e2e_complete: boolean;
  evidence_e2e_complete: boolean;
  thresholds: Record<string, boolean>;
  thresholdBasis: Record<string, string>;
  blockers: string[];
  traceCounts: {
    totalRows: number;
    admittedRealProductTraces: number;
    judgedProductRows: number;
    selectedProductTraces: number;
  };
};

const MAX_HARM_RATE = 0.05;
const MIN_WIN_RATE = 0.6;

export function deriveDecision(rows: NormalizedRow[], engineeringE2EComplete = false): DecisionSummary {
  const productRows = rows.filter((row) => row.admitted && row.provenanceType === "real" && row.countsAsProductEvidence === true);
  const explicitlySelectedRows = productRows.filter((row) => row.productSelected === true);
  const scoredRows = explicitlySelectedRows.length > 0 ? explicitlySelectedRows : productRows;
  const admittedRealProductTraces = new Set(productRows.map((row) => row.traceId)).size;
  const selectedProductTraces = new Set(scoredRows.map((row) => row.traceId)).size;
  const judgedProductRows = productRows.filter((row) => row.outcome !== "unjudged").length;
  const judgedScoredRows = scoredRows.filter((row) => row.outcome !== "unjudged").length;
  const requiredSlicesPresent = REQUIRED_SLICES.every((slice) => productRows.some((row) => row.slices.includes(slice)));
  const privacyComplete = productRows.length > 0 && productRows.every((row) => row.privacyScrubbed === true);
  const hasJudgments = productRows.length > 0 && productRows.every((row) => row.outcome !== "unjudged");
  const presentSlices = REQUIRED_SLICES.filter((slice) => productRows.some((row) => row.slices.includes(slice))).length;
  const privacyReadyRows = productRows.filter((row) => row.privacyScrubbed === true).length;
  const harmCount = scoredRows.filter((row) => row.outcome === "harm" || row.harmFlags.length > 0).length;
  const winCount = scoredRows.filter((row) => row.outcome === "win").length;
  const harm = rate(harmCount, scoredRows.length);
  const wins = rate(winCount, scoredRows.length);

  const thresholds = {
    minimum_40_real_admitted_traces: admittedRealProductTraces >= PRODUCT_TRACE_THRESHOLD,
    required_slices_present: requiredSlicesPresent,
    privacy_scrubbed_for_product_rows: privacyComplete,
    judging_complete_for_product_rows: hasJudgments && judgedScoredRows === scoredRows.length,
    harm_rate_at_or_below_5_percent: harm.percent !== null && harm.percent <= MAX_HARM_RATE,
    win_rate_at_or_above_60_percent: wins.percent !== null && wins.percent >= MIN_WIN_RATE,
  };

  const thresholdBasis = {
    minimum_40_real_admitted_traces: `${admittedRealProductTraces}/${PRODUCT_TRACE_THRESHOLD} traces`,
    required_slices_present: `${presentSlices}/${REQUIRED_SLICES.length} required slices`,
    privacy_scrubbed_for_product_rows: `${privacyReadyRows}/${productRows.length} product rows`,
    judging_complete_for_product_rows: `${judgedProductRows}/${productRows.length} product rows; ${judgedScoredRows}/${scoredRows.length} selected policy rows`,
    harm_rate_at_or_below_5_percent: `${harmCount}/${scoredRows.length} selected policy harms; max ${MAX_HARM_RATE * 100}%`,
    win_rate_at_or_above_60_percent: `${winCount}/${scoredRows.length} selected policy wins; min ${MIN_WIN_RATE * 100}%`,
  };

  const blockers = Object.entries(thresholds)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
  const evidence_e2e_complete = thresholds.minimum_40_real_admitted_traces
    && thresholds.required_slices_present
    && thresholds.privacy_scrubbed_for_product_rows
    && thresholds.judging_complete_for_product_rows;

  let decision = "BLOCKED: no product decision; collect real admitted redacted traces and complete judging.";
  let decisionKind: DecisionSummary["decisionKind"] = "blocked";

  if (evidence_e2e_complete && !thresholds.harm_rate_at_or_below_5_percent) {
    decision = "PAUSE: product evidence exists, but harm threshold is exceeded.";
    decisionKind = "pause";
  } else if (evidence_e2e_complete && thresholds.harm_rate_at_or_below_5_percent && !thresholds.win_rate_at_or_above_60_percent) {
    decision = "CONTINUE GATED: evidence is complete, but utility threshold is not met.";
    decisionKind = "continue_gated";
  } else if (evidence_e2e_complete && thresholds.harm_rate_at_or_below_5_percent && thresholds.win_rate_at_or_above_60_percent) {
    decision = "CONTINUE: thresholds support a gated product path.";
    decisionKind = "continue";
  }

  return {
    decision,
    decisionKind,
    engineering_e2e_complete: engineeringE2EComplete,
    evidence_e2e_complete,
    thresholds,
    thresholdBasis,
    blockers,
    traceCounts: {
      totalRows: rows.length,
      admittedRealProductTraces,
      judgedProductRows,
      selectedProductTraces,
    },
  };
}

export function renderDecisionMemo(decision: DecisionSummary, sourceDescription: string): string {
  const thresholdRows = Object.entries(decision.thresholds)
    .map(([name, passed]) => `| ${name} | ${passed ? "pass" : "fail"} | ${decision.thresholdBasis[name] ?? "n/a"} |`)
    .join("\n");
  const blockers = decision.blockers.length > 0 ? decision.blockers.map((blocker) => `- ${blocker}`).join("\n") : "- none";

  return `<!-- GENERATED by packages/results-site/src/generate.ts; do not hand-edit. -->
# 30-Day Decision

${decision.traceCounts.admittedRealProductTraces < PRODUCT_TRACE_THRESHOLD ? "**NOT PRODUCT EVIDENCE**\n\n" : ""}Decision: **${decision.decision}**

Source: ${sourceDescription}

## Thresholds

| Threshold | Status | Basis / Denominator |
|---|---|---|
${thresholdRows}

## Blockers

${blockers}

## Gate State

| Gate | Value | Basis / Denominator |
|---|---:|---|
| engineering_e2e_complete | ${decision.engineering_e2e_complete} | ${decision.engineering_e2e_complete ? "1/1" : "0/1"} RUN_STATE engineering gate |
| evidence_e2e_complete | ${decision.evidence_e2e_complete} | ${decision.blockers.length === 0 ? "all" : Object.keys(decision.thresholds).length - decision.blockers.length}/${Object.keys(decision.thresholds).length} thresholds |
| admitted real product traces | ${decision.traceCounts.admittedRealProductTraces}/${PRODUCT_TRACE_THRESHOLD} | required traces |
| selected product policy traces | ${decision.traceCounts.selectedProductTraces}/${decision.traceCounts.admittedRealProductTraces} | selected policy traces / admitted traces |
| judged product rows | ${decision.traceCounts.judgedProductRows}/${decision.traceCounts.totalRows} | judged rows / all rows |
`;
}
