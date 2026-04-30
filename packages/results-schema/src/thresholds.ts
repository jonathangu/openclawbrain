import { type Backend, type ProductOutcome } from "./rubric.ts";
import { type BackendSummary, type ResultsSummary } from "./summary.ts";

export type ThresholdDecision =
  | { status: "recommended"; recommended_product_outcome: ProductOutcome; backend: Backend | "hybrid" | "runtime-health"; blockers: []; tied_backends: []; reason: string }
  | { status: "tie"; recommended_product_outcome: null; backend: null; blockers: []; tied_backends: Backend[]; reason: string }
  | { status: "blocked"; recommended_product_outcome: null; backend: null; blockers: string[]; tied_backends: []; reason: string };

export function applyThresholds(summary: ResultsSummary): ThresholdDecision {
  const blockers = [...summary.blockers];
  if (!summary.evidence_e2e_complete) blockers.push("evidence_e2e_complete=false");
  if (blockers.length) return blocked(blockers, "Evidence blockers prevent a product recommendation.");

  const none = requireBackend(summary, "none");
  const correction = requireBackend(summary, "correction-only");
  const heuristics = requireBackend(summary, "correction+heuristics");
  const full = requireBackend(summary, "full-ocb");
  const missing = [none, correction, heuristics, full].filter((x): x is string => typeof x === "string");
  if (missing.length) return blocked(missing, "Missing backend summaries.");
  const n=none as BackendSummary, c=correction as BackendSummary, h=heuristics as BackendSummary, f=full as BackendSummary;

  const positiveSlices = summary.backends.flatMap((b)=>b.slices.filter((s)=>s.meanNetTaskUtility>0).map((s)=>s.slice));
  if (new Set(positiveSlices).size < 4) return recommendation("F", "none", "Pause: no backend shows positive net task utility in at least 4 slices.");

  const fullBeatsCorrection25 = f.primaryMeanNetTaskUtility >= c.primaryMeanNetTaskUtility * 1.25;
  const correctionCapturesLess75 = c.primaryMeanNetTaskUtility < f.primaryMeanNetTaskUtility * 0.75;
  const fullWinsPrimarySlices = winCount(f, c) >= 2;
  const falseFireDeltaOk = f.falseFireRate - c.falseFireRate <= 0.05;
  const stalePositive = (f.staleMemoryConflictMeanNetTaskUtility ?? -Infinity) > 0;
  const noCorrectionRegression = (f.correctionFollowupMeanNetTaskUtility ?? -Infinity) >= (c.correctionFollowupMeanNetTaskUtility ?? Infinity);
  const costOk = f.costPerUtilityPoint === null || c.costPerUtilityPoint === null || f.costPerUtilityPoint <= c.costPerUtilityPoint * 1.25 || f.primaryMeanNetTaskUtility > c.primaryMeanNetTaskUtility;
  const fullFlagship = fullBeatsCorrection25 && correctionCapturesLess75 && fullWinsPrimarySlices && falseFireDeltaOk && stalePositive && noCorrectionRegression && costOk;
  if (fullFlagship) return recommendation("A", "full-ocb", "Full OCB satisfies all V5 flagship thresholds.");

  const correctionDefault =
    c.primaryMeanNetTaskUtility >= f.primaryMeanNetTaskUtility * 0.75 ||
    ((c.correctionFollowupMeanNetTaskUtility ?? -Infinity) >= (f.correctionFollowupMeanNetTaskUtility ?? Infinity) && (c.staleMemoryConflictMeanNetTaskUtility ?? -Infinity) >= (f.staleMemoryConflictMeanNetTaskUtility ?? Infinity)) ||
    f.falseFireRate - c.falseFireRate > 0.05 ||
    fullWinsPrimarySlices < 2;
  if (correctionDefault) return recommendation("B", "correction-only", "Correction-only satisfies a V5 default condition.");

  const heuristicsDefault = h.primaryMeanNetTaskUtility > c.primaryMeanNetTaskUtility && h.primaryMeanNetTaskUtility >= f.primaryMeanNetTaskUtility * 0.85 && h.falseFireRate <= f.falseFireRate && h.averageHarmDelta <= f.averageHarmDelta;
  if (heuristicsDefault) return recommendation("C", "correction+heuristics", "Correction+heuristics satisfies all V5 default conditions.");

  if (f.slices.some((s)=>["retrieval-heavy","tool-heavy"].includes(s.slice) && s.meanNetTaskUtility > 0) && !fullFlagship) {
    return recommendation("D", "hybrid", "Hybrid default + slice-gated full OCB is indicated by secondary-slice gains without primary-slice flagship proof.");
  }

  if (n.averageNetTaskUtility >= Math.max(c.averageNetTaskUtility, h.averageNetTaskUtility, f.averageNetTaskUtility)) return recommendation("E", "runtime-health", "Runtime health / verification layer only: memory backends do not beat no-memory baseline.");
  return recommendation("F", "none", "Pause until better traces exist.");
}

function requireBackend(summary: ResultsSummary, backend: Backend): BackendSummary | string { return summary.backends.find((b)=>b.backend===backend) ?? `missing backend ${backend}`; }
function winCount(a: BackendSummary, b: BackendSummary): number { return ["correction-follow-up","continuation","stale-memory-conflict"].filter((slice)=> (a.slices.find((s)=>s.slice===slice)?.meanNetTaskUtility ?? -Infinity) > (b.slices.find((s)=>s.slice===slice)?.meanNetTaskUtility ?? -Infinity)).length; }
function blocked(blockers: string[], reason: string): ThresholdDecision { return { status: "blocked", recommended_product_outcome: null, backend: null, blockers: [...new Set(blockers)], tied_backends: [], reason }; }
function recommendation(outcome: ProductOutcome, backend: Backend | "hybrid" | "runtime-health", reason: string): ThresholdDecision { return { status: "recommended", recommended_product_outcome: outcome, backend, blockers: [], tied_backends: [], reason }; }
