import { isSyntheticOrSmoke, LOW_N_TRACE_THRESHOLD, REQUIRED_BACKENDS, REQUIRED_SLICES } from "./examples.ts";
import type { NormalizedRow } from "./examples.ts";
import { formatNumber, formatPercent, mean, rate, uncertaintyLabel } from "./uncertainty.ts";

export type TableBlock = {
  title: string;
  markdown: string;
};

export type ScoreboardStats = {
  totalRows: number;
  totalTraces: number;
  productEvidenceRows: number;
  syntheticOrSmokeRows: number;
  negativeRows: number;
  unjudgedRows: number;
};

export function buildStats(rows: NormalizedRow[]): ScoreboardStats {
  return {
    totalRows: rows.length,
    totalTraces: new Set(rows.map((row) => row.traceId)).size,
    productEvidenceRows: rows.filter((row) => row.countsAsProductEvidence && row.provenanceType === "real").length,
    syntheticOrSmokeRows: rows.filter(isSyntheticOrSmoke).length,
    negativeRows: rows.filter((row) => row.negativeResult).length,
    unjudgedRows: rows.filter((row) => row.outcome === "unjudged").length,
  };
}

export function renderAllTables(rows: NormalizedRow[]): TableBlock[] {
  return [
    { title: "Data Status", markdown: renderDataStatusTable(rows) },
    { title: "Backend Outcomes", markdown: renderBackendOutcomeTable(rows) },
    { title: "Required Slice Coverage", markdown: renderSliceCoverageTable(rows) },
    { title: "Negative Results", markdown: renderNegativeResultsTable(rows) },
    { title: "Smoke And Synthetic Rows", markdown: renderSmokeTable(rows) },
  ];
}

export function renderDataStatusTable(rows: NormalizedRow[]): string {
  const total = rows.length;
  const admitted = rows.filter((row) => row.admitted).length;
  const realProduct = rows.filter((row) => row.admitted && row.provenanceType === "real" && row.countsAsProductEvidence).length;
  const synthetic = rows.filter((row) => row.provenanceType === "synthetic" || row.mode === "smoke").length;
  const privacyReady = rows.filter((row) => row.privacyScrubbed === true).length;
  const unjudged = rows.filter((row) => row.outcome === "unjudged").length;

  return markdownTable(
    ["Metric", "Count / Denominator", "Status"],
    [
      ["ledger rows", `${total}/${total}`, total === 0 ? "EMPTY" : "present"],
      ["admitted rows", `${admitted}/${total}`, admitted === 0 ? "EMPTY" : "visible"],
      ["real product-evidence rows", `${realProduct}/${total}`, realProduct === 0 ? "BLOCKED" : "candidate evidence"],
      ["synthetic or smoke rows", `${synthetic}/${total}`, synthetic > 0 ? "NOT PRODUCT EVIDENCE" : "none"],
      ["privacy-scrubbed rows", `${privacyReady}/${total}`, privacyReady === total && total > 0 ? "complete" : "incomplete or unavailable"],
      ["unjudged rows", `${unjudged}/${total}`, unjudged > 0 ? "judging incomplete" : "none visible"],
    ],
  );
}

export function renderBackendOutcomeTable(rows: NormalizedRow[]): string {
  const backendNames = [...new Set([...REQUIRED_BACKENDS, ...rows.map((row) => row.backend)])];
  const body = backendNames.map((backend) => {
    const backendRows = rows.filter((row) => row.backend === backend);
    const denominator = backendRows.length;
    const winRate = rate(backendRows.filter((row) => row.outcome === "win").length, denominator);
    const lossRate = rate(backendRows.filter((row) => row.outcome === "loss").length, denominator);
    const harmRate = rate(backendRows.filter((row) => row.outcome === "harm" || row.harmFlags.length > 0).length, denominator);
    const utilityMean = mean(backendRows.map((row) => row.utilityDelta).filter((value): value is number => value !== null));
    return [
      backend,
      `${denominator}/${rows.length}`,
      winRate.label,
      lossRate.label,
      harmRate.label,
      formatNumber(utilityMean),
      uncertaintyLabel(denominator),
    ];
  });

  return markdownTable(["Backend", "Rows / Total", "Wins", "Losses", "Harms", "Mean Utility Δ", "Uncertainty"], body);
}

export function renderSliceCoverageTable(rows: NormalizedRow[]): string {
  const allSlices = [...new Set([...REQUIRED_SLICES, ...rows.flatMap((row) => row.slices)])];
  const body = allSlices.map((slice) => {
    const traceIds = new Set(rows.filter((row) => row.slices.includes(slice)).map((row) => row.traceId));
    const sliceRows = rows.filter((row) => row.slices.includes(slice));
    const judged = sliceRows.filter((row) => row.outcome !== "unjudged").length;
    const negatives = sliceRows.filter((row) => row.negativeResult).length;
    return [
      slice,
      `${traceIds.size}/${Math.max(LOW_N_TRACE_THRESHOLD, traceIds.size)}`,
      `${judged}/${sliceRows.length}`,
      `${negatives}/${sliceRows.length}`,
      uncertaintyLabel(traceIds.size),
    ];
  });

  return markdownTable(["Slice", "Distinct Traces / Low-N Denominator", "Judged Rows", "Negative Rows", "Visibility"], body);
}

export function renderNegativeResultsTable(rows: NormalizedRow[]): string {
  const negativeRows = rows.filter((row) => row.negativeResult || row.harmFlags.length > 0);
  const denominator = rows.length;
  if (negativeRows.length === 0) {
    return markdownTable(
      ["Trace", "Backend", "Slice", "Outcome", "Utility Δ", "Failure / Harm Modes", "Count / Denominator"],
      [["none observed", "n/a", "n/a", "n/a", "n/a", "No negative rows in available ledger; absence is not proof of safety.", `0/${denominator}`]],
    );
  }

  return markdownTable(
    ["Trace", "Backend", "Slice", "Outcome", "Utility Δ", "Failure / Harm Modes", "Count / Denominator"],
    negativeRows.map((row, index) => [
      row.traceId,
      row.backend,
      row.slices.join(", "),
      row.outcome,
      formatNumber(row.utilityDelta),
      [...row.failureModes, ...row.harmFlags].join(", ") || "negative utility/result",
      `${index + 1}/${denominator}`,
    ]),
  );
}

export function renderSmokeTable(rows: NormalizedRow[]): string {
  const smokeRows = rows.filter(isSyntheticOrSmoke);
  if (smokeRows.length === 0) {
    return markdownTable(
      ["Mode", "Rows / Total", "Evidence Status"],
      [["synthetic/smoke", `0/${rows.length}`, "none visible"]],
    );
  }

  const modes = [...new Set(smokeRows.map((row) => `${row.provenanceType}:${row.mode}`))];
  return markdownTable(
    ["Mode", "Rows / Total", "Evidence Status"],
    modes.map((mode) => {
      const count = smokeRows.filter((row) => `${row.provenanceType}:${row.mode}` === mode).length;
      return [mode, `${count}/${rows.length}`, "NOT PRODUCT EVIDENCE / SYNTHETIC PIPELINE VALIDATION ONLY"];
    }),
  );
}

export function renderRateWithInterval(numerator: number, denominator: number): string {
  const summary = rate(numerator, denominator);
  if (summary.percent === null) return summary.label;
  return `${summary.label}; 95% Wilson ${formatPercent(summary.wilsonLow)}–${formatPercent(summary.wilsonHigh)}`;
}

function markdownTable(headers: string[], rows: string[][]): string {
  return [
    `| ${headers.map(escapeCell).join(" | ")} |`,
    `| ${headers.map(() => "---").join(" | ")} |`,
    ...rows.map((row) => `| ${row.map(escapeCell).join(" | ")} |`),
  ].join("\n");
}

function escapeCell(value: string): string {
  return String(value).replaceAll("|", "\\|").replaceAll("\n", "<br>");
}
