import { LOW_N_TRACE_THRESHOLD } from "./examples.ts";

export type RateSummary = {
  numerator: number;
  denominator: number;
  percent: number | null;
  wilsonLow: number | null;
  wilsonHigh: number | null;
  label: string;
};

export function rate(numerator: number, denominator: number): RateSummary {
  if (denominator <= 0) {
    return { numerator, denominator, percent: null, wilsonLow: null, wilsonHigh: null, label: `${numerator}/${denominator} (n/a)` };
  }
  const percent = numerator / denominator;
  const [wilsonLow, wilsonHigh] = wilsonInterval(numerator, denominator);
  return {
    numerator,
    denominator,
    percent,
    wilsonLow,
    wilsonHigh,
    label: `${numerator}/${denominator} (${formatPercent(percent)})`,
  };
}

export function formatPercent(value: number | null): string {
  return value === null ? "n/a" : `${(value * 100).toFixed(1)}%`;
}

export function formatNumber(value: number | null): string {
  return value === null || !Number.isFinite(value) ? "n/a" : value.toFixed(3);
}

export function uncertaintyLabel(denominator: number): string {
  if (denominator === 0) return "EMPTY SLICE — 0 denominator";
  if (denominator < LOW_N_TRACE_THRESHOLD) return `LOW-N — ${denominator}/${LOW_N_TRACE_THRESHOLD} minimum`;
  return "reportable with caution";
}

export function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function wilsonInterval(numerator: number, denominator: number): [number, number] {
  if (denominator <= 0) return [0, 0];
  const z = 1.96;
  const proportion = numerator / denominator;
  const denominatorTerm = 1 + (z * z) / denominator;
  const centre = proportion + (z * z) / (2 * denominator);
  const margin = z * Math.sqrt((proportion * (1 - proportion) + (z * z) / (4 * denominator)) / denominator);
  return [Math.max(0, (centre - margin) / denominatorTerm), Math.min(1, (centre + margin) / denominatorTerm)];
}
