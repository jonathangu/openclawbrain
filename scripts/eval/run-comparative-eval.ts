#!/usr/bin/env tsx

import path from "node:path";
import process from "node:process";
import {
  DEFAULT_COMPARATIVE_EVAL_MANIFEST_PATH,
  runComparativeEval,
  type ComparativeEvalPolicyThresholdsV1,
  type RunComparativeEvalInput,
} from "../../src/eval/comparative-eval-runner.ts";

function usage(): void {
  process.stderr.write(
    [
      "Usage: tsx scripts/eval/run-comparative-eval.ts [options]",
      "",
      "Options:",
      `  --manifest <path>          Manifest path. Defaults to ${DEFAULT_COMPARATIVE_EVAL_MANIFEST_PATH}`,
      "  --output-dir <path>        Output root for runner artifacts.",
      "  --scratch-root-dir <path>  Scratch parent for replay runs.",
      "  --worked-trace-limit <n>   Limit the number of worked traces in traces/_lane/worked-traces.md.",
      "  --max-failed-traces <n>    Maximum failed traces allowed before the gate becomes partial.",
      "  --min-candidate-trace-tie-or-better-rate <0-1>",
      "                             Minimum candidate trace tie-or-better rate versus the baseline.",
      "  --max-candidate-mean-quality-regression <n>",
      "                             Maximum allowed mean quality regression for the candidate versus the baseline.",
      "  --max-candidate-tie-promotion-delta <n>",
      "                             Maximum tie-trace promotion churn allowed for the candidate versus the baseline.",
      "  --min-baseline-mean-quality-gain-vs-floor <n>",
      "                             Minimum mean quality gain the baseline must hold over the floor mode.",
      "  --help                     Show this help.",
      "",
      "Outputs:",
      "  report.json",
      "  scorecard.json",
      "  summary.md",
      "  source-manifest.json",
      "  traces/<trace-id>/... per-trace replay proof bundles",
      "  traces/_lane/... aggregate replay lane artifacts",
    ].join("\n") + "\n",
  );
}

function normalizeCliString(value: string | undefined): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed.length === 0 ? null : trimmed;
}

function parseWorkedTraceLimit(value: string | null): number {
  if (value === null) {
    throw new Error("--worked-trace-limit requires a value");
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 1 || !Number.isInteger(parsed)) {
    throw new Error("--worked-trace-limit must be a positive integer");
  }
  return parsed;
}

function parseNonNegativeIntegerArg(value: string | null, fieldName: string): number {
  if (value === null) {
    throw new Error(`${fieldName} requires a value`);
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0 || !Number.isInteger(parsed)) {
    throw new Error(`${fieldName} must be a non-negative integer`);
  }
  return parsed;
}

function parseNumericArg(value: string | null, fieldName: string): number {
  if (value === null) {
    throw new Error(`${fieldName} requires a value`);
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${fieldName} must be a finite number`);
  }
  return parsed;
}

function parseRateArg(value: string | null, fieldName: string): number {
  const parsed = parseNumericArg(value, fieldName);
  if (parsed < 0 || parsed > 1) {
    throw new Error(`${fieldName} must be between 0 and 1`);
  }
  return parsed;
}

function parseArgs(argv: string[]): RunComparativeEvalInput {
  const parsed: RunComparativeEvalInput & {
    maxFailedTraceCount?: number;
    minCandidateTraceTieOrBetterRateVsBaseline?: number;
    maxCandidateMeanQualityRegressionVsBaseline?: number;
    maxCandidateTiePromotionDeltaVsBaseline?: number;
    minBaselineMeanQualityGainVsFloor?: number;
  } = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--manifest":
        parsed.manifestPath = normalizeCliString(argv[index + 1]) ?? undefined;
        index += 1;
        break;
      case "--output-dir":
        parsed.outputDir = normalizeCliString(argv[index + 1]) ?? undefined;
        index += 1;
        break;
      case "--scratch-root-dir":
        parsed.scratchRootDir = normalizeCliString(argv[index + 1]) ?? undefined;
        index += 1;
        break;
      case "--worked-trace-limit":
        parsed.workedTraceLimit = parseWorkedTraceLimit(normalizeCliString(argv[index + 1]));
        index += 1;
        break;
      case "--max-failed-traces":
        parsed.maxFailedTraceCount = parseNonNegativeIntegerArg(
          normalizeCliString(argv[index + 1]),
          "--max-failed-traces",
        );
        index += 1;
        break;
      case "--min-candidate-trace-tie-or-better-rate":
        parsed.minCandidateTraceTieOrBetterRateVsBaseline = parseRateArg(
          normalizeCliString(argv[index + 1]),
          "--min-candidate-trace-tie-or-better-rate",
        );
        index += 1;
        break;
      case "--max-candidate-mean-quality-regression":
        parsed.maxCandidateMeanQualityRegressionVsBaseline = parseNumericArg(
          normalizeCliString(argv[index + 1]),
          "--max-candidate-mean-quality-regression",
        );
        index += 1;
        break;
      case "--max-candidate-tie-promotion-delta":
        parsed.maxCandidateTiePromotionDeltaVsBaseline = parseNumericArg(
          normalizeCliString(argv[index + 1]),
          "--max-candidate-tie-promotion-delta",
        );
        index += 1;
        break;
      case "--min-baseline-mean-quality-gain-vs-floor":
        parsed.minBaselineMeanQualityGainVsFloor = parseNumericArg(
          normalizeCliString(argv[index + 1]),
          "--min-baseline-mean-quality-gain-vs-floor",
        );
        index += 1;
        break;
      case "--help":
      case "-h":
        usage();
        process.exit(0);
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return {
    ...parsed,
    manifestPath: parsed.manifestPath ? path.resolve(parsed.manifestPath) : undefined,
    outputDir: parsed.outputDir ? path.resolve(parsed.outputDir) : undefined,
    scratchRootDir: parsed.scratchRootDir ? path.resolve(parsed.scratchRootDir) : undefined,
    policy: {
      ...(parsed.maxFailedTraceCount === undefined ? {} : { maxFailedTraceCount: parsed.maxFailedTraceCount }),
      ...(parsed.minCandidateTraceTieOrBetterRateVsBaseline === undefined
        ? {}
        : { minCandidateTraceTieOrBetterRateVsBaseline: parsed.minCandidateTraceTieOrBetterRateVsBaseline }),
      ...(parsed.maxCandidateMeanQualityRegressionVsBaseline === undefined
        ? {}
        : { maxCandidateMeanQualityRegressionVsBaseline: parsed.maxCandidateMeanQualityRegressionVsBaseline }),
      ...(parsed.maxCandidateTiePromotionDeltaVsBaseline === undefined
        ? {}
        : { maxCandidateTiePromotionDeltaVsBaseline: parsed.maxCandidateTiePromotionDeltaVsBaseline }),
      ...(parsed.minBaselineMeanQualityGainVsFloor === undefined
        ? {}
        : { minBaselineMeanQualityGainVsFloor: parsed.minBaselineMeanQualityGainVsFloor }),
    } satisfies Partial<ComparativeEvalPolicyThresholdsV1>,
  };
}

function printCliSummary(descriptor: ReturnType<typeof runComparativeEval>): void {
  const failedChecks = descriptor.scorecard.policy.checks.filter((check) => check.status === "fail");
  process.stdout.write(
    [
      `Comparative eval runner: ${descriptor.report.status}`,
      `Comparative eval gate: ${descriptor.scorecard.policy.status}`,
      `manifestPath: ${descriptor.report.manifestPath}`,
      `manifestId: ${descriptor.report.manifestId ?? "null"}`,
      `traceCount: ${descriptor.report.successfulTraceCount}/${descriptor.report.requestedTraceCount}`,
      `outputDir: ${descriptor.outputDir}`,
      `report: ${descriptor.reportPath}`,
      `scorecard: ${descriptor.scorecardPath}`,
      `summary: ${descriptor.summaryPath}`,
      ...descriptor.scorecard.policy.checks.map((check) => `${check.id}: ${check.status}`),
      ...(failedChecks.length === 0 ? [] : [`failedChecks: ${failedChecks.map((check) => check.id).join(",")}`]),
    ].join("\n") + "\n",
  );
}

try {
  const descriptor = runComparativeEval(parseArgs(process.argv.slice(2)));
  printCliSummary(descriptor);
  process.exitCode = descriptor.scorecard.policy.status === "pass" ? 0 : 1;
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}
