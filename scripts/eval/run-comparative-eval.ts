#!/usr/bin/env tsx

import path from "node:path";
import process from "node:process";
import {
  DEFAULT_COMPARATIVE_EVAL_MANIFEST_PATH,
  runComparativeEval,
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

function parseArgs(argv: string[]): RunComparativeEvalInput {
  const parsed: RunComparativeEvalInput = {};
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
  };
}

function printCliSummary(descriptor: ReturnType<typeof runComparativeEval>): void {
  process.stdout.write(
    [
      `Comparative eval runner: ${descriptor.report.status}`,
      `manifestPath: ${descriptor.report.manifestPath}`,
      `manifestId: ${descriptor.report.manifestId ?? "null"}`,
      `traceCount: ${descriptor.report.successfulTraceCount}/${descriptor.report.requestedTraceCount}`,
      `outputDir: ${descriptor.outputDir}`,
      `report: ${descriptor.reportPath}`,
      `scorecard: ${descriptor.scorecardPath}`,
      `summary: ${descriptor.summaryPath}`,
    ].join("\n") + "\n",
  );
}

try {
  const descriptor = runComparativeEval(parseArgs(process.argv.slice(2)));
  printCliSummary(descriptor);
  process.exitCode = descriptor.report.status === "ok" ? 0 : 1;
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}
