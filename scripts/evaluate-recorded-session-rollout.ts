#!/usr/bin/env tsx

import path from "node:path";
import process from "node:process";
import {
  DEFAULT_LEARNED_ROUTE_ROLLOUT_BAR,
  discoverRecordedSessionReplayTracePaths,
  evaluateRecordedSessionReplayRollout,
  formatRecordedSessionReplayRolloutVerdict,
} from "../src/recorded-session-rollout-proof.ts";

function usage(): void {
  process.stderr.write(
    [
      "Usage: tsx scripts/evaluate-recorded-session-rollout.ts [options]",
      "",
      "Options:",
      "  --trace-root <path>   Recursively scan for recorded-session-replay trace.json files.",
      "  --trace <path>        Evaluate one explicit recorded_session_trace.v1 trace.",
      "  --json                Emit JSON instead of the text summary.",
      "  --help                Show this help.",
      "",
      "Defaults:",
      "  If no --trace-root or --trace is passed, the script scans docs/evidence.",
      "",
      "Rollout bar:",
      `  minEligibleTraceCount=${DEFAULT_LEARNED_ROUTE_ROLLOUT_BAR.minEligibleTraceCount}`,
      `  minCleanWinRate=${DEFAULT_LEARNED_ROUTE_ROLLOUT_BAR.minCleanWinRate}`,
      `  minAverageMarginVsVectorOnly=${DEFAULT_LEARNED_ROUTE_ROLLOUT_BAR.minAverageMarginVsVectorOnly}`,
      `  minAverageMarginVsGraphPriorOnly=${DEFAULT_LEARNED_ROUTE_ROLLOUT_BAR.minAverageMarginVsGraphPriorOnly}`,
      `  maxLossCountVsVectorOnly=${DEFAULT_LEARNED_ROUTE_ROLLOUT_BAR.maxLossCountVsVectorOnly}`,
      `  maxLossCountVsGraphPriorOnly=${DEFAULT_LEARNED_ROUTE_ROLLOUT_BAR.maxLossCountVsGraphPriorOnly}`,
    ].join("\n") + "\n",
  );
}

function dedupe(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function parseArgs(argv: string[]): { traceRoots: string[]; tracePaths: string[]; json: boolean } {
  const traceRoots: string[] = [];
  const tracePaths: string[] = [];
  let json = false;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--trace-root":
        traceRoots.push(path.resolve(argv[index + 1] ?? ""));
        index += 1;
        break;
      case "--trace":
        tracePaths.push(path.resolve(argv[index + 1] ?? ""));
        index += 1;
        break;
      case "--json":
        json = true;
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

  return { traceRoots, tracePaths, json };
}

function main(): void {
  const { traceRoots, tracePaths, json } = parseArgs(process.argv.slice(2));
  const resolvedTraceRoots =
    traceRoots.length === 0 && tracePaths.length === 0 ? [path.resolve("docs/evidence")] : traceRoots;
  const discoveredTracePaths = resolvedTraceRoots.flatMap((traceRoot) => discoverRecordedSessionReplayTracePaths(traceRoot));
  const resolvedTracePaths = dedupe([...tracePaths, ...discoveredTracePaths]).sort((left, right) => left.localeCompare(right));

  if (resolvedTracePaths.length === 0) {
    throw new Error("No recorded-session-replay trace.json files found");
  }

  const verdict = evaluateRecordedSessionReplayRollout(resolvedTracePaths);
  if (json) {
    process.stdout.write(`${JSON.stringify(verdict, null, 2)}\n`);
  } else {
    process.stdout.write(formatRecordedSessionReplayRolloutVerdict(verdict));
  }
  if (!verdict.ok) {
    process.exitCode = 1;
  }
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
}
