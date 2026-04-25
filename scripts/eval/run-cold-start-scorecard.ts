#!/usr/bin/env tsx

import path from "node:path";
import process from "node:process";
import {
  DEFAULT_COLD_START_PRIOR_ARTIFACT_DIR,
  DEFAULT_COMPARATIVE_EVAL_MANIFEST_PATH,
  runColdStartScorecardV1,
  type RunColdStartScorecardInputV1,
} from "../../src/eval/cold-start-scorecard.ts";

function usage(): void {
  process.stderr.write([
    "Usage: tsx scripts/eval/run-cold-start-scorecard.ts [options]",
    "",
    "Options:",
    `  --manifest <path>            Manifest path. Defaults to ${DEFAULT_COMPARATIVE_EVAL_MANIFEST_PATH}`,
    `  --candidate-artifact <path>  Cold-start candidate artifact. Defaults to ${DEFAULT_COLD_START_PRIOR_ARTIFACT_DIR}`,
    "  --output-dir <path>          Output root for scorecard artifacts.",
    "  --scratch-root-dir <path>    Scratch parent for replay runs.",
    "  --generated-at <iso>         Stable timestamp for deterministic reports.",
    "  --help                       Show this help.",
  ].join("\n") + "\n");
}

function parseArgs(argv: string[]): RunColdStartScorecardInputV1 {
  const input: RunColdStartScorecardInputV1 = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--manifest":
        input.manifestPath = path.resolve(argv[++index] ?? "");
        break;
      case "--candidate-artifact":
        input.candidateArtifactDir = path.resolve(argv[++index] ?? "");
        break;
      case "--output-dir":
        input.outputDir = path.resolve(argv[++index] ?? "");
        break;
      case "--scratch-root-dir":
        input.scratchRootDir = path.resolve(argv[++index] ?? "");
        break;
      case "--generated-at":
        input.generatedAt = argv[++index] ?? undefined;
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
  return input;
}

try {
  const scorecard = await runColdStartScorecardV1(parseArgs(process.argv.slice(2)));
  const coldStartPrior = scorecard.modes.find((mode) => mode.mode === "cold_start_prior");
  const graphPrior = scorecard.modes.find((mode) => mode.mode === "graph_prior_only");
  process.stdout.write([
    `Cold-start scorecard: ${scorecard.verdict.usefulContextWin && scorecard.verdict.noOverfireVsGraphPrior ? "ok" : "partial"}`,
    `traceCount: ${scorecard.traceCount}`,
    `cold_start_prior phrase hits: ${coldStartPrior?.totalPhraseHitCount ?? "n/a"}/${coldStartPrior?.totalPhraseCount ?? "n/a"}`,
    `graph_prior_only phrase hits: ${graphPrior?.totalPhraseHitCount ?? "n/a"}/${graphPrior?.totalPhraseCount ?? "n/a"}`,
    `summary: ${scorecard.verdict.summary}`,
  ].join("\n") + "\n");
  process.exitCode = scorecard.verdict.usefulContextWin && scorecard.verdict.noOverfireVsGraphPrior ? 0 : 1;
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
}
