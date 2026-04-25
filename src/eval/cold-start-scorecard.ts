import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type {
  ComparativeEvalModeScorecardRowV1,
  ComparativeEvalRunnerDescriptor,
} from "./comparative-eval-runner.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..");

export const DEFAULT_COMPARATIVE_EVAL_MANIFEST_PATH = path.resolve(
  repoRoot,
  "evals",
  "recorded-session-replay",
  "canonical-frozen-20",
  "manifest.json",
);

export const COLD_START_SCORECARD_CONTRACT_V1 = "cold_start_scorecard.v1";

export const DEFAULT_COLD_START_PRIOR_ARTIFACT_DIR = path.resolve(
  "artifacts",
  "activation-first-gating-retune",
  "T-20260419-269",
  "candidate-artifact",
);

export interface ColdStartScorecardModeRowV1 {
  mode: "no_brain" | "graph_prior_only" | "cold_start_prior_single" | "cold_start_prior" | "learned_route";
  sourceEval: "baseline" | "cold_start_prior_single" | "cold_start_prior";
  sourceMode: string;
  meanQualityScore: number | null;
  totalPhraseHitCount: number;
  totalPhraseCount: number;
  phraseHitRate: number | null;
  totalSelectedContextBlockCount: number;
  totalSelectedContextChars: number;
  estimatedPromptTokens: number;
  warningCount: number;
}

export interface ColdStartScorecardDeltaV1 {
  leftMode: ColdStartScorecardModeRowV1["mode"];
  rightMode: ColdStartScorecardModeRowV1["mode"];
  meanQualityDeltaRightMinusLeft: number | null;
  phraseHitDeltaRightMinusLeft: number;
  selectedContextBlockDeltaRightMinusLeft: number;
  selectedContextCharDeltaRightMinusLeft: number;
  promptTokenDeltaRightMinusLeft: number;
}

export interface ColdStartScorecardV1 {
  contract: typeof COLD_START_SCORECARD_CONTRACT_V1;
  generatedAt: string;
  manifestPath: string;
  candidateArtifactDir: string;
  traceCount: number;
  modes: ColdStartScorecardModeRowV1[];
  deltas: ColdStartScorecardDeltaV1[];
  verdict: {
    usefulContextWin: boolean;
    noOverfireVsGraphPrior: boolean;
    summary: string;
  };
}

export interface RunColdStartScorecardInputV1 {
  manifestPath?: string;
  outputDir?: string;
  scratchRootDir?: string;
  candidateArtifactDir?: string;
  generatedAt?: string;
}

function findMode(descriptor: ComparativeEvalRunnerDescriptor, mode: string): ComparativeEvalModeScorecardRowV1 {
  const row = descriptor.scorecard.modes.find((candidate) => candidate.mode === mode);
  if (!row) {
    throw new Error(`comparative eval ${descriptor.outputDir} is missing mode ${mode}`);
  }
  return row;
}

function mapMode(
  mode: ColdStartScorecardModeRowV1["mode"],
  sourceEval: ColdStartScorecardModeRowV1["sourceEval"],
  sourceMode: string,
  row: ComparativeEvalModeScorecardRowV1,
): ColdStartScorecardModeRowV1 {
  return {
    mode,
    sourceEval,
    sourceMode,
    meanQualityScore: row.meanQualityScore,
    totalPhraseHitCount: row.totalPhraseHitCount,
    totalPhraseCount: row.totalPhraseCount,
    phraseHitRate: row.phraseHitRate,
    totalSelectedContextBlockCount: row.totalSelectedContextBlockCount,
    totalSelectedContextChars: row.totalSelectedContextChars,
    estimatedPromptTokens: row.estimatedPromptTokens,
    warningCount: row.totalWarningCount,
  };
}

function delta(
  left: ColdStartScorecardModeRowV1,
  right: ColdStartScorecardModeRowV1,
): ColdStartScorecardDeltaV1 {
  return {
    leftMode: left.mode,
    rightMode: right.mode,
    meanQualityDeltaRightMinusLeft: left.meanQualityScore === null || right.meanQualityScore === null
      ? null
      : Number((right.meanQualityScore - left.meanQualityScore).toFixed(6)),
    phraseHitDeltaRightMinusLeft: right.totalPhraseHitCount - left.totalPhraseHitCount,
    selectedContextBlockDeltaRightMinusLeft:
      right.totalSelectedContextBlockCount - left.totalSelectedContextBlockCount,
    selectedContextCharDeltaRightMinusLeft: right.totalSelectedContextChars - left.totalSelectedContextChars,
    promptTokenDeltaRightMinusLeft: right.estimatedPromptTokens - left.estimatedPromptTokens,
  };
}

export function buildColdStartScorecardV1(params: {
  generatedAt: string;
  manifestPath: string;
  candidateArtifactDir: string;
  baseline: ComparativeEvalRunnerDescriptor;
  coldStartPriorSingle: ComparativeEvalRunnerDescriptor;
  coldStartPrior: ComparativeEvalRunnerDescriptor;
}): ColdStartScorecardV1 {
  const noBrain = mapMode("no_brain", "baseline", "no_brain", findMode(params.baseline, "no_brain"));
  const graphPrior = mapMode("graph_prior_only", "baseline", "graph_prior_only", findMode(params.baseline, "graph_prior_only"));
  const learnedRoute = mapMode("learned_route", "baseline", "learned_route", findMode(params.baseline, "learned_route"));
  const coldStartSingle = mapMode(
    "cold_start_prior_single",
    "cold_start_prior_single",
    "learned_route",
    findMode(params.coldStartPriorSingle, "learned_route"),
  );
  const coldStartPrior = mapMode(
    "cold_start_prior",
    "cold_start_prior",
    "learned_route",
    findMode(params.coldStartPrior, "learned_route"),
  );
  const modes = [noBrain, graphPrior, coldStartSingle, coldStartPrior, learnedRoute];
  const deltas = [
    delta(noBrain, coldStartPrior),
    delta(coldStartSingle, coldStartPrior),
    delta(graphPrior, coldStartPrior),
    delta(coldStartPrior, learnedRoute),
  ];
  const singleToPrior = deltas.find((entry) => entry.leftMode === "cold_start_prior_single" && entry.rightMode === "cold_start_prior");
  const graphToPrior = deltas.find((entry) => entry.leftMode === "graph_prior_only" && entry.rightMode === "cold_start_prior");
  const usefulContextWin = (singleToPrior?.phraseHitDeltaRightMinusLeft ?? 0) > 0
    || (singleToPrior?.meanQualityDeltaRightMinusLeft ?? 0) > 0;
  const noOverfireVsGraphPrior = coldStartPrior.totalSelectedContextBlockCount <= graphPrior.totalSelectedContextBlockCount
    && coldStartPrior.totalSelectedContextChars <= graphPrior.totalSelectedContextChars
    && coldStartPrior.totalPhraseHitCount >= graphPrior.totalPhraseHitCount;

  return {
    contract: COLD_START_SCORECARD_CONTRACT_V1,
    generatedAt: params.generatedAt,
    manifestPath: path.resolve(params.manifestPath),
    candidateArtifactDir: path.resolve(params.candidateArtifactDir),
    traceCount: params.baseline.report.successfulTraceCount,
    modes,
    deltas,
    verdict: {
      usefulContextWin,
      noOverfireVsGraphPrior,
      summary: usefulContextWin && noOverfireVsGraphPrior
        ? `cold_start_prior recovers useful context versus the single-block prior and matches graph_prior_only recall without selecting more context (${graphToPrior?.selectedContextCharDeltaRightMinusLeft ?? "unknown"} chars).`
        : "cold_start_prior scorecard is partial; see deltas for the exact boundary.",
    },
  };
}

export function renderColdStartScorecardMarkdownV1(scorecard: ColdStartScorecardV1): string {
  const lines: string[] = [];
  lines.push("# Cold-start scorecard");
  lines.push("");
  lines.push(`Generated: ${scorecard.generatedAt}`);
  lines.push(`Manifest: \`${scorecard.manifestPath}\``);
  lines.push(`Candidate artifact: \`${scorecard.candidateArtifactDir}\``);
  lines.push(`Trace count: ${scorecard.traceCount}`);
  lines.push("");
  lines.push("## Mode summary");
  lines.push("");
  lines.push("| mode | mean quality | phrase hits | selected blocks | selected chars | prompt tokens | warnings |");
  lines.push("| --- | ---: | ---: | ---: | ---: | ---: | ---: |");
  for (const mode of scorecard.modes) {
    lines.push(`| ${mode.mode} | ${mode.meanQualityScore ?? "n/a"} | ${mode.totalPhraseHitCount}/${mode.totalPhraseCount} | ${mode.totalSelectedContextBlockCount} | ${mode.totalSelectedContextChars} | ${mode.estimatedPromptTokens} | ${mode.warningCount} |`);
  }
  lines.push("");
  lines.push("## Deltas");
  lines.push("");
  lines.push("Positive quality/phrase deltas mean the right-hand mode improved; negative context deltas mean it used less context.");
  lines.push("");
  lines.push("| comparison | quality Δ | phrase-hit Δ | block Δ | char Δ | token Δ |");
  lines.push("| --- | ---: | ---: | ---: | ---: | ---: |");
  for (const item of scorecard.deltas) {
    lines.push(`| ${item.rightMode} vs ${item.leftMode} | ${item.meanQualityDeltaRightMinusLeft ?? "n/a"} | ${item.phraseHitDeltaRightMinusLeft} | ${item.selectedContextBlockDeltaRightMinusLeft} | ${item.selectedContextCharDeltaRightMinusLeft} | ${item.promptTokenDeltaRightMinusLeft} |`);
  }
  lines.push("");
  lines.push("## Verdict");
  lines.push("");
  lines.push(`- Useful-context win: \`${scorecard.verdict.usefulContextWin}\``);
  lines.push(`- No overfire vs graph_prior_only: \`${scorecard.verdict.noOverfireVsGraphPrior}\``);
  lines.push(`- Summary: ${scorecard.verdict.summary}`);
  lines.push("");
  lines.push("## Honest boundary");
  lines.push("");
  lines.push("This is a frozen replay scorecard over checked-in sanitized/replayable traces. `cold_start_prior` maps to the candidate-artifact `learned_route` replay override, not to the served learned-route hot path. The served `learned_route` baseline remains stronger where a learned router is already available; this lane improves the cold-start prior/selection fallback and keeps the served/publish boundary unchanged.");
  lines.push("");
  return `${lines.join("\n")}\n`;
}

export async function runColdStartScorecardV1(input: RunColdStartScorecardInputV1 = {}): Promise<ColdStartScorecardV1> {
  const outputDir = path.resolve(input.outputDir ?? path.join("artifacts", "cold-start-scorecard"));
  const scratchRootDir = path.resolve(input.scratchRootDir ?? path.join(outputDir, "scratch"));
  const manifestPath = path.resolve(input.manifestPath ?? DEFAULT_COMPARATIVE_EVAL_MANIFEST_PATH);
  const candidateArtifactDir = path.resolve(input.candidateArtifactDir ?? DEFAULT_COLD_START_PRIOR_ARTIFACT_DIR);
  const policy = {
    maxFailedTraceCount: 20,
    minCandidateTraceTieOrBetterRateVsBaseline: 0,
    maxCandidateMeanQualityRegressionVsBaseline: 999,
    minBaselineMeanQualityGainVsFloor: -999,
    maxCandidateTiePromotionDeltaVsBaseline: 999,
  };
  mkdirSync(outputDir, { recursive: true });
  const { runComparativeEval } = await import("./comparative-eval-runner.ts");
  const baseline = runComparativeEval({
    manifestPath,
    outputDir: path.join(outputDir, "baseline"),
    scratchRootDir: path.join(scratchRootDir, "baseline"),
    workedTraceLimit: 3,
    policy,
  });
  const coldStartPriorSingle = runComparativeEval({
    manifestPath,
    outputDir: path.join(outputDir, "cold-start-prior-single"),
    scratchRootDir: path.join(scratchRootDir, "cold-start-prior-single"),
    workedTraceLimit: 3,
    learnedRouteCandidateArtifactDir: candidateArtifactDir,
    learnedRouteCandidateArtifactMode: "selection_override",
    learnedRouteCandidateMaxCandidateIds: 1,
    policy,
  });
  const coldStartPrior = runComparativeEval({
    manifestPath,
    outputDir: path.join(outputDir, "cold-start-prior"),
    scratchRootDir: path.join(scratchRootDir, "cold-start-prior"),
    workedTraceLimit: 3,
    learnedRouteCandidateArtifactDir: candidateArtifactDir,
    learnedRouteCandidateArtifactMode: "selection_override",
    policy,
  });
  const scorecard = buildColdStartScorecardV1({
    generatedAt: input.generatedAt ?? new Date().toISOString(),
    manifestPath,
    candidateArtifactDir,
    baseline,
    coldStartPriorSingle,
    coldStartPrior,
  });
  writeFileSync(path.join(outputDir, "scorecard.json"), `${JSON.stringify(scorecard, null, 2)}\n`, "utf8");
  writeFileSync(path.join(outputDir, "summary.md"), renderColdStartScorecardMarkdownV1(scorecard), "utf8");
  return scorecard;
}
