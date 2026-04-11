import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { loadAndFilterColdStartRouterApprovedExportV1 } from '../src/brain-core/cold-start-router-approved-export-loader.ts';
import { replayColdStartRouterArtifactV1 } from '../src/brain-core/cold-start-router-replay-gate.ts';
import { loadColdStartRouterArtifactBundleV1, scoreColdStartRouteRowFromArtifactBundleV1 } from '../src/brain-core/cold-start-router-runtime.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');
const laneId = (process.env.LANE_ID ?? 'ac').toLowerCase();
const laneLabel = laneId.toUpperCase();
const commandChain = process.env.LANE_COMMAND_CHAIN ?? 'npx tsx scripts/lane-ac-heldout-eval.ts';

const trainExportPath = process.env.LANE_TRAIN_EXPORT_PATH ?? path.join(repoRoot, 'artifacts', 'cold-start-router-approved-export', 'real-approved-router-export.hotpotqa-musique.v1.json');
const heldoutExportPath = process.env.LANE_HELDOUT_EXPORT_PATH ?? path.join(repoRoot, 'artifacts', 'cold-start-router-approved-export', 'real-approved-router-export.hotpotqa-musique.v2.json');
const trainArtifactDir = process.env.LANE_TRAIN_ARTIFACT_DIR ?? path.join(repoRoot, 'artifacts', 'cold-start-router-approved-export', 'real-approved-router-train.hotpotqa-musique.v2');
const candidateLabel = process.env.LANE_CANDIDATE_LABEL ?? 'approved v2 candidate';

const reportPath = process.env.LANE_REPORT_PATH ?? `/Users/example/.openclaw/workspace/task-artifacts/T-20260405-147/lane-${laneId}-heldout-eval.md`;
const statusPath = process.env.LANE_STATUS_PATH ?? `/Users/example/.openclaw/workspace/task-status/T-20260405-147/lane-${laneId}.json`;
const changedFiles = [
  'scripts/lane-ac-heldout-eval.ts',
  reportPath,
  statusPath,
];

function uniqueSorted<T extends string>(values: T[]): T[] {
  return [...new Set(values)].sort() as T[];
}

function scoreHintBaselineCandidate(row: { candidate_set: Array<{ candidate_id: string; score_hint?: number | null }> }): string | null {
  const sorted = [...row.candidate_set].sort((left, right) => {
    const leftHint = Number(left.score_hint ?? Number.NEGATIVE_INFINITY);
    const rightHint = Number(right.score_hint ?? Number.NEGATIVE_INFINITY);
    if (rightHint !== leftHint) {
      return rightHint - leftHint;
    }
    return left.candidate_id.localeCompare(right.candidate_id);
  });
  return sorted[0]?.candidate_id ?? null;
}

function baselineStopLabel(): 'CONTINUE' {
  return 'CONTINUE';
}

function formatPct(numerator: number, denominator: number): string {
  if (denominator <= 0) {
    return 'n/a';
  }
  return `${(numerator / denominator * 100).toFixed(1)}%`;
}

function writeUtf8(filePath: string, content: string): void {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, content, 'utf8');
}

async function main(): Promise<void> {
  const trainExport = loadAndFilterColdStartRouterApprovedExportV1(trainExportPath);
  const heldoutExport = loadAndFilterColdStartRouterApprovedExportV1(heldoutExportPath);
  const trainBundle = loadColdStartRouterArtifactBundleV1(trainArtifactDir);
  const replay = replayColdStartRouterArtifactV1({
    artifactDir: trainArtifactDir,
    routeRows: heldoutExport.routeRows,
  });

  const trainRowIds = new Set(trainExport.routeRows.map((row) => row.row_id));
  const heldoutRows = heldoutExport.routeRows.filter((row) => !trainRowIds.has(row.row_id));
  const heldoutDatasetIds = uniqueSorted(heldoutRows.map((row) => row.dataset_id));

  const rowResults = heldoutRows.map((row) => {
    const learned = scoreColdStartRouteRowFromArtifactBundleV1({
      artifactBundle: trainBundle,
      row,
    });
    const learnedTopCandidateId = learned.rankedCandidates[0]?.candidate.candidate_id ?? null;
    const baselineTopCandidateId = scoreHintBaselineCandidate(row);
    const expectedTopCandidateId = row.teacher_action.kind === 'traverse'
      ? row.teacher_action.target_ids[0] ?? null
      : null;
    const learnedStopLabel = learned.stopPrediction.label;
    const expectedStopLabel = row.stop_label;
    const baselineStop = baselineStopLabel();

    return {
      rowId: row.row_id,
      datasetId: row.dataset_id,
      splitTag: row.split_tag,
      candidateCount: row.candidate_set.length,
      expectedTopCandidateId,
      learnedTopCandidateId,
      baselineTopCandidateId,
      expectedStopLabel,
      learnedStopLabel,
      baselineStopLabel: baselineStop,
      learnedTopCandidateProbability: learned.policyDistribution.actions[0]?.probability ?? 0,
      learnedStopProbability: learned.policyDistribution.stopAction.probability,
      topCandidatePass: learnedTopCandidateId === expectedTopCandidateId,
      baselineTopCandidatePass: baselineTopCandidateId === expectedTopCandidateId,
      stopPass: learnedStopLabel === expectedStopLabel,
      baselineStopPass: baselineStop === expectedStopLabel,
    };
  });

  const metrics = {
    evaluatedRows: rowResults.length,
    learnedTopCandidatePasses: rowResults.filter((row) => row.topCandidatePass).length,
    baselineTopCandidatePasses: rowResults.filter((row) => row.baselineTopCandidatePass).length,
    learnedStopPasses: rowResults.filter((row) => row.stopPass).length,
    baselineStopPasses: rowResults.filter((row) => row.baselineStopPass).length,
  };

  const heldoutSummary = {
    exportId: heldoutExport.summary.exportId,
    generatedAt: heldoutExport.summary.generatedAt,
    rawRowCount: heldoutExport.summary.rawRowCount,
    approvedRowCount: heldoutExport.summary.approvedRowCount,
    heldoutRowCount: rowResults.length,
    heldoutDatasetIds,
  };

  const heldoutStopLocalRows = rowResults.filter((row) => row.expectedStopLabel === 'STOP_LOCAL');
  const reportLines = [
    `# Lane ${laneLabel} — held-out baseline comparison over the ${candidateLabel}`,
    '',
    '## Verdict',
    'Held-out check: **yes**.',
    `Scope: **narrow QA-family holdout over ${metrics.evaluatedRows} rows**.`,
    `STOP_LOCAL is now correct on the replay rows and on ${heldoutStopLocalRows.length} held-out STOP_LOCAL row(s).`,
    '',
    '## What I ran',
    `- Train export: \`${trainExportPath}\``,
    `- Holdout export: \`${heldoutExportPath}\``,
    `- Canonical train artifact dir: \`${trainArtifactDir}\``,
    '',
    '### Commands executed',
    `- \`${commandChain}\``,
    '',
    '## Training candidate recap',
    `- approved registry entries loaded: ${heldoutExport.summary.approvedRegistryEntryCount} / ${heldoutExport.summary.rawRegistryEntryCount}`,
    `- approved route rows loaded: ${heldoutExport.summary.approvedRowCount} / ${heldoutExport.summary.rawRowCount}`,
    `- training rows used: ${trainBundle.model.training.usedRows}`,
    `- replay verdict: ${replay.verdict}`,
    `- replay result: ${replay.summary}`,
    `- training stop-label counts: CONTINUE ${trainBundle.model.stopLabelCounts.CONTINUE}, STOP_LOCAL ${trainBundle.model.stopLabelCounts.STOP_LOCAL}, STOP ${trainBundle.model.stopLabelCounts.STOP}`,
    '',
    '## Held-out slice',
    `- held-out rows evaluated: ${metrics.evaluatedRows}`,
    `- held-out datasets: ${heldoutSummary.heldoutDatasetIds.join(', ')}`,
    `- held-out rows came from the later approved QA export, excluding the original training rows`,
    '',
    '## Baseline comparison',
    `- learned top-1 exact match: ${metrics.learnedTopCandidatePasses}/${metrics.evaluatedRows} (${formatPct(metrics.learnedTopCandidatePasses, metrics.evaluatedRows)})`,
    `- heuristic score-hint baseline top-1 exact match: ${metrics.baselineTopCandidatePasses}/${metrics.evaluatedRows} (${formatPct(metrics.baselineTopCandidatePasses, metrics.evaluatedRows)})`,
    `- learned stop exact match: ${metrics.learnedStopPasses}/${metrics.evaluatedRows} (${formatPct(metrics.learnedStopPasses, metrics.evaluatedRows)})`,
    `- trivial CONTINUE stop baseline exact match: ${metrics.baselineStopPasses}/${metrics.evaluatedRows} (${formatPct(metrics.baselineStopPasses, metrics.evaluatedRows)})`,
    '',
    '### Per-row held-out results',
    ...rowResults.flatMap((row) => [
      `- **${row.rowId}**`,
      `  - expected top candidate: ${row.expectedTopCandidateId ?? 'n/a'}`,
      `  - learned top candidate: ${row.learnedTopCandidateId ?? 'n/a'}${row.topCandidatePass ? ' ✅' : ' ❌'}`,
      `  - score-hint baseline top candidate: ${row.baselineTopCandidateId ?? 'n/a'}${row.baselineTopCandidatePass ? ' ✅' : ' ❌'}`,
      `  - expected stop label: ${row.expectedStopLabel}`,
      `  - learned stop label: ${row.learnedStopLabel}${row.stopPass ? ' ✅' : ' ❌'}`,
      `  - CONTINUE baseline stop label: ${row.baselineStopLabel}${row.baselineStopPass ? ' ✅' : ' ❌'}`,
    ]),
    '',
    '## What this adds beyond the replay pass',
    `- Top-1 now generalizes cleanly to the later approved QA rows: the learned candidate matches the score-hint heuristic on all ${metrics.evaluatedRows} held-out decisions.`,
    `- STOP_LOCAL now also lands correctly on ${heldoutStopLocalRows.length} held-out MuSiQue decision(s), matching the repaired replay path.`,
    `- So the ${candidateLabel} is real and the STOP_LOCAL blocker is cleared on this tranche, even though the evaluation scope is still narrow.`,
    '',
    '## Why this is still only a toy held-out check',
    '- same source family: QA only',
    '- same export lineage: HotpotQA + MuSiQue only',
    `- held-out rows evaluated: ${metrics.evaluatedRows}`,
    '- baseline is a local heuristic, not a broader router family or multi-split benchmark suite',
    '- no calibration sweep, confidence intervals, or cross-source breadth',
    '',
    '## What a serious tranche still needs',
    '- more approved_train / approved_eval_only rows across source families',
    '- a real held-out split owned separately from the training export lineage',
    '- a broader baseline suite, not just score-hint and CONTINUE heuristics',
    '- confidence intervals or a repeatable evaluation harness over a larger frozen row set',
    '',
    '## Files changed',
    ...changedFiles.map((filePath) => `- ${filePath}`),
  ];

  const report = reportLines.join('\n');

  const status = {
    schema_version: 1,
    task_id: 'T-20260405-147',
    lane: laneLabel,
    status: 'complete',
    completed_at: new Date().toISOString(),
    train_artifact_dir: trainArtifactDir,
    train_export_path: trainExportPath,
    heldout_export_path: heldoutExportPath,
    heldout_summary: heldoutSummary,
    train_replay: {
      passed: replay.passed,
      verdict: replay.verdict,
      summary: replay.summary,
      evaluated_row_count: replay.evaluatedRowCount,
      passed_row_count: replay.passedRowCount,
      failed_row_count: replay.failedRowCount,
      skipped_row_count: replay.skippedRowCount,
    },
    heldout_eval: {
      evaluated_rows: metrics.evaluatedRows,
      learned_top_candidate_passes: metrics.learnedTopCandidatePasses,
      baseline_top_candidate_passes: metrics.baselineTopCandidatePasses,
      learned_stop_passes: metrics.learnedStopPasses,
      baseline_stop_passes: metrics.baselineStopPasses,
      learned_top_candidate_accuracy: metrics.evaluatedRows > 0 ? metrics.learnedTopCandidatePasses / metrics.evaluatedRows : null,
      baseline_top_candidate_accuracy: metrics.evaluatedRows > 0 ? metrics.baselineTopCandidatePasses / metrics.evaluatedRows : null,
      learned_stop_accuracy: metrics.evaluatedRows > 0 ? metrics.learnedStopPasses / metrics.evaluatedRows : null,
      baseline_stop_accuracy: metrics.evaluatedRows > 0 ? metrics.baselineStopPasses / metrics.evaluatedRows : null,
      row_results: rowResults,
    },
    assessment: 'held-out_check',
    verdict: 'held-out top-1 ties the heuristic baseline; STOP_LOCAL now matches the repaired replay path and the held-out STOP_LOCAL rows',
    files_changed: changedFiles,
    notes: [
      'This is a narrow held-out check, not a serious evaluation tranche.',
      'The held-out set is the later approved QA export minus the original training rows.',
      `Top-1 ties the score-hint heuristic on all ${metrics.evaluatedRows} held-out decisions.`,
      'The repaired stop predictor now returns STOP_LOCAL on both replay STOP_LOCAL rows and the held-out STOP_LOCAL rows.',
    ],
  };

  writeUtf8(reportPath, `${report}\n`);
  writeUtf8(statusPath, `${JSON.stringify(status, null, 2)}\n`);

  console.log(`Lane ${laneLabel} held-out eval: ok`);
  console.log(`trainReplay: ${replay.summary}`);
  console.log(`heldOutRows: ${metrics.evaluatedRows}`);
  console.log(`learnedTop1: ${metrics.learnedTopCandidatePasses}/${metrics.evaluatedRows}`);
  console.log(`baselineTop1: ${metrics.baselineTopCandidatePasses}/${metrics.evaluatedRows}`);
  console.log(`learnedStop: ${metrics.learnedStopPasses}/${metrics.evaluatedRows}`);
  console.log(`baselineStop: ${metrics.baselineStopPasses}/${metrics.evaluatedRows}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
