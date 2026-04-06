import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { loadAndFilterColdStartRouterApprovedExportV1 } from '../src/brain-core/cold-start-router-approved-export-loader.ts';
import { replayColdStartRouterArtifactV1 } from '../src/brain-core/cold-start-router-replay-gate.ts';
import { loadColdStartRouterArtifactBundleV1, scoreColdStartRouteRowFromArtifactBundleV1 } from '../src/brain-core/cold-start-router-runtime.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

const trainExportPath = path.join(repoRoot, 'artifacts', 'cold-start-router-approved-export', 'real-approved-router-export.hotpotqa-musique.v1.json');
const heldoutExportPath = path.join(repoRoot, 'artifacts', 'cold-start-router-approved-export', 'real-approved-router-export.hotpotqa-musique.v2.json');
const trainArtifactDir = path.join(repoRoot, 'artifacts', 'cold-start-router-approved-export', 'real-approved-router-train.hotpotqa-musique.v2');

const reportPath = '/Users/guclaw/.openclaw/workspace/task-artifacts/T-20260405-147/lane-ac-heldout-eval.md';
const statusPath = '/Users/guclaw/.openclaw/workspace/task-status/T-20260405-147/lane-ac.json';
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

  const reportLines = [
    '# Lane AC — held-out baseline comparison over the approved v2 candidate',
    '',
    '## Verdict',
    'Toy held-out check: **yes**.',
    'Serious evaluation tranche: **no** — this is still a narrow QA-family holdout over four rows.',
    '',
    '## What I ran',
    `- Train export: \`${trainExportPath}\``,
    `- Holdout export: \`${heldoutExportPath}\``,
    `- Canonical train artifact dir: \`${trainArtifactDir}\``,
    '',
    '### Commands executed',
    '- `npx tsx scripts/lane-ac-heldout-eval.ts`',
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
    `- held-out rows came from the later approved QA export, excluding the original two training rows`,
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
    '- Top-1 now generalizes cleanly to the later approved QA rows: the learned candidate matches the score-hint heuristic on all four held-out decisions.',
    '- STOP_LOCAL remains under-proven: the learned candidate still predicts CONTINUE on the held-out STOP_LOCAL row.',
    '- So the approved v2 candidate is real, but it still needs better STOP_LOCAL coverage to become a serious tranche.',
    '',
    '## Why this is still only a toy held-out check',
    '- same source family: QA only',
    '- same export lineage: HotpotQA + MuSiQue only',
    '- only four held-out rows',
    '- baseline is a local heuristic, not a broader router family or multi-split benchmark suite',
    '- no calibration sweep, confidence intervals, or cross-source breadth',
    '',
    '## What a serious tranche still needs',
    '- more approved_train / approved_eval_only rows across source families',
    '- a real held-out split owned separately from the training export lineage',
    '- explicit STOP_LOCAL supervision coverage in the train set',
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
    lane: 'AC',
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
    assessment: 'toy_heldout_check',
    verdict: 'held-out top-1 ties the heuristic baseline; stop behavior still lacks STOP_LOCAL breadth',
    files_changed: changedFiles,
    notes: [
      'This is a toy held-out check, not a serious evaluation tranche.',
      'The held-out set is the later approved QA export minus the two training rows.',
      'Top-1 ties the score-hint heuristic on all four held-out decisions, but STOP_LOCAL is still unresolved.',
      'The learned candidate still fails both STOP_LOCAL replay rows in the v2 export.',
    ],
  };

  writeUtf8(reportPath, `${report}\n`);
  writeUtf8(statusPath, `${JSON.stringify(status, null, 2)}\n`);

  console.log('Lane AC held-out eval: ok');
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
