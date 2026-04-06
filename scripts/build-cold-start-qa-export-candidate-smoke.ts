#!/usr/bin/env node

import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildColdStartQaSnapshotExportCandidateV1,
  loadColdStartQaSnapshotExportCandidateV1,
  summarizeColdStartQaSnapshotExportCandidateV1,
  writeColdStartQaSnapshotExportCandidateV1,
} from "../src/brain-core/cold-start-qa-export-candidate.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const workspaceRoot = path.resolve(repoRoot, "..");
const registryPath = path.join(repoRoot, "data", "cold-start", "registry.bootstrap.json");
const outputDir = path.join(repoRoot, "artifacts", "cold-start-qa-export-candidate");
const outputPath = path.join(outputDir, "under-review-qa-export-candidate.v1.json");

async function main(): Promise<void> {
  const candidate = buildColdStartQaSnapshotExportCandidateV1({
    registryPath,
    workspaceRoot,
    generatedAt: "2026-04-05T18:10:00Z",
    sampleCount: 1,
  });

  mkdirSync(outputDir, { recursive: true });
  writeColdStartQaSnapshotExportCandidateV1(outputPath, candidate);

  const loaded = loadColdStartQaSnapshotExportCandidateV1(outputPath);
  const summary = summarizeColdStartQaSnapshotExportCandidateV1(loaded);

  console.log("Cold-start QA snapshot export candidate smoke: ok");
  console.log(`outputPath: ${outputPath}`);
  console.log(`summary: ${JSON.stringify(summary)}`);
  console.log(`reviewStatus: ${loaded.review_status}`);
  console.log(`routeRows: ${loaded.route_rows.length}`);
  console.log(`datasets: ${loaded.datasets.map((dataset) => dataset.datasetId).join(", ")}`);
  console.log(`snapshotRefs: ${loaded.datasets.map((dataset) => dataset.snapshotRef).join(", ")}`);
  if (loaded.review_status !== "under_review") {
    throw new Error(`export candidate must remain under_review, got ${loaded.review_status}`);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
