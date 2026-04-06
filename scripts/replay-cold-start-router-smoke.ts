#!/usr/bin/env node

import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { compileColdStartDocsQaSourceBundleFromFileV1 } from "../src/brain-core/cold-start-data-compiler.ts";
import { trainColdStartRouterArtifactV1 } from "../src/brain-core/cold-start-router-trainer.ts";
import { replayColdStartRouterArtifactV1 } from "../src/brain-core/cold-start-router-replay-gate.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const sampleBundlePath = path.join(repoRoot, "artifacts", "cold-start-router-sample", "docs-qa-sample.raw.json");

const tempRoots: string[] = [];

function createTempRoot(label: string): string {
  const root = mkdtempSync(path.join(os.tmpdir(), `${label}-`));
  tempRoots.push(root);
  return root;
}

function cleanup(): void {
  while (tempRoots.length > 0) {
    rmSync(tempRoots.pop()!, { recursive: true, force: true });
  }
}

async function main(): Promise<void> {
  const bundle = compileColdStartDocsQaSourceBundleFromFileV1({ bundlePath: sampleBundlePath, repoRoot });
  const outputDir = createTempRoot("cold-start-router-replay-smoke");
  mkdirSync(outputDir, { recursive: true });

  trainColdStartRouterArtifactV1({
    artifactId: "router-artifact-replay-smoke",
    artifactVersion: "0.0.1",
    packType: "base",
    compatibleRuntimeVersion: "openclawbrain-runtime@0.3.8",
    registryEntries: [bundle.registryEntry],
    routeRows: bundle.routeRows,
    outputDir,
    routerIdentity: "router:smoke:replay",
    createdAt: "2026-04-05T17:35:00Z",
    trainingDataRefs: [bundle.registryEntry.dataset_id],
    replayGateRefs: ["replay:docs-qa-sample:tiny-gate"],
  });

  const verdict = replayColdStartRouterArtifactV1({
    artifactDir: outputDir,
    routeRows: bundle.routeRows,
  });

  console.log("Cold-start router replay smoke: ok");
  console.log(`verdict: ${verdict.verdict}`);
  console.log(`summary: ${verdict.summary}`);
  console.log(`manifestChecksum: ${verdict.manifestSummary?.checksum ?? "n/a"}`);
  console.log(`passedRows: ${verdict.passedRowCount}/${verdict.evaluatedRowCount}`);
  console.log(`firstRow: ${verdict.rowResults[0]?.rowId ?? "n/a"}`);

  if (!verdict.passed) {
    throw new Error(`replay gate failed: ${verdict.summary}`);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
}).finally(() => {
  cleanup();
});
