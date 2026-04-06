import { mkdtempSync, mkdirSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  buildColdStartQaSnapshotExportCandidateV1,
  loadColdStartQaSnapshotExportCandidateV1,
  summarizeColdStartQaSnapshotExportCandidateV1,
  writeColdStartQaSnapshotExportCandidateV1,
} from "../../src/brain-core/cold-start-qa-export-candidate.js";
import {
  validateDataRegistryEntryV1,
  validateRouteDecisionRowV1,
} from "../../src/brain-core/cold-start-router-contracts.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "../..");
const workspaceRoot = path.resolve(repoRoot, "..");
const registryPath = path.join(repoRoot, "data", "cold-start", "registry.bootstrap.json");

function createTempRoot(label: string): string {
  return mkdtempSync(path.join(os.tmpdir(), `${label}-`));
}

describe("cold-start QA snapshot export candidate", () => {
  it("builds a real snapshot-backed under-review export candidate from HotpotQA and MuSiQue", () => {
    const candidate = buildColdStartQaSnapshotExportCandidateV1({
      registryPath,
      workspaceRoot,
      generatedAt: "2026-04-05T18:00:00Z",
      sampleCount: 1,
    });

    expect(candidate).toMatchObject({
      contract: "cold_start_qa_snapshot_export_candidate.v1",
      export_id: "cold-start-qa-snapshot-export-candidate-v1",
      review_status: "under_review",
      source_registry_path: registryPath,
    });
    expect(candidate.datasets).toHaveLength(2);
    expect(candidate.registry_entries).toHaveLength(2);
    expect(candidate.route_rows).toHaveLength(2);
    expect(candidate.registry_entries.every((entry) => validateDataRegistryEntryV1(entry).valid)).toBe(true);
    expect(candidate.route_rows.every((row) => validateRouteDecisionRowV1(row).valid)).toBe(true);
    expect(candidate.route_rows.every((row) => row.provenance.review_status === "under_review")).toBe(true);
    expect(candidate.route_rows.map((row) => row.dataset_id).sort()).toEqual(["hotpotqa_v1", "musique_v1"]);

    const [hotpotqa, musique] = candidate.datasets;
    expect(hotpotqa.datasetFamily).toBe("hotpotqa");
    expect(hotpotqa.datasetId).toBe("hotpotqa_v1");
    expect(hotpotqa.compilation.report).toMatchObject({
      datasetId: "hotpotqa_v1",
      acceptedRowCount: 1,
      rejectedRowCount: 0,
    });
    expect(hotpotqa.compilation.routeRows[0]?.teacher_action.kind).toBe("traverse");
    expect(hotpotqa.compilation.routeRows[0]?.candidate_set.length).toBeGreaterThan(1);

    expect(musique.datasetFamily).toBe("musique");
    expect(musique.datasetId).toBe("musique_v1");
    expect(musique.compilation.report).toMatchObject({
      datasetId: "musique_v1",
      acceptedRowCount: 1,
      rejectedRowCount: 0,
    });
    expect(musique.compilation.routeRows[0]?.teacher_action.kind).toBe("traverse");
    expect(musique.compilation.routeRows[0]?.candidate_set.some((candidate) => candidate.candidate_id.startsWith("musique_v1:support:"))).toBe(true);

    const summary = summarizeColdStartQaSnapshotExportCandidateV1(candidate);
    expect(summary).toMatchObject({
      exportId: "cold-start-qa-snapshot-export-candidate-v1",
      reviewStatus: "under_review",
      datasetCount: 2,
      routeRowCount: 2,
      registryEntryCount: 2,
      datasetIds: ["hotpotqa_v1", "musique_v1"],
    });
    expect(summary.snapshotRefs).toEqual([
      "openclawbrain/data/cold-start/snapshots/hotpotqa_v1/hotpotqa.github.io__20260405/manifest.json",
      "openclawbrain/data/cold-start/snapshots/musique_v1/stonybrooknlp__musique__922ac98f19a2/manifest.json",
    ]);
    expect(candidate.notes.some((note) => note.includes("under review only"))).toBe(true);
    expect(candidate.notes.some((note) => note.includes("approved_train"))).toBe(true);
  });

  it("expands the export with STOP_LOCAL-rich MuSiQue rows when asked for a larger sample set", () => {
    const candidate = buildColdStartQaSnapshotExportCandidateV1({
      registryPath,
      workspaceRoot,
      generatedAt: "2026-04-05T18:02:00Z",
      hotpotSampleCount: 3,
      musiqueSampleCount: 11,
    });

    expect(candidate.route_rows).toHaveLength(14);
    expect(candidate.route_rows.filter((row) => row.dataset_id === "hotpotqa_v1")).toHaveLength(3);
    expect(candidate.route_rows.filter((row) => row.dataset_id === "musique_v1")).toHaveLength(11);
    expect(candidate.route_rows.filter((row) => row.stop_label === "STOP_LOCAL")).toHaveLength(2);
    expect(candidate.route_rows.every((row) => validateRouteDecisionRowV1(row).valid)).toBe(true);
    expect(candidate.route_rows.every((row) => row.provenance.review_status === "under_review")).toBe(true);
    expect(candidate.route_rows[3]?.dataset_id).toBe("musique_v1");
    expect(candidate.route_rows[3]?.stop_label).toBe("STOP_LOCAL");

    const summary = summarizeColdStartQaSnapshotExportCandidateV1(candidate);
    expect(summary.routeRowCount).toBe(14);
    expect(summary.sampleStrategies).toEqual([
      "first-3-examples-from-hotpotqa-dev-distractor-snapshot",
      "first-11-supporting-examples-from-musique-dev-snapshot-stoplocal-aware",
    ]);
  });

  it("round-trips the export candidate through a file without inventing approved_train state", () => {
    const candidate = buildColdStartQaSnapshotExportCandidateV1({
      registryPath,
      workspaceRoot,
      generatedAt: "2026-04-05T18:05:00Z",
      sampleCount: 1,
    });

    const tempRoot = createTempRoot("cold-start-qa-export-candidate");
    mkdirSync(tempRoot, { recursive: true });
    const filePath = path.join(tempRoot, "candidate.json");
    writeColdStartQaSnapshotExportCandidateV1(filePath, candidate);

    const loaded = loadColdStartQaSnapshotExportCandidateV1(filePath);
    expect(loaded).toMatchObject({
      contract: "cold_start_qa_snapshot_export_candidate.v1",
      review_status: "under_review",
      export_id: candidate.export_id,
    });
    expect(loaded.route_rows.every((row) => row.provenance.review_status === "under_review")).toBe(true);
    expect(loaded.route_rows.some((row) => row.dataset_id === "hotpotqa_v1")).toBe(true);
    expect(loaded.route_rows.some((row) => row.dataset_id === "musique_v1")).toBe(true);

    rmSync(tempRoot, { recursive: true, force: true });
  });
});
