import { fileURLToPath } from "node:url";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  compileColdStartRepoSnapshotExportCandidateV1,
  loadColdStartRepoSnapshotIndexV1,
  summarizeColdStartRepoSnapshotExportCandidateV1,
} from "../../src/brain-core/cold-start-repo-export-candidate.js";
import {
  validateDataRegistryEntryV1,
  validateGraphCompilerContractV1,
} from "../../src/brain-core/cold-start-router-contracts.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..");

describe("cold-start repo snapshot export candidate", () => {
  it("compiles the real snapshot index into a bridge artifact without claiming approved_train", () => {
    const snapshotIndex = loadColdStartRepoSnapshotIndexV1(repoRoot);
    expect(snapshotIndex.schema_version).toBe(1);
    expect(snapshotIndex.snapshots).toHaveLength(4);
    expect(snapshotIndex.snapshots.map((snapshot) => snapshot.dataset_id)).toContain("toolmind_v1");

    const bundle = compileColdStartRepoSnapshotExportCandidateV1({
      repoRoot,
      candidateId: "cold-start-repo-snapshot-export-candidate-smoke",
      generatedAt: "2026-04-05T18:25:00-07:00",
    });

    expect(bundle.contract).toBe("cold_start_repo_export_candidate.v1");
    expect(bundle.snapshot_index.snapshots).toHaveLength(3);
    expect(bundle.snapshot_index.snapshots.map((snapshot) => snapshot.dataset_id)).not.toContain("toolmind_v1");
    expect(bundle.report).toMatchObject({
      candidate_id: "cold-start-repo-snapshot-export-candidate-smoke",
      snapshot_count: 3,
      manifest_count: 3,
      registry_entry_count: 3,
      exact_snapshot_file_count: 19,
      data_file_count: 13,
      route_seed_count: 13,
      materialized_snapshot_count: 3,
      rights_caveat_count: 3,
      approved_train_claim: false,
      route_row_count: 0,
      snapshot_backed: true,
      bridge_state: "snapshot-backed",
    });

    expect(bundle.report.dataset_ids).toEqual([
      "swe_bench_v1",
      "repo_bench_python_v1.1",
      "repo_bench_java_v1.1",
    ]);
    expect(bundle.report.caveats).toHaveLength(3);
    expect(bundle.report.caveats.join(" \n")).toContain("approved_train claim");
    expect(bundle.registry_entries).toHaveLength(3);
    expect(bundle.registry_entries.every((entry) => validateDataRegistryEntryV1(entry).valid)).toBe(true);
    expect(bundle.registry_entries.map((entry) => entry.approval_status)).toEqual([
      "under_review",
      "under_review",
      "under_review",
    ]);

    expect(bundle.graph_compiler.source_family).toBe("repo");
    expect(validateGraphCompilerContractV1(bundle.graph_compiler).valid).toBe(true);
    expect(bundle.graph_compiler.provenance_rules).toContain("row-level route export stays pending until parquet extraction and rights review complete");
    expect(bundle.graph_compiler.output_neighborhood_pack.pack_id).toContain("cold-start-repo-snapshot-export-candidate-smoke");

    expect(bundle.route_seeds).toHaveLength(13);
    expect(bundle.route_seeds.every((seed) => seed.bridge_status === "pending_row_extraction")).toBe(true);
    expect(bundle.route_seeds.filter((seed) => seed.dataset_id === "swe_bench_v1")).toHaveLength(3);
    expect(bundle.route_seeds.filter((seed) => seed.dataset_id === "repo_bench_python_v1.1")).toHaveLength(4);
    expect(bundle.route_seeds.filter((seed) => seed.dataset_id === "repo_bench_java_v1.1")).toHaveLength(6);
    expect(bundle.route_seeds.some((seed) => seed.dataset_id === "swe_bench_v1" && seed.split_name === "test")).toBe(true);
    expect(bundle.route_seeds.some((seed) => seed.dataset_id === "repo_bench_python_v1.1" && seed.split_name === "cross_file_random")).toBe(true);
    expect(bundle.route_seeds.some((seed) => seed.dataset_id === "repo_bench_java_v1.1" && seed.split_name === "in_file")).toBe(true);

    expect(summarizeColdStartRepoSnapshotExportCandidateV1(bundle)).toMatchObject({
      candidateId: "cold-start-repo-snapshot-export-candidate-smoke",
      snapshotCount: 3,
      manifestCount: 3,
      registryEntryCount: 3,
      routeSeedCount: 13,
      snapshotBacked: true,
      bridgeState: "snapshot-backed",
      dataFileCount: 13,
      rightsCaveatCount: 3,
    });
  });
});
