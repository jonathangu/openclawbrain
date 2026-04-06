import { fileURLToPath } from "node:url";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  loadAndFilterColdStartRouterApprovedExportV1,
  summarizeColdStartRouterApprovedExportV1,
} from "../../src/brain-core/cold-start-router-approved-export-loader.js";

const approvedExportPath = fileURLToPath(
  new URL("../../artifacts/cold-start-router-approved-export/approved-router-export.fixture.v1.json", import.meta.url),
);

const repoRoot = path.resolve(path.dirname(approvedExportPath), "..", "..");

describe("cold-start router approved export loader", () => {
  it("filters the curated export down to approved_train rows only", () => {
    const loaded = loadAndFilterColdStartRouterApprovedExportV1(approvedExportPath);

    expect(loaded.summary).toMatchObject({
      exportId: "cold-start-router-approved-export-fixture-v1",
      generatedAt: "2026-04-05T16:10:00Z",
      rawRegistryEntryCount: 2,
      approvedRegistryEntryCount: 1,
      rawRowCount: 4,
      approvedRowCount: 2,
      skippedRegistryEntryCount: 1,
      skippedRowCount: 2,
      approvedDatasetIds: ["router_fixture_train_v1"],
    });

    expect(loaded.registryEntries).toHaveLength(1);
    expect(loaded.registryEntries[0]?.dataset_id).toBe("router_fixture_train_v1");
    expect(loaded.routeRows).toHaveLength(2);
    expect(loaded.routeRows.every((row) => row.dataset_id === "router_fixture_train_v1")).toBe(true);
    expect(loaded.routeRows.every((row) => row.provenance.review_status === "approved_train")).toBe(true);

    expect(loaded.skippedRegistryEntries).toHaveLength(1);
    expect(loaded.skippedRegistryEntries[0]).toMatchObject({
      datasetId: "router_fixture_holdout_v1",
    });
    expect(loaded.skippedRegistryEntries[0]?.reason).toContain("under_review");

    expect(loaded.skippedRows).toHaveLength(2);
    expect(loaded.skippedRows.map((row) => row.rowId)).toEqual([
      "router_fixture_row_003",
      "router_fixture_row_004",
    ]);
    expect(loaded.skippedRows[0]?.reason).toContain("only approved_train rows are eligible");
    expect(loaded.skippedRows[1]?.reason).toContain("governed source-intake layer");

    expect(summarizeColdStartRouterApprovedExportV1(loaded)).toMatchObject({
      bundle: {
        exportId: "cold-start-router-approved-export-fixture-v1",
      },
      summary: {
        approvedRowCount: 2,
      },
    });

    expect(repoRoot).toContain("openclawbrain");
  });
});
