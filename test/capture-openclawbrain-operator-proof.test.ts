import path from "node:path";
import { describe, expect, it } from "vitest";
import { DEFAULT_OUTPUT_PARENT, resolveOutputDir, resolveTeacherV3ProofOutputDir } from "../scripts/capture-openclawbrain-operator-proof.mjs";

describe("capture-openclawbrain-operator-proof", () => {
  it("defaults to the shared workspace artifacts root", () => {
    const outputDir = resolveOutputDir({ outputDir: null });

    expect(outputDir.startsWith(`${DEFAULT_OUTPUT_PARENT}${path.sep}`)).toBe(true);
    expect(path.basename(outputDir)).toMatch(/^operator-proof-\d{8}-\d{6}Z$/);
  });

  it("defaults teacher-v3 proof bundles beneath the shared artifacts tree", () => {
    const outputDir = resolveTeacherV3ProofOutputDir({
      bundleStartedAt: new Date("2026-04-03T18:26:00Z"),
    });

    expect(path.dirname(outputDir)).toBe(path.join(path.resolve(".."), "artifacts", "teacher-v3-proof"));
    expect(path.basename(outputDir)).toBe("teacher-v3-proof-20260403-182600Z");
  });

  it("respects an explicit output directory override", () => {
    expect(resolveOutputDir({ outputDir: "custom/proof-dir" })).toBe(path.resolve("custom/proof-dir"));
    expect(resolveTeacherV3ProofOutputDir({ outputDir: "custom/teacher-v3-proof" })).toBe(
      path.resolve("custom/teacher-v3-proof"),
    );
  });
});
