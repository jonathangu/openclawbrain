import path from "node:path";
import { describe, expect, it } from "vitest";
import { DEFAULT_OUTPUT_PARENT, resolveOutputDir } from "../scripts/capture-openclawbrain-operator-proof.mjs";

describe("capture-openclawbrain-operator-proof", () => {
  it("defaults to the shared workspace artifacts root", () => {
    const outputDir = resolveOutputDir({ outputDir: null });

    expect(outputDir.startsWith(`${DEFAULT_OUTPUT_PARENT}${path.sep}`)).toBe(true);
    expect(path.basename(outputDir)).toMatch(/^operator-proof-\d{8}-\d{6}Z$/);
  });

  it("respects an explicit output directory override", () => {
    expect(resolveOutputDir({ outputDir: "custom/proof-dir" })).toBe(path.resolve("custom/proof-dir"));
  });
});
