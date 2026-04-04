import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  buildTeacherV3PromotableExamples,
  writeTeacherV3PromotableExamples,
} from "../scripts/teacher-v3-promotable-examples.mjs";

const tempDirs: string[] = [];

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("teacher v3 promotable worked examples", () => {
  it("builds compiler and lint examples with honest reviewable/promotable boundaries", () => {
    const root = mkdtempSync(join(tmpdir(), "teacher-v3-promotable-examples-build-"));
    tempDirs.push(root);

    const examples = buildTeacherV3PromotableExamples(root);

    expect(examples.compiler.proposal.status).toBe("promoted");
    expect(examples.lint.proposal.status).toBe("promotable");
    expect(examples.compiler.proposalSummary.hasReplaySummary).toBe(true);
    expect(examples.lint.proposalSummary.hasReplaySummary).toBe(true);
    expect(examples.compiler.proposalSummary.hasProofBundle).toBe(false);
    expect(examples.lint.proposalSummary.hasProofBundle).toBe(false);
    expect(examples.compiler.replaySummary.status).toBe("promotable");
    expect(examples.lint.replaySummary.status).toBe("promotable");
    expect(examples.compiler.proofBundle.verdictReport.verdict).toBe("reviewable");
    expect(examples.lint.proofBundle.verdictReport.verdict).toBe("reviewable");
    expect(examples.compiler.proofBundle.statusReport.canaryRollout.enabled).toBe(false);
    expect(examples.lint.proofBundle.statusReport.canaryRollout.enabled).toBe(false);
    expect(examples.compiler.proofBundle.statusReport.canaryActivationGuard.requested).toBe(false);
    expect(examples.lint.proofBundle.statusReport.canaryActivationGuard.requested).toBe(false);
    expect(examples.compiler.proofBundle.statusReport.canaryActivationGuard.proofReady).toBe(true);
    expect(examples.lint.proofBundle.statusReport.canaryActivationGuard.proofReady).toBe(true);
    expect(examples.compiler.proofBundle.statusReport.canaryActivationGuard.rollbackReady).toBe(true);
    expect(examples.lint.proofBundle.statusReport.canaryActivationGuard.rollbackReady).toBe(true);
    expect(examples.compiler.example.targetStateNotes.join(" ")).toContain("canary plan is explicit");
    expect(examples.lint.example.targetStateNotes.join(" ")).toContain("target-state overlays");
  });

  it("writes the checked bundle layout to disk", () => {
    const root = mkdtempSync(join(tmpdir(), "teacher-v3-promotable-examples-write-"));
    tempDirs.push(root);

    const result = writeTeacherV3PromotableExamples(root);

    const manifest = JSON.parse(readFileSync(join(root, "manifest.json"), "utf8"));
    expect(manifest.contract).toBe("teacher_v3_promotable_examples_manifest.v1");
    expect(manifest.lanes.compiler.proposalStatus).toBe("promoted");
    expect(manifest.lanes.lint.proposalStatus).toBe("promotable");
    expect(manifest.lanes.compiler.reviewMode).toBe("promotable");
    expect(manifest.lanes.lint.reviewMode).toBe("promotable");

    expect(readFileSync(join(root, "compiler", "example.md"), "utf8")).toContain("Compiler worked example");
    expect(readFileSync(join(root, "lint", "example.md"), "utf8")).toContain("Lint worked example");

    expect(JSON.parse(readFileSync(join(root, "compiler", "proof-bundle", "verdict.json"), "utf8"))).toMatchObject({
      verdict: "reviewable",
      severity: "info",
    });
    expect(JSON.parse(readFileSync(join(root, "lint", "proof-bundle", "verdict.json"), "utf8"))).toMatchObject({
      verdict: "reviewable",
      severity: "info",
    });

    expect(result.manifest.lanes.compiler.proofBundleDigest.files["summary.md"]).toMatch(/^sha256:/);
    expect(result.manifest.lanes.lint.proofBundleDigest.files["summary.md"]).toMatch(/^sha256:/);
  });
});
