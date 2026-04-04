import { mkdirSync, readFileSync, rmSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildTeacherV3ProofBundle,
  TEACHER_V3_PROOF_BUNDLE_LAYOUT,
  writeTeacherV3ProofBundle,
} from "../scripts/teacher-v3-proof-bundle.mjs";

const runtimeStatus = {
  serveState: "serving_active_pack",
  currentPackVersion: 7,
  currentPackPromotedAt: "2026-04-03T18:00:00Z",
  currentPackMetadata: {
    reason: "current promoted pack",
    kind: "promoted_pack",
  },
  teacherConfigured: true,
  teacherProvider: "openai",
  teacherModel: "gpt-4.1",
  operatorHealth: {
    status: "healthy",
    detail: "runtime and proof surface are aligned",
  },
  learningHealth: {
    status: "healthy",
    detail: "learning loop is steady",
  },
  routeTraceCount: 12,
  supervisionCount: 4,
  recentTraceCount: 2,
  pendingLabels: 1,
  pendingObservations: 3,
  lastCompileReportSummary: "compile report is bounded",
  lastAssemblyDecision: {
    summary: "assembly summary",
    verdict: "approved",
  },
  lastPrefetchDecision: {
    summary: "prefetch summary",
    verdict: "ready",
  },
  lastPromotionReason: "current pack promoted",
  lastPromotionVerdict: {
    verdict: "promoted",
    summary: "promotion succeeded",
  },
  lastReplayFailureReason: null,
  lastReplayGateVerdict: {
    verdict: "pass",
    summary: "replay gate passed",
  },
};

const operatorProof = {
  bundleDir: "artifacts/operator-proof-20260403-182600Z",
  command: "openclawbrain proof --openclaw-home ~/.openclaw",
  summary: "operator proof summary",
  verdict: {
    verdict: "success_and_proven",
    severity: "info",
    why: "runtime truth and proof truth were aligned",
    missingProofs: [],
  },
  runtimeLoadProofPath: "~/.openclaw/activation/attachment-truth/runtime-load-proofs.json",
  runtimeLoadProofExists: true,
  stepCount: 5,
  postBundleCount: 2,
};

const docsTruth = {
  path: "docs/architecture/teacher-v3-proof.md",
  title: "Teacher v3 proposal reporting / proof surfaces",
  summary: "Design-only mapping of shipped truth to target-state proof surfaces.",
};

describe("teacher v3 proof bundle writer", () => {
  it("emits the bounded five-file bundle from real runtime/proof inputs", () => {
    const outputDir = path.join(process.cwd(), "scratch", "teacher-v3-proof-bundle-test");
    rmSync(outputDir, { recursive: true, force: true });
    mkdirSync(path.dirname(outputDir), { recursive: true });

    const bundle = buildTeacherV3ProofBundle({
      bundleStartedAt: "2026-04-03T18:26:00Z",
      outputDir,
      runtimeStatusCommand: "npx @openclawbrain/cli status --openclaw-home ~/.openclaw --detailed",
      runtimeStatus,
      operatorProofCommand: "openclawbrain proof --openclaw-home ~/.openclaw",
      operatorProof,
      docsTruth,
      producerVersion: "openclawbrain@0.3.8",
    });

    expect(Object.keys(bundle.files).sort()).toEqual([
      TEACHER_V3_PROOF_BUNDLE_LAYOUT.proposalReport,
      TEACHER_V3_PROOF_BUNDLE_LAYOUT.status,
      TEACHER_V3_PROOF_BUNDLE_LAYOUT.summary,
      TEACHER_V3_PROOF_BUNDLE_LAYOUT.surfaceMap,
      TEACHER_V3_PROOF_BUNDLE_LAYOUT.verdict,
    ].sort());
    expect(bundle.surfaceMap.counts).toMatchObject({
      observedSurfaceCount: 3,
      shippedSurfaceCount: 3,
      targetSurfaceCount: 5,
      totalSurfaceCount: 8,
    });
    expect(bundle.statusReport.recommendations).toHaveLength(3);
    expect(bundle.proposalReport.proposal).toMatchObject({
      proposalClass: "lint",
      proposalLane: "lint",
      status: "validated",
      reviewMode: "promotable",
      recordSource: "runtime-capture",
    });
    expect(bundle.proposalReport.gate1Seam).toMatchObject({
      present: false,
      recordSource: "runtime-capture",
    });
    expect(bundle.proposalReport.replayGate).toMatchObject({
      proposalClass: "lint",
      reviewMode: "promotable",
    });
    expect(Object.keys(bundle.proposalReport.replayGate.dimensions).sort()).toEqual([
      "attributionFloor",
      "boundedness",
      "reversibility",
      "truthInvariants",
    ]);
    expect(bundle.proposalReport.publicationSafeArtifacts).toHaveLength(5);
    expect(bundle.verdictReport).toMatchObject({
      verdict: "reviewable",
      severity: "info",
      targetStateOnly: true,
    });
    expect(bundle.summaryMarkdown).toContain("Teacher v3 proof bundle");
    expect(bundle.summaryMarkdown).toContain("Gate 1 seam");

    const writeResult = writeTeacherV3ProofBundle(outputDir, bundle);
    expect(writeResult.writtenFiles).toHaveLength(5);

    const summaryPath = path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.summary);
    const statusPath = path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.status);
    const surfaceMapPath = path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.surfaceMap);
    const proposalReportPath = path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.proposalReport);
    const verdictPath = path.join(outputDir, TEACHER_V3_PROOF_BUNDLE_LAYOUT.verdict);

    expect(() => JSON.parse(bundle.files[TEACHER_V3_PROOF_BUNDLE_LAYOUT.status])).not.toThrow();
    expect(() => JSON.parse(bundle.files[TEACHER_V3_PROOF_BUNDLE_LAYOUT.surfaceMap])).not.toThrow();
    expect(() => JSON.parse(bundle.files[TEACHER_V3_PROOF_BUNDLE_LAYOUT.proposalReport])).not.toThrow();
    expect(() => JSON.parse(bundle.files[TEACHER_V3_PROOF_BUNDLE_LAYOUT.verdict])).not.toThrow();

    expect(readFileSync(summaryPath, "utf8")).toContain("runtime truth");
    expect(readFileSync(statusPath, "utf8")).toContain("teacher_v3_proof_bundle_status.v1");
    expect(readFileSync(surfaceMapPath, "utf8")).toContain("runtime-truth");
    expect(readFileSync(proposalReportPath, "utf8")).toContain("teacher_v3_proposal_report.v1");
    expect(readFileSync(verdictPath, "utf8")).toContain("reviewable");
  });
});
